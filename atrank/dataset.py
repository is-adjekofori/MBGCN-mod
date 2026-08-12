"""Sequence datasets for ATRank on KuaiLive.

Reads dataset/click_interaction/sequences.npz (built by derive_sequence_dataset.py)
and dataset/click_interaction/candidates.npz (the shared eval pools). Carving rules
implement the fair-comparison contract:

  * Held-out val/test clicks are the LAST TWO click events per eval user (by
    position, not timestamp) -> matches MBGCN's train.txt byte-for-byte.
  * History for a target = the user's events strictly before the target's slice
    position, with held-out clicks removed. Auxiliary events are thus causal
    (only those before the target) -- the intrinsic sequential vs graph
    difference against MBGCN, which aggregates the whole auxiliary graph.
  * Training positives = every non-held-out click (all-position, with dups),
    matching ATRank's native next-item training and MBGCN's positive set.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Exponential elapsed-time buckets, in MINUTES. KuaiLive spans ~21 days, so the
# authors' day-granularity buckets collapse; minutes give fine resolution on the
# recent tail and coarse resolution across the multi-day bulk. 15 thresholds ->
# bucket ids 0..15 -> one-hot depth 16.
TIME_GAPS = np.array(
    [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    dtype=np.int64,
)
N_TIME_BUCKETS = len(TIME_GAPS) + 1  # 16
N_ACTIONS = 5                        # click, comment, like, gift, negative


def bucketize_minutes(elapsed_min):
    """elapsed (minutes) -> bucket id in [0, 15] = # thresholds met/exceeded."""
    return np.searchsorted(TIME_GAPS, elapsed_min, side="right")


class _SeqStore:
    """Loads sequences.npz once and exposes per-user event slices + boundaries."""

    def __init__(self, path, allowed_actions=None):
        d = np.load(os.path.join(path, "sequences.npz"))
        self.offsets = d["offsets"]
        self.ev_streamer = d["ev_streamer"]
        self.ev_ts = d["ev_ts"]
        self.ev_action = d["ev_action"]
        self.val_ts = d["val_ts"]
        self.test_ts = d["test_ts"]
        self.val_item = d["val_item"]
        self.test_item = d["test_item"]
        d.close()
        self.user_num = self.offsets.shape[0] - 1
        # actions kept in the *history* (target is always a click). Default: all.
        self.allowed = (
            None if allowed_actions is None else np.asarray(sorted(allowed_actions))
        )
        self._locate_clicks()

    def _locate_clicks(self):
        """Per user, record click positions; derive held-out + train targets.

        held_val_pos / held_test_pos: slice-local index of the 2nd-last / last
        click for eval users (-1 otherwise). train_user / train_pos: flat list of
        (user, slice-local target position) for every non-held-out click that has
        a non-empty history.
        """
        off = self.offsets
        is_eval = self.val_ts >= 0
        self.held_val_pos = np.full(self.user_num, -1, dtype=np.int64)
        self.held_test_pos = np.full(self.user_num, -1, dtype=np.int64)
        tu, tp = [], []
        for u in range(self.user_num):
            s, e = off[u], off[u + 1]
            cpos = np.nonzero(self.ev_action[s:e] == 0)[0]  # click positions (local)
            if is_eval[u]:
                if cpos.shape[0] < 2:
                    continue
                self.held_val_pos[u] = cpos[-2]
                self.held_test_pos[u] = cpos[-1]
                targets = cpos[:-2]
            else:
                targets = cpos
            for p in targets:
                if p >= 1:                     # need a non-empty prefix
                    tu.append(u)
                    tp.append(int(p))
        self.train_user = np.asarray(tu, dtype=np.int64)
        self.train_pos = np.asarray(tp, dtype=np.int64)

    def history(self, u, target_pos, target_ts, max_len, drop_pos=None,
                per_action_caps=None):
        """Return (streamer, action, timebucket) arrays for u's carved history.

        Events are the user's slice [0:target_pos], minus any index in drop_pos
        (used to remove the held-out val click from a test history), optionally
        filtered to allowed actions, then truncated.

        Truncation mode:
          * per_action_caps=None -> global recency window: keep the most recent
            `max_len` events of any type (headline / vanilla ATRank).
          * per_action_caps={action_id: cap} -> keep the most recent `cap` events
            of EACH action independently (cap<=0 drops that action entirely;
            actions absent from the dict are kept in full). This decouples click
            depth from negative volume, so an all-vs-no-negative ablation isn't
            confounded by how much click history the window happens to hold.
        Both modes preserve chronological order.
        """
        s = self.offsets[u]
        idx = np.arange(target_pos)
        if drop_pos is not None:
            idx = idx[~np.isin(idx, drop_pos)]
        strm = self.ev_streamer[s + idx]
        act = self.ev_action[s + idx]
        ts = self.ev_ts[s + idx]
        if self.allowed is not None:
            keep = np.isin(act, self.allowed)
            strm, act, ts = strm[keep], act[keep], ts[keep]
        if per_action_caps is not None:
            keepm = np.zeros(strm.shape[0], dtype=bool)
            for a in np.unique(act):
                pos_a = np.nonzero(act == a)[0]
                cap = per_action_caps.get(int(a), None)
                if cap is None:                 # unspecified action -> keep all
                    keepm[pos_a] = True
                elif cap > 0:                    # keep most recent `cap`
                    keepm[pos_a[-cap:]] = True
                # cap <= 0 -> drop every event of this action
            strm, act, ts = strm[keepm], act[keepm], ts[keepm]
        elif strm.shape[0] > max_len:
            strm, act, ts = strm[-max_len:], act[-max_len:], ts[-max_len:]
        elapsed_min = np.maximum(0, (target_ts - ts)) // 60000
        tb = bucketize_minutes(elapsed_min).astype(np.int64)
        return strm.astype(np.int64), act.astype(np.int64), tb


class ATRankTrainDataset(Dataset):
    """One item = (user, carved history, positive click, uniform negative)."""

    def __init__(self, store: _SeqStore, item_num, max_len=200,
                 per_action_caps=None):
        self.store = store
        self.item_num = item_num
        self.max_len = max_len
        self.caps = per_action_caps
        self.train_user = store.train_user
        self.train_pos = store.train_pos
        self.newit()

    def newit(self):
        """Resample one uniform negative per training pair (matches MBGCN)."""
        n = self.train_user.shape[0]
        self.neg = np.random.randint(0, self.item_num, size=n).astype(np.int64)

    def __len__(self):
        return self.train_user.shape[0]

    def __getitem__(self, i):
        u = int(self.train_user[i])
        p = int(self.train_pos[i])
        s = self.store.offsets[u]
        target_ts = int(self.store.ev_ts[s + p])
        pos = int(self.store.ev_streamer[s + p])
        strm, act, tb = self.store.history(
            u, p, target_ts, self.max_len, per_action_caps=self.caps)
        return {
            "hist_streamer": strm,
            "hist_action": act,
            "hist_time": tb,
            "pos": pos,
            "neg": int(self.neg[i]),
        }


class ATRankTestDataset(Dataset):
    """One item = (user, [pos, negatives...] candidates, carved history).

    Reuses the shared candidates.npz so ranking is over the identical pools as
    MBGCN. History is carved at the held-out boundary; for the test target the
    val click is removed so the click information set matches MBGCN's train set.
    """

    def __init__(self, store: _SeqStore, path, task="test", max_len=200,
                 per_action_caps=None):
        self.store = store
        self.max_len = max_len
        self.caps = per_action_caps
        self.task = task
        d = np.load(os.path.join(path, "candidates.npz"))
        pre = "val" if task == "validation" else "test"
        self.users = d[pre + "_user"].astype(np.int64)
        self.pos = d[pre + "_item"].astype(np.int64)
        self.neg = d[pre + "_neg"]                 # [N, num_neg] int32
        self.num_neg = int(d["num_neg"])
        d.close()

    def __len__(self):
        return self.users.shape[0]

    def __getitem__(self, i):
        u = int(self.users[i])
        cand = np.empty(1 + self.num_neg, dtype=np.int64)
        cand[0] = self.pos[i]                      # positive at column 0
        cand[1:] = self.neg[i]
        if self.task == "validation":
            tpos = int(self.store.held_val_pos[u])
            tts = int(self.store.val_ts[u])
            drop = None
        else:
            tpos = int(self.store.held_test_pos[u])
            tts = int(self.store.test_ts[u])
            drop = np.array([self.store.held_val_pos[u]])  # exclude val click
        strm, act, tb = self.store.history(
            u, tpos, tts, self.max_len, drop_pos=drop, per_action_caps=self.caps)
        return {
            "user": u,
            "cand": cand,
            "hist_streamer": strm,
            "hist_action": act,
            "hist_time": tb,
        }


def _pad_histories(items):
    """Pad variable-length histories to the batch max. Returns tensors + mask."""
    lens = [max(1, it["hist_streamer"].shape[0]) for it in items]
    L = max(lens)
    B = len(items)
    strm = np.zeros((B, L), np.int64)
    act = np.zeros((B, L), np.int64)
    tb = np.zeros((B, L), np.int64)
    mask = np.zeros((B, L), np.bool_)
    for b, it in enumerate(items):
        n = it["hist_streamer"].shape[0]
        if n > 0:
            strm[b, :n] = it["hist_streamer"]
            act[b, :n] = it["hist_action"]
            tb[b, :n] = it["hist_time"]
            mask[b, :n] = True
        else:
            mask[b, 0] = True   # degenerate: keep one padded (zero) valid slot
    return (
        torch.from_numpy(strm),
        torch.from_numpy(act),
        torch.from_numpy(tb),
        torch.from_numpy(mask),
    )


def train_collate(items):
    strm, act, tb, mask = _pad_histories(items)
    pos = torch.tensor([it["pos"] for it in items], dtype=torch.long)
    neg = torch.tensor([it["neg"] for it in items], dtype=torch.long)
    return strm, act, tb, mask, pos, neg


def test_collate(items):
    strm, act, tb, mask = _pad_histories(items)
    users = torch.tensor([it["user"] for it in items], dtype=torch.long)
    cand = torch.from_numpy(np.stack([it["cand"] for it in items], axis=0))
    return users, cand, strm, act, tb, mask
