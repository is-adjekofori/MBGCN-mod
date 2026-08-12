#!/usr/bin/env python3
"""Derive a per-user chronological SEQUENCE view for ATRank.

The graph-shaped ``dataset/click_interaction/`` files (train/val/test.txt, the
per-behaviour .txt matrices, candidates.npz) drop timestamps, but ATRank needs a
temporally ordered, multi-behaviour event log per user. This script rebuilds that
log from the raw KuaiLive CSVs, aligned *byte-for-byte* with the click dataset so
the two models compare on the same problem:

  * identical id space   -> same --id-base shift to 0-indexed (item = streamer)
  * identical eval users + boundaries -> read from validation_meta.csv /
    test_meta.csv (which derive_click_dataset.py already wrote), NOT recomputed,
    so candidates.npz rows still line up.

What it writes to <out>/sequences.npz  (uncompressed, so it can be mmap'd):

  offsets      int64 [user_num + 1]   CSR pointers: user u's events are the slice
                                       [offsets[u] : offsets[u+1]] of the ev_* arrays
  ev_streamer  int32 [N]              streamer id (0-indexed) of each event
  ev_ts        int64 [N]              event timestamp (raw ms epoch)
  ev_action    int8  [N]              action id: 0=click,1=comment,2=like,3=gift,4=negative
  val_ts       int64 [user_num]       validation-target timestamp per user (-1 = not an eval user)
  test_ts      int64 [user_num]       test-target timestamp per user (-1 = not an eval user)
  val_item     int32 [user_num]       validation-target streamer per user (-1 = none)
  test_item    int32 [user_num]       test-target streamer per user (-1 = none)

Events are sorted by (user, timestamp) so each user's slice is already chronological.

IMPORTANT — this script only records the *complete* event log + the held-out
boundaries. It does NOT carve histories or drop held-out clicks; that logic lives
in the ATRank dataset loader, where the fairness rule is applied per target.

Held-out clicks MUST be identified BY POSITION, not by timestamp: for each eval
user (val_ts >= 0), the held-out val/test clicks are the *last two click events*
in their chronological slice (== derive_click_dataset.py's events[-2], events[-1]).
A timestamp rule (ts < val_ts) is WRONG: 3,454 eval users have val_ts == test_ts
(concurrent clicks), and a ts rule would also drop training clicks tied at that
instant. The position rule reproduces train.txt exactly (4,862,719 clicks with
dups; 2,763,921 deduped; 46,796 held out). Because clicks are loaded first and
lexsort is stable, each user's click subsequence preserves click.csv tie-order,
so "last two clicks" matches the graph split byte-for-byte.

Given a target click at chronological position p in a user's slice, the loader's
history is: all events before p, MINUS the two held-out click positions. Auxiliary
events are thus taken causally (only those before the target) — the intrinsic
sequential/graph difference vs MBGCN, which aggregates the whole auxiliary graph.

Memory: the machine is small (~3 GB free), so everything is done with
preallocated / concatenated NumPy arrays (int32/int64/int8), never Python lists
of tuples. Peak footprint ~1 GB for the full 18M-event log.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def read_data_size(src):
    with open(os.path.join(src, "data_size.txt")) as f:
        u, i = f.readline().strip().split()
    return int(u), int(i)


def load_behaviour(path, action_id, shift, user_num, item_num, chunksize=2_000_000):
    """Stream one behaviour CSV -> (user, streamer, ts, action) int arrays.

    Reads in chunks with explicit dtypes to keep the parser's peak memory bounded,
    shifts ids to 0-indexed, and drops any row that falls outside the id range.
    """
    u_parts, s_parts, t_parts = [], [], []
    n_bad = 0
    reader = pd.read_csv(
        path,
        usecols=["user_id", "streamer_id", "timestamp"],
        dtype={"user_id": "int64", "streamer_id": "int64", "timestamp": "int64"},
        chunksize=chunksize,
    )
    for chunk in reader:
        u = chunk["user_id"].to_numpy() + shift
        s = chunk["streamer_id"].to_numpy() + shift
        t = chunk["timestamp"].to_numpy()
        ok = (u >= 0) & (u < user_num) & (s >= 0) & (s < item_num)
        n_bad += int((~ok).sum())
        u_parts.append(u[ok].astype(np.int32, copy=False))
        s_parts.append(s[ok].astype(np.int32, copy=False))
        t_parts.append(t[ok])
    u = np.concatenate(u_parts) if u_parts else np.empty(0, np.int32)
    s = np.concatenate(s_parts) if s_parts else np.empty(0, np.int32)
    t = np.concatenate(t_parts) if t_parts else np.empty(0, np.int64)
    a = np.full(u.shape[0], action_id, dtype=np.int8)
    return u, s, t, a, n_bad


def load_meta(out_dir, name, user_num):
    """Read <name>_meta.csv (user,item,live_id,timestamp) -> dense per-user arrays."""
    df = pd.read_csv(
        os.path.join(out_dir, name + "_meta.csv"),
        usecols=["user", "item", "timestamp"],
        dtype={"user": "int64", "item": "int64", "timestamp": "int64"},
    )
    ts = np.full(user_num, -1, dtype=np.int64)
    item = np.full(user_num, -1, dtype=np.int32)
    u = df["user"].to_numpy()
    ts[u] = df["timestamp"].to_numpy()
    item[u] = df["item"].to_numpy().astype(np.int32)
    return ts, item, len(df)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", default="KuaiLive", help="raw CSV directory")
    ap.add_argument("--out", default="dataset/click_interaction",
                    help="dataset dir (holds *_meta.csv; sequences.npz written here)")
    ap.add_argument("--target", default="click", help="target behaviour (action id 0)")
    ap.add_argument("--aux", default="comment,like,gift,negative",
                    help="comma-separated auxiliary behaviours (ids 1..)")
    ap.add_argument("--id-base", type=int, default=1,
                    help="raw id base (1 => shift to 0-indexed), must match derive_click_dataset.py")
    args = ap.parse_args()

    user_num, item_num = read_data_size(args.src)
    shift = -args.id_base
    behaviours = [args.target] + [a for a in args.aux.split(",") if a]
    action_map = {b: i for i, b in enumerate(behaviours)}
    log(f"users={user_num:,} items={item_num:,}")
    log(f"actions: {action_map}")

    # ---- load every behaviour into flat arrays --------------------------------
    u_all, s_all, t_all, a_all = [], [], [], []
    counts = {}
    for b in behaviours:
        path = os.path.join(args.src, b + ".csv")
        log(f"[load] {b} ({os.path.getsize(path) / 1e6:.0f} MB)")
        u, s, t, a, bad = load_behaviour(path, action_map[b], shift, user_num, item_num)
        u_all.append(u); s_all.append(s); t_all.append(t); a_all.append(a)
        counts[b] = {"events": int(u.shape[0]), "skipped": bad}
        log(f"       {u.shape[0]:,} events ({bad:,} skipped)")

    ev_user = np.concatenate(u_all); del u_all
    ev_streamer = np.concatenate(s_all); del s_all
    ev_ts = np.concatenate(t_all); del t_all
    ev_action = np.concatenate(a_all); del a_all
    total = ev_user.shape[0]
    log(f"[merge] {total:,} total events")

    # ---- sort by (user, timestamp) so each user's slice is chronological ------
    log("[sort] lexsort by (user, ts)")
    order = np.lexsort((ev_ts, ev_user))          # primary user, secondary ts
    ev_streamer = ev_streamer[order]
    ev_ts = ev_ts[order]
    ev_action = ev_action[order]
    ev_user = ev_user[order]
    del order

    # ---- CSR offsets over the full 0..user_num-1 range ------------------------
    counts_per_user = np.bincount(ev_user, minlength=user_num)
    offsets = np.zeros(user_num + 1, dtype=np.int64)
    np.cumsum(counts_per_user, out=offsets[1:])
    del ev_user, counts_per_user

    # ---- held-out boundaries (reuse derive_click_dataset.py's split) ----------
    val_ts, val_item, n_val = load_meta(args.out, "validation", user_num)
    test_ts, test_item, n_test = load_meta(args.out, "test", user_num)
    log(f"[bounds] eval users: val={n_val:,} test={n_test:,}")

    # ---- save -----------------------------------------------------------------
    out_path = os.path.join(args.out, "sequences.npz")
    np.savez(
        out_path,
        offsets=offsets,
        ev_streamer=ev_streamer,
        ev_ts=ev_ts,
        ev_action=ev_action,
        val_ts=val_ts,
        test_ts=test_ts,
        val_item=val_item,
        test_item=test_item,
    )
    log(f"[done] wrote {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB)")

    manifest = {
        "src": args.src,
        "out": out_path,
        "user_num": user_num,
        "item_num": item_num,
        "id_base": args.id_base,
        "action_map": action_map,
        "total_events": int(total),
        "per_behaviour": counts,
        "eval_users_val": int(n_val),
        "eval_users_test": int(n_test),
        "notes": {
            "sorted_by": "(user, timestamp) ascending",
            "carving": "history/held-out-click removal applied in the ATRank "
                       "dataset loader, NOT here",
            "alignment": "boundaries read from *_meta.csv so candidates.npz rows "
                         "still align by row index",
        },
    }
    with open(os.path.join(args.out, "sequences_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
