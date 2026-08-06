#!/usr/bin/env python3
"""Build the time-valid negative candidate table for sampled-candidate eval.

For each evaluation instance (user u, held-out positive streamer p, at time t),
sample N negative streamers that were LIVE at time t (a room with
start <= t <= end) and that u never interacted with. This reproduces the
KuaiLive paper's protocol (sec 5.1.2): rank the positive against a fixed pool
of time-valid negatives rather than the full 452k catalogue.

Output (per split) is an int32 array [num_instances, N] aligned row-for-row with
validation.txt / test.txt, saved into a single .npz. The table is seeded, so it
is deterministic and can be shared identically across MBGCN and ATRank.

Method: a sweep line over room start(+1)/end(-1) events with the eval timestamps
interleaved as queries. Tie order at equal timestamps is start < query < end, so
membership is inclusive [start, end]. This is O((rooms + queries) log(...))
instead of scanning 11.8M rooms per query.
"""
import argparse
import csv
import json
import os
import sys
import time
from array import array

import numpy as np


def log(m):
    print(m, file=sys.stderr, flush=True)


def load_rooms(path, id_base):
    """Parse room.csv -> (streamer0, start, end) int64 numpy arrays."""
    sid = array("q")
    start = array("q")
    end = array("q")
    with open(path, newline="") as f:
        r = csv.reader(f)
        h = next(r)
        ci_s, ci_start, ci_end = (h.index("streamer_id"),
                                  h.index("start_timestamp"),
                                  h.index("end_timestamp"))
        for row in r:
            sid.append(int(row[ci_s]) - id_base)
            start.append(int(row[ci_start]))
            end.append(int(row[ci_end]))
    return (np.frombuffer(sid, dtype=np.int64),
            np.frombuffer(start, dtype=np.int64),
            np.frombuffer(end, dtype=np.int64))


def load_meta(path):
    """Return users, items, timestamps (int64) in file order."""
    u, it, ts = array("q"), array("q"), array("q")
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            u.append(int(row[0]))
            it.append(int(row[1]))
            ts.append(int(row[3]))
    return (np.frombuffer(u, dtype=np.int64),
            np.frombuffer(it, dtype=np.int64),
            np.frombuffer(ts, dtype=np.int64))


def load_user_positives(train_path, val_users, val_items, test_users, test_items):
    """user -> python set of streamers to exclude from its negatives."""
    pos = {}
    with open(train_path) as f:
        for line in f:
            a, b = line.split()
            pos.setdefault(int(a), set()).add(int(b))
    for us, it in ((val_users, val_items), (test_users, test_items)):
        for u, i in zip(us.tolist(), it.tolist()):
            pos.setdefault(u, set()).add(i)
    return pos


def build(rooms, users, items, ts, user_pos, num_neg, seed):
    sid, start, end = rooms
    n = len(ts)
    R = len(sid)
    rng = np.random.default_rng(seed)

    # --- event stream: starts(+1, type0), ends(-1, type2), queries(type1) ----
    # sort key = time*4 + type keeps start < query < end at equal timestamps.
    ev_time = np.concatenate([start, end, ts])
    ev_type = np.concatenate([np.zeros(R, np.int64),
                              np.full(R, 2, np.int64),
                              np.ones(n, np.int64)])
    ev_payload = np.concatenate([sid, sid, np.arange(n, dtype=np.int64)])
    order = np.argsort(ev_time * 4 + ev_type, kind="stable")
    ev_type = ev_type[order]
    ev_payload = ev_payload[order]

    out = np.empty((n, num_neg), dtype=np.int32)
    active = {}                       # streamer -> overlapping-room count
    shortfalls = 0
    done = 0
    t0 = time.time()
    for etype, pay in zip(ev_type.tolist(), ev_payload.tolist()):
        if etype == 0:                # room start
            active[pay] = active.get(pay, 0) + 1
        elif etype == 2:              # room end
            c = active.get(pay, 0) - 1
            if c <= 0:
                active.pop(pay, None)
            else:
                active[pay] = c
        else:                         # query
            u = users[pay]
            excl = user_pos.get(u, ())
            keys = np.fromiter(active.keys(), np.int32, len(active))
            # oversample to survive exclusion removals, then trim to num_neg
            k = min(len(keys), num_neg + len(excl) + 64)
            draw = rng.choice(keys, size=k, replace=False)
            if excl:
                draw = draw[~np.isin(draw, np.fromiter(excl, np.int32, len(excl)))]
            if len(draw) >= num_neg:
                out[pay] = draw[:num_neg]
            else:                     # extremely rare: pad from live set
                shortfalls += 1
                pad = rng.choice(keys, size=num_neg, replace=True)
                pad[:len(draw)] = draw
                out[pay] = pad
            done += 1
            if done % 5000 == 0:
                log(f"    {done:,}/{n:,} queries  ({time.time()-t0:.0f}s)")
    return out, shortfalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="KuaiLive/room.csv")
    ap.add_argument("--data", default="dataset/click_interaction")
    ap.add_argument("--num-neg", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--id-base", type=int, default=1,
                    help="raw streamer id base (room.csv is 1-indexed)")
    ap.add_argument("--out", default=None,
                    help="output .npz (default <data>/candidates.npz)")
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.data, "candidates.npz")

    log("[1/4] loading rooms")
    t = time.time()
    rooms = load_rooms(args.rooms, args.id_base)
    log(f"      {len(rooms[0]):,} rooms in {time.time()-t:.0f}s")

    log("[2/4] loading eval meta")
    vu, vi, vts = load_meta(os.path.join(args.data, "validation_meta.csv"))
    tu, ti, tts = load_meta(os.path.join(args.data, "test_meta.csv"))
    log(f"      validation {len(vu):,}  test {len(tu):,}")

    log("[3/4] loading user positives (exclusions)")
    user_pos = load_user_positives(os.path.join(args.data, "train.txt"),
                                   vu, vi, tu, ti)

    # process both splits in one sweep, then slice back
    users = np.concatenate([vu, tu])
    items = np.concatenate([vi, ti])
    ts = np.concatenate([vts, tts])
    log(f"[4/4] sweeping {len(ts):,} instances x {args.num_neg} negatives")
    negs, shortfalls = build(rooms, users, items, ts, user_pos,
                             args.num_neg, args.seed)

    nval = len(vu)
    val_neg = negs[:nval]
    test_neg = negs[nval:]

    np.savez_compressed(
        out_path,
        val_neg=val_neg, val_user=vu, val_item=vi,
        test_neg=test_neg, test_user=tu, test_item=ti,
        num_neg=np.int64(args.num_neg), seed=np.int64(args.seed),
    )
    manifest = {
        "out": out_path, "num_neg": args.num_neg, "seed": args.seed,
        "validation_instances": int(nval), "test_instances": int(len(tu)),
        "shortfalls": int(shortfalls),
        "note": "row i aligns with row i of validation.txt / test.txt",
    }
    with open(os.path.join(args.data, "candidates_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"[done] saved {out_path}  (shortfalls={shortfalls})")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
