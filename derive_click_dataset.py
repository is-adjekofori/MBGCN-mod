#!/usr/bin/env python3
"""Derive an MBGCN-format dataset from the raw KuaiLive CSVs.

Target behaviour = CLICK, split with a leave-one-out (LOO) strategy, matching
the KuaiLive paper's top-K protocol (SIGIR '26, sec 5.1.2). See DATASET notes
in the thesis for the rationale (click is the abundant, recommender-actionable
signal; gift was the sparsest behaviour).

What it writes to <out>/ :

  data_size.txt          "<user_num> <item_num>"   (item = streamer)
  click.txt              training clicks, WITH duplicates  (graph -> intensity)
  train.txt              deduped training (u,i) click pairs (BPR positives)
  validation.txt         one line per eval user: their 2nd-last click
  test.txt               one line per eval user: their last click
  comment.txt like.txt gift.txt negative.txt
                         auxiliary behaviours, whole, WITH duplicates
  validation_meta.csv    user,item,live_id,timestamp   (for time-valid negs)
  test_meta.csv          user,item,live_id,timestamp
  manifest.json          all parameters + row counts for reproducibility

Design decisions (all overridable via CLI):
  * IDs in the raw CSVs are 1-indexed contiguous; we shift to 0-indexed to
    match the [user_num x item_num] matrices the model builds.
  * A user needs >= --min-eval-clicks clicks to be evaluated; otherwise all of
    their clicks go to train (so validation and test share the same user set).
  * click.txt keeps duplicate (u, streamer) events so the coalesced relation
    matrix carries interaction *intensity* into the behaviour weight alpha.
    train.txt is deduped so BPR treats each preference once.
  * Auxiliary behaviours are kept whole (not per-user time-filtered), matching
    the reference MBGCN implementation.

Nothing here loads a whole CSV into memory except the target's per-user click
lists; every other file is streamed line by line.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def read_data_size(src):
    with open(os.path.join(src, "data_size.txt")) as f:
        u, i = f.readline().strip().split()
    return int(u), int(i)


def col_index(header, name):
    return header.index(name)


def stream_csv(path):
    """Yield rows (list of str) from a CSV, header first."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        yield header
        for row in reader:
            yield row


def derive_target(src, target, user_shift, item_shift, user_num, item_num):
    """Load click events per user, LOO-split, return split dicts.

    Returns (train_events, val_line, test_line, val_meta, test_meta, stats)
    where train_events is a list of (u0, i0) WITH duplicates, val/test are lists
    of (u0, i0), and *_meta lists carry (u0, i0, live_id, timestamp).
    """
    path = os.path.join(src, target + ".csv")
    gen = stream_csv(path)
    header = next(gen)
    ui = col_index(header, "user_id")
    si = col_index(header, "streamer_id")
    li = col_index(header, "live_id")
    ti = col_index(header, "timestamp")

    # per user: list of (timestamp, item0, live_id)
    per_user = defaultdict(list)
    n_rows = 0
    n_bad = 0
    for row in gen:
        try:
            u = int(row[ui]) + user_shift
            s = int(row[si]) + item_shift
            ts = int(row[ti])
            live = row[li]
        except (ValueError, IndexError):
            n_bad += 1
            continue
        if not (0 <= u < user_num and 0 <= s < item_num):
            n_bad += 1
            continue
        per_user[u].append((ts, s, live))
        n_rows += 1
    log(f"  read {n_rows:,} {target} events ({n_bad:,} skipped) "
        f"over {len(per_user):,} users")
    return per_user, n_rows, n_bad


def stream_behaviour(src, name, user_shift, item_shift, user_num, item_num,
                     out_path):
    """Stream an auxiliary behaviour CSV -> '<u0> <i0>' lines, dups preserved."""
    path = os.path.join(src, name + ".csv")
    gen = stream_csv(path)
    header = next(gen)
    ui = col_index(header, "user_id")
    si = col_index(header, "streamer_id")
    n = 0
    n_bad = 0
    with open(out_path, "w") as out:
        for row in gen:
            try:
                u = int(row[ui]) + user_shift
                s = int(row[si]) + item_shift
            except (ValueError, IndexError):
                n_bad += 1
                continue
            if not (0 <= u < user_num and 0 <= s < item_num):
                n_bad += 1
                continue
            out.write(f"{u} {s}\n")
            n += 1
    log(f"  wrote {n:,} {name} edges ({n_bad:,} skipped) -> {out_path}")
    return n, n_bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="KuaiLive", help="raw CSV directory")
    ap.add_argument("--out", default="dataset/click_interaction",
                    help="output dataset directory")
    ap.add_argument("--target", default="click", help="target behaviour")
    ap.add_argument("--aux", default="comment,like,gift,negative",
                    help="comma-separated auxiliary behaviours")
    ap.add_argument("--min-eval-clicks", type=int, default=3,
                    help="min target events for a user to be in val AND test")
    ap.add_argument("--id-base", type=int, default=1,
                    help="raw ID base (1 => shift to 0-indexed)")
    args = ap.parse_args()

    user_num, item_num = read_data_size(args.src)
    shift = -args.id_base
    os.makedirs(args.out, exist_ok=True)
    log(f"users={user_num:,} items={item_num:,}  out={args.out}")

    # ---- target: load, LOO split -----------------------------------------
    log(f"[target] {args.target}: loading + leave-one-out split")
    per_user, n_target_rows, n_target_bad = derive_target(
        args.src, args.target, shift, shift, user_num, item_num)

    train_dupe = 0          # click.txt lines (with duplicates)
    train_pairs = set()     # deduped (u,i) for train.txt / BPR
    val_lines, test_lines = [], []
    val_meta, test_meta = [], []
    n_eval_users = 0
    n_train_only_users = 0

    click_txt = open(os.path.join(args.out, args.target + ".txt"), "w")
    for u, events in per_user.items():
        events.sort(key=lambda e: e[0])         # ascending by timestamp
        if len(events) >= args.min_eval_clicks:
            test_ev = events[-1]
            val_ev = events[-2]
            train_ev = events[:-2]
            test_lines.append((u, test_ev[1]))
            val_lines.append((u, val_ev[1]))
            test_meta.append((u, test_ev[1], test_ev[2], test_ev[0]))
            val_meta.append((u, val_ev[1], val_ev[2], val_ev[0]))
            n_eval_users += 1
        else:
            train_ev = events                    # everything to train
            n_train_only_users += 1
        for ts, s, live in train_ev:
            click_txt.write(f"{u} {s}\n")        # duplicates preserved
            train_dupe += 1
            train_pairs.add((u, s))
    click_txt.close()
    log(f"  eval users (>= {args.min_eval_clicks} clicks): {n_eval_users:,}; "
        f"train-only users: {n_train_only_users:,}")
    log(f"  {args.target}.txt training events (with dups): {train_dupe:,}")
    log(f"  train.txt deduped (u,i) pairs: {len(train_pairs):,}")

    # ---- train.txt (deduped BPR positives) -------------------------------
    with open(os.path.join(args.out, "train.txt"), "w") as f:
        for u, s in sorted(train_pairs):
            f.write(f"{u} {s}\n")

    # ---- validation.txt / test.txt + meta --------------------------------
    def write_split(name, lines, meta):
        with open(os.path.join(args.out, name + ".txt"), "w") as f:
            for u, s in lines:
                f.write(f"{u} {s}\n")
        with open(os.path.join(args.out, name + "_meta.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["user", "item", "live_id", "timestamp"])
            w.writerows(meta)
    write_split("validation", val_lines, val_meta)
    write_split("test", test_lines, test_meta)
    log(f"  validation.txt: {len(val_lines):,}  test.txt: {len(test_lines):,}")

    # ---- auxiliary behaviours (whole, dups preserved) --------------------
    aux = [a for a in args.aux.split(",") if a]
    aux_counts = {}
    for name in aux:
        log(f"[aux] {name}")
        n, bad = stream_behaviour(args.src, name, shift, shift,
                                  user_num, item_num,
                                  os.path.join(args.out, name + ".txt"))
        aux_counts[name] = {"edges": n, "skipped": bad}

    # ---- data_size.txt + manifest ----------------------------------------
    with open(os.path.join(args.out, "data_size.txt"), "w") as f:
        f.write(f"{user_num} {item_num}\n")

    manifest = {
        "src": args.src,
        "user_num": user_num,
        "item_num": item_num,
        "target": args.target,
        "aux": aux,
        "id_base": args.id_base,
        "min_eval_clicks": args.min_eval_clicks,
        "split": "leave-one-out (last=test, 2nd-last=validation)",
        "target_raw_events": n_target_rows,
        "target_skipped": n_target_bad,
        "click_graph_events_with_dups": train_dupe,
        "train_pairs_deduped": len(train_pairs),
        "eval_users": n_eval_users,
        "train_only_users": n_train_only_users,
        "validation_instances": len(val_lines),
        "test_instances": len(test_lines),
        "aux_counts": aux_counts,
        "notes": {
            "target_txt": "click.txt keeps duplicates -> intensity in alpha",
            "train_txt": "deduped (u,i) BPR positives",
            "aux": "kept whole, not per-user time-filtered",
            "ids": "shifted to 0-indexed",
        },
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log("[done] wrote manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
