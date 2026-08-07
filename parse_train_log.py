#!/usr/bin/env python3
"""Extract per-validation metrics from a saved training stdout log into a CSV.

The training loop prints one block of ``<Metric>:<value>`` lines per validation
(e.g. ``Recall10:0.4755``), and a ``New Record! <v> @ epoch <N>!`` line on
record-breaking validations. This scrapes those into a tidy CSV you can chart.

Use it for a run that finished BEFORE the CSV logging was added (i.e. whose only
record is the notebook/cell output). Save that output to a file first, e.g. copy
it into ``run.log``, then:

    python parse_train_log.py run.log --out metrics_from_log.csv --eval-every 5

Columns: validation_index, epoch, then every metric (Recall5, NDCG5, ... ).
``epoch`` comes from the "@ epoch N" line when present, otherwise it is inferred
as ``(validation_index) * eval_every + (eval_every - 1)`` (0-based epochs, matching
the trainer's ``(epoch+1) % eval_every == 0`` schedule).
"""
import argparse
import csv
import re
import sys

METRIC_RE = re.compile(r"^([A-Za-z]+\d+):(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")
EPOCH_RE = re.compile(r"@ epoch (\d+)")


def parse(path):
    blocks = []          # list of (metrics_dict, epoch_or_None)
    cur = {}
    cur_epoch = None
    with open(path) as f:
        for line in f:
            m = METRIC_RE.match(line.strip())
            if m:
                name, val = m.group(1), float(m.group(2))
                if name in cur:                      # repeat -> new validation
                    blocks.append((cur, cur_epoch))
                    cur, cur_epoch = {}, None
                cur[name] = val
                continue
            e = EPOCH_RE.search(line)
            if e and cur:                            # tag current block's epoch
                cur_epoch = int(e.group(1))
    if cur:
        blocks.append((cur, cur_epoch))
    return blocks


def order_metrics(names):
    def key(n):
        m = re.match(r"([A-Za-z]+)(\d+)", n)
        kind, k = m.group(1), int(m.group(2))
        rank = {"Recall": 0, "NDCG": 1, "MRR": 2}.get(kind, 3)
        return (k, rank, n)
    return sorted(names, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--out", default="metrics_from_log.csv")
    ap.add_argument("--eval-every", type=int, default=5,
                    help="validation cadence, to infer epoch when not printed")
    args = ap.parse_args()

    blocks = parse(args.logfile)
    if not blocks:
        print("No validation blocks found. Is this the right log file?",
              file=sys.stderr)
        sys.exit(1)

    metric_names = order_metrics({n for b, _ in blocks for n in b})
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["validation_index", "epoch"] + metric_names)
        for i, (metrics, epoch) in enumerate(blocks):
            if epoch is None:
                epoch = i * args.eval_every + (args.eval_every - 1)
            w.writerow([i, epoch] + [metrics.get(n, "") for n in metric_names])

    print(f"Parsed {len(blocks)} validations -> {args.out}")
    print("metrics:", ", ".join(metric_names))
    last, le = blocks[-1]
    print(f"last validation (epoch~{le if le is not None else '?'}): "
          + ", ".join(f"{n}={last.get(n)}" for n in metric_names))


if __name__ == "__main__":
    main()
