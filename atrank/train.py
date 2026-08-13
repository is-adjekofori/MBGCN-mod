"""Train / evaluate ATRank on KuaiLive under the MBGCN fair-comparison contract.

Reuses loss.bprloss (identical BPR pairwise loss + L2 scaling), the
utils.EarlyStopManager (identical early-stopping rule), and
metrics.Sampled{Recall,NDCG,MRR} (identical sampled ranking metrics) so the only
thing that differs from the MBGCN run is the architecture.

Protocol matched to MBGCN: validate every --eval_every epochs; early-stop on val
Recall@10 with --es_patience (grace period epoch<=10); cap at --epochs. A full
checkpoint is written EVERY validation cycle (not only on a new best), so an
interrupted Colab session resumes from where it stopped, not just the last best.

Real training is meant for a GPU (Colab), like MBGCN; on CPU use --smoke to
sanity-check the full path on a tiny subset.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss import bprloss                                    # noqa: E402
from metrics import SampledRecall, SampledNDCG, SampledMRR  # noqa: E402
from utils import EarlyStopManager                          # noqa: E402
from atrank.dataset import (                                # noqa: E402
    _SeqStore, ATRankTrainDataset, ATRankTestDataset, train_collate, test_collate,
)
from atrank.model import ATRank                             # noqa: E402


def build_metrics():
    # Recall@10 first -> it is the leaderboard / early-stopping metric (as in MBGCN).
    return {
        "Recall10": SampledRecall(10), "NDCG10": SampledNDCG(10), "MRR10": SampledMRR(10),
        "Recall20": SampledRecall(20), "NDCG20": SampledNDCG(20), "MRR20": SampledMRR(20),
        "Recall5": SampledRecall(5), "NDCG5": SampledNDCG(5), "MRR5": SampledMRR(5),
    }


def read_data_size(path, name):
    with open(os.path.join(path, name, "data_size.txt")) as f:
        u, i = f.readline().strip().split()
    return int(u), int(i)


def append_csv(path, header, row):
    """Append one row, writing the header first if the file is new."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def prune_csv(path, start_epoch):
    """On resume, drop rows with epoch >= start_epoch so we don't duplicate them
    (mirrors MBGCN's VisManager.resume_from)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header, data = rows[0], rows[1:]
    ei = header.index("epoch")
    kept = [r for r in data if r and int(r[ei]) < start_epoch]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)


def parse_caps(spec):
    """'0:50,1:30,2:30,3:30,4:0' -> {0:50,1:30,2:30,3:30,4:0}; '' -> None."""
    spec = (spec or "").strip()
    if not spec:
        return None
    caps = {}
    for part in spec.split(","):
        a, c = part.split(":")
        caps[int(a)] = int(c)
    return caps


@torch.no_grad()
def evaluate(model, loader, device, metrics, cand_chunk, tag="val"):
    model.eval()
    for m in metrics.values():
        m.start()
    for users, cand, strm, act, tb, mask in tqdm(loader, desc=tag, dynamic_ncols=True):
        scores = model.score_candidates(
            strm.to(device), act.to(device), tb.to(device), mask.to(device),
            cand.to(device), chunk=cand_chunk,
        )
        pos_score = scores[:, 0:1]
        rank = (scores[:, 1:] > pos_score).sum(dim=1) + 1     # strict >, ties -> pos
        for m in metrics.values():
            m(rank.cpu())
    for m in metrics.values():
        m.stop()
    return {k: v._metric for k, v in metrics.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="dataset")
    ap.add_argument("--dataset_name", default="click_interaction")
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--behaviors", default="0,1,2,3,4",
                    help="action ids kept in history (0=click..4=negative)")
    ap.add_argument("--per_action_caps", default="",
                    help="per-behaviour caps 'aid:cap,...' e.g. 0:50,1:30,2:30,3:30,4:50; "
                         "empty => global max_len (headline). cap 0 drops that behaviour.")
    ap.add_argument("--id_dim", type=int, default=64)
    ap.add_argument("--action_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_blocks", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--reg", type=float, default=5e-5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=400, help="epoch cap (early stop ends it sooner)")
    ap.add_argument("--eval_every", type=int, default=5, help="validate every N epochs")
    ap.add_argument("--es_patience", type=int, default=10,
                    help="stop after this many validations with no new best (grace: epoch<=10)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--test_batch_size", type=int, default=64)
    ap.add_argument("--cand_chunk", type=int, default=1024)
    ap.add_argument("--log_every", type=int, default=200,
                    help="update the tqdm loss postfix every N batches (0=off; throttles GPU sync)")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrain_path", default="",
                    help="dir with MF model.pkl -> warm-start item_emb (init parity with MBGCN)")
    ap.add_argument("--save", default="output/click_interaction/atrank")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <save>.ckpt if present (full state)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU sanity run: few train batches + few eval users")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if (args.gpu and torch.cuda.is_available()) else "cpu")
    print(f"device={device}  args={vars(args)}", flush=True)

    user_num, item_num = read_data_size(args.path, args.dataset_name)
    dpath = os.path.join(args.path, args.dataset_name)
    allowed = [int(x) for x in args.behaviors.split(",") if x != ""]
    caps = parse_caps(args.per_action_caps)
    store = _SeqStore(dpath, allowed_actions=allowed)
    print(f"users={user_num:,} items={item_num:,} train_targets={len(store.train_user):,} "
          f"caps={caps}", flush=True)

    trainset = ATRankTrainDataset(store, item_num, max_len=args.max_len, per_action_caps=caps)
    valset = ATRankTestDataset(store, dpath, task="validation", max_len=args.max_len,
                               per_action_caps=caps)
    testset = ATRankTestDataset(store, dpath, task="test", max_len=args.max_len,
                                per_action_caps=caps)

    if args.smoke:
        trainset = Subset(trainset, list(range(min(2048, len(trainset)))))
        valset = Subset(valset, list(range(min(128, len(valset)))))
        testset = Subset(testset, list(range(min(128, len(testset)))))

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_collate, num_workers=args.num_workers)
    val_loader = DataLoader(valset, batch_size=args.test_batch_size, shuffle=False,
                            collate_fn=test_collate, num_workers=args.num_workers)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                             collate_fn=test_collate, num_workers=args.num_workers)

    model = ATRank(item_num, id_dim=args.id_dim, action_dim=args.action_dim,
                   hidden=args.hidden, num_heads=args.num_heads,
                   num_blocks=args.num_blocks, dropout=args.dropout,
                   reg=args.reg).to(device)
    if args.pretrain_path:
        model.load_mf_item(args.pretrain_path)
        print(f"warm-started item_emb from {args.pretrain_path}/model.pkl", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    metrics = build_metrics()
    es = EarlyStopManager(args)          # identical rule to MBGCN (es_patience, grace<=10)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    ckpt_path = args.save + ".ckpt"
    best_path = args.save + ".pkl"       # best-validation model, loaded for final test
    metrics_csv = args.save + ".metrics.csv"   # wide: epoch + every metric (per validation)
    scalars_csv = args.save + ".scalars.csv"   # long: epoch,name,value (metrics + loss)

    best, best_epoch, start_epoch = -1.0, -1, 0

    # ---- resume from a mid-run checkpoint (full state) -----------------------
    if args.resume and os.path.exists(ckpt_path):
        # our own checkpoint (contains numpy RNG state) -> full unpickle
        c = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["optimizer"])
        start_epoch = c["epoch"] + 1
        best, best_epoch = c["best_metric"], c["best_epoch"]
        es.count, es.max_metric = c["es_count"], c["es_max_metric"]
        np.random.set_state(c["np_rng"])
        torch.set_rng_state(c["torch_rng"])
        prune_csv(metrics_csv, start_epoch)     # drop rows >= resume point
        prune_csv(scalars_csv, start_epoch)
        print(f">>> resumed from {ckpt_path}: next epoch {start_epoch}, "
              f"best Recall@10={best:.4f}@{best_epoch}, es_count={es.count}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if not isinstance(trainset, Subset):
            trainset.newit()                       # resample negatives each epoch
        t0 = time.time()
        total = torch.zeros((), device=device)   # accumulate on-device (sync only when logging)
        nb = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for strm, act, tb, mask, pos, neg in pbar:
            strm, act, tb, mask = (x.to(device) for x in (strm, act, tb, mask))
            pos, neg = pos.to(device), neg.to(device)
            logits, l2 = model(strm, act, tb, mask, pos, neg)
            loss = bprloss((logits, l2), logits.shape[0], "mean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach()
            nb += 1
            if args.log_every and nb % args.log_every == 0:
                pbar.set_postfix(loss=f"{(total/nb).item():.4f}")   # throttled GPU sync

        avg_loss = (total / max(nb, 1)).item()
        is_last = epoch == args.epochs - 1
        if (epoch + 1) % args.eval_every != 0 and not is_last:
            print(f"[epoch {epoch}] loss={avg_loss:.4f} time={time.time()-t0:.1f}s",
                  flush=True)
            continue

        val = evaluate(model, val_loader, device, metrics, args.cand_chunk, tag="val")
        r10 = val["Recall10"]
        print(f"[epoch {epoch}] loss={avg_loss:.4f} time={time.time()-t0:.1f}s "
              f"| val Recall@10={r10:.4f} NDCG@10={val['NDCG10']:.4f} "
              f"Recall@20={val['Recall20']:.4f}", flush=True)

        # durable logs (survive a crash; same wide schema as MBGCN's metrics.csv)
        keys = list(metrics.keys())
        append_csv(metrics_csv, ["epoch"] + keys, [epoch] + [val[k] for k in keys])
        append_csv(scalars_csv, ["epoch", "name", "value"], [epoch, "loss", avg_loss])
        for k in keys:
            append_csv(scalars_csv, ["epoch", "name", "value"], [epoch, k, val[k]])

        if r10 > best:
            best, best_epoch = r10, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val,
                        "args": vars(args)}, best_path)
            print(f"  new best Recall@10={best:.4f} -> {best_path}", flush=True)

        stop = es.step(r10, epoch)

        # Full checkpoint EVERY validation cycle so a crash resumes from here.
        torch.save({
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "epoch": epoch, "best_metric": best, "best_epoch": best_epoch,
            "es_count": es.count, "es_max_metric": es.max_metric,
            "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "args": vars(args),
        }, ckpt_path)

        if stop:
            print(f">>> early stop at epoch {epoch} (best Recall@10={best:.4f}@{best_epoch})",
                  flush=True)
            break

    if os.path.exists(best_path):
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=False)["model"])
        print(f"loaded best model (Recall@10={best:.4f}@{best_epoch}) for test", flush=True)
    test = evaluate(model, test_loader, device, metrics, args.cand_chunk, tag="test")
    print("=== TEST ===", flush=True)
    for k, v in test.items():
        print(f"{k}: {v:.4f}", flush=True)


if __name__ == "__main__":
    main()
