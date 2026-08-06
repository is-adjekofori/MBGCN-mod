#!/usr/bin/env python3
"""
Dataset-agnostic sparsity investigation for MBGCN-style interaction data.

The tool reads a dataset directory laid out like the MBGCN datasets:

    <dir>/data_size.txt         ->  "<user_num> <item_num>"
    <dir>/<behaviour>.txt       ->  lines of "<user_id> <item_id>"  (e.g. click, gift, ...)
    <dir>/train.txt             ->  target-behaviour training interactions
    <dir>/validation.txt        ->  validation ground truth
    <dir>/test.txt              ->  test ground truth

It computes interaction density, degree (activity/popularity) distributions,
long-tail concentration, catalogue coverage, and -- most importantly for
evaluation -- how many users are actually *rank-able* and how many
validation/test items are cold (never seen in training).

Outputs (written to <out>/):
    report.md          human-readable report (intended for the thesis)
    stats.json         machine-readable version of every number in the report
    per_file_stats.csv one row per interaction file
    *.png              optional plots (requires matplotlib; --no-plots to skip)

The report is about a SINGLE dataset only -- no cross-dataset comparison.

Usage:
    python sparsity_report.py --dir dataset/nw_interaction
    python sparsity_report.py --path dataset --dataset nw_interaction --out reports/kuailive
    python sparsity_report.py --dir dataset/Tmall --behaviors buy,cart,collect,click
"""

import argparse
import json
import math
import os
from datetime import datetime

import numpy as np

# ---- files that are never treated as auxiliary behaviours -------------------
DEFAULT_SPLITS = ["train", "validation", "test"]
DEFAULT_TARGET = "train"
EVAL_SPLITS = ["validation", "test"]  # splits we assess for rank-ability
NON_BEHAVIOUR = {"data_size"}  # metadata, not interactions


# =============================================================================
# IO
# =============================================================================
def read_data_size(dataset_dir):
    with open(os.path.join(dataset_dir, "data_size.txt")) as f:
        user_num, item_num = f.readline().strip().split()
    return int(user_num), int(item_num)


def read_edges(path):
    """Stream a "<user> <item>" file into two int64 arrays (raw, not deduped)."""
    users, items = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            users.append(int(parts[0]))
            items.append(int(parts[1]))
    return np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)


def discover_behaviour_files(dataset_dir, splits):
    """All top-level *.txt files that are neither splits nor metadata."""
    skip = set(splits) | NON_BEHAVIOUR
    found = []
    for fn in sorted(os.listdir(dataset_dir)):
        if not fn.endswith(".txt"):
            continue
        stem = fn[:-4]
        if stem in skip:
            continue
        found.append(stem)
    return found


# =============================================================================
# Metrics
# =============================================================================
def gini(counts):
    """Gini coefficient of a 1-D array of non-negative counts (0=uniform,1=all-in-one)."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    total = x.sum()
    if n == 0 or total == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1.0) / n)


def top_share(counts, fracs=(0.01, 0.05, 0.10, 0.20)):
    """Fraction of interactions held by the top f-fraction of nodes (by degree)."""
    c = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
    n = c.size
    total = c.sum()
    out = {}
    for fr in fracs:
        if n == 0 or total == 0:
            out[fr] = float("nan")
        else:
            k = max(1, int(math.ceil(fr * n)))
            out[fr] = float(c[:k].sum() / total)
    return out


def describe(values):
    """Summary stats of a 1-D array (assumed already filtered to the population)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {
            k: float("nan")
            for k in (
                "count",
                "mean",
                "std",
                "min",
                "p25",
                "p50",
                "p75",
                "p90",
                "p95",
                "p99",
                "max",
            )
        }
    return {
        "count": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "p25": float(np.percentile(v, 25)),
        "p50": float(np.percentile(v, 50)),
        "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(v.max()),
    }


def edge_stats(name, users, items, U, I):
    """Full sparsity/degree profile of one interaction file."""
    raw = users.size
    # keep only in-range edges for node-level stats; report the rest
    valid = (users >= 0) & (users < U) & (items >= 0) & (items < I)
    invalid = int((~valid).sum())
    u, it = users[valid], items[valid]

    codes = u.astype(np.int64) * I + it.astype(np.int64)
    uniq = np.unique(codes)
    n_unique = int(uniq.size)
    n_dup = int(u.size - n_unique)

    # degrees over the *unique* edges (duplicates shouldn't inflate degree)
    uu = (uniq // I).astype(np.int64)
    ui = (uniq % I).astype(np.int64)
    user_deg = np.bincount(uu, minlength=U)[:U]
    item_deg = np.bincount(ui, minlength=I)[:I]

    users_active = int((user_deg > 0).sum())
    items_active = int((item_deg > 0).sum())

    density = n_unique / (U * I) if U and I else float("nan")

    return {
        "name": name,
        "raw_edges": int(raw),
        "unique_edges": n_unique,
        "duplicate_edges": n_dup,
        "invalid_edges": invalid,
        "density": density,
        "sparsity_pct": 100.0 * (1.0 - density),
        "one_in_n": (1.0 / density) if density > 0 else float("inf"),
        "users_active": users_active,
        "users_active_frac": users_active / U if U else float("nan"),
        "users_inactive": U - users_active,
        "items_active": items_active,
        "items_active_frac": items_active / I if I else float("nan"),
        "items_inactive": I - items_active,
        "user_degree_all": describe(user_deg),  # incl. zero-degree users
        "user_degree_active": describe(user_deg[user_deg > 0]),
        "item_degree_all": describe(item_deg),
        "item_degree_active": describe(item_deg[item_deg > 0]),
        "user_gini": gini(user_deg[user_deg > 0]),
        "item_gini": gini(item_deg[item_deg > 0]),
        "user_top_share": top_share(user_deg[user_deg > 0]),
        "item_top_share": top_share(item_deg[item_deg > 0]),
        # arrays kept for plotting / feasibility / union (not serialised to JSON)
        "_user_deg": user_deg,
        "_item_deg": item_deg,
        "_user_set": set(uu.tolist()),
        "_item_set": set(ui.tolist()),
        "_codes": uniq,
    }


def eval_feasibility(ev, train_user_set, train_item_set, U, I):
    """How rank-able is an evaluation split, given the training catalogue."""
    user_deg = ev["_user_deg"]
    rankable = ev["_user_set"]  # users with >=1 positive
    n_rankable = len(rankable)

    # positives per rank-able user
    pos_per_user = user_deg[user_deg > 0]

    # user cold-start: rank-able eval users unseen in training
    cold_users = rankable - train_user_set
    # item cold-start: eval items never seen in training
    eval_items = ev["_item_set"]
    cold_items = eval_items - train_item_set

    # fraction of eval *interactions* whose item is cold (uncrankable by pure CF)
    item_deg = ev["_item_deg"]
    cold_item_mask = np.zeros(I, dtype=bool)
    if cold_items:
        cold_item_mask[
            np.fromiter(cold_items, dtype=np.int64, count=len(cold_items))
        ] = True
    total_edges = int(item_deg.sum())
    cold_edges = int(item_deg[cold_item_mask].sum()) if total_edges else 0

    return {
        "rankable_users": n_rankable,
        "rankable_users_frac": n_rankable / U if U else float("nan"),
        "empty_users": U - n_rankable,
        "empty_users_frac": (U - n_rankable) / U if U else float("nan"),
        "positives_per_rankable_user": describe(pos_per_user),
        "cold_start_users": len(cold_users),
        "cold_start_users_frac_of_rankable": (
            (len(cold_users) / n_rankable) if n_rankable else float("nan")
        ),
        "cold_items": len(cold_items),
        "cold_items_frac_of_eval_items": (
            (len(cold_items) / len(eval_items)) if eval_items else float("nan")
        ),
        "cold_edges": cold_edges,
        "cold_edges_frac": (cold_edges / total_edges) if total_edges else float("nan"),
    }


def random_recall_reference(I, ks=(10, 20, 40, 80)):
    """E[Recall@k] for a uniformly random ranker = k / item_num (independent of #positives)."""
    return {int(k): (k / I if I else float("nan")) for k in ks}


# =============================================================================
# Plots (optional)
# =============================================================================
def make_plots(out_dir, dataset_name, target_stats, feas):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib missing -> skip silently
        return []

    made = []

    def rankfreq(deg, title, fname, color):
        d = np.sort(deg[deg > 0])[::-1]
        if d.size == 0:
            return
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.loglog(np.arange(1, d.size + 1), d, color=color)
        ax.set_xlabel("rank")
        ax.set_ylabel("degree (interactions)")
        ax.set_title(title)
        fig.tight_layout()
        p = os.path.join(out_dir, fname)
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(fname)

    rankfreq(
        target_stats["_item_deg"],
        f"{dataset_name}: item popularity (target)",
        "item_rankfreq.png",
        "#c0392b",
    )
    rankfreq(
        target_stats["_user_deg"],
        f"{dataset_name}: user activity (target)",
        "user_rankfreq.png",
        "#2c3e50",
    )

    # Lorenz curve of item interactions (target)
    d = np.sort(target_stats["_item_deg"].astype(np.float64))
    if d.sum() > 0:
        cum = np.cumsum(d) / d.sum()
        x = np.linspace(0, 1, cum.size)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(x, cum, color="#c0392b", label="items")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="equality")
        ax.set_xlabel("cumulative fraction of items (least→most popular)")
        ax.set_ylabel("cumulative fraction of interactions")
        ax.set_title(f"{dataset_name}: item Lorenz curve (target)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "item_lorenz.png"), dpi=130)
        plt.close(fig)
        made.append("item_lorenz.png")

    # rank-able users per eval split
    if feas:
        names = list(feas.keys())
        vals = [feas[n]["rankable_users_frac"] for n in names]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(names, vals, color="#27ae60")
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of users that are rank-able")
        ax.set_title(f"{dataset_name}: evaluation coverage")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "eval_coverage.png"), dpi=130)
        plt.close(fig)
        made.append("eval_coverage.png")

    return made


# =============================================================================
# Report rendering
# =============================================================================
def pct(x):
    return (
        "n/a"
        if (x is None or (isinstance(x, float) and math.isnan(x)))
        else f"{100.0 * x:.4f}%"
    )


def num(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.4f}"


def describe_row(label, d):
    return (
        f"| {label} | {num(d['mean'])} | {num(d['p50'])} | {num(d['p90'])} | "
        f"{num(d['p99'])} | {num(d['max'])} | {num(d['std'])} |"
    )


def render_markdown(
    ds_name, U, I, target, per_file, feas, combined, rand_ref, plot_files
):
    L = []
    W = L.append
    W(f"# Sparsity Report — `{ds_name}`")
    W("")
    W(
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        f"This report concerns `{ds_name}` only._"
    )
    W("")

    # ---- 1. Overview
    W("## 1. Dataset overview")
    W("")
    W(f"- **Users:** {U:,}")
    W(f"- **Items:** {I:,}")
    W(f"- **User × Item cells:** {U * I:,}")
    files_str = ", ".join("`" + s["name"] + "`" for s in per_file)
    W(f"- **Interaction files:** {files_str}")
    W("")
    W(
        "**Random-ranker reference** — the expected Recall@k of a *uniformly random* "
        "ranker is `k / item_num`, i.e. the floor any model must beat:"
    )
    W("")
    W("| k | E[Recall@k] random |")
    W("|---|---|")
    for k, v in rand_ref.items():
        W(f"| {k} | {v:.3e} |")
    W("")
    min_k = min(rand_ref)
    W(
        f"> With {I:,} items, the random baseline for Recall@{min_k} is only "
        f"{rand_ref[min_k]:.2e}. A learning-but-early model can therefore print "
        "`0.000` at typical printing precision for many epochs; near-zero metrics "
        "alone are not evidence of a broken model at this catalogue size."
    )
    W("")

    # ---- 2. Density per file
    W("## 2. Interaction density")
    W("")
    W(
        "Density = unique interactions / (users × items). `1 in N` = one observed "
        "interaction per N possible user–item cells."
    )
    W("")
    W(
        "| File | Interactions | Unique | Density | Sparsity | 1 in N | Users touched | Items touched |"
    )
    W(
        "|------|-------------:|-------:|--------:|---------:|-------:|--------------:|--------------:|"
    )
    for s in per_file:
        W(
            f"| `{s['name']}` | {s['raw_edges']:,} | {s['unique_edges']:,} | "
            f"{s['density']:.3e} | {s['sparsity_pct']:.6f}% | {s['one_in_n']:,.0f} | "
            f"{s['users_active']:,} ({pct(s['users_active_frac'])}) | "
            f"{s['items_active']:,} ({pct(s['items_active_frac'])}) |"
        )
    W("")

    # ---- 3. Target behaviour deep-dive
    W(f"## 3. Target behaviour deep-dive (`{target['name']}`)")
    W("")
    W(
        f"- **Catalogue coverage:** {target['items_active']:,} / {I:,} items "
        f"({pct(target['items_active_frac'])}) appear in training. The remaining "
        f"{target['items_inactive']:,} items have **no** target signal and cannot be "
        "learned by collaborative filtering."
    )
    W(
        f"- **Active users:** {target['users_active']:,} / {U:,} "
        f"({pct(target['users_active_frac'])})."
    )
    W(
        f"- **Popularity skew (Gini):** items {target['item_gini']:.4f}, "
        f"users {target['user_gini']:.4f} (0 = uniform, 1 = maximally concentrated)."
    )
    W("")
    W(
        "**Popularity concentration (share of interactions held by the most active nodes):**"
    )
    W("")
    W("| Top fraction | Items' interaction share | Users' interaction share |")
    W("|---|---|---|")
    for fr in (0.01, 0.05, 0.10, 0.20):
        W(
            f"| top {int(fr*100)}% | {pct(target['item_top_share'][fr])} | "
            f"{pct(target['user_top_share'][fr])} |"
        )
    W("")
    W("**Degree distributions (target):**")
    W("")
    W("| Population | mean | median | p90 | p99 | max | std |")
    W("|---|---|---|---|---|---|---|")
    W(describe_row("interactions / active user", target["user_degree_active"]))
    W(describe_row("interactions / active item", target["item_degree_active"]))
    W("")

    # ---- 4. Evaluation feasibility (the key section)
    W("## 4. Evaluation feasibility")
    W("")
    W(
        "A user can only contribute to ranking metrics if they have at least one "
        "ground-truth positive in the split. An item can only be retrieved by a CF "
        "model if it was seen during training (otherwise it is *cold*)."
    )
    W("")
    for split, f in feas.items():
        W(f"### `{split}`")
        W("")
        W(
            f"- **Rank-able users:** {f['rankable_users']:,} / {U:,} "
            f"(**{pct(f['rankable_users_frac'])}**). "
            f"{f['empty_users']:,} users ({pct(f['empty_users_frac'])}) have no "
            "positive here and are dead weight in the evaluation loop."
        )
        W(
            f"- **Positives per rank-able user:** mean {num(f['positives_per_rankable_user']['mean'])}, "
            f"median {num(f['positives_per_rankable_user']['p50'])}, "
            f"max {num(f['positives_per_rankable_user']['max'])}."
        )
        W(
            f"- **Cold-start users:** {f['cold_start_users']:,} rank-able users "
            f"({pct(f['cold_start_users_frac_of_rankable'])} of rank-able) never appear "
            "in training — no personalised signal exists for them."
        )
        W(
            f"- **Cold items:** {f['cold_items']:,} of this split's items "
            f"({pct(f['cold_items_frac_of_eval_items'])}) are never seen in training. "
            f"They account for {pct(f['cold_edges_frac'])} of this split's positives, "
            "which are effectively impossible for a pure-CF model to rank."
        )
        W("")

    # ---- 5. Per-behaviour distributions
    W("## 5. Activity & popularity across all behaviours")
    W("")
    W(
        "| Behaviour | Interactions | Density | User Gini | Item Gini | "
        "med. user deg | med. item deg | inactive users | inactive items |"
    )
    W("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in per_file:
        W(
            f"| `{s['name']}` | {s['unique_edges']:,} | {s['density']:.3e} | "
            f"{s['user_gini']:.3f} | {s['item_gini']:.3f} | "
            f"{num(s['user_degree_active']['p50'])} | {num(s['item_degree_active']['p50'])} | "
            f"{pct(1 - s['users_active_frac'])} | {pct(1 - s['items_active_frac'])} |"
        )
    W("")

    # ---- 6. Combined coverage
    W("## 6. Combined multi-behaviour coverage")
    W("")
    W("Union of every interaction file (all behaviours + splits):")
    W("")
    W(f"- **Unique user–item interactions:** {combined['unique_edges']:,}")
    W(
        f"- **Combined density:** {combined['density']:.3e} "
        f"(1 in {combined['one_in_n']:,.0f})"
    )
    W(
        f"- **Users touched by *some* behaviour:** {combined['users_active']:,} "
        f"({pct(combined['users_active_frac'])})"
    )
    W(
        f"- **Items touched by *some* behaviour:** {combined['items_active']:,} "
        f"({pct(combined['items_active_frac'])})"
    )
    W("")

    # ---- plots
    if plot_files:
        W("## 7. Figures")
        W("")
        for p in plot_files:
            W(f"![{p}]({p})")
            W("")

    # ---- definitions
    W("## Appendix — definitions")
    W("")
    W(
        "- **Density** = unique interactions ÷ (users × items). **Sparsity** = 1 − density."
    )
    W("- **Active / touched** node = has ≥ 1 interaction in that file.")
    W(
        "- **Gini** of the degree distribution over active nodes: 0 = every node equal, "
        "1 = all interactions on one node (long-tail severity)."
    )
    W("- **Rank-able user** = has ≥ 1 ground-truth positive in the split.")
    W("- **Cold-start user** = rank-able in the split but never seen in `train`.")
    W(
        "- **Cold item** = present in the split but never seen in `train` "
        "(unretrievable by pure collaborative filtering)."
    )
    W("- **E[Recall@k] random** = k ÷ item_num.")
    W("")
    return "\n".join(L)


# =============================================================================
# JSON-safe stripping (drop the private array/set fields)
# =============================================================================
def clean_for_json(d):
    if isinstance(d, dict):
        return {
            k: clean_for_json(v) for k, v in d.items() if not str(k).startswith("_")
        }
    if isinstance(d, (list, tuple)):
        return [clean_for_json(v) for v in d]
    if isinstance(d, np.generic):
        return d.item()
    return d


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Dataset-agnostic sparsity report.")
    ap.add_argument(
        "--dir", help="Path to the dataset directory (contains data_size.txt)."
    )
    ap.add_argument("--path", default=".", help="Data root (used with --dataset).")
    ap.add_argument("--dataset", help="Dataset folder name under --path.")
    ap.add_argument(
        "--behaviours",
        "--behaviors",
        dest="behaviours",
        help="Comma-separated behaviour file stems. Default: auto-discover.",
    )
    ap.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated split file stems (default: train,validation,test).",
    )
    ap.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Target-behaviour split for coverage/cold-start (default: train).",
    )
    ap.add_argument("--out", help="Output directory (default: <dir>/sparsity_report).")
    ap.add_argument("--no-plots", action="store_true", help="Skip PNG plots.")
    args = ap.parse_args()

    if args.dir:
        dataset_dir = args.dir
    elif args.dataset:
        dataset_dir = os.path.join(args.path, args.dataset)
    else:
        ap.error("provide --dir OR --dataset")
    ds_name = os.path.basename(os.path.normpath(dataset_dir))
    out_dir = args.out or os.path.join(dataset_dir, "sparsity_report")
    os.makedirs(out_dir, exist_ok=True)

    splits = [s for s in args.splits.split(",") if s]
    if args.behaviours:
        behaviours = [b for b in args.behaviours.split(",") if b]
    else:
        behaviours = discover_behaviour_files(dataset_dir, splits)

    U, I = read_data_size(dataset_dir)
    print(f"[{ds_name}] users={U:,} items={I:,}")
    print(f"  behaviours: {behaviours}")
    print(f"  splits: {splits}  target: {args.target}")

    # order files: behaviours first, then splits (only those that exist)
    file_order = behaviours + [s for s in splits if s not in behaviours]

    per_file = []
    stats_by_name = {}
    for name in file_order:
        fp = os.path.join(dataset_dir, name + ".txt")
        if not os.path.exists(fp):
            print(f"  ! missing, skipping: {fp}")
            continue
        u, it = read_edges(fp)
        s = edge_stats(name, u, it, U, I)
        per_file.append(s)
        stats_by_name[name] = s
        print(
            f"  read {name}: {s['raw_edges']:,} edges "
            f"({s['unique_edges']:,} unique, density {s['density']:.2e})"
        )

    if args.target not in stats_by_name:
        ap.error(f"target '{args.target}' not found among files: {list(stats_by_name)}")
    target = stats_by_name[args.target]
    train_user_set = target["_user_set"]
    train_item_set = target["_item_set"]

    # evaluation feasibility for eval splits that exist
    feas = {}
    for split in EVAL_SPLITS:
        if split in stats_by_name and split != args.target:
            feas[split] = eval_feasibility(
                stats_by_name[split], train_user_set, train_item_set, U, I
            )

    # combined union across all interaction files (reuse per-file unique codes)
    if per_file:
        union_codes = np.unique(np.concatenate([s["_codes"] for s in per_file]))
    else:
        union_codes = np.array([], dtype=np.int64)
    comb_users = int(np.unique(union_codes // I).size) if union_codes.size else 0
    comb_items = int(np.unique(union_codes % I).size) if union_codes.size else 0
    comb_density = union_codes.size / (U * I) if U and I else float("nan")
    combined = {
        "unique_edges": int(union_codes.size),
        "density": comb_density,
        "one_in_n": (1.0 / comb_density) if comb_density > 0 else float("inf"),
        "users_active": comb_users,
        "users_active_frac": comb_users / U if U else float("nan"),
        "items_active": comb_items,
        "items_active_frac": comb_items / I if I else float("nan"),
    }

    rand_ref = random_recall_reference(I)

    # plots
    plot_files = []
    if not args.no_plots:
        plot_files = make_plots(out_dir, ds_name, target, feas)
        if not plot_files:
            print("  (matplotlib unavailable or no data -> plots skipped)")

    # render + write
    md = render_markdown(
        ds_name, U, I, target, per_file, feas, combined, rand_ref, plot_files
    )
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(md)

    payload = {
        "dataset": ds_name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "user_num": U,
        "item_num": I,
        "random_recall_reference": {str(k): v for k, v in rand_ref.items()},
        "per_file": [clean_for_json(s) for s in per_file],
        "evaluation_feasibility": clean_for_json(feas),
        "combined": combined,
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # per-file CSV
    cols = [
        "name",
        "raw_edges",
        "unique_edges",
        "duplicate_edges",
        "invalid_edges",
        "density",
        "sparsity_pct",
        "one_in_n",
        "users_active",
        "users_active_frac",
        "items_active",
        "items_active_frac",
        "user_gini",
        "item_gini",
    ]
    with open(os.path.join(out_dir, "per_file_stats.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for s in per_file:
            f.write(",".join(str(s[c]) for c in cols) + "\n")

    print(f"\nWrote report to: {out_dir}")
    print(f"  - report.md")
    print(f"  - stats.json")
    print(f"  - per_file_stats.csv")
    for p in plot_files:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
