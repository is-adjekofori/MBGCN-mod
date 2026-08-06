# MBGCN on KuaiLive — Modifications, Issues, and Mitigations

This document records the significant changes made to the reference MBGCN
implementation in order to benchmark it on the **KuaiLive** live-streaming
dataset (`nw_interaction`: 23,772 users × 452,621 items), together with the
issue that motivated each change. It is intended as a reference for the thesis
write-up.

The original model targets the **Tmall** and **Beibei** datasets. KuaiLive is
~38× larger in the item dimension than Tmall (452,621 vs 11,953 items), which is
the root cause of most of the problems below: memory blow-ups, slow computation,
and near-zero early metrics.

---

## 0. Baseline context

- **Model:** MBGCN (Multi-Behavior Graph Convolutional Network), Jin et al.,
  SIGIR 2020. Two scoring paths: a behavior-aware **user–item** score (`score1`)
  and an item-relevance **item–item** score (`score2`), combined via `λ`.
- **Datasets used:** Tmall (reference), KuaiLive `nw_interaction` (this work).
- **KuaiLive behaviours:** `gift, comment, like, click, negative`
  (counts: 45k / 115k / 105k / 2.42M / 3.90M). Target split: train 72,646 /
  val 6,360 / test 6,533.

---

## 1. Item–item propagation: sparse-matmul bypass (correctness verified)

**Issue.** The original code materialises a precomputed item–item co-interaction
graph (`item_<behaviour>.pth`) and its degree vector for the item–item
propagation. For KuaiLive the item–item graph is `452,621 × 452,621`, which is
infeasible to build/store on the available hardware.

**Modification.** Bypass the intermediate item graph and compute the propagation
directly from the sparse user–item matrix `R` (per behaviour):

```
item_graph_degree     = Rᵀ (R · 1)                    # = rowsum(Rᵀ R)
tmp_item_propagation  = (Rᵀ (R · E)) / degree · W       # = D⁻¹ (Rᵀ R) E W
```

**Why it is correct (mathematical justification).** The item–item co-interaction
graph is exactly `S = Rᵀ R` (entry `S_ij` = number of users who interacted with
both items i and j under that behaviour). Using the identities
`Rᵀ(R·1) = rowsum(RᵀR)` and `Rᵀ(R·E) = (RᵀR)E = S·E`, the bypass computes
`D⁻¹ S E W` — identical to first materialising `S` and normalising. This matches
the paper's mean-aggregation over co-behaviored items (§3.4/§3.5.2). **The
modification does not alter the model's mathematics.**

**Caveat noted for the thesis.** `S = RᵀR` retains the diagonal (self-loops,
`S_ii` = item's user-degree) and uses weighted co-occurrence counts. If the
original precomputed graph was binarised or had its diagonal removed, the
normalisation differs slightly. For dense-interaction live-streaming data the
self-loop weight (~`1/avg_user_degree`) is small, so the effect is minor.

**Files:** `model.py` (`forward`, `propagate`).

---

## 2. Memory: densification of user–item matrices ("memory bomb")

**Issue.** `TrainDataset.__calculate_user_behaviour` called `.to_dense()` on each
`[23,772 × 452,621]` behaviour matrix (twice per behaviour) to compute per-user
and per-item interaction counts. Each dense matrix is ~**43 GB** in float32 —
an immediate OOM on GPU and catastrophic swapping on CPU.

**Mitigation.** Replace the dense reductions with sparse ones:

```python
user_deg = torch.sparse.sum(mat, dim=1).to_dense().unsqueeze(-1)  # per-user count
item_deg = torch.sparse.sum(mat, dim=0).to_dense().unsqueeze(-1)  # per-item count
```

Numerically identical (verified: duplicate COO entries and zero rows behave the
same), peak memory drops from ~43 GB/behaviour to a few MB.

**Files:** `dataset.py` (`__calculate_user_behaviour`).

---

## 3. Compute: redundant per-minibatch recomputation of item-graph degree

**Issue.** `item_graph_degree = Rᵀ(R·1)` depends only on the static relation
matrices, not on the (changing) embeddings, yet it was recomputed on every
minibatch of both training and evaluation.

**Mitigation.** Precompute it once at model construction
(`__precompute_item_graph_degree`, under `torch.no_grad()`) and cache per
behaviour; `forward` and `propagate` read the cache. Bit-identical results, no
gradient impact (the quantity was always constant w.r.t. parameters).

**Files:** `model.py` (`__init__`, `__precompute_item_graph_degree`, `forward`,
`propagate`).

---

## 4. Evaluation: full-graph propagation recomputed every validation batch

**Issue.** For MBGCN, validation called `model.evaluate(users)` per batch, and
`evaluate` recomputed the **entire** user/item graph propagation from scratch
each call. With ~47 validation batches and 5 behaviours, the item–item
propagation over 452k items ran ~235× per validation instead of once — the
single largest source of validation slowdown.

**Mitigation.** Split the monolithic `evaluate` into:
- **`propagate(task="test")`** — runs all user-independent graph work **once**
  (per-behaviour `tmp_item_propagation`, `user_feature`, `item_feature`,
  behaviour projections);
- **`evaluate(propagate_result, user)`** — thin per-batch scoring (index by the
  batch's users, do the final `[batch × items]` matmuls).

Validation now calls `propagate` once outside the batch loop (mirrors the MF
path). Removed the redundant `multi3_validation`; both models share
`validation()`. Results are bit-identical (same operations and accumulation
order); only the *timing* of computation changed. The `Rᵀ(R·E)` propagation
recompute collapses from ~235× to 5× per validation.

**Files:** `model.py` (new `propagate`, rewritten `evaluate`), `train.py`.

---

## 5. Evaluation: dense ground-truth / train-mask and cadence

**Issue.** Even after (4), each validation batch built dense `[batch × 452,621]`
one-hot tensors for the ground truth and the train mask via `scipy .toarray()`
on the CPU, then transferred them to the GPU — ~1.85 GB host→device **per
batch** (~87 GB per validation), largely serialised with `num_workers=0`.
Additionally, validation ran every epoch, roughly doubling wall-clock.

**Mitigations (three parts).**

**(a) Validate every N epochs.** New `--eval_every` flag (default 1). The
training loop runs validation / leaderboard / early-stop only every `eval_every`
epochs (and on the final epoch). Also fixed a **metric-dictionary key
collision**: `"NDCG80"`/`"MRR80"` were being overwritten by the `@500` metrics,
silently dropping NDCG@80 and MRR@80; keys are now `"NDCG500"`/`"MRR500"`.

**(b) Sparse train-mask.** Instead of a dense mask, the training items are
removed via in-place `pred[tm_rows, tm_cols] -= BIGNUM` using the user's sparse
train-item indices. Same effect (masked items pushed out of top-k), a few ints
per user instead of ~0.9 GB/batch.

**(c) Sparse ground-truth membership.** `TestDataset` now yields each user's
positive/train item **indices** (CSR row slices) instead of dense one-hots; a
`test_collate` function flattens them into batch-local `(row, col)` tensors.
`metrics.get_is_hit` encodes top-k picks as `row*item_num + col` codes and tests
membership with `torch.isin` against the positives — no dense one-hot is ever
built. `num_pos` comes from per-user counts.

**Correctness.** The `row*item_num + col` encoding is a bijection for
`col ∈ [0, item_num)`, so `isin` membership equals the old dense lookup exactly.
Verified with a randomised simulation (200 trials × top-k {5,10,20} × mask
{on,off}): `is_hit` and `num_pos` are identical to the dense implementation.

**Files:** `dataset.py` (`TestDataset.__getitem__`, `test_collate`),
`metrics.py` (`get_is_hit`, `Recall`/`NDCG`/`MRR` now take sparse `gt`),
`train.py` (`run_metrics`, `validation`, `test`, loop), `main.py`, `MBGCN.sh`.

---

## 6. Reproducibility: run scripts made robust

**Issue.** `MBGCN.sh` was ad-hoc and had real bugs:
- The per-behaviour MGNN weights were hardcoded and duplicated (`mgnn_weight4`
  passed twice) to reach 5 weights for KuaiLive's 5 behaviours — fragile.
- The `message_dropout` / `node_dropout` sweep loops varied only the experiment
  *name*; the values were **never passed** to `main.py`, so every run silently
  used the 0.2 defaults.

**Mitigation.** Rewrote `MBGCN.sh`:
- `set -euo pipefail`, all variables quoted.
- `build_mgnn_weight_args` derives one `--mgnn_weight` per behaviour from the
  `relation` list (default 1.0 each; optional `mgnn_weights=(...)` override with a
  length check). Verified: 5 weights for KuaiLive, 4 for Tmall, error on
  mismatch.
- Actually passes `--message_dropout` / `--node_dropout`, plus `--epoch`,
  `--batch_size`, `--test_batch_size`, `--loss_mode`, `--gpu`, `--eval_every`.
- Per-experiment naming via `exp_tag`.

**Files:** `MBGCN.sh`.

---

## 7. Removed the Colab session-checkpoint / resume feature

**Issue.** A session-checkpoint/resume mechanism had been added to survive free
Colab session timeouts. It also contained two latent bugs: a periodic-save guard
`if epoch + 1 % 70 == 0` (operator precedence — never true) and a resume path
referencing an undefined flag `flags_obj.checkpoint_path`.

**Mitigation.** Removed entirely (no longer needed on the paid plan): the
periodic save, `resume_from_checkpoint`, the `session_checkpoint_path` flag and
resume branch, and the associated `self.resume`/`self.start_epoch` state.
Training now runs a single clean pass from epoch 0. The separate **best-model
save** (`ContentManager.model_save`, triggered on a new record in
`update_leaderboard`) is retained.

**Files:** `train.py`, `main.py`, `MBGCN.sh`.

---

## 8. Two-stage MF → MBGCN pretraining pipeline

**Issue / motivation.** MBGCN is designed to be initialised from pretrained MF
embeddings (as in the original paper). The KuaiLive config was using random
initialisation (`create_embeddings='True'`) and pointed `pretrain_path` at a
**Tmall** MF output (incompatible: 11,953 vs 452,621 items). On a large, sparse
catalog, a warm start typically improves convergence speed and stability, and MF
provides a cheap baseline for diagnosing the near-zero-metric behaviour.

**Mitigation.**
- Added **`MF_KuaiLive.sh`** (stage 1): trains MF on `nw_interaction` with a
  fixed output name `nw_interaction-MF-pretrain` and `embedding_size=64`.
  (Passes a single small behaviour `relation='gift'` only, since MF does not use
  the behaviour graphs — its embeddings come from `train.txt` + BPR negatives —
  so the full relation set is unnecessary overhead for this stage.)
- Updated **`MBGCN_KuaiLive.sh`** (stage 2): `create_embeddings='False'` and
  `pretrain_path='output/nw_interaction/nw_interaction-MF-pretrain'` (verified to
  match MF's save directory). Embeddings are loaded, `F.normalize`d, and kept
  trainable (`pretrain_frozen=False`) so MBGCN fine-tunes them.

**Run order:** `bash MF_KuaiLive.sh` → `bash MBGCN_KuaiLive.sh`.

**Files:** `MF_KuaiLive.sh` (new), `MBGCN_KuaiLive.sh`.

---

## Analyses / decisions (no code change, but relevant to the thesis)

### A. Near-zero early metrics: scale, not (necessarily) a bug
Expected Recall@80 for a random model over 452,621 items ≈ 80/452,621 ≈ 1.8e-4,
which prints as ~0.000. Combined with a hyper-sparse target (density 6.7e-6) and
an item space 38× larger than Tmall, slow/near-zero early metrics are largely a
**scale** effect. Recommended diagnostics: watch `mgnn_weight` for NaNs; log
metrics at higher precision or larger k; and compare against the MF baseline (if
MF is also ~0, the cause is scale rather than an MBGCN defect).

### B. The `negative` behaviour — kept, with a documented caveat
The `negative` behaviour (largest signal, 3.90M edges) was retained on the
hypothesis that MBGCN can learn a **negative** behaviour weight (`w_t < 0`) to
"repel" disliked items. Analysis shows this is *partially* supported but has two
mathematical risks worth discussing in the thesis:
1. The behaviour weight normalisation `α_ut = n_ut w_t / Σ_m n_um w_m` divides by
   a **signed** per-user sum. A large-magnitude negative `w_negative` can drive
   the denominator to zero or flip its sign, causing per-user weight explosions
   and sign inversions of the *positive* behaviours. The scheme implicitly
   assumes `w_t ≥ 0`. Gradient descent therefore tends to park `w_negative`
   near/above 0 rather than at an "ideal repel" value.
2. The item-based score (`score2`) has **no** behaviour weight — every behaviour
   contributes with a fixed `+1/|behaviours|` — so the sign mechanism only exists
   in `score1`. The two scoring paths treat negatives inconsistently.

**Planned experiment:** ablate with/without `negative` and inspect the learned
`w_negative`. A "decoupled signed-repel" reformulation (positive behaviours
normalised among themselves; negatives added as a separate `-softplus(β)` term
outside the normalisation) is a possible extension that makes the hypothesis
numerically safe.

### C. Item-side aggregation is unnormalised (noted, unchanged)
`item_feature = train_matrixᵀ · user_embedding` is an unnormalised sum over a
item's users, whereas the user side is mean-normalised. This asymmetry appears in
the reference implementation and is retained; on a 452k-item catalog it induces a
popularity scaling worth noting when interpreting results.

---

## Summary table

| # | Area | Issue | Mitigation | Correctness |
|---|------|-------|-----------|-------------|
| 1 | Item–item prop. | 452k² item graph infeasible | Direct sparse `D⁻¹(RᵀR)EW` | Proven equivalent |
| 2 | Memory | `.to_dense()` ~43 GB/behaviour | Sparse `torch.sparse.sum` | Identical |
| 3 | Compute | degree recomputed per batch | Cache once at init | Bit-identical |
| 4 | Eval compute | full propagation per batch | `propagate` once + thin `evaluate` | Bit-identical |
| 5a | Eval cadence | validate every epoch; key clash | `--eval_every`; fix metric keys | Same metrics |
| 5b | Eval memory | dense train-mask/batch | Sparse in-place mask | Identical |
| 5c | Eval memory | dense ground-truth/batch | Sparse `isin` membership | Verified identical |
| 6 | Tooling | fragile script; dropout not passed | Robust `MBGCN.sh` | N/A |
| 7 | Tooling | unneeded resume + latent bugs | Removed | N/A |
| 8 | Init | random init; wrong pretrain path | MF→MBGCN pipeline | N/A |

*Files touched:* `model.py`, `dataset.py`, `metrics.py`, `train.py`, `main.py`,
`MBGCN.sh`, `MBGCN_KuaiLive.sh`, `MF_KuaiLive.sh`. See also `MODEL_REVIEW.md` for
the initial correctness/efficiency review.
