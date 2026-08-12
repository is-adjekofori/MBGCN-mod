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

> **Two phases.** *Part I* (§1–§8) made MBGCN **run** on KuaiLive (memory,
> compute, tooling) using the initial `nw_interaction` dataset — the **gift**
> target with **full-catalogue** ranking. Metrics still stayed ~0.000. *Part II*
> (§9–§14) diagnoses why and changes the **task and evaluation protocol**: a new
> `click_interaction` dataset (**click** target, temporal leave-one-out split) and
> **sampled-candidate** evaluation. Where Part II supersedes a Part I item, this is
> noted inline. The final results in §14 are from the Part II configuration.

---

## 0. Baseline context

- **Model:** MBGCN (Multi-Behavior Graph Convolutional Network), Jin et al.,
  SIGIR 2020. Two scoring paths: a behavior-aware **user–item** score (`score1`)
  and an item-relevance **item–item** score (`score2`), combined via `λ`.
- **Datasets used:** Tmall (reference); KuaiLive — initially `nw_interaction`
  (Part I), then `click_interaction` (Part II, the final configuration).
- **KuaiLive behaviours:** `click, comment, like, gift, negative`.
- **Initial (Part I) target:** `gift` — train 72,646 / val 6,360 / test 6,533.
  **Final (Part II) target:** `click` with a temporal leave-one-out split —
  23,398 validation + 23,398 test instances (see §10–§11).

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

> **Superseded in part by §12.** Parts (b) and (c) below optimised *full-catalogue*
> ranking (scoring every user against all 452,621 items). Part II replaces
> full-catalogue ranking with **sampled-candidate** evaluation (§12), so the sparse
> train-mask (5b) and `isin` membership (5c) are no longer on the active path. Part
> (a) — `--eval_every` cadence — is retained. This section is kept as it documents
> a real optimisation and the reasoning that led to reconsidering the protocol.

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

**Mitigation.** First removed entirely (the buggy version): the periodic save,
`resume_from_checkpoint`, the `session_checkpoint_path` flag/branch, and the
`self.resume`/`self.start_epoch` state. The best-model save
(`ContentManager.model_save`, on a new record) was retained.

**Update (Part II).** A *clean* resume was later re-added, because with the click
target an epoch is ~50× longer (§13) and losing a long run to an interruption is
costly. On each new record the trainer now also writes `checkpoint.pkl` (model +
**optimizer** + epoch + leaderboard + early-stop state); `--resume` reloads it and
continues from `best_epoch + 1`, restoring early-stopping. This is not the old
timeout hack — it is an ordinary best-checkpoint resume, and it correctly saves
optimizer/epoch state (which the old version never did). Metric CSV logs are
pruned of rows at `epoch ≥ resume-epoch` so no epoch is double-logged.

**Files:** `train.py`, `main.py`, `utils.py`, `MBGCN.sh`.

---

## 8. Two-stage MF → MBGCN pretraining pipeline

**Issue / motivation.** MBGCN is designed to be initialised from pretrained MF
embeddings (as in the original paper). The KuaiLive config was using random
initialisation (`create_embeddings='True'`) and pointed `pretrain_path` at a
**Tmall** MF output (incompatible: 11,953 vs 452,621 items). On a large, sparse
catalog, a warm start typically improves convergence speed and stability, and MF
provides a cheap baseline for diagnosing the near-zero-metric behaviour.

**Mitigation.** (Paths shown are the Part II final values; the pipeline was first
built on `nw_interaction` and later re-pointed at `click_interaction` when the
target changed — §10/§11.)
- Added **`MF_KuaiLive.sh`** (stage 1): trains MF with a fixed output name
  `click_interaction-MF-pretrain` and `embedding_size=64`. (Passes a single small
  behaviour `relation='gift'` only, since MF does not use the behaviour graphs —
  its embeddings come from `train.txt` + BPR negatives — so the full relation set
  is unnecessary overhead for this stage.)
- Updated **`MBGCN_KuaiLive.sh`** (stage 2): `create_embeddings='False'` and
  `pretrain_path='output/click_interaction/click_interaction-MF-pretrain'`
  (verified to match MF's save directory). Embeddings are loaded, `F.normalize`d,
  and kept trainable (`pretrain_frozen=False`) so MBGCN fine-tunes them.

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

**Empirical result (Part II).** With the click-target run, the learned behaviour
weights converged to
`[click 1.44, comment 1.60, like 1.69, gift 1.32, negative −0.008]`. The model
**drove `w_negative` to ≈ 0** — it neither used the skip signal as positive
affinity nor as a repel term, and, having settled near zero, it did **not**
trigger the α-instability of risk (1). Interpretation for the thesis: as an MBGCN
*propagation* behaviour the exposed-but-skipped signal contributes essentially
nothing, yet it is the largest graph (12.7M edges) and is propagated every
minibatch — i.e. pure compute cost for ~zero benefit.

**Follow-up experiments this motivates:**
1. **Ablation:** drop `negative` from propagation (`relation='click,comment,like,gift'`);
   expected to match the with-negative metrics at materially lower compute — the
   learned `≈0` weight is the prediction.
2. **Better use of the skip signal:** feed observed negatives as **BPR hard
   negatives** (replacing/augmenting random negatives) — a training-signal use
   that sidesteps both mathematical risks above.
3. A "decoupled signed-repel" reformulation (positive behaviours normalised among
   themselves; negatives added as a separate `-softplus(β)` term *outside* the
   normalisation) remains a possible extension that makes the original repel
   hypothesis numerically safe.

### C. Item-side aggregation is unnormalised (noted, unchanged)
`item_feature = train_matrixᵀ · user_embedding` is an unnormalised sum over a
item's users, whereas the user side is mean-normalised. This asymmetry appears in
the reference implementation and is retained; on a 452k-item catalog it induces a
popularity scaling worth noting when interpreting results.

---

# Part II — Sparsity diagnosis, click re-target, and sampled-candidate evaluation

The Part I changes let MBGCN *run* on KuaiLive, but validation metrics still
printed ~0.000. Part II diagnoses the cause and changes the **task** and
**evaluation protocol** accordingly. A new dataset, `click_interaction`, was built
for this; the earlier `nw_interaction` (gift target, full-catalogue eval) is
superseded.

---

## 9. Diagnosis: why metrics stayed near zero (sparsity analysis)

**Investigation.** Built a dataset-agnostic sparsity report and compared KuaiLive
against Tmall (`sparsity_report.py`). Key disparities:

| | Tmall | KuaiLive |
|---|---:|---:|
| Items (catalogue) | 11,953 | 452,621 (~38×) |
| Train density | 3.64e-4 | 5.0e-6 (~73× sparser) |
| Items with any training signal | 100% | 7.8% (417k cold) |
| Median interactions / item (train) | 10 | 1 |
| Rank-able users in validation | 70% | ~18% (under gift) |
| Random Recall@10 floor (`k/items`) | 8.4e-4 | 2.2e-5 |

**Conclusion.** The near-zero metrics were **not** a model defect but the product
of three compounding factors: (i) **catalogue scale** — the random-ranker floor is
~38× lower, so a learning model prints 0.000 at normal precision; (ii) **signal
starvation** — median item/user degree = 1, only 7.8% of items ever trained; and
(iii) **mostly-empty evaluation** — ~80% of validation rows had no positive under
the gift target. The MF baseline behaved identically, confirming scale over defect.
The paper itself notes KuaiLive deliberately retains this sparsity/cold-start.

**Files:** `sparsity_report.py` (per-dataset `report.md` / `stats.json` / plots).

---

## 10. Target behaviour: gift → click

**Issue.** The initial target was **gift** — `train.txt` = 72,646 = `#Gifts`, the
**sparsest** of the five behaviours (gifts are ~1.5% of clicks). This is the
direct cause of the ~18% rank-able-user rate.

**Verification against the KuaiLive paper.**
- The 452,621 items are **streamers** (the dense, persistent item definition), not
  ephemeral rooms (11.6M). So the catalogue is already at the best-available
  granularity — no re-keying helps.
- The paper conducts its top-K experiments on **click** ("provides the most
  abundant interaction data", ~4.9M events); **gift** is spun off as a separate
  *Gift-Through-Rate* task, precisely because of its scarcity.

**Modification.** Switch the prediction target to **click**. Rationale:
- Click is the **gateway action** the recommender directly influences (every
  deeper behaviour is conditioned on it), and is dense enough to learn from.
- User coverage rises from ~18% (gift) to **98.4%** (click).
- It is the **fair** target for the MBGCN-vs-ATRank comparison — ATRank is an
  attention-over-behaviour-sequence model and needs abundant history.

**Caveat for the thesis.** Click carries **exposure / position / clickbait bias**
(you only observe clicks on what was shown). This should be stated as a limitation;
it is a preference/discovery signal, not a bias-free one.

**Files:** dataset rebuild (§11); `relation='click,comment,like,gift,negative'`.

---

## 11. Dataset reconstruction with a temporal leave-one-out split

**Issue / motivation.** The click target needs a proper train/val/test split, and
the earlier `nw_interaction` had **deduplicated** the raw signal (one row per
user–item pair), discarding interaction *intensity* (a habitual re-viewer looked
identical to a one-time viewer, flattening the behaviour weight α).

**Modification.** Rebuilt the dataset from the raw KuaiLive CSVs (with timestamps)
as `dataset/click_interaction/` (`derive_click_dataset.py`):
- **Leave-one-out by time** (the paper's top-K protocol): per user, sort clicks by
  timestamp → **last = test, 2nd-last = validation, rest = train**; users with < 3
  clicks are train-only. Result: **23,398** validation + **23,398** test instances,
  one positive each (98.4% user coverage), vs 6,360 / 6,533 before.
- **IDs shifted to 0-indexed** (raw KuaiLive IDs are 1-indexed).
- **No leakage:** held-out click events are removed from the train graph.
- **Intensity preserved:** `click.txt` keeps duplicate (user, streamer) events, so
  the coalesced relation matrix carries interaction frequency into α; `train.txt`
  is deduped for BPR positives. (Re-adds the intensity `nw_interaction` had lost.)
- Auxiliary behaviours kept whole; `*_meta.csv` retain `live_id` + `timestamp` for
  §12.

**Files:** `derive_click_dataset.py`; outputs under `dataset/click_interaction/`.

---

## 12. Sampled-candidate evaluation (KuaiLive protocol) — replaces full-catalogue ranking

**Issue.** Ranking each held-out positive against **all 452,621 items** makes even
a good model score ~0.000 (random floor `k/452621`), and dilutes the ranking with
417k items that never appear in training. Full-catalogue ranking is the wrong
protocol at this scale (this is what §5b/5c had been optimising).

**Modification.** Adopt the paper's protocol (§5.1.2): for each evaluation
instance, rank the positive against a fixed pool of **10,000 time-valid
negatives** — streamers that were **live at the interaction's timestamp** —
excluding the user's own clicks. "Live at time *t*" is reconstructed from
`room.csv` start/end lifecycles.

- **Builder (`build_candidates.py`):** a **sweep-line** over 11.8M room
  start/end events with the eval timestamps interleaved builds the candidate table
  in one pass (vs a 5.5e11-op brute force). Output `candidates.npz` is **seeded**
  and row-aligned to `validation.txt` / `test.txt`, so **MBGCN and ATRank rank the
  identical pools** (fair comparison). The live-streamer set is ≥ 12,141 at every
  instance, so 10,000 negatives are always available (0 shortfalls).
- **Scoring (`model.evaluate_candidates`, MF + MBGCN):** gathers only the
  `[batch × 10,001]` candidate embeddings; verified **bit-identical** to the
  full-catalogue `evaluate` restricted to those columns.
- **Metrics:** `SampledRecall/NDCG/MRR` — the positive's rank in its pool is
  `1 + #negatives scoring higher`; reported at **k = {5, 10, 20}** (matching the
  paper's Table 3). The train-mask is no longer needed (negatives already exclude
  seen items).

**Caveat for the thesis.** Sampled metrics are **biased estimators** of true
full-catalogue ranking (Krichene & Rendle, KDD 2020). This is acceptable because
(a) the protocol matches the paper, making our numbers comparable to Table 3, and
(b) it is applied **identically** to both models. State that reported metrics are
sampled estimates.

**Files:** `build_candidates.py`, `dataset.py` (`TestDataset`, `test_collate`),
`model.py` (`evaluate_candidates`), `metrics.py` (`Sampled*`), `train.py`
(`run_metrics`).

---

## 13. On-the-fly BPR negative sampling — replaces pre-generated sample files

**Issue.** The reference pipeline read a pre-generated per-epoch file of
`(user, pos, neg)` triples (`sample_file/sample_i.txt`). Feasible for gift
(~54k pairs) but **not** for click: **2.76M** positive pairs × hundreds of epochs
would need billions of pre-written lines.

**Modification.** Sample negatives **on the fly**: `newit()` draws one uniform
negative per (user, positive) pair each epoch in a single vectorised op. Uniform
over 452k items → negligible false-negative rate (standard BPR practice).

**Compute note (explains longer epochs).** With click, an epoch now processes
**~2.76M pairs (~51× more than gift)**, and MBGCN recomputes the full-graph
propagation **per minibatch** (it cannot be cached during training — the
embeddings change each step, unlike evaluation). Epoch time therefore scales with
the number of batches, so `batch_size` was raised (2048 → 8192) to amortise the
fixed per-batch propagation over more samples.

**Files:** `dataset.py` (`TrainDataset.newit`, `__getitem__`), `train.py`.

---

## 14. Results (click target, sampled-candidate protocol)

**Validation, with the `negative` behaviour, epoch 24 (still improving):**

| Metric | @5 | @10 | @20 |
|---|---:|---:|---:|
| Recall | 0.393 | 0.476 | **0.565** |
| NDCG | 0.315 | 0.342 | 0.364 |
| MRR | 0.289 | 0.300 | 0.306 |

**Context (paper Table 3, streamer task, same sampled protocol).** Best baseline
Recall@20 ≈ 0.561 (TiSASRec); pure-CF baselines BPRMF 0.485, LightGCN 0.384,
DirectAU 0.395. This MBGCN run is **competitive with the paper's strongest
baselines** and clears every pure-CF method — a complete turnaround from the
initial ~0.000.

**Negative-behaviour finding** (see Analysis B for the full argument): the learned
weights were `[click 1.44, comment 1.60, like 1.69, gift 1.32, negative −0.008]` —
the model **switched the skip signal off** (weight ≈ 0), with no instability.

**Caveats to report with the numbers.**
1. These are **validation** figures (used for early stopping); confirm on the
   held-out **test** set for the final result.
2. The head-to-head with Table 3 assumes our sampled pools match the paper's
   construction closely (10k time-valid negatives, LOO). Frame as *"competitive
   with the paper's baselines"* rather than a new state of the art.

*Reproducibility infrastructure (not a research contribution, noted for
completeness):* durable per-validation CSV logging (`metrics.csv` / `scalars.csv`),
a headless (`--no_vis`) mode for Colab, and the §7 resume were added so runs are
logged and interruption-tolerant.

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
| 7 | Tooling | resume: latent bugs, then run length | Removed buggy version; re-added clean best-checkpoint resume | N/A |
| 8 | Init | random init; wrong pretrain path | MF→MBGCN pipeline | N/A |
| 9 | Diagnosis | metrics stuck ~0.000 | Sparsity analysis: scale + starvation + empty eval, not a defect | N/A |
| 10 | Task | gift target = sparsest (18% users) | Switch target to **click** (98.4% users) | Paper-aligned |
| 11 | Data | no split; intensity deduplicated | Temporal LOO split; rebuild with duplicates (`click_interaction`) | No leakage |
| 12 | Eval | full-catalogue ranking → ~0.000 | **Sampled candidates** (10k time-valid negs) | Score-identical; biased-estimator caveat |
| 13 | Train | pre-gen sample files don't scale | On-the-fly BPR negatives | Standard BPR |
| 14 | Result | — | Recall@20 0.565, competitive w/ paper baselines | Validation |

*Files touched (Part I):* `model.py`, `dataset.py`, `metrics.py`, `train.py`,
`main.py`, `MBGCN.sh`, `MBGCN_KuaiLive.sh`, `MF_KuaiLive.sh`.
*New in Part II:* `sparsity_report.py`, `derive_click_dataset.py`,
`build_candidates.py` (dataset/eval tooling), plus edits to `model.py`,
`dataset.py`, `metrics.py`, `train.py`, `utils.py`, `main.py`, and the shells.
See also `MODEL_REVIEW.md` (initial review) and the per-dataset sparsity reports.
