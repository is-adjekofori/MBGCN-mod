# Project Documentation Index

Working documents for the thesis *"Comparative Evaluation of Graph-Based and
Sequential Multi-Behaviour Recommendation Architectures in a Live Streaming
Domain"* (Adjekofori Israel Oghenefejiro, PSC2207846).

The study compares two architecture families on KuaiLive:
**MBGCN** (graph-based, Part I — built) and **ATRank** (sequential/attention,
Part II — in design).

## Documents

| File | What it is | When to read it |
|------|-----------|-----------------|
| [`THESIS_WRITEUP_CHANGES.md`](THESIS_WRITEUP_CHANGES.md) | **Change list** reconciling `Thesis_Chapter1_Draft2.md` with what was actually built/decided. Flags the gift→click and room→streamer gaps and the ID-only side-feature decision. | **Start here** before editing the thesis. |
| [`THESIS_MODIFICATIONS.md`](THESIS_MODIFICATIONS.md) | The engineering change-log. §1–8 = Part I (MBGCN fixes: sparse ops, memory, eval split, resume, MF pretraining). §9–14 = Part II (gift→click re-target, temporal LOO, sampled-candidate eval, BPR negatives, results). | To see *why* the code is the way it is. |
| [`MODEL_REVIEW.md`](MODEL_REVIEW.md) | Correctness + efficiency review of the MBGCN implementation (item–item bypass proof, the 43 GB densification bug, the `negative`-behaviour concern). | Background on the Part I fixes. |
| [`COLAB_SETUP.md`](COLAB_SETUP.md) | How to run the click-target MBGCN pipeline on Google Colab (Drive data layout, pretrain → train). | To reproduce a run. |

## Related files (not in `docs/`)

- `../Thesis_Chapter1_Draft2.md` — the thesis draft being reconciled (kept at repo root).
- `../README.md` — upstream MBGCN authors' README (Jin et al., 2020); not our work.
- `../reference/` — vendored **read-only** third-party source. Currently
  `reference/atrank_upstream/` = the original ATRank authors' TensorFlow code
  (Zhou et al., 2018, Apache 2.0), kept as a reading reference for our PyTorch
  reimplementation. See `../reference/README.md`.
- `../memory/click-retarget-pipeline.md` — persistent memory note on the click re-target pipeline.

## Current decisions locked

- **Target behaviour:** click (matches the implemented MBGCN pipeline and `candidates.npz`).
- **Item granularity:** `streamer_id` (persistent); room ephemerality captured via time-valid candidate pools.
- **Side features:** ID-only for the primary comparison; side-feature ablation kept as a separate, labelled experiment.
- **Fairness contract:** ATRank reuses the identical split, candidate pools, target, and metrics as MBGCN — architecture is the only moving part.
