# Reference code (read-only, vendored)

Third-party source kept here **only as a reading reference** for our own
implementation. Nothing in this directory is imported, run, or modified by our
pipeline. It is namespaced under `reference/` precisely so it cannot collide with
our flat MBGCN files or our future ATRank package (both define `model.py`,
`train.py`, etc.).

## `atrank_upstream/` — original ATRank authors' code

- **Paper:** Zhou, Bai, Song, Liu, Zhao, Chen, Gao. *ATRank: An Attention-Based
  User Behavior Modeling Framework for Recommendation.* AAAI 2018.
  (arXiv:1711.06632 — the PDF we studied).
- **Source:** https://github.com/jinze1994/ATRank.git
- **Vendored commit:** `2031fd112c0bf5c84beba284492bc4c7dcf476e7` (2018-05-11)
- **License:** Apache 2.0 (see `atrank_upstream/LICENSE`, preserved verbatim).
- **Framework:** TensorFlow 1.x + Python 3.6 (old `tf.contrib` era). We do **not**
  run this; we read it to get the math and data-flow exactly right, then
  reimplement in PyTorch against KuaiLive.

### Which folders matter to us

| Folder | What it is | Relevance |
|--------|-----------|-----------|
| `multi/` | **Heterogeneous multi-behaviour** ATRank (the private-dataset version from the paper). `model.py` is the biggest (30 KB). | **Highest** — KuaiLive is multi-behaviour; this is the closest analogue to what we build. |
| `atrank/` | Single-behaviour ATRank on the public Amazon Electronics data. | High — cleanest reference for the core self-attention + vanilla-attention + scoring path. |
| `bpr/`, `cnn/`, `rnn/`, `rnn_att/` | The paper's baselines. | Low — context only; note `bpr/` shows their pairwise setup. |
| `utils/` | Amazon data download + id-remap preprocessing. | Low — Amazon-specific; our data pipeline is separate. |

Per the upstream README, the multi-behaviour dataset in the paper is private, so
`multi/` was never runnable end-to-end publicly — it is read as the reference for
the heterogeneous-behaviour construction only.

## How this relates to our work

Our implementation is a **PyTorch reimplementation adapted to KuaiLive**, not a
fork of this code. It will live in its own top-level package (planned: `atrank/`
at the repo root), obey the fair-comparison contract with MBGCN (same click
target, same LOO split, same `candidates.npz` pools, same metrics), and use
ID-only object embeddings. See `docs/THESIS_WRITEUP_CHANGES.md` (A1–A4) and
`docs/THESIS_MODIFICATIONS.md` for the design decisions.
