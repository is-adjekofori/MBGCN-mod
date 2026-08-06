# Running MBGCN (click target) on Google Colab

Code comes from GitHub; the derived dataset lives on Google Drive. You do **not**
need the raw `KuaiLive/` CSVs on Colab — training only reads
`dataset/click_interaction/`.

## 0. One-time: put the derived data on Drive

From your local machine, upload the whole folder

```
dataset/click_interaction/      # click/train/validation/test.txt, aux .txt,
                                # candidates.npz (~1.4GB), data_size.txt, manifest.json
```

to somewhere on Drive, e.g. `MyDrive/fyp/click_interaction/`.

(Keep the raw CSVs + `derive_click_dataset.py` / `build_candidates.py` archived on
Drive too, for reproducibility — but they are not needed at train time.)

## 1. Clone the code

```python
!git clone https://github.com/<you>/<repo>.git /content/MBGCN_t
%cd /content/MBGCN_t
```

## 2. Install the few missing deps

```python
# torch/numpy/pandas/scipy/tqdm ship with Colab; this adds the rest.
!pip install -q absl-py setproctitle
```

## 3. Mount Drive and link the data into place

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('dataset', exist_ok=True)
# adjust the source path to wherever you uploaded it:
src = '/content/drive/MyDrive/fyp/click_interaction'
dst = '/content/MBGCN_t/dataset/click_interaction'
if not os.path.exists(dst):
    os.symlink(src, dst)
print(sorted(os.listdir(dst)))   # should list candidates.npz, train.txt, ...
```

`--path dataset` then resolves to the symlink, so the shell scripts work unchanged.

## 4. Train

Stage 1 — pretrain MF embeddings (writes `output/click_interaction/click_interaction-MF-pretrain/model.pkl`):

```python
!bash MF_KuaiLive.sh
```

Stage 2 — MBGCN, warm-started from those embeddings:

```python
!bash MBGCN_KuaiLive.sh
```

Both scripts pass `--no_vis=true`, so metrics stream to the cell output (no visdom
server needed). Reported metrics are Recall/NDCG/MRR @ {5, 10, 20} over the
sampled 10,001-candidate pools — directly comparable to the paper's Table 3
(streamer task: healthy Recall@20 ~ 0.38-0.49).

## 5. Persist outputs (optional)

`output/` is git-ignored and lives on the ephemeral Colab disk. To keep the best
model / logs across sessions, copy them to Drive when done:

```python
!cp -r output "/content/drive/MyDrive/fyp/output_$(date +%Y%m%d_%H%M)"
```

## Notes / gotchas

- **RAM:** `TestDataset` loads `val_neg` + `test_neg` (~1.9 GB total) at startup.
  Fine on a standard Colab runtime; if you hit limits, use a High-RAM runtime.
- **GPU:** set `Runtime > Change runtime type > GPU`. The shells already pass
  `--gpu=true --gpu_id 0`.
- **test_batch_size=256** is set for the sampled-eval candidate gather; lower it
  if you see CUDA OOM during validation, raise it to go faster on a big GPU.
- **Regenerating data on Colab instead of uploading:** upload the raw `KuaiLive/`
  CSVs to Drive, then run `!python derive_click_dataset.py --src KuaiLive --out
  dataset/click_interaction` and `!python build_candidates.py` (~17 min total).
