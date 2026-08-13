#!/usr/bin/env bash
# Train ATRank on KuaiLive (click target, ID-only) under the MBGCN fair contract.
#
# Prereqs in dataset/click_interaction/: sequences.npz (derive_sequence_dataset.py)
# and candidates.npz. For warm-start (init parity with MBGCN), run MF_KuaiLive.sh
# first so output/click_interaction/click_interaction-MF-pretrain/model.pkl exists.
#
# Usage:  bash atrank/ATRank_KuaiLive.sh <mode> [batch]
#   mode  = headline | ablation_neg | ablation_noneg   (default: headline)
#   batch = 256 | 512 | 1024                            (default: 256)
#           lr is auto-scaled with batch (larger batch = fewer, lower-noise
#           updates per epoch, so lr goes up to keep per-epoch progress):
#             256 -> lr 1e-3   512 -> lr 1.5e-3   1024 -> lr 2e-3
# Each (mode,batch) writes to its OWN output prefix, so presets don't clobber
# each other's checkpoints/logs and --resume picks up the right run.
# Meant for a GPU box / Colab.
set -e
cd "$(dirname "$0")/.."

MODE="${1:-headline}"
BATCH="${2:-256}"
PRETRAIN="output/click_interaction/click_interaction-MF-pretrain"   # MF warm-start (option a)

# batch -> auto-scaled lr preset
case "${BATCH}" in
  256)  LR=1e-3   ;;
  512)  LR=1.5e-3 ;;
  1024) LR=2e-3   ;;
  *) echo "unknown batch '${BATCH}' (use: 256 | 512 | 1024)" >&2; exit 1 ;;
esac
echo ">>> mode=${MODE} batch=${BATCH} lr=${LR}"

COMMON=(
  --path dataset --dataset_name click_interaction
  --id_dim 64 --action_dim 64 --hidden 128
  --num_heads 8 --num_blocks 1 --dropout 0.1
  --reg 5e-5
  --epochs 400 --eval_every 2 --es_patience 5
  --batch_size "${BATCH}" --lr "${LR}"
  --test_batch_size 64 --cand_chunk 1024
  --num_workers 2 --gpu --resume
  --pretrain_path "${PRETRAIN}"
)

SAVE="output/click_interaction/atrank-${MODE}-b${BATCH}"

case "${MODE}" in
  headline)
    # RQ1: vanilla ATRank, global recency window, all 5 behaviours.
    python3 -m atrank.train "${COMMON[@]}" \
      --max_len 200 --behaviors 0,1,2,3,4 \
      --save "${SAVE}"
    ;;
  ablation_neg)
    # Ablation A: per-behaviour caps, negatives PRESENT (cap 50). Click depth fixed.
    python3 -m atrank.train "${COMMON[@]}" \
      --behaviors 0,1,2,3,4 --per_action_caps 0:50,1:30,2:30,3:30,4:50 \
      --save "${SAVE}"
    ;;
  ablation_noneg)
    # Ablation B: identical caps but negatives REMOVED (cap 0). Only diff vs A.
    python3 -m atrank.train "${COMMON[@]}" \
      --behaviors 0,1,2,3,4 --per_action_caps 0:50,1:30,2:30,3:30,4:0 \
      --save "${SAVE}"
    ;;
  *)
    echo "unknown mode '${MODE}' (use: headline | ablation_neg | ablation_noneg)" >&2
    exit 1
    ;;
esac
