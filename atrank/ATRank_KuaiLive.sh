#!/usr/bin/env bash
# Train ATRank on KuaiLive (click target, ID-only) under the MBGCN fair contract.
#
# Prereqs in dataset/click_interaction/: sequences.npz (derive_sequence_dataset.py)
# and candidates.npz. For warm-start (init parity with MBGCN), run MF_KuaiLive.sh
# first so output/click_interaction/click_interaction-MF-pretrain/model.pkl exists.
#
# Usage:  bash atrank/ATRank_KuaiLive.sh [headline|ablation_neg|ablation_noneg]
# Meant for a GPU box / Colab. Every run resumes from its own .ckpt if interrupted.
set -e
cd "$(dirname "$0")/.."

MODE="${1:-headline}"
PRETRAIN="output/click_interaction/click_interaction-MF-pretrain"   # MF warm-start (option a)

COMMON=(
  --path dataset --dataset_name click_interaction
  --id_dim 64 --action_dim 64 --hidden 128
  --num_heads 8 --num_blocks 1 --dropout 0.1
  --reg 5e-5 --lr 1e-3
  --epochs 400 --eval_every 5 --es_patience 10
  --batch_size 256 --test_batch_size 64 --cand_chunk 1024
  --num_workers 2 --gpu --resume
  --pretrain_path "${PRETRAIN}"
)

case "${MODE}" in
  headline)
    # RQ1: vanilla ATRank, global recency window, all 5 behaviours.
    python3 -m atrank.train "${COMMON[@]}" \
      --max_len 200 --behaviors 0,1,2,3,4 \
      --save output/click_interaction/atrank-headline
    ;;
  ablation_neg)
    # Ablation A: per-behaviour caps, negatives PRESENT (cap 50). Click depth fixed.
    python3 -m atrank.train "${COMMON[@]}" \
      --behaviors 0,1,2,3,4 --per_action_caps 0:50,1:30,2:30,3:30,4:50 \
      --save output/click_interaction/atrank-ablation-neg
    ;;
  ablation_noneg)
    # Ablation B: identical caps but negatives REMOVED (cap 0). Only diff vs A.
    python3 -m atrank.train "${COMMON[@]}" \
      --behaviors 0,1,2,3,4 --per_action_caps 0:50,1:30,2:30,3:30,4:0 \
      --save output/click_interaction/atrank-ablation-noneg
    ;;
  *)
    echo "unknown mode '${MODE}' (use: headline | ablation_neg | ablation_noneg)" >&2
    exit 1
    ;;
esac
