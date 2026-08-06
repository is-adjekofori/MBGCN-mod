#!/bin/bash
#
# Stage 2 of the KuaiLive pipeline: train MBGCN, warm-started from the MF
# embeddings produced by MF_KuaiLive.sh (run that first).
#
source ./MBGCN.sh

# shell file for Kuailive (click target, sampled-candidate eval)
dataset_name='click_interaction'
relation='click,comment,like,gift,negative'   # target = click (train.txt), rest auxiliary
path='dataset'

# Sampled eval gathers [test_batch_size x (1 + 10000)] candidate embeddings;
# 256 keeps that comfortably within GPU memory (see build_candidates.py).
test_batch_size='256'

# Warm start: load the pretrained MF user/item embeddings as initialisation
# instead of random init. pretrain_path is the directory containing model.pkl
# (see MF_KuaiLive.sh -> output/<dataset_name>/<name>). embedding_size and the
# dataset must match the MF run.
create_embeddings='False'
pretrain_path='output/click_interaction/click_interaction-MF-pretrain'
no_vis='true'    # Colab: log to stdout instead of a visdom server

run
