#!/bin/bash
#
# Stage 1 of the KuaiLive pipeline: pretrain a plain MF model. Its learned
# user/item embeddings are then used to initialise MBGCN.
#
#   1. bash MF_KuaiLive.sh        # this file  -> writes output/<ds>/<name>/model.pkl
#   2. bash MBGCN_KuaiLive.sh     # loads those embeddings as its starting point
#
set -euo pipefail

model='MF'
dataset_name='click_interaction'
path='dataset'

# MF does NOT use the behaviour graphs at all -- its embeddings come only from
# the target interactions in train.txt (+ the sampled BPR negatives). But
# TrainDataset still builds a relation matrix per behaviour, so we pass just the
# smallest behaviour file here to keep data loading fast. (The MBGCN stage uses
# the full relation list.)
relation='gift'

create_embeddings='True'          # MF itself starts from random embeddings
gpu='true'
gpu_id='0'
embedding_size='64'               # MUST match the MBGCN embedding_size
lr='1e-2'
L2='1e-4'
epoch='400'
eval_every='5'
es_patience='10'
batch_size='8192'              # ~2.76M click pairs/epoch; large batch cuts overhead
test_batch_size='256'          # sampled eval gathers [B x (1+10000)] candidates
num_workers='2'

# Fixed name so MBGCN_KuaiLive.sh can reference the output directory.
name='click_interaction-MF-pretrain'

echo ">>> pretraining MF: ${name}"
python main.py \
    --name "${name}" \
    --model "${model}" \
    --gpu="${gpu}" \
    --gpu_id "${gpu_id}" \
    --dataset_name "${dataset_name}" \
    --path "${path}" \
    --no_vis=true \
    --relation "${relation}" \
    --create_embeddings "${create_embeddings}" \
    --embedding_size "${embedding_size}" \
    --lr "${lr}" \
    --L2_norm "${L2}" \
    --epoch "${epoch}" \
    --eval_every "${eval_every}" \
    --es_patience "${es_patience}" \
    --batch_size "${batch_size}" \
    --test_batch_size "${test_batch_size}" \
    --num_workers "${num_workers}"

echo ">>> MF pretraining done."
echo ">>> best model saved at: output/${dataset_name}/${name}/model.pkl"
echo ">>> point MBGCN_KuaiLive.sh pretrain_path at: output/${dataset_name}/${name}"
