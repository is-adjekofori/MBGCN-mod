#!/bin/bash
#
# Shared launcher for MBGCN / MF experiments.
#
# Dataset-specific wrappers (e.g. MBGCN_KuaiLive.sh) do:
#     source ./MBGCN.sh      # loads defaults + the `run` function
#     dataset_name='...'      # override whatever they need
#     relation='a,b,c'
#     run                     # launch with the current settings
#
set -euo pipefail

# ---- defaults (override any of these before calling `run`) ----
model='MBGCN'
dataset_name='Tmall'
gpu='true'                       # absl bool: use --gpu=true / --gpu=false
gpu_id='0'
create_embeddings='False'
es_patience='10'
embedding_size='64'
lamb='1'
# MBGCN recomputes the full-graph propagation once PER BATCH, so that fixed cost
# is amortised over the batch. With the click target (~2.76M pairs/epoch) a large
# batch means far fewer propagations per epoch -> much faster epochs. Drop this if
# you hit CUDA OOM (e.g. 4096 on a 16GB T4).
num_workers='2'                  # overlap CPU batch prep with GPU compute
path='./'
epoch='400'
eval_every='5'                   # run (expensive) validation every N epochs
batch_size='8192'
test_batch_size='512'
loss_mode='mean'
no_vis='false'                   # set 'true' on headless envs (Colab) to skip visdom
resume='false'                   # 'true' to continue from the best saved checkpoint
exp_tag='fyp'                    # suffix used in the experiment name

relation='buy,cart,collect,click'

# Per-behaviour MGNN weights, in the SAME order as `relation`.
# Leave empty () to default every behaviour to 1.0. To override, set e.g.
#   mgnn_weights=(1 1 0.5 -0.2)
# The length is validated against the number of behaviours in `relation`.
mgnn_weights=()

pretrain_path='/content/drive/MyDrive/Personal/Project/MBGCN/output/Tmall/Tmall-MF_lr1e-2-L1e-2-size64@jinbowen/'

lr='3e-4'
L2='1e-4'
message_dropout=('0.2')
node_dropout=('0.2')

# Build one "--mgnn_weight <w>" pair per behaviour into the _mgnn_args array.
build_mgnn_weight_args() {
    local IFS=','
    read -ra _rels <<< "${relation}"
    local n=${#_rels[@]}

    if [ "${n}" -eq 0 ]; then
        echo "ERROR: 'relation' is empty." >&2
        exit 1
    fi

    _mgnn_args=()
    if [ ${#mgnn_weights[@]} -eq 0 ]; then
        local i
        for ((i = 0; i < n; i++)); do
            _mgnn_args+=(--mgnn_weight 1)
        done
    else
        if [ ${#mgnn_weights[@]} -ne "${n}" ]; then
            echo "ERROR: mgnn_weights has ${#mgnn_weights[@]} entries but 'relation' has ${n} behaviours (${relation})." >&2
            exit 1
        fi
        local w
        for w in "${mgnn_weights[@]}"; do
            _mgnn_args+=(--mgnn_weight "${w}")
        done
    fi
}

run() {
    build_mgnn_weight_args

    local md nd name
    for md in "${message_dropout[@]}"; do
        for nd in "${node_dropout[@]}"; do
            name="${dataset_name}-${model}_lr${lr}-L${L2}-size${embedding_size}-lamb${lamb}-md${md}-nd${nd}@${exp_tag}"

            echo ">>> launching: ${name}"
            python main.py \
                --name "${name}" \
                --model "${model}" \
                --gpu="${gpu}" \
                --gpu_id "${gpu_id}" \
                --dataset_name "${dataset_name}" \
                --path "${path}" \
                --L2_norm "${L2}" \
                --lr "${lr}" \
                --epoch "${epoch}" \
                --eval_every "${eval_every}" \
                --batch_size "${batch_size}" \
                --test_batch_size "${test_batch_size}" \
                --loss_mode "${loss_mode}" \
                --no_vis="${no_vis}" \
                --resume="${resume}" \
                --create_embeddings "${create_embeddings}" \
                --es_patience "${es_patience}" \
                --embedding_size "${embedding_size}" \
                "${_mgnn_args[@]}" \
                --lamb "${lamb}" \
                --relation "${relation}" \
                --message_dropout "${md}" \
                --node_dropout "${nd}" \
                --pretrain_path "${pretrain_path}" \
                --num_workers "${num_workers}"
        done
    done
}
