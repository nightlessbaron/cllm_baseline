#!/bin/bash
# Paper-faithful Phase 1 of Jacobi Forcing Coder training.
# Derived from train_jacobi_forcing_coder_n32.sh with paper's (arXiv 2512.14681)
# hyperparameters substituted:
#   - learning_rate: 1e-5 (script default) -> 1e-6 (paper)
#   - model_max_length: 16384 (script default) -> 2048 (paper)
#   - num_train_epochs=1 (script default) -> max_steps=10000 (paper, per phase)
#   - dataset: n32w16 (script default) -> n16w16 (paper Phase 1)
#   - block/window: n=32 w=16 -> n=16 w=16 (paper Phase 1)
# Phase 2 will continue from Phase 1's final checkpoint with n=32 w=16 data.
#
# wandb: online (not offline). Requires `wandb login` to have been run once.

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=cllm2_training
export WANDB_RUN_NAME="coder-phase1-n16w16-paperhparams"
# Leave WANDB_MODE unset -> online.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

REPRO_ROOT=/mnt/weka/home/varad.pimpalkhute/diffusion

model_path="${REPRO_ROOT}/models/base-coder"
trajectory_file="${REPRO_ROOT}/data/OpenCodeInstruct_training_data_n16w16/40k_samples_merged_data_v2_8_30_opencodeinstruct_progressive_noise_cyclic_all.jsonl"
output_path="${REPRO_ROOT}/runs/coder-phase1-n16w16"
n_token_seq_size=16            # paper Phase 1: block size n=16
qlora=False

mkdir -p "${output_path}"

cd ${REPRO_ROOT}/JacobiForcing/JacobiForcing

torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=101 \
    --rdzv_endpoint='localhost:5666' \
    --master_port 10000 \
    train/soft_flexattn_train_cllm_multiblock.py \
    --target_model_path "${model_path}" \
    --data_path "${trajectory_file}" \
    --output_dir "${output_path}" \
    --max_new_tokens ${n_token_seq_size} \
    --bf16 True \
    --report_to wandb \
    --do_train \
    --max_steps 10000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --qlora ${qlora}
