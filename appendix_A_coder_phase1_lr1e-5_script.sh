#!/bin/bash
# Paper-faithful Phase 1 of Jacobi Forcing Coder — but with LR=1e-5 (10x paper's 1e-6).
# Hypothesis: paper's 1e-6 is calibrated for resuming from an unreleased intermediate;
# starting from vanilla Qwen2.5-Coder-7B-Instruct needs more aggressive LR over the same
# 10k-step budget. Prior run at 1e-6 gave only ~1.12x inference speedup vs paper's 4x.
# See docs/reproduction_notes.md §9.3 for the full hypothesis.
#
# Other hparams match paper: bs=4 (4 GPUs x bs=1), max_len=2048, n=16, 10k steps.
# All 10 save_steps=1000 checkpoints retained (+ final top-level save = 11 states).

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=cllm2_training
export WANDB_RUN_NAME="coder-phase1-n16w16-lr1e-5"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

REPRO_ROOT=/mnt/weka/home/varad.pimpalkhute/diffusion
SHARED_ROOT=/mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline

# Models + data live on shared fs; upstream code (with our patches) is local.
model_path="${SHARED_ROOT}/models/base-coder"
trajectory_file="${SHARED_ROOT}/data/OpenCodeInstruct_training_data_n16w16/40k_samples_merged_data_v2_8_30_opencodeinstruct_progressive_noise_cyclic_all.jsonl"
output_path="${SHARED_ROOT}/runs/coder-phase1-n16w16-lr1e-5"
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
    --save_total_limit 15 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --qlora ${qlora}
