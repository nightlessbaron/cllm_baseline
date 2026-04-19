#!/bin/bash
# Paper-faithful Phase 1 of Jacobi Forcing Math — LR=1e-5 variant (10x paper's 1e-6).
# See train_coder_phase1_n16w16_lr1e-5.sh for the LR-bump rationale.
# Dual NaN guard in CllmTrainer._one_pass_losses_step remains active; needed because
# Math + bf16 + vanilla Qwen2.5-Math-7B-Instruct init produces sporadic grad NaN
# (non-deterministic; see docs/reproduction_notes.md §5.4).

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=cllm2_training
export WANDB_RUN_NAME="math-phase1-n16w16-lr1e-5"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

REPRO_ROOT=/mnt/weka/home/varad.pimpalkhute/diffusion
SHARED_ROOT=/mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline

# Models + data live on shared fs; upstream code (with our patches) is local.
model_path="${SHARED_ROOT}/models/base-math"
trajectory_file="${SHARED_ROOT}/data/OpenThoughts_Math_training_data_n16w16/40k_samples_seed42_maxlen2048.jsonl"
output_path="${SHARED_ROOT}/runs/math-phase1-n16w16-lr1e-5"
n_token_seq_size=16
qlora=False

mkdir -p "${output_path}"

cd ${REPRO_ROOT}/JacobiForcing/JacobiForcing

# Different master_port + rdzv_id than Coder to avoid collision if co-scheduled.
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=102 \
    --rdzv_endpoint='localhost:5667' \
    --master_port 20000 \
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
