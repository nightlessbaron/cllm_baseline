#!/bin/bash
# Paper-faithful Phase 1 of Jacobi Forcing Math training.
# Mirrors train_coder_phase1_n16w16.sh with math paths + base-math model.
# Paper recipe (arXiv 2512.14681, Section 4.1):
#   - LR=1e-6, bs=4, max_seq_len=2048, 10k steps
#   - Phase 1: n=16, w=16
#   - Starting model: Qwen2.5-Math-7B-Instruct
#
# Data note: OpenThoughts_Math_training_data_n16w16 ships as parquet (not JSONL
# like the Coder dataset) and is bucket-sorted by sequence length. We first
# filter to rows with complete_training_sequence_ids length <= 2048 (paper's
# max_seq_len — drops 10.4% of the 250k-row dataset), then draw a uniform
# random 40k sample at seed=42. Output: 40k_samples_seed42_maxlen2048.jsonl.
# Covers all 54 length buckets (310-1040 rows/bucket). This mirrors upstream's
# Coder 40k_samples file (which was already pre-filtered for <=2048); upstream
# did not ship an equivalent subset for Math. See Appendix A in docs/training_plan.md.

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=cllm2_training
export WANDB_RUN_NAME="math-phase1-n16w16-paperhparams"
# WANDB_MODE unset -> online

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

REPRO_ROOT=/mnt/weka/home/varad.pimpalkhute/diffusion

model_path="${REPRO_ROOT}/models/base-math"
trajectory_file="${REPRO_ROOT}/data/OpenThoughts_Math_training_data_n16w16/40k_samples_seed42_maxlen2048.jsonl"
output_path="${REPRO_ROOT}/runs/math-phase1-n16w16"
n_token_seq_size=16            # paper Phase 1: block size n=16
qlora=False

mkdir -p "${output_path}"

cd ${REPRO_ROOT}/JacobiForcing/JacobiForcing

# Use different master_port + rdzv_id than Coder Phase 1 (master_port=10000, rdzv_id=101)
# to avoid collision if ever scheduled on the same node.
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
    --save_total_limit 5 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --qlora ${qlora}
