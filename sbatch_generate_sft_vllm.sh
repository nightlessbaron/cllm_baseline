#!/bin/bash
#SBATCH --job-name=jf-sft-vllm
#SBATCH --partition=main
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --array=0-31
#SBATCH --output=/mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline/data/openthoughts3_math_sft/slurm-%A_%a.out
#SBATCH --error=/mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline/data/openthoughts3_math_sft/slurm-%A_%a.err

# 32 array tasks × 1 node × 8 GPUs = 256 parallel vLLM generator processes.
# Each process handles 1/256 of the data; shard filename = sft_shard_NNNN_of_0256.jsonl

mkdir -p /mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline/data/openthoughts3_math_sft

source /mnt/weka/home/varad.pimpalkhute/diffusion/env_vllm.sh

bash /mnt/weka/home/varad.pimpalkhute/diffusion/scripts/dataset_prep/generate_sft_vllm.sh
