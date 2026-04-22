#!/bin/bash
#SBATCH --job-name=jf-gen-traj-math
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --array=0-31
#SBATCH --output=/mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline/data/openthoughts3_math_greedy_blk16/slurm-%A_%a.out
#SBATCH --error=/mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline/data/openthoughts3_math_greedy_blk16/slurm-%A_%a.err

# 32 array tasks × 1 node × 8 GPUs = 256 parallel generator processes.
# Each task handles its stride of the bucket list (see launcher).

mkdir -p /mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline/data/openthoughts3_math_greedy_blk16

source /mnt/weka/home/varad.pimpalkhute/diffusion/env.sh

bash /mnt/weka/home/varad.pimpalkhute/diffusion/JacobiForcing/generate_trajectory/generation/generate_trajectory_openthoughts3_math_greedy.sh
