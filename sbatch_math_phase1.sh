#!/bin/bash
#SBATCH --job-name=jf-math-phase1
#SBATCH --partition=main
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --output=/mnt/weka/home/varad.pimpalkhute/diffusion/runs/math-phase1-n16w16/slurm-%j.out
#SBATCH --error=/mnt/weka/home/varad.pimpalkhute/diffusion/runs/math-phase1-n16w16/slurm-%j.err

# Phase 1 of paper-faithful Jacobi Forcing Math training.
# Mirror of Coder Phase 1 sbatch: 1 exclusive node, 4 GPUs used via
# CUDA_VISIBLE_DEVICES inside the script (other 4 idle to match paper's bs=4).

mkdir -p /mnt/weka/home/varad.pimpalkhute/diffusion/runs/math-phase1-n16w16

source /mnt/weka/home/varad.pimpalkhute/diffusion/env.sh

bash /mnt/weka/home/varad.pimpalkhute/diffusion/JacobiForcing/JacobiForcing/scripts/train/train_math_phase1_n16w16.sh
