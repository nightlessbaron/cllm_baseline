#!/bin/bash
#SBATCH --job-name=jf-coder-phase1
#SBATCH --partition=main
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --output=/mnt/weka/home/varad.pimpalkhute/diffusion/runs/coder-phase1-n16w16/slurm-%j.out
#SBATCH --error=/mnt/weka/home/varad.pimpalkhute/diffusion/runs/coder-phase1-n16w16/slurm-%j.err

# Phase 1 of paper-faithful Jacobi Forcing Coder training.
# Allocates 1 node (whole node for exclusive access), runs 4-GPU training
# (pinned via CUDA_VISIBLE_DEVICES inside the training script — the remaining
# 4 GPUs in the exclusive allocation stay idle to match the paper's batch=4).

mkdir -p /mnt/weka/home/varad.pimpalkhute/diffusion/runs/coder-phase1-n16w16

source /mnt/weka/home/varad.pimpalkhute/diffusion/env.sh

bash /mnt/weka/home/varad.pimpalkhute/diffusion/JacobiForcing/JacobiForcing/scripts/train/train_coder_phase1_n16w16.sh
