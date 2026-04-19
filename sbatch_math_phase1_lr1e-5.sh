#!/bin/bash
#SBATCH --job-name=jf-math-p1-lr1e5
#SBATCH --partition=main
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --output=/mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline/runs/math-phase1-n16w16-lr1e-5/slurm-%j.out
#SBATCH --error=/mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline/runs/math-phase1-n16w16-lr1e-5/slurm-%j.err

mkdir -p /mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline/runs/math-phase1-n16w16-lr1e-5

source /mnt/weka/home/varad.pimpalkhute/diffusion/env.sh

bash /mnt/weka/home/varad.pimpalkhute/diffusion/JacobiForcing/JacobiForcing/scripts/train/train_math_phase1_n16w16_lr1e-5.sh
