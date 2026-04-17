# JacobiForcing reproduction — environment sourced per run.
# Usage: source /mnt/weka/home/varad.pimpalkhute/diffusion/env.sh

export REPRO_ROOT=/mnt/weka/home/varad.pimpalkhute/diffusion
export REPO=$REPRO_ROOT/JacobiForcing

# Determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

# NCCL — verbose on warnings only; single-node so no interface pinning
export NCCL_DEBUG=WARN

# Activate the conda env for this session
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jacobi_forcing
