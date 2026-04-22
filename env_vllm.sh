# Environment for vLLM-based SFT data generation.
# Uses a separate conda env (sync-rl-v1) to avoid flash-attn/torch ABI conflicts
# with the jacobi_forcing env used for JacobiForcing training.
# Usage: source /mnt/weka/home/varad.pimpalkhute/diffusion/env_vllm.sh

export REPRO_ROOT=/mnt/weka/home/varad.pimpalkhute/diffusion
export REPO=$REPRO_ROOT/JacobiForcing

# NCCL — verbose on warnings only
export NCCL_DEBUG=WARN

# Activate the vLLM conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-serving
