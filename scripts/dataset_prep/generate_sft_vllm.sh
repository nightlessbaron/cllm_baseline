#!/bin/bash
# Launches one vLLM generator process per GPU on the current node, each processing
# its own shard of the data. Uses SLURM_ARRAY_TASK_ID to pick the node's stride.
#
# Total shards = SLURM_ARRAY_TASK_COUNT * N_GPUS.
# Process (task_id, gpu_id) handles shard_idx = task_id * N_GPUS + gpu_id.

set -e

REPRO_ROOT=/mnt/weka/home/varad.pimpalkhute/diffusion
SHARED_ROOT=/mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline

MODEL_PATH="${MODEL_PATH:-${SHARED_ROOT}/models/base-math}"
INPUT_DIR="${INPUT_DIR:-${SHARED_ROOT}/data/openthoughts3_math_boxed}"
OUTPUT_DIR="${OUTPUT_DIR:-${SHARED_ROOT}/data/openthoughts3_math_sft}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3584}"
N_GPUS="${N_GPUS:-8}"

N_TASK="${SLURM_ARRAY_TASK_COUNT:-1}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS=$(( N_TASK * N_GPUS ))

mkdir -p "${OUTPUT_DIR}"
cd "${REPRO_ROOT}"

echo "[task ${TASK_ID}/${N_TASK}] firing ${N_GPUS} vLLM processes  num_shards=${NUM_SHARDS}"

for g in $(seq 0 $((N_GPUS - 1))); do
    shard_idx=$(( TASK_ID * N_GPUS + g ))
    log="${OUTPUT_DIR}/sft_shard_$(printf '%04d' $shard_idx).gen.log"
    echo "[task ${TASK_ID}][gpu ${g}] shard_idx=${shard_idx}/${NUM_SHARDS}  log=${log}"
    CUDA_VISIBLE_DEVICES=${g} python3 \
        scripts/dataset_prep/generate_sft_vllm.py \
        --input_dir "${INPUT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --model "${MODEL_PATH}" \
        --max_total_len "${MAX_TOTAL_LEN}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --shard_idx "${shard_idx}" \
        --num_shards "${NUM_SHARDS}" \
        > "${log}" 2>&1 &
done

wait
echo "[task ${TASK_ID}/${N_TASK}] all ${N_GPUS} shards finished."
