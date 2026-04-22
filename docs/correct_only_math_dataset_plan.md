# Correct-Only Math Dataset — Plan & Outcome

Started as a plan to generate clean correctness-filtered Jacobi trajectories for
Phase-1 training; ended with a fully-built pipeline plus a **pivot to pure SFT
via vLLM** when we realised Jacobi generation from a base model gives no speedup
advantage at data-gen time. This doc records both the original plan, the actual
implementation, and the learnings so we don't re-derive them.

---

## 1. Inputs / outputs at a glance

| | |
|---|---|
| Base model | `Qwen2.5-Math-7B-Instruct` at `${SHARED_ROOT}/models/base-math` (same weights used as `--target_model_path` in `train_math_phase1_n16w16_lr1e-5_constant.sh`) |
| Raw source | `open-thoughts/OpenThoughts3-1.2M` (120 parquet shards, 1.2M rows). Math domain = shards **25-109** (85 shards, 850k rows, all from `ai2-adapt-dev/openmath-2-math`) |
| Filtered source artifact | `melhoushi/OpenThoughts3_math_boxed` — math rows with extractable `\boxed{}` gold, **306k rows** (~36% of 850k) |
| Final SFT artifact | `melhoushi/OpenThoughts3_math_qwen_sft` — **80,670** correct `(prompt, response)` JSONL pairs (26.15% accept rate) |
| Wall time (SFT gen) | **~4 min on 32 nodes × 8 H100** (14 prompts/sec/GPU via vLLM greedy) |

---

## 2. Pipeline — what actually runs

Two parallel pipelines exist: **SFT (shipped)** and **Jacobi (built, archived)**. SFT is what produced the uploaded dataset.

### 2a. SFT pipeline (shipped)

```
open-thoughts/OpenThoughts3-1.2M  (HF, 120 parquet shards)
        │
        │   scripts/dataset_prep/filter_openthoughts3_math.py
        │     --start 25 --end 110  --boxed_only
        ▼
${SHARED_ROOT}/data/openthoughts3_math_boxed/*.parquet   (306k rows)
        │   (also uploaded to melhoushi/OpenThoughts3_math_boxed)
        │
        │   scripts/dataset_prep/generate_sft_vllm.py
        │     --max_total_len 4096  --max_new_tokens 3584
        │     --shard_idx k --num_shards 256  (greedy: temperature=0)
        │   launched by scripts/dataset_prep/generate_sft_vllm.sh + sbatch_generate_sft_vllm.sh
        ▼
${SHARED_ROOT}/data/openthoughts3_math_sft/sft_shard_*.jsonl   (256 shards)
        │   (each process: vLLM continuous-batched greedy + math_verify correctness filter)
        │
        │   cat *.jsonl > merged.jsonl
        ▼
${SHARED_ROOT}/data/openthoughts3_math_sft_merged.jsonl  (80,670 lines, 399MB)
        │
        │   huggingface_hub.upload_file(...)
        ▼
melhoushi/OpenThoughts3_math_qwen_sft (public HF dataset)
```

**Per-record schema (final JSONL)**:
```json
{
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "problem": "...",        // raw user turn
  "response": "...",       // model's greedy CoT + \boxed{answer}
  "gold_answer": "42",     // extracted from source
  "pred_answer": "42",     // extracted from response's last \boxed{}
  "source": "ai2-adapt-dev/openmath-2-math"
}
```

### 2b. Jacobi pipeline (built, not shipped)

Files are in place under `JacobiForcing/generate_trajectory/` and `sbatch_generate_trajectory_openthoughts3_math.sh`. Works end-to-end with batch_size=8. Not run to completion because we chose SFT.

Stages:
1. `data/0_bucketing_openthoughts3_math_boxed.py` — prompt-only bucketing with Qwen2.5-Math tokenizer, bucket_size=1200 → 258 buckets
2. `generation/generate_trajectory_openthoughts3_math_greedy.py` — Jacobi greedy trajectory gen, bs=8 with min-acceptance (see §5)
3. `data/1_5_openthoughts3_correctness_filter.py` — `\boxed{}` extraction + math_verify, per-`data_id` decision (keeps all iteration records of correct trajectories)
4. Stage 2 / Stage 3: unchanged upstream `2_prepare_efficient_cllm_training_data_new_progressive_noise_cyclic.py` / length filter

**Measured**: 12.8 s/prompt amortized at bs=8 on one H100, 2.3× speedup vs upstream bs=1. Projected full run: ~4 hr on 256 GPUs. Not launched.

---

## 3. File inventory

### Created / edited this session

| Path | Role |
|---|---|
| `scripts/dataset_prep/filter_openthoughts3_math.py` | OpenThoughts3 → math-only + boxed filter + optional HF upload |
| `scripts/dataset_prep/generate_sft_vllm.py` | vLLM greedy SFT generator with math_verify filter |
| `scripts/dataset_prep/generate_sft_vllm.sh` | Per-node 8-GPU launcher |
| `sbatch_generate_sft_vllm.sh` | 32-array sbatch, 4 hr walltime |
| `env_vllm.sh` | Separate conda env loader (`vllm-serving`) — avoids flash-attn/torch ABI conflict with `jacobi_forcing` env |
| `JacobiForcing/generate_trajectory/data/0_bucketing_openthoughts3_math_boxed.py` | Jacobi Stage 0 bucketer for our filtered source |
| `JacobiForcing/generate_trajectory/data/1_5_openthoughts3_correctness_filter.py` | Jacobi Stage 1.5 correctness filter |
| `JacobiForcing/generate_trajectory/generation/generate_trajectory_openthoughts3_math_greedy.py` | Jacobi Stage 1 trajectory generator (bs=1 / bs=8) |
| `JacobiForcing/generate_trajectory/generation/generate_trajectory_openthoughts3_math_greedy.sh` | Jacobi per-node launcher, supports Slurm array stride partition |
| `sbatch_generate_trajectory_openthoughts3_math.sh` | Jacobi 32-array sbatch |

### Upstream files patched (in `JacobiForcing/generate_trajectory/generation/qwen2_modeling_jacobi_forcing_greedy.py`)

Three edits to `get_jacobi_forward_trajectory_greedy` to support batch_size > 1:
- **Line 356** (prefill softmax+argmax): removed the `.squeeze(0).unsqueeze(0)` dance that was a no-op at B=1 but silently added a phantom dim at B>1.
- **Line 426** (decode softmax): same fix.
- **Line 437** (min-acceptance): `num_accepted = int(accepted[0])` → `int(accepted.min())` so KV-cache deletion stays consistent across a heterogeneous batch. Over-accepting samples re-propose their extra tokens next iteration — valid Jacobi trajectory, just more iterations.

---

## 4. Data characteristics (measured)

From 306k filtered OpenThoughts3-math prompts:

| Outcome | Count | % |
|---|---|---|
| **Correct** (math_verify matches gold) | 80,670 | **26.15%** |
| No `\boxed{}` emitted in response | 69,135 | 22.41% |
| Mismatch (`\boxed{}` present but wrong) | 158,677 | 51.44% |

Filter funnel from raw OpenThoughts3 math:
- 850k math rows
- → 306k have extractable `\boxed{}` in the gpt turn (36%)
- → 80k survive greedy+correctness (26% of 306k, 9% of 850k)

---

## 5. Key learnings (save the next session a week)

1. **"Corrupted" in JacobiForcing ≠ "incorrect".** It refers to **intermediate Jacobi refinement states** — partially-converged blocks that haven't stabilised. The Jacobi Forcing objective *needs* these; they are the training signal. Filtering by correctness should operate on the **final** `teacher_output_ids` per `data_id`, not on intermediate `answer_trajectory_ids`.

2. **Jacobi decoding on a base model degenerates to AR.** The base Qwen2.5-Math hasn't been trained to accept multi-token Jacobi blocks, so each step accepts ~1-2 tokens. Wall clock is near-AR. Jacobi's speed benefit requires a Jacobi-trained model. → **For data generation from a base model, use vLLM AR. ~20-30× faster.**

3. **`\boxed{}` coverage in OpenThoughts3 math is only 36%.** The other 64% of rows have valid reasoning + a final answer, but the answer isn't `\boxed{}`-wrapped. Recovering them would require fuzzy-matching prompts against `nvidia/OpenMathInstruct-2`'s `expected_answer` field (the upstream grandparent dataset). Not worth it at our scale — 306k × 26% = 80k already covers 2× our 40k training need.

4. **Batch>1 in upstream Jacobi has three hidden bugs.** Don't just bump a flag:
   - `LogitsProcessorList` wrap used `.squeeze(0).unsqueeze(0)` which assumed B=1 (prefill & decode both).
   - `num_accepted = int(accepted[0])` picks only the first batch element's acceptance count — must be `int(accepted.min())` so the KV cache deletion is consistent.
   - `still_active` masking of input_ids without matching KV-cache surgery is incorrect — safer to run all samples to EOS-or-max and skip recording records for post-EOS iterations.

5. **Prompt-length bucketing doesn't help vLLM.** vLLM's continuous batching handles variable-length prompts natively (PagedAttention). We only bucket for Jacobi's fixed-batch-shape requirement.

6. **Separate conda env for vLLM.** `flash-attn` in the `jacobi_forcing` env is pinned to the training-era torch ABI; vLLM brings a different torch and the symbol mismatch (`undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`) crashes engine init. Solution: `vllm-serving` conda env loaded via `env_vllm.sh`. Do **not** `pip uninstall flash-attn` in the training env.

7. **vLLM engine-init race.** Starting 8 vLLM engines simultaneously on one node occasionally deadlocks (~0.8% failure rate in our 256-shard run: 2/256 failed). Design retries into the workflow — the script already supports resumable shard-by-shard reruns.

8. **Slurm `--array=0-31` does run as 32 concurrent nodes** on this cluster (not serialised), as long as the cluster policy doesn't throttle arrays. Confirmed by running 32 concurrent job IDs in both runs.

9. **Trajectory storage scales with generation length.** Upstream stored `teacher_output_ids` on every iteration record → ~32 MB/sample at 16k generation cap (10 TB for 306k). Fix: store the final `teacher_output_ids` only on the **last record** per `data_id`. Our generator does this; upstream's "longest-teacher" hack is dropped because we evaluate correctness per-iteration instead.

10. **Qwen2.5-Math default system prompt is already correct.** Don't pass an explicit `system` message — the tokenizer's default chat template injects *"Please reason step by step, and put your final answer within \\boxed{}."*. Passing `"You are Qwen, created by Alibaba Cloud..."` (as upstream scripts do) is a silent distribution shift vs the base model's training.

---

## 6. What's NOT done

- **SFT training loop** using the uploaded dataset is not wired up. Would be something like `trl.SFTTrainer` with `model_max_length=4096` reading `melhoushi/OpenThoughts3_math_qwen_sft`. New training script needed — the existing `train_math_phase1_*.sh` scripts are Jacobi trainers, not SFT.
- **Jacobi trajectory run at full scale** — we built and piloted bs=8 (2.3× speedup, 25% accept rate) but didn't commit 256 GPUs for 4 hr since SFT was a better path for our goal.
- **Recovering the 64% non-`\boxed{}` rows** via OpenMathInstruct-2 fuzzy match. Feasible but not pursued.

---

## 7. Reproduction cheatsheet

```bash
# (a) Filter OpenThoughts3 math → local parquets (~1hr on cpuonly)
srun --partition=cpuonly --time=2:00:00 --cpus-per-task=8 --mem=32G \
  python scripts/dataset_prep/filter_openthoughts3_math.py \
    --start 25 --end 110 --boxed_only \
    --out ${SHARED_ROOT}/data/openthoughts3_math_boxed

# (b) [optional] Upload filtered data to HF
python scripts/dataset_prep/filter_openthoughts3_math.py --upload \
  --out ${SHARED_ROOT}/data/openthoughts3_math_boxed \
  --repo_id melhoushi/OpenThoughts3_math_boxed \
  --token <HF_TOKEN>

# (c) Generate SFT pairs via vLLM (~4 min wall on 32×8 H100)
sbatch sbatch_generate_sft_vllm.sh

# (d) Merge + upload
cat ${SHARED_ROOT}/data/openthoughts3_math_sft/sft_shard_*.jsonl \
  > ${SHARED_ROOT}/data/openthoughts3_math_sft_merged.jsonl

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('melhoushi/OpenThoughts3_math_qwen_sft', repo_type='dataset', token='<TOKEN>', exist_ok=True)
api.upload_file(path_or_fileobj='${SHARED_ROOT}/data/openthoughts3_math_sft_merged.jsonl',
                path_in_repo='data/openthoughts3_math_qwen_sft.jsonl',
                repo_id='melhoushi/OpenThoughts3_math_qwen_sft',
                repo_type='dataset', token='<TOKEN>')
"
```
