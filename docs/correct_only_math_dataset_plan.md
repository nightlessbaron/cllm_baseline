# Correct-Only OpenThoughts Math Dataset — Plan

**Goal.** Replace the current noisy OpenThoughts-2 trajectory dataset with a **clean** variant where every retained trajectory corresponds to a prompt whose greedy-generated answer **matches the gold final answer**. Use the same base model we used for Phase-1 training (`Qwen2.5-Math-7B-Instruct` at `${SHARED_ROOT}/models/base-math`), greedy decoding, cap at **2048 total tokens**. Hypothesis: trajectories from incorrect generations teach the model to refine towards wrong answers, so filtering out ~50% (typical math pass-rate at base) should shrink the dataset but improve signal.

---

## 1. Current pipeline (math branch)

| Stage | File | Input → Output |
|---|---|---|
| 0 | `generate_trajectory/data/0_bucketing_openthought2.py` | OpenThoughts-2 `.parquet` (prompt + `<\|begin_of_solution\|>` gold) → bucketed `.json` of **prompt strings only** |
| 1 | `generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.py` (or nongreedy) | Bucket JSON → per-iteration trajectory JSON with `prompt_ids`, `answer_trajectory_ids` (Jacobi refinements), `teacher_output_ids` |
| 2 | `generate_trajectory/data/2_prepare_efficient_cllm_training_data_new_progressive_noise_cyclic.py` | Trajectory JSON → training JSONL (`data_id`, `prompt_ids`, `complete_training_sequence_ids`, `prompt_ids_len`, `traj_position_indices`) |
| 3 | `generate_trajectory/data/3_postprocessing_data_length_filtering.py` (+ optional `3_downsample_dataset.py`) | Filter by length, downsample |
| Train | `JacobiForcing/train/soft_flexattn_train_cllm_multiblock.py` | Loads Stage-2 JSONL via `JacobianDataset` (`soft_flexattn_train_cllm_multiblock.py:98-143`) |

Critical observation: **the gold answer is discarded at Stage 0.** Any correctness check must re-inject it.

---

## 2. Design — what to change

### 2a. Keep the Jacobi trajectory; add a correctness filter at Stage 1

We do **not** switch to SFT. We reuse the existing Stage-1 greedy generator and Stage-2 cyclic prep verbatim — the only addition is a "correct vs. not" label per sample, and we drop the not-correct ones before Stage 2.

Why preserve Jacobi trajectories: the whole training objective (`train_cllm_multiblock.py`) depends on `answer_trajectory_ids`. Without trajectories we're training plain SFT, which is already covered by `2_prepare_baseline_training_data_sft.py`. That's a different experiment; this doc is about clean-trajectory Jacobi Forcing.

### 2b. File-by-file changes

Add (do **not** edit existing files — this is a parallel fork of the pipeline, so old runs stay reproducible):

```
generate_trajectory/data/
    0_bucketing_openthought2_with_gold.py          # NEW (fork of 0_bucketing_openthought2.py)
    -1_openthought2_correctness_filter.py          # NEW (post-Stage-1 filter)
generate_trajectory/generation/
    generate_trajectory_openthoughts_greedy.py     # NEW (fork of opencodeinstruct_greedy.py,
                                                   #   drops the longest-teacher hack, uses Math chat template,
                                                   #   caps max_new_seq_len so prompt+response ≤ 2048)
    generate_trajectory_openthoughts_greedy.sh     # NEW launcher
```

Changes per file:

**`0_bucketing_openthought2_with_gold.py`** — fork of the existing bucketer. Preserve the `<|begin_of_solution|>...<|end_of_solution|>` text next to the prompt. Output bucket format becomes an array of dicts:
```json
[{"prompt": "...", "gold_solution": "...", "gold_answer": "42"}, ...]
```
`gold_answer` extracted via `\boxed{...}` regex; fall back to last-number heuristic if absent. (Point the tokenizer at `${SHARED_ROOT}/models/base-math` — current file hard-codes `/checkpoint/lhu/models/OpenThinker2-7B` at line 15 which is wrong for our setup.)

**`generate_trajectory_openthoughts_greedy.py`** — fork of `generate_trajectory_opencodeinstruct_greedy.py`. Changes:
- Line 98 system prompt: replace `"You are Qwen, created by Alibaba Cloud..."` with Qwen2.5-Math's default (`"Please reason step by step, and put your final answer within \\boxed{}."`) — matches the base model's training distribution.
- Line 85: load a bucket of dicts (prompt+gold) instead of list of strings.
- Line 122 break condition: change `max_new_seq_len` so that **total** (prompt + response) ≤ 2048. Concretely: `effective_max_new = 2048 - input_ids.size(1)`; skip the sample if `effective_max_new < 128`.
- Line 170-176 JSON dict: add `"gold_answer"` field carried through from the bucket.
- Line 184-191 "longest-teacher" trick: **remove**. That picks the longest generation across iterations as the reference — fine for distillation but it corrupts correctness labels because the "longest" may not be the one we correctness-checked. Instead, keep each iteration's own `teacher_output_ids` and check correctness per-iteration. Or simpler: keep only `itr_0` per data_id (first block of N=16 tokens is deterministic for greedy anyway).

**`-1_openthought2_correctness_filter.py`** — NEW script, runs between Stage 1 and Stage 2. For each trajectory record:
1. Decode `teacher_output_ids` → string.
2. Extract final answer (`\boxed{...}` or last number).
3. Compare to `gold_answer` using `math_verify` (already handles equivalent forms like `\\frac{1}{2}` vs `0.5`) — fall back to sympy `simplify(a - b) == 0` if `math_verify` not installed.
4. Keep record only if equal. Write filtered JSON.

Outputs `trajectory_*_CORRECT.json` which Stage 2 consumes unchanged.

### 2c. What Stage 2 sees

Stage 2 (`2_prepare_efficient_cllm_training_data_new_progressive_noise_cyclic.py`) does not need to change. It operates on `answer_trajectory_ids` / `teacher_output_ids` fields that are still present. The only difference is the input now has ~50% fewer records (the correct ones).

One caveat at `2_prepare_efficient_cllm_training_data_new_progressive_noise_cyclic.py:108`: `cycle_len` is derived from trajectory length. If we remove `itr_0`/keep-all policy differs from the prior pipeline, confirm `traj_position_indices` still ends up in-range. Recommend keeping *all* iterations per correct sample (i.e. if the prompt's `itr_0` and `itr_3` both produced correct answers, keep both; drop `itr_1`/`itr_2` if they didn't). That preserves the cyclic sampling semantics.

---

## 3. Model & generation config

| Param | Value | Reason |
|---|---|---|
| Base model | `${SHARED_ROOT}/models/base-math` (= `Qwen2.5-Math-7B-Instruct`) | Same weights used as `--target_model_path` in `train_math_phase1_n16w16_lr1e-5.sh:20` |
| `n_token_seq_len` | **16** | Matches Phase-1 training block size (`--max_new_tokens 16`) |
| `max_new_seq_len` | `2048 - len(prompt_ids)` per sample, rounded down to multiple of 16 | User constraint: "2048 max length" (total) |
| Sampling | **greedy** (`get_jacobi_forward_trajectory_greedy`, argmax) | No top-p / temperature — deterministic, matches user ask |
| Chat template | Qwen2.5-Math default system prompt | Avoid distribution shift from base model training |
| Attn impl | `flash_attention_2`, bf16 | Match training setup |
| Parallelism | 4 GPUs × 1 prompt/GPU, one bucket file per GPU | Matches existing launcher pattern (`generate_trajectory_opencodeinstruct_greedy.sh:23-41`) |

---

## 4. Concrete launch sequence

Assume OpenThoughts-2 parquet files live at `${SHARED_ROOT}/data/OpenThoughts2_raw/*.parquet` (verify; if not, point to wherever they are — the current `40k_samples_seed42_maxlen2048.jsonl` is post-processed and won't work as input).

```bash
# --- Stage 0: bucket with gold ---
python generate_trajectory/data/0_bucketing_openthought2_with_gold.py \
    --input_path  ${SHARED_ROOT}/data/OpenThoughts2_raw \
    --output_path ${SHARED_ROOT}/data/OpenThoughts_Math_bucketed_with_gold \
    --bucket_size 5000 \
    --n_workers 16 \
    --think_format

# --- Stage 1: greedy trajectory generation (4 GPUs in parallel, 1 bucket each) ---
bash generate_trajectory/generation/generate_trajectory_openthoughts_greedy.sh

# --- Stage 1.5: correctness filter ---
python generate_trajectory/data/-1_openthought2_correctness_filter.py \
    --input_dir  ${SHARED_ROOT}/data/openthoughts_generated_trajectory_blk16 \
    --output_dir ${SHARED_ROOT}/data/openthoughts_generated_trajectory_blk16_correct

# --- Stage 2: cyclic progressive noise training prep (unchanged) ---
python generate_trajectory/data/2_prepare_efficient_cllm_training_data_new_progressive_noise_cyclic.py \
    --input_dir  ${SHARED_ROOT}/data/openthoughts_generated_trajectory_blk16_correct \
    --output_path ${SHARED_ROOT}/data/OpenThoughts_Math_training_data_n16w16_correct_only \
    --n_token_seq_length 16 \
    --half_cap_idx 8

# --- Stage 3: length filter to 2048 (drop any residuals over cap) ---
python generate_trajectory/data/3_postprocessing_data_length_filtering.py \
    --input  ${SHARED_ROOT}/data/OpenThoughts_Math_training_data_n16w16_correct_only \
    --output ${SHARED_ROOT}/data/OpenThoughts_Math_training_data_n16w16_correct_only_filtered \
    --max_len 2048

# --- Training: point sbatch at the new JSONL ---
# In train_math_phase1_n16w16_lr1e-5_constant.sh, change trajectory_file to the new JSONL.
```

---

## 5. Correctness check: math-specific notes

- **`\boxed{}` extraction** is the primary signal. Regex: `r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}"` (handles one level of nested braces like `\boxed{\frac{1}{2}}`).
- **Equivalence**: `math_verify.parse` → compare. `math_verify` is designed exactly for this (handles `0.5 == 1/2 == \frac{1}{2}`, trailing whitespace, etc.). Install via `pip install math-verify`.
- **Fallback when `math_verify` fails to parse**: `sympy.simplify(sp.sympify(a) - sp.sympify(b)) == 0`. If *that* fails, string-normalize (strip whitespace/commas) and do literal compare.
- **Empty / no-`\boxed{}` generations**: count as incorrect. Do not try to rescue with last-number heuristic — that inflates false positives.

Log the correctness rate per bucket — if it's <30% at base, reconsider (maybe Qwen2.5-Math-7B-Instruct zero-shot on OpenThoughts-2 is too hard and we need a smarter extraction, or we need to switch to a different base, or the `<think>` chain-of-thought template is hurting).

---

## 6. Risks / open questions

1. **Where are the raw OpenThoughts-2 parquets?** The existing training JSONL is post-processed. Need the source parquets to re-bucket with gold. If not available locally, `hf download open-thoughts/OpenThoughts-2 --repo-type dataset`.
2. **`longest-teacher` removal** (Stage 1, line 184-191) may have been there to handle a specific pathology — verify by running a small pilot (100 prompts) and confirming trajectories look sane without it.
3. **Correctness rate shrinks dataset**. If base model is ~50% on OpenThoughts-2 and we started from 40k samples, we'll end with ~20k. Phase-1 Math was trained on 40k for 10k steps; with 20k samples we'd see each prompt ~2× per epoch at the same step budget — might want to halve `max_steps` or rerun at 10k to compare like-for-like. Surface this decision before launching.
4. **Determinism**: greedy Jacobi is deterministic given model + prompt, **except** for the `q_sampled = random.choices(...)` at `generate_trajectory_opencodeinstruct_greedy.py:146` which injects random tokens as the initial Jacobi guess. Set `random.seed(42)` at script entry so correctness filtering is reproducible across reruns.
5. **Coder parallel**: if the math experiment works, the Coder equivalent needs test-execution-based correctness (not answer matching) — much heavier. Out of scope here.

---

## 7. Milestones

- [ ] Confirm parquet location and `math_verify` availability in env
- [ ] Fork Stage-0 bucketer, verify it carries gold on 100 samples
- [ ] Fork Stage-1 greedy generator, run on 1 small bucket (100 prompts) to verify trajectory format + correctness filter
- [ ] Measure correctness rate — sanity check vs. public eval numbers for Qwen2.5-Math-7B-Instruct on OpenThoughts-style problems
- [ ] Full run: 4 GPUs, 40k prompts → expect ~4–6 hours
- [ ] Stage 2 + 3, length check
- [ ] Launch training with `--lr_scheduler_type constant --learning_rate 1e-5` and new JSONL, new output dir `math-phase1-n16w16-lr1e-5-constant-correctonly`
