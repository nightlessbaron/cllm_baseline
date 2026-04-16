# JacobiForcing Baseline Reproduction — Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce two documentation artifacts — `docs/codebase_explanation.md` (method-first walkthrough of the JacobiForcing repo) and `docs/training_plan.md` (stage-oriented runbook to reproduce Coder 7B and Math 7B checkpoints via Option A on 8× H200) — that together let the user defend a JacobiForcing baseline in their own paper.

**Architecture:** Two independent markdown files under `docs/`. Doc 1 is structured as *method concept → code pointer*; Doc 2 is structured as *stages* (Environment → Source → Data → Smoke test → Coder train → Math train → HumanEval → MATH500 → Reporting → Appendices A, B). Every upstream file path referenced in either doc must be verified against the fetched upstream file tree (captured in the spec's §4). Every command must be copy-pasteable.

**Tech Stack:** Markdown only for the artifacts. Git for versioning in `/mnt/weka/home/varad.pimpalkhute/diffusion/`. Upstream is [hao-ai-lab/JacobiForcing](https://github.com/hao-ai-lab/JacobiForcing) at its current `main`.

**Spec reference:** `docs/superpowers/specs/2026-04-16-jacobiforcing-baseline-reproduction-design.md` (commits `0b8e4e3`, `314e6b1`).

**Adaptation note (docs, not code):** The writing-plans skill assumes TDD for code. This plan is for documentation, so the TDD analog is "validation": each task's verification step checks that (a) every upstream file path cited exists in the upstream tree, (b) no placeholders remain, (c) commands are syntactically runnable. Where the skill template says "write failing test," this plan says "state the acceptance bullets the section must satisfy" — same role, different medium.

**Working directory:** `/mnt/weka/home/varad.pimpalkhute/diffusion/`. All relative paths below are relative to this directory unless absolute.

---

## Reference — Upstream file paths (authoritative list)

These paths were fetched via the GitHub tree API during brainstorming and are the *only* upstream paths the docs may cite. Any task that needs to cite something not in this list must first verify the path exists upstream (the verification steps explain how).

```
# Training
JacobiForcing/scripts/train/baseline_sft_train.sh
JacobiForcing/scripts/train/ds_config.json
JacobiForcing/scripts/train/ds_config_cpu_offloading.json
JacobiForcing/scripts/train/train_clean_context_conditioned_cllm_openthinker2_n64.sh
JacobiForcing/scripts/train/train_jacobi_forcing_coder_n32.sh
JacobiForcing/scripts/train/train_jacobi_forcing_coder_n64.sh
JacobiForcing/train/baseline_sft_train.py
JacobiForcing/train/cllm_trainer.py
JacobiForcing/train/soft_flexattn_cllm_trainer.py
JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock.py
JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock_window.py
JacobiForcing/train/soft_flexattn_train_cllm.py
JacobiForcing/train/soft_flexattn_train_cllm_multiblock.py
JacobiForcing/train/soft_flexattn_train_cllm_multiblock_window.py
JacobiForcing/train/train_cllm.py

# Inference entry points
JacobiForcing/ar_inference_baseline.py
JacobiForcing/jacobi_forcing_inference_humaneval.py
JacobiForcing/jacobi_forcing_inference_MR_humaneval.py
JacobiForcing/jacobi_forcing_inference_MR_humaneval_config_grid_search.py
JacobiForcing/jacobi_forcing_inference_MATH500.py
JacobiForcing/scripts/inference/scanning_hyperparameter_jacobi_decoding_mr.sh

# Data pipeline
generate_trajectory/data/0_bucketing_opencodeinstruct.py
generate_trajectory/data/0_bucketing_openthought2.py
generate_trajectory/data/2_prepare_efficient_cllm_training_data_progressive_noise_window.py
generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.py
generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.sh
generate_trajectory/generation/qwen2_modeling_jacobi_forcing_greedy.py
generate_trajectory/generation/qwen2_modeling_jacobi_forcing_nongreedy_blk32.py

# Modeling
modeling/cllm2_qwen2_modeling_kv_terminate_on_eos_improved.py
modeling/cllm2_qwen2_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified.py

# Inference engine (lightweight)
inference_engine/config.py
inference_engine/llm.py
inference_engine/sampling_params.py
inference_engine/engine/block_manager.py
inference_engine/engine/jacobi_decoding.py
inference_engine/engine/jacobi_decoding_nongreedy.py
inference_engine/engine/jacobi_decoding_nongreedy_on_policy.py
inference_engine/engine/llm_engine.py
inference_engine/engine/model_runner.py
inference_engine/engine/scheduler.py
inference_engine/engine/sequence.py
inference_engine/layers/*.py
inference_engine/models/qwen3.py
inference_engine/tests/test_jacobi_decoding_greedy.py
inference_engine/tests/test_jacobi_decoding_nongreedy.py

# Applications
applications/jacobi_model_chat.py
applications/jacobi_streaming_driver.py

# Root
README.md
requirements.txt
LICENSE
```

**How to verify any new path during plan execution:**
```bash
curl -s "https://api.github.com/repos/hao-ai-lab/JacobiForcing/contents/<dir-path>?ref=main" | jq -r '.[].path'
```
or fetch the full tree once at the start of execution and grep locally.

**HuggingFace dataset/model list (for Stage 2 references):**
- `JacobiForcing/OpenCodeInstruct_training_data_n32w16` (Coder, n=32, w=16, 2.16M rows)
- `JacobiForcing/OpenThoughts_Math_training_data_n64w32` (Math, n=64, w=32, 230k rows)
- `JacobiForcing/JacobiForcing_Coder_7B_v1` (published Coder checkpoint, for comparison)
- `JacobiForcing/JacobiForcing_Math_7B_v1` (published Math checkpoint)
- `JacobiForcing/JacobiForcing_Coder_7B_v0` (intermediate, **source of the training shards' trajectories** — cite in provenance notes)
- `Qwen/Qwen2.5-Coder-7B-Instruct` (base Coder)
- `Qwen/Qwen2.5-Math-7B-Instruct` (base Math)

---

## File structure

Created in this plan (relative to `/mnt/weka/home/varad.pimpalkhute/diffusion/`):
- `docs/codebase_explanation.md` — Doc 1
- `docs/training_plan.md` — Doc 2

Unchanged: everything else.

---

## Task 1: Scaffold Doc 1

**Files:**
- Create: `docs/codebase_explanation.md`

- [ ] **Step 1: Define acceptance bullets for the scaffold**

The scaffold must contain: H1 title, a ≤3-sentence TL;DR paragraph, a table of contents with 5 sections matching the spec, a short "Scope" paragraph stating the doc is concept-first with code pointers, and a "Conventions" paragraph stating that code paths are relative to the upstream repo root `hao-ai-lab/JacobiForcing`.

- [ ] **Step 2: Write the scaffold**

Create `docs/codebase_explanation.md` with the exact content below:

```markdown
# JacobiForcing — Codebase Explanation

**TL;DR.** Jacobi Forcing trains an autoregressive LLM to behave as a *causal parallel decoder* by finetuning it on noisy future blocks drawn from Jacobi fixed-point trajectories. The repo ships the full pipeline: bucketing, trajectory generation, noise-window preparation, noise-conditioned training, and a lightweight inference engine with rejection recycling. This document walks the method conceptually and points to the code that implements each stage.

## Contents

1. [What Jacobi Forcing is](#1-what-jacobi-forcing-is)
2. [The five method stages](#2-the-five-method-stages)
3. [The model-code layer](#3-the-model-code-layer)
4. [Repo map](#4-repo-map)
5. [Quirks and gotchas for reproduction](#5-quirks-and-gotchas-for-reproduction)

## Scope and conventions

This document is **concept-first**: each method stage is explained before its code pointer. It is *not* a directory reference — see §4 for that. All code paths are relative to the upstream repo root `hao-ai-lab/JacobiForcing` at the commit recorded in `docs/training_plan.md` Stage 1. Paths beginning with `JacobiForcing/` refer to the **nested subdirectory** inside the repo (see §5).
```

- [ ] **Step 3: Verify scaffold**

Run: `head -40 docs/codebase_explanation.md`
Expected: shows the H1, TL;DR, contents, scope sections as written above. No TBDs.

- [ ] **Step 4: Commit**

```bash
git add docs/codebase_explanation.md
git commit -m "docs(codebase_explanation): scaffold with TOC and scope"
```

---

## Task 2: Doc 1 §1 — What Jacobi Forcing is

**Files:**
- Modify: `docs/codebase_explanation.md` (append §1)

- [ ] **Step 1: Acceptance bullets**

§1 must explain, in 400–600 words:
- The autoregressive baseline problem (one token per forward pass, wall-clock bottleneck on reasoning/coding).
- Why diffusion LMs address speed but trade accuracy and lose KV-cache friendliness.
- What "Jacobi decoding" is at the method level (fixed-point iteration on a block of future tokens: start from a guess, run the model, refine, repeat until the block is self-consistent).
- The AR-to-diffusion *mismatch* — base AR models were never trained on noisy future blocks, so Jacobi rollouts converge slowly or wrongly.
- Jacobi Forcing's answer: keep the causal AR backbone, add a training signal on noisy future blocks along the Jacobi trajectory. Result: multi-token-per-forward decoding while preserving left-to-right causality and KV cache.
- One sentence on reported speedups (4.5× tokens/forward, 4× wall-clock on coding/math) with the paper citation arXiv 2512.14681.

- [ ] **Step 2: Draft §1**

Append to `docs/codebase_explanation.md`:

```markdown

## 1. What Jacobi Forcing is

### The problem

Autoregressive decoding produces one token per forward pass. For a 7B model answering a multi-hundred-token coding or math prompt, that's hundreds of serial GPU calls — each one underutilizing the hardware. Discrete-diffusion language models sidestep this by decoding many tokens in parallel, but they abandon the causal, left-to-right structure that makes standard AR serving efficient (KV cache, paged attention, streaming). They also tend to lose accuracy on long-form reasoning.

### Jacobi decoding (the idea that pre-dates Jacobi Forcing)

Jacobi decoding treats next-block generation as a fixed-point problem. Pick a block size *n*. Guess *n* future tokens (often by copying or simple heuristic). Run the model once to get *n* next-token predictions conditioned on the guess. Replace the guess with the predictions. Repeat. When the block stops changing, it is a fixed point: the model agrees that these are the tokens it would have produced autoregressively. Any prefix of the fixed point that matches the pure-AR continuation can be committed. On a well-trained model, many iterations produce several committable tokens per forward pass.

The catch: *base* AR models were not trained on noisy future-block inputs. The first few Jacobi iterations pass the model nonsense-looking prefixes, and it produces nonsense conditioning for the tokens after them. Convergence is slow, quality drops, and the theoretical speedup doesn't materialize.

### The Jacobi Forcing fix

Jacobi Forcing patches the AR-to-diffusion mismatch **by training the model to expect noisy future blocks**. Concretely: roll out Jacobi trajectories from a base AR model on real prompts; along each trajectory, pair every intermediate (noisy) state with the final (clean) state; train the model to predict the clean continuation given the noisy input. A progressive noise schedule varies how noisy the input block is across training examples. The backbone stays causal, the KV cache keeps working, but the model now tolerates — and accelerates through — noisy future-token context.

The published results (arXiv 2512.14681) show up to 4.5× tokens/forward and 4× wall-clock speedup on coding (HumanEval) and math (GSM8K/MATH) benchmarks with near-AR generation quality.

### Why this matters for reproduction

Two properties of the recipe shape everything downstream. First, the training data is *model-dependent*: trajectories depend on whoever generated them. The authors' released training shards were rolled from an intermediate `JacobiForcing_Coder_7B_v0` checkpoint, not from the base Qwen — a fact worth disclosing in any paper that uses this as a baseline. Second, the noise-conditioning hyperparameters (block size `n`, window size `w`, noise-ratio schedule) are baked into the training shards and must match the training script's expectations exactly. Mismatches produce silent accuracy loss.
```

- [ ] **Step 3: Verify §1**

Run: `wc -w docs/codebase_explanation.md` — expect ≥ ~700 words total so far.
Run: `grep -c "TBD\|TODO\|FIXME" docs/codebase_explanation.md` — expect 0.
Run: `grep -c "arXiv 2512.14681" docs/codebase_explanation.md` — expect ≥ 1.

- [ ] **Step 4: Commit**

```bash
git add docs/codebase_explanation.md
git commit -m "docs(codebase_explanation): add §1 method overview"
```

---

## Task 3: Doc 1 §2 — The five method stages

**Files:**
- Modify: `docs/codebase_explanation.md` (append §2)

- [ ] **Step 1: Acceptance bullets**

§2 must have five subsections, in this order, each 150–300 words:
- §2.1 **Data bucketing** — explain length bucketing rationale (packing efficiency for FlexAttention + DeepSpeed), point to `generate_trajectory/data/0_bucketing_opencodeinstruct.py` (Coder) and `generate_trajectory/data/0_bucketing_openthought2.py` (Math).
- §2.2 **Trajectory generation** — explain Jacobi rollout, point to `generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.py` (+ `.sh` wrapper) and the greedy/nongreedy Qwen2 rollout models at `generate_trajectory/generation/qwen2_modeling_jacobi_forcing_{greedy,nongreedy_blk32}.py`. **Must include the caveat:** "Under Option A (used by this reproduction), this stage is not executed — the HF shards ship its output."
- §2.3 **Training-sequence prep with progressive noise window** — explain the `n_token_seq_length`, `window_size`, `min_noisy_ratio`, `max_noisy_ratio`, `strategy=progressive` knobs from `generate_trajectory/data/2_prepare_efficient_cllm_training_data_progressive_noise_window.py`. **Must include the caveat:** "The HF shards already encode this step's output (columns `prompt_ids`, `complete_training_sequence_ids`, `traj_position_indices`)."
- §2.4 **Noise-conditioned training** — explain which trainer variant matches which recipe. Pointers: `JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock_window.py` (primary trainer for the windowed-progressive recipe used by Option A), `train_cllm.py` (entry point), `soft_flexattn_train_cllm_multiblock_window.py` (train loop). Mention the two DeepSpeed configs (`ds_config.json`, `ds_config_cpu_offloading.json`) and when each applies.
- §2.5 **Inference with rejection recycling** — explain Multi-block Rejection Recycling (MR): try parallel blocks in flight, reject non-converged ones, recycle compute. Pointers: `inference_engine/engine/jacobi_decoding.py`, `jacobi_decoding_nongreedy.py`, entry points `JacobiForcing/jacobi_forcing_inference_humaneval.py` (plain) and `jacobi_forcing_inference_MR_humaneval.py` (with MR). Hyperparameters to call out: block size `n`, block count `K`, pool size, activation ratio `r`.

- [ ] **Step 2: Draft §2**

Append the five subsections to `docs/codebase_explanation.md`. Prose is written during execution; every file path above **must** appear verbatim and every caveat must be included. A minimum acceptable skeleton:

```markdown

## 2. The five method stages

### 2.1 Data bucketing

<150–300 words explaining length bucketing and its packing role>

**Code:**
- `generate_trajectory/data/0_bucketing_opencodeinstruct.py` — Coder bucketing.
- `generate_trajectory/data/0_bucketing_openthought2.py` — Math bucketing.

### 2.2 Trajectory generation

<150–300 words explaining Jacobi rollout>

> **Option A note:** This stage is not executed in the reproduction — the HF dataset shards ship its output.

**Code:**
- `generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.py` (+ `.sh` wrapper).
- `generate_trajectory/generation/qwen2_modeling_jacobi_forcing_greedy.py` — greedy Jacobi rollout on Qwen2.
- `generate_trajectory/generation/qwen2_modeling_jacobi_forcing_nongreedy_blk32.py` — nongreedy variant.

### 2.3 Training-sequence prep with progressive noise window

<150–300 words explaining the knobs>

> **Option A note:** The HF shards already encode this step's output (columns `prompt_ids`, `complete_training_sequence_ids`, `traj_position_indices`).

**Code:**
- `generate_trajectory/data/2_prepare_efficient_cllm_training_data_progressive_noise_window.py`

**Key arguments:** `--n_token_seq_length`, `--window_size`, `--min_noisy_ratio`, `--max_noisy_ratio`, `--strategy progressive`.

### 2.4 Noise-conditioned training

<150–300 words explaining the trainer and which variant pairs with Option A>

**Code:**
- `JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock_window.py` — primary trainer for the windowed-progressive recipe.
- `JacobiForcing/train/soft_flexattn_train_cllm_multiblock_window.py` — train loop.
- `JacobiForcing/train/train_cllm.py` — top-level entry point.
- `JacobiForcing/scripts/train/ds_config.json` — DeepSpeed config for standard (no-offload) multi-GPU training.
- `JacobiForcing/scripts/train/ds_config_cpu_offloading.json` — ZeRO-3 CPU offload config for lower-VRAM setups (not needed on H200).

### 2.5 Inference with rejection recycling

<150–300 words explaining MR inference>

**Code:**
- `inference_engine/engine/jacobi_decoding.py` — greedy Jacobi decoding.
- `inference_engine/engine/jacobi_decoding_nongreedy.py` — nongreedy variant.
- `JacobiForcing/jacobi_forcing_inference_humaneval.py` — plain HumanEval inference entry.
- `JacobiForcing/jacobi_forcing_inference_MR_humaneval.py` — HumanEval with MR.

**Hyperparameters:** block size `n` (default 64 for inference), block count `K` (default 2), pool size (default 4), activation ratio `r` (default 0.85).
```

- [ ] **Step 3: Verify §2**

Run this check to confirm every cited upstream path appears in the spec's reference list (which matches the upstream tree):

```bash
spec=docs/superpowers/specs/2026-04-16-jacobiforcing-baseline-reproduction-design.md
doc=docs/codebase_explanation.md
# Extract every backticked path under a § 2 section and check each exists upstream.
# If paths are wrong, the grep in spec tree will miss them.
grep -oE '`[a-zA-Z0-9_/.-]+\.(py|sh|json)`' "$doc" | sort -u | while read -r p; do
  stripped=${p//\`/}
  curl -sf -o /dev/null "https://raw.githubusercontent.com/hao-ai-lab/JacobiForcing/main/$stripped" \
    && echo "OK  $stripped" \
    || echo "MISS $stripped"
done
```
Expected: every line is `OK`. Any `MISS` must be fixed before moving on.

Run: `grep -c "Option A note" docs/codebase_explanation.md` — expect ≥ 2 (§2.2 and §2.3).
Run: `grep -cE "TBD|TODO|FIXME|<[0-9]" docs/codebase_explanation.md` — expect 0 (no placeholders; the `<150–300 words>` markers are already replaced by real prose at this point).

- [ ] **Step 4: Commit**

```bash
git add docs/codebase_explanation.md
git commit -m "docs(codebase_explanation): add §2 five method stages"
```

---

## Task 4: Doc 1 §3 + §4 — Model-code layer and repo map

**Files:**
- Modify: `docs/codebase_explanation.md` (append §3 and §4)

- [ ] **Step 1: Acceptance bullets**

§3 must (in ≤250 words) explain the two modeling files and when each is used:
- `modeling/cllm2_qwen2_modeling_kv_terminate_on_eos_improved.py` — standard KV cache + EOS termination; used by plain Jacobi inference.
- `modeling/cllm2_qwen2_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified.py` — multiblock lookahead variant; used by MR inference.

§4 must be a single markdown table mapping every top-level directory in the repo to a one-line purpose. Directories to cover (per the upstream tree):
- `applications/`, `generate_trajectory/`, `inference_engine/`, `JacobiForcing/`, `modeling/`

(Plus root-level `requirements.txt`, `README.md`, `LICENSE`.)

- [ ] **Step 2: Draft §3 and §4**

Append to `docs/codebase_explanation.md`:

```markdown

## 3. The model-code layer

Two Qwen2-based model files sit under `modeling/`. They share a common base but differ in their decoding machinery; picking the right one at inference time matters.

- **`modeling/cllm2_qwen2_modeling_kv_terminate_on_eos_improved.py`** — standard KV-cache-aware causal forward pass with early EOS termination. Used by the plain HumanEval / MATH500 inference entry points.
- **`modeling/cllm2_qwen2_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified.py`** — adds the multiblock lookahead plumbing needed by MR inference (parallel in-flight blocks with rejection). Used by `jacobi_forcing_inference_MR_humaneval.py`.

Training loads whichever model file matches the trainer variant selected by the script being run (§2.4).

## 4. Repo map

One-line purpose for each top-level location. Use this as a lookup, not a narrative — see §1 and §2 for the conceptual story.

| Path                        | Purpose                                                                                 |
| --------------------------- | --------------------------------------------------------------------------------------- |
| `applications/`             | Demo scripts: chat driver, streaming driver.                                            |
| `generate_trajectory/`      | Data pipeline: bucketing (stage 0), trajectory rollout (stage 1), noise-window prep (stage 2), tools. |
| `inference_engine/`         | Lightweight inference engine (FlashAttention, paged KV, CUDA graph, tensor parallel).   |
| `JacobiForcing/` (nested)   | Training scripts, trainer classes, inference entry points, DeepSpeed configs.           |
| `modeling/`                 | Qwen2-based model implementations (standard and multiblock-lookahead).                  |
| `requirements.txt`          | Python dependencies (install under Python 3.12).                                        |
| `README.md`                 | Upstream overview and quickstart — note: path typos corrected in §5.                    |
| `LICENSE`                   | Apache 2.0.                                                                             |
```

- [ ] **Step 3: Verify §3 and §4**

Run: `grep -c "^| " docs/codebase_explanation.md` — expect ≥ 8 rows in the table (header + 7+ entries).
Run: `grep -c "modeling/cllm2_qwen2" docs/codebase_explanation.md` — expect ≥ 2.
Run: the upstream path check from Task 3 Step 3 over the whole file. Expect all `OK`.

- [ ] **Step 4: Commit**

```bash
git add docs/codebase_explanation.md
git commit -m "docs(codebase_explanation): add §3 model layer and §4 repo map"
```

---

## Task 5: Doc 1 §5 — Quirks and gotchas

**Files:**
- Modify: `docs/codebase_explanation.md` (append §5)

- [ ] **Step 1: Acceptance bullets**

§5 must enumerate, as a bulleted list, each of these quirks with a 1–3 sentence explanation of *why it matters for reproduction*:
1. Nested `JacobiForcing/JacobiForcing/` directory — the repo's root is `JacobiForcing/` and contains a subdirectory also named `JacobiForcing/` housing training scripts and inference entry points.
2. Upstream README cites `scripts/train/train_jacobi_forcing_coder_n32.sh`; the actual path is `JacobiForcing/scripts/train/train_jacobi_forcing_coder_n32.sh`.
3. Math training script is named `train_clean_context_conditioned_cllm_openthinker2_n64.sh` — the `openthinker2` naming is a legacy carryover; this is the script that pairs with `OpenThoughts_Math_training_data_n64w32`.
4. Python 3.12 pin; newer Python versions are not guaranteed to work (requirements built against 3.12).
5. Upstream does not pin HF revisions for base models or datasets — our reproduction pins them in `docs/training_plan.md` Stage 1 and Stage 2.
6. Flash-attention version drift can shift loss values at the 4th decimal across identical runs — captured in Stage 0's `versions.lock`.
7. The HF training shards encode trajectories rolled from `JacobiForcing_Coder_7B_v0` (intermediate checkpoint), not from base Qwen — disclosable provenance in the paper methods section.

- [ ] **Step 2: Draft §5**

Append to `docs/codebase_explanation.md`:

```markdown

## 5. Quirks and gotchas for reproduction

These are the details that bit us (or would have) during reproduction. Each matters because it affects either which command you run, which numbers you compare against, or what you disclose in the methods section of a paper that cites this work as a baseline.

- **Nested `JacobiForcing/JacobiForcing/`.** The repo root `JacobiForcing/` contains a subdirectory *also* named `JacobiForcing/` that houses the training scripts, trainer classes, inference entry points, and DeepSpeed configs. Absolute paths in this doc and in `docs/training_plan.md` use the correct nested form (e.g. `JacobiForcing/JacobiForcing/scripts/train/...`).
- **README path error.** The upstream README shows `scripts/train/train_jacobi_forcing_coder_n32.sh`. The actual path after clone is `JacobiForcing/scripts/train/train_jacobi_forcing_coder_n32.sh`. Copying the README command verbatim from within the repo root will `No such file`.
- **Legacy math script name.** The math training script is `train_clean_context_conditioned_cllm_openthinker2_n64.sh`. The `openthinker2` naming is a carryover from an earlier method; this *is* the script to use with `OpenThoughts_Math_training_data_n64w32`. Don't assume unrelated.
- **Python 3.12 pin.** `requirements.txt` is pinned against Python 3.12. Newer versions (3.13+) are untested; older versions will fail to resolve some deps.
- **Unpinned HF revisions upstream.** Neither the base Qwen models nor the HF training datasets have pinned revisions in the upstream README. Our reproduction pins both at download time — see Stage 1 and Stage 2 in `docs/training_plan.md`.
- **flash-attention version drift.** Flash-attention, torch, and transformers version changes can shift training loss at the 4th decimal across otherwise-identical runs. Stage 0's `versions.lock` captures the installed set.
- **Training-data provenance (disclosable).** The HF training shards were rolled from `JacobiForcing_Coder_7B_v0` — an intermediate checkpoint, not the base Qwen and not the published `_v1`. A paper that uses this as a baseline should disclose that the reproduction re-used the authors' v0-generated trajectories rather than re-rolling from the base model.
```

- [ ] **Step 3: Verify §5**

Run: `grep -c '^- \*\*' docs/codebase_explanation.md` — expect ≥ 7 bullet markers (matches the 7 quirks).
Run: `grep -c "openthinker2" docs/codebase_explanation.md` — expect ≥ 1.
Run: `grep -c "Coder_7B_v0" docs/codebase_explanation.md` — expect ≥ 1.
Run: `grep -cE "TBD|TODO|FIXME" docs/codebase_explanation.md` — expect 0.

- [ ] **Step 4: Commit**

```bash
git add docs/codebase_explanation.md
git commit -m "docs(codebase_explanation): add §5 quirks and gotchas"
```

---

## Task 6: Scaffold Doc 2

**Files:**
- Create: `docs/training_plan.md`

- [ ] **Step 1: Acceptance bullets**

Scaffold must contain: H1 title, TL;DR paragraph (≤3 sentences), "Target hardware" callout (single-node 8× H200, `ds_config.json`, no offload), the "Reproducibility contract" paragraph stating that the plan pins HF revisions and records `versions.lock`, a contents list with stages 0 through 7 plus Appendix A and B, and a "Conventions" paragraph.

- [ ] **Step 2: Write scaffold**

Create `docs/training_plan.md` with:

```markdown
# JacobiForcing — Training Reproduction Plan (Option A)

**TL;DR.** This runbook takes an 8× H200 node from an empty environment to a reproduced JacobiForcing **Coder 7B** checkpoint (HumanEval) and **Math 7B** checkpoint (MATH500) using upstream's Option A path (pre-generated HuggingFace training shards). Every stage records provenance (HF revision hashes, `versions.lock`) so the resulting numbers are citable as a baseline in a research paper.

## Target hardware

Single-node **8× H200** (141 GB HBM3e each). Configuration uses `JacobiForcing/scripts/train/ds_config.json` with no CPU offload. Multinode is available but is **not** the default path — it introduces NCCL/allreduce ordering variables that can shift loss curves vs. upstream. If you switch to multinode, log the change in Appendix A.

## Reproducibility contract

- Every downloaded artifact (datasets, base models, repo) is pinned at download time by recording its commit hash.
- Installed package versions are captured in `versions.lock` at the end of Stage 0.
- Training is configured deterministically where feasible (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, fixed seeds from upstream script).
- A smoke test (Stage 2.5) validates the pipeline before committing GPU-hours to the full run.

## Contents

- [Stage 0 — Environment](#stage-0--environment)
- [Stage 1 — Source and base models](#stage-1--source-and-base-models)
- [Stage 2 — Option A data download](#stage-2--option-a-data-download)
- [Stage 2.5 — Smoke test](#stage-25--smoke-test)
- [Stage 3 — Coder training (7B, n32w16)](#stage-3--coder-training-7b-n32w16)
- [Stage 4 — Math training (7B, n64w32)](#stage-4--math-training-7b-n64w32)
- [Stage 5 — HumanEval on Coder checkpoint](#stage-5--humaneval-on-coder-checkpoint)
- [Stage 6 — MATH500 on Math checkpoint](#stage-6--math500-on-math-checkpoint)
- [Stage 7 — Reporting](#stage-7--reporting)
- [Appendix A — Deviations from upstream recipe](#appendix-a--deviations-from-upstream-recipe)
- [Appendix B — Failure-mode runbook](#appendix-b--failure-mode-runbook)

## Conventions

- `$REPRO_ROOT` = `/mnt/weka/home/varad.pimpalkhute/diffusion` (this plan's working directory).
- `$REPO` = `$REPRO_ROOT/JacobiForcing` (created in Stage 1).
- All `bash` blocks are copy-pasteable. Each stage ends with a **Reproducibility checklist** (what to record) and **Expected artifacts** (what the stage produces).
```

- [ ] **Step 3: Verify scaffold**

Run: `grep -c "^## Stage" docs/training_plan.md` — expect 0 so far (TOC uses plain list).
Run: `grep -c "^- \[Stage " docs/training_plan.md` — expect 8 (Stage 0–7).
Run: `grep -c "^- \[Appendix " docs/training_plan.md` — expect 2.

- [ ] **Step 4: Commit**

```bash
git add docs/training_plan.md
git commit -m "docs(training_plan): scaffold with hardware, contract, TOC"
```

---

## Task 7: Doc 2 Stage 0 — Environment

**Files:**
- Modify: `docs/training_plan.md` (append Stage 0)

- [ ] **Step 1: Acceptance bullets**

Stage 0 must include:
- `conda create -n jacobi_forcing python=3.12 -y` and `conda activate jacobi_forcing`.
- `git lfs install`.
- `huggingface-cli login` — reminder to store token.
- Determinism env vars exported into `~/.bashrc` or the run script: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, `CUDA_LAUNCH_BLOCKING` left unset (performance), `NCCL_DEBUG=WARN`.
- A CUDA/driver check command (`nvidia-smi`) with expected output notes (H200, driver ≥ 550, CUDA 12.x).
- Final step: `pip freeze > $REPRO_ROOT/versions.lock` after requirements install.
- Reproducibility checklist: `versions.lock` created; `nvidia-smi` output saved.
- Expected artifact: `$REPRO_ROOT/versions.lock`.

- [ ] **Step 2: Draft Stage 0**

Append:

````markdown

## Stage 0 — Environment

### Prereqs

- Sudo access not required; conda and HF CLI installed for the user.
- Outbound HTTPS to huggingface.co and github.com.

### Commands

```bash
# 1. Conda env
conda create -n jacobi_forcing python=3.12 -y
conda activate jacobi_forcing

# 2. Git LFS (needed for Stage 2 dataset download)
git lfs install

# 3. HuggingFace auth — paste a token with read access
huggingface-cli login

# 4. Determinism + NCCL env vars (add to run scripts; avoid global export if sharing the node)
cat >> ~/.bashrc <<'EOF'
# JacobiForcing reproduction — determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export NCCL_DEBUG=WARN
EOF
source ~/.bashrc

# 5. GPU / driver sanity
nvidia-smi | tee $REPRO_ROOT/nvidia-smi.txt
```

Expected from `nvidia-smi`: **8 × NVIDIA H200**, driver ≥ 550, CUDA ≥ 12.4.

### After Stage 1's `pip install -r requirements.txt` has run, finalize the lock

```bash
pip freeze > $REPRO_ROOT/versions.lock
```

### Reproducibility checklist

- [ ] `$REPRO_ROOT/nvidia-smi.txt` saved.
- [ ] `$REPRO_ROOT/versions.lock` created (after Stage 1 deps install).
- [ ] Determinism env vars present in current shell (`echo $CUBLAS_WORKSPACE_CONFIG` returns `:4096:8`).

### Expected artifacts

- `$REPRO_ROOT/nvidia-smi.txt`
- `$REPRO_ROOT/versions.lock` (created at end of Stage 1)
````

- [ ] **Step 3: Verify Stage 0**

Run: `grep -c "CUBLAS_WORKSPACE_CONFIG=:4096:8" docs/training_plan.md` — expect ≥ 1.
Run: `grep -c "huggingface-cli login" docs/training_plan.md` — expect ≥ 1.
Run: `grep -c "versions.lock" docs/training_plan.md` — expect ≥ 2.

- [ ] **Step 4: Commit**

```bash
git add docs/training_plan.md
git commit -m "docs(training_plan): add Stage 0 environment"
```

---

## Task 8: Doc 2 Stage 1 — Source and base models

**Files:**
- Modify: `docs/training_plan.md`

- [ ] **Step 1: Acceptance bullets**

Stage 1 must:
- Clone upstream: `git clone https://github.com/hao-ai-lab/JacobiForcing.git $REPO`.
- Record commit hash: `cd $REPO && git rev-parse HEAD > $REPRO_ROOT/upstream_commit.txt`.
- Install deps: `pip install -r $REPO/requirements.txt`.
- Refresh `versions.lock` after install.
- Download both base models with revision capture:
  - `huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir $REPRO_ROOT/models/base-coder --local-dir-use-symlinks False`
  - `huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct --local-dir $REPRO_ROOT/models/base-math --local-dir-use-symlinks False`
  - Record each model's `git rev-parse HEAD` from the downloaded directory (HF cache exposes it) into `base_model_revisions.txt`.
- Reproducibility checklist: `upstream_commit.txt`, `base_model_revisions.txt`, `versions.lock`.

- [ ] **Step 2: Draft Stage 1**

Append:

````markdown

## Stage 1 — Source and base models

### Commands

```bash
# 1. Clone upstream and pin commit
git clone https://github.com/hao-ai-lab/JacobiForcing.git $REPO
cd $REPO
git rev-parse HEAD > $REPRO_ROOT/upstream_commit.txt
cat $REPRO_ROOT/upstream_commit.txt    # record in your paper's methods section

# 2. Install Python deps (inside the conda env from Stage 0)
pip install -r requirements.txt

# 3. Refresh versions.lock now that deps are installed
pip freeze > $REPRO_ROOT/versions.lock

# 4. Download base models and record their HF revisions
mkdir -p $REPRO_ROOT/models

huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
    --local-dir $REPRO_ROOT/models/base-coder \
    --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct \
    --local-dir $REPRO_ROOT/models/base-math \
    --local-dir-use-symlinks False

# HF stores a commit hash inside the snapshot dir — capture it.
{
    echo "base-coder: $(cat $REPRO_ROOT/models/base-coder/.git/HEAD 2>/dev/null || git -C $REPRO_ROOT/models/base-coder rev-parse HEAD 2>/dev/null || echo 'NOT_A_GIT_REPO — use `huggingface-cli download --revision` output or the `commit_hash` field in HF cache')"
    echo "base-math:  $(cat $REPRO_ROOT/models/base-math/.git/HEAD 2>/dev/null  || git -C $REPRO_ROOT/models/base-math rev-parse HEAD 2>/dev/null  || echo 'NOT_A_GIT_REPO — see note above')"
} > $REPRO_ROOT/base_model_revisions.txt
```

> **Note on HF revision capture.** `huggingface-cli download` into a non-bare `--local-dir` does not always leave a `.git` dir behind. If the above returns `NOT_A_GIT_REPO`, instead capture the revision by running:
> `huggingface-cli scan-cache | grep Qwen2.5-Coder-7B-Instruct` — the commit hash is in the output.
> Or re-download with `--revision main` pinned and record the resolved hash from the CLI output.

### Reproducibility checklist

- [ ] `$REPRO_ROOT/upstream_commit.txt` contains a 40-char SHA.
- [ ] `$REPRO_ROOT/base_model_revisions.txt` contains both model revisions (40-char SHA each).
- [ ] `$REPRO_ROOT/versions.lock` refreshed after `pip install`.

### Expected artifacts

- `$REPO/` (cloned repo)
- `$REPRO_ROOT/upstream_commit.txt`
- `$REPRO_ROOT/models/base-coder/`, `$REPRO_ROOT/models/base-math/`
- `$REPRO_ROOT/base_model_revisions.txt`
- `$REPRO_ROOT/versions.lock` (refreshed)
````

- [ ] **Step 3: Verify Stage 1**

Run: `grep -c "upstream_commit.txt" docs/training_plan.md` — expect ≥ 2.
Run: `grep -c "base_model_revisions.txt" docs/training_plan.md` — expect ≥ 2.
Run: `grep -c "huggingface-cli download" docs/training_plan.md` — expect ≥ 2.

- [ ] **Step 4: Commit**

```bash
git add docs/training_plan.md
git commit -m "docs(training_plan): add Stage 1 source and base models"
```

---

## Task 9: Doc 2 Stage 2 + Stage 2.5 — Data download and smoke test

**Files:**
- Modify: `docs/training_plan.md`

- [ ] **Step 1: Acceptance bullets**

**Stage 2** must:
- `git lfs clone` (or `huggingface-cli download`) both training datasets.
- Record each dataset's HF commit into `dataset_revisions.txt`.
- Verify row counts: Coder ≈ 2.16M, Math ≈ 230k.

**Stage 2.5** must:
- Slice 100 samples from the Coder dataset into a temp shard.
- Run the trainer with `--max_steps 10 --save_steps 5 --logging_steps 1` overrides on one GPU (or all 8 with a small effective batch).
- Pass criteria: run exits cleanly, a checkpoint appears at `tmp_smoke/checkpoint-5`, loss at step 10 is lower than at step 1 (any amount).
- Failure handling: if crash, inspect `NCCL_DEBUG=INFO` output and the `Appendix B` failure table.
- **Explicit:** smoke test is one-GPU if feasible (faster to iterate); if only multi-GPU works on this script, use all 8.

- [ ] **Step 2: Draft Stages 2 and 2.5**

Append:

````markdown

## Stage 2 — Option A data download

### Commands

```bash
mkdir -p $REPRO_ROOT/data
cd $REPRO_ROOT/data

# Coder (n=32, w=16)
git lfs clone https://huggingface.co/datasets/JacobiForcing/OpenCodeInstruct_training_data_n32w16
# Math (n=64, w=32)
git lfs clone https://huggingface.co/datasets/JacobiForcing/OpenThoughts_Math_training_data_n64w32

# Record commit hashes
{
  echo "coder: $(git -C OpenCodeInstruct_training_data_n32w16 rev-parse HEAD)"
  echo "math:  $(git -C OpenThoughts_Math_training_data_n64w32 rev-parse HEAD)"
} > $REPRO_ROOT/dataset_revisions.txt

cat $REPRO_ROOT/dataset_revisions.txt
```

### Row-count sanity check

```bash
python - <<'EOF'
from datasets import load_from_disk
import os
root = os.environ['REPRO_ROOT']
for name, approx in [
    ('OpenCodeInstruct_training_data_n32w16', 2_160_000),
    ('OpenThoughts_Math_training_data_n64w32', 230_000),
]:
    p = f"{root}/data/{name}"
    # datasets stored as parquet on HF — load via datasets.load_dataset
    from datasets import load_dataset
    ds = load_dataset('parquet', data_files=f"{p}/**/*.parquet", split='train')
    print(f"{name}: {len(ds):,} rows (expected ~{approx:,})")
EOF
```

Expected output:
- `OpenCodeInstruct_training_data_n32w16: 2,16x,xxx rows (expected ~2,160,000)`
- `OpenThoughts_Math_training_data_n64w32: 23x,xxx rows (expected ~230,000)`

### Reproducibility checklist

- [ ] `$REPRO_ROOT/dataset_revisions.txt` contains two 40-char SHAs.
- [ ] Both `len(ds)` values within 1% of the expected approximations above.

### Expected artifacts

- `$REPRO_ROOT/data/OpenCodeInstruct_training_data_n32w16/`
- `$REPRO_ROOT/data/OpenThoughts_Math_training_data_n64w32/`
- `$REPRO_ROOT/dataset_revisions.txt`

## Stage 2.5 — Smoke test

Purpose: validate the data loader, DeepSpeed launch, checkpoint write, and a basic loss decrease before committing full GPU-hours.

### Commands

```bash
# 1. Slice 100 samples from the Coder dataset into a small debug shard
python - <<'EOF'
from datasets import load_dataset
import os
root = os.environ['REPRO_ROOT']
src = f"{root}/data/OpenCodeInstruct_training_data_n32w16"
dst = f"{root}/data/coder_smoke_100"
ds = load_dataset('parquet', data_files=f"{src}/**/*.parquet", split='train')
ds.select(range(100)).save_to_disk(dst)
print(f"Wrote {dst}")
EOF

# 2. Copy the real training script to a smoke-test variant and override:
#    - data path -> coder_smoke_100
#    - max_steps=10, save_steps=5, logging_steps=1
#    - output_dir -> $REPRO_ROOT/tmp_smoke
cp $REPO/JacobiForcing/scripts/train/train_jacobi_forcing_coder_n32.sh \
   $REPO/JacobiForcing/scripts/train/smoke_coder_n32.sh

# Apply the override edits (the engineer must read the script and identify
# its dataset arg and training-args env vars, then edit smoke_coder_n32.sh:
#   - point dataset path at $REPRO_ROOT/data/coder_smoke_100
#   - add `--max_steps 10 --save_steps 5 --logging_steps 1`
#   - set output_dir to $REPRO_ROOT/tmp_smoke
# These edits are script-specific; see Stage 3 for the full arg surface.)

# 3. Run it on all 8 GPUs (single-GPU launch requires changing the DeepSpeed
#    config, which itself adds variance — safer to keep 8-GPU topology).
cd $REPO
bash JacobiForcing/scripts/train/smoke_coder_n32.sh 2>&1 | tee $REPRO_ROOT/tmp_smoke/smoke.log
```

### Pass criteria

- Exit code 0.
- `$REPRO_ROOT/tmp_smoke/checkpoint-5/` exists and contains `pytorch_model*` or `model.safetensors`.
- Loss at step 10 < loss at step 1 (by any amount). Extract from `smoke.log`:

```bash
grep -E "'loss':" $REPRO_ROOT/tmp_smoke/smoke.log | head -1
grep -E "'loss':" $REPRO_ROOT/tmp_smoke/smoke.log | tail -1
```

### If the smoke test fails

See Appendix B. Most common causes: data-arg mismatch (wrong column name after Option A preprocessing), missing flash-attn wheel, NCCL init timeout.

### Reproducibility checklist

- [ ] `$REPRO_ROOT/tmp_smoke/smoke.log` archived.
- [ ] Loss at step 10 < loss at step 1 recorded explicitly.

### Expected artifacts

- `$REPRO_ROOT/tmp_smoke/checkpoint-5/`, `checkpoint-10/` (saved)
- `$REPRO_ROOT/tmp_smoke/smoke.log`
- `$REPO/JacobiForcing/scripts/train/smoke_coder_n32.sh` (delete after smoke test passes OR keep for traceability — record choice in Appendix A)
````

- [ ] **Step 3: Verify Stages 2 and 2.5**

Run: `grep -c "OpenCodeInstruct_training_data_n32w16" docs/training_plan.md` — expect ≥ 3.
Run: `grep -c "OpenThoughts_Math_training_data_n64w32" docs/training_plan.md` — expect ≥ 2.
Run: `grep -c "smoke" docs/training_plan.md` — expect ≥ 6.
Run: `grep -cE "TBD|TODO|FIXME" docs/training_plan.md` — expect 0.

- [ ] **Step 4: Commit**

```bash
git add docs/training_plan.md
git commit -m "docs(training_plan): add Stage 2 data download and Stage 2.5 smoke test"
```

---

## Task 10: Doc 2 Stage 3 — Coder training

**Files:**
- Modify: `docs/training_plan.md`

- [ ] **Step 1: Acceptance bullets**

Stage 3 must:
- Reference the exact script path: `JacobiForcing/scripts/train/train_jacobi_forcing_coder_n32.sh` (inside `$REPO`).
- Describe the minimum required edits (point dataset path at `$REPRO_ROOT/data/OpenCodeInstruct_training_data_n32w16`, point base model at `$REPRO_ROOT/models/base-coder`, output_dir at `$REPRO_ROOT/runs/coder-n32`).
- **Not** change any hyperparameter the script sets (LR, schedule, epochs, micro-bs, grad-accum, seed, warmup). A note states this.
- Launch command: `bash JacobiForcing/scripts/train/train_jacobi_forcing_coder_n32.sh 2>&1 | tee $REPRO_ROOT/runs/coder-n32/train.log`.
- Expected wall-clock estimate: state as "record during first run; upstream blog/paper's compute is a reference lower bound."
- Reproducibility checklist: `train.log` saved, final checkpoint path recorded, any edits to the script logged into Appendix A.

- [ ] **Step 2: Draft Stage 3**

Append:

````markdown

## Stage 3 — Coder training (7B, n32w16)

### Script used

`$REPO/JacobiForcing/scripts/train/train_jacobi_forcing_coder_n32.sh` — pairs with `OpenCodeInstruct_training_data_n32w16`. DeepSpeed config: `JacobiForcing/scripts/train/ds_config.json` (no CPU offload).

### Required edits — and ONLY these

The upstream script hard-codes paths. Edit these three, and only these three:

1. Dataset path → `$REPRO_ROOT/data/OpenCodeInstruct_training_data_n32w16`
2. Base model path → `$REPRO_ROOT/models/base-coder`
3. Output dir → `$REPRO_ROOT/runs/coder-n32`

**Do not change** LR, LR schedule, warmup, epoch count, micro-batch size, gradient accumulation, seed, or any model-architecture arg. H200 has ample VRAM to raise micro-bs, but doing so changes the *effective* batch and breaks comparability with upstream. Frozen hparams are the point of a baseline reproduction.

Record every edit you make as a unified diff in Appendix A:
```bash
cd $REPO
git diff JacobiForcing/scripts/train/train_jacobi_forcing_coder_n32.sh \
    > $REPRO_ROOT/appendix_A_coder_script.diff
```

### Launch

```bash
mkdir -p $REPRO_ROOT/runs/coder-n32
cd $REPO
bash JacobiForcing/scripts/train/train_jacobi_forcing_coder_n32.sh \
    2>&1 | tee $REPRO_ROOT/runs/coder-n32/train.log
```

### Monitoring

- **wandb/tensorboard.** If the upstream trainer initializes wandb, it will prompt; paste the API key or set `WANDB_MODE=offline` for reproducibility. Record the run ID.
- Expected first-epoch behavior: loss decreasing monotonically after the first ~100 steps; no NaNs; grad norm in a reasonable range (script-dependent, typically < 10).

### Wall-clock expectation

Upstream does not publish exact wall-clock; record yours on first run. For 8× H200 on ~2.16M training sequences at the script's default epoch count, expect on the order of tens of hours (record and report in the paper).

### Reproducibility checklist

- [ ] `$REPRO_ROOT/runs/coder-n32/train.log` archived.
- [ ] `$REPRO_ROOT/runs/coder-n32/checkpoint-*/` present (final + periodic).
- [ ] `$REPRO_ROOT/appendix_A_coder_script.diff` saved.
- [ ] wandb/tensorboard run ID recorded (or `WANDB_MODE=offline` log archived).
- [ ] Final-epoch loss recorded.

### Expected artifacts

- `$REPRO_ROOT/runs/coder-n32/train.log`
- `$REPRO_ROOT/runs/coder-n32/checkpoint-<final>/` (final Coder checkpoint)
- `$REPRO_ROOT/appendix_A_coder_script.diff`
````

- [ ] **Step 3: Verify Stage 3**

Run: `grep -c "train_jacobi_forcing_coder_n32.sh" docs/training_plan.md` — expect ≥ 2.
Run: `grep -c "Do not change" docs/training_plan.md` — expect ≥ 1.
Run: `grep -c "appendix_A_coder_script.diff" docs/training_plan.md` — expect ≥ 2.

- [ ] **Step 4: Commit**

```bash
git add docs/training_plan.md
git commit -m "docs(training_plan): add Stage 3 Coder training"
```

---

## Task 11: Doc 2 Stage 4 — Math training

**Files:**
- Modify: `docs/training_plan.md`

- [ ] **Step 1: Acceptance bullets**

Stage 4 must:
- Reference the exact script: `JacobiForcing/scripts/train/train_clean_context_conditioned_cllm_openthinker2_n64.sh` (inside `$REPO`).
- Note the `openthinker2` naming carryover.
- Same edit discipline as Stage 3: only paths (dataset, base model, output dir); no hparam changes.
- Capture the edits via `git diff` into `appendix_A_math_script.diff`.
- Launch command identical in shape to Stage 3, with the math paths substituted.

- [ ] **Step 2: Draft Stage 4**

Append:

````markdown

## Stage 4 — Math training (7B, n64w32)

### Script used

`$REPO/JacobiForcing/scripts/train/train_clean_context_conditioned_cllm_openthinker2_n64.sh` — pairs with `OpenThoughts_Math_training_data_n64w32`. The `openthinker2` in the filename is a legacy naming carryover; this *is* the current math training script. Same DeepSpeed config as Stage 3 (`ds_config.json`).

### Required edits — and ONLY these

1. Dataset path → `$REPRO_ROOT/data/OpenThoughts_Math_training_data_n64w32`
2. Base model path → `$REPRO_ROOT/models/base-math`
3. Output dir → `$REPRO_ROOT/runs/math-n64`

Same discipline as Stage 3: no hparam changes. Record every edit:
```bash
cd $REPO
git diff JacobiForcing/scripts/train/train_clean_context_conditioned_cllm_openthinker2_n64.sh \
    > $REPRO_ROOT/appendix_A_math_script.diff
```

### Launch

```bash
mkdir -p $REPRO_ROOT/runs/math-n64
cd $REPO
bash JacobiForcing/scripts/train/train_clean_context_conditioned_cllm_openthinker2_n64.sh \
    2>&1 | tee $REPRO_ROOT/runs/math-n64/train.log
```

### Monitoring and expectations

Same shape as Stage 3: wandb/tensorboard optional, record the run ID, expect monotonic loss decrease after warmup, track grad norm. The Math dataset is ~10× smaller (230k vs 2.16M), so wall-clock is materially shorter than Coder — record yours.

### Reproducibility checklist

- [ ] `$REPRO_ROOT/runs/math-n64/train.log` archived.
- [ ] `$REPRO_ROOT/runs/math-n64/checkpoint-*/` present.
- [ ] `$REPRO_ROOT/appendix_A_math_script.diff` saved.
- [ ] wandb/tensorboard run ID recorded.
- [ ] Final-epoch loss recorded.

### Expected artifacts

- `$REPRO_ROOT/runs/math-n64/train.log`
- `$REPRO_ROOT/runs/math-n64/checkpoint-<final>/` (final Math checkpoint)
- `$REPRO_ROOT/appendix_A_math_script.diff`
````

- [ ] **Step 3: Verify Stage 4**

Run: `grep -c "train_clean_context_conditioned_cllm_openthinker2_n64.sh" docs/training_plan.md` — expect ≥ 2.
Run: `grep -c "appendix_A_math_script.diff" docs/training_plan.md` — expect ≥ 2.

- [ ] **Step 4: Commit**

```bash
git add docs/training_plan.md
git commit -m "docs(training_plan): add Stage 4 Math training"
```

---

## Task 12: Doc 2 Stage 5 + Stage 6 — Evaluation

**Files:**
- Modify: `docs/training_plan.md`

- [ ] **Step 1: Acceptance bullets**

**Stage 5 (HumanEval)** must:
- Point at `$REPO/JacobiForcing/jacobi_forcing_inference_humaneval.py` (plain) AND `$REPO/JacobiForcing/jacobi_forcing_inference_MR_humaneval.py` (with MR).
- Include the MR hparams explicitly: `n=64, K=2, pool_size=4, r=0.85`.
- Also run `$REPO/JacobiForcing/ar_inference_baseline.py` on the same checkpoint as the AR-speedup denominator.
- Record pass@1, tokens/forward, wall-clock speedup.

**Stage 6 (MATH500)** must:
- Point at `$REPO/JacobiForcing/jacobi_forcing_inference_MATH500.py`.
- Record accuracy and speedup vs `ar_inference_baseline.py`.

Both stages must include the explicit note that MBPP and GSM8K are out of scope here.

- [ ] **Step 2: Draft Stage 5 and Stage 6**

Append:

````markdown

## Stage 5 — HumanEval on Coder checkpoint

### Scripts used

- `$REPO/JacobiForcing/jacobi_forcing_inference_humaneval.py` — plain Jacobi decoding.
- `$REPO/JacobiForcing/jacobi_forcing_inference_MR_humaneval.py` — Jacobi decoding with Multi-block Rejection Recycling.
- `$REPO/JacobiForcing/ar_inference_baseline.py` — AR baseline on the *same* checkpoint (the denominator for the speedup ratio).

### MR hyperparameters (upstream defaults)

| Parameter       | Value | Meaning                                                    |
| --------------- | ----- | ---------------------------------------------------------- |
| Block size `n`  | 64    | Tokens decoded per Jacobi iteration.                       |
| Block count `K` | 2     | Parallel in-flight blocks.                                 |
| Pool size       | 4     | Candidate-block pool for rejection recycling.              |
| Activation `r`  | 0.85  | Fraction of block activated per iteration.                 |

### Commands

```bash
export CODER_CKPT=$REPRO_ROOT/runs/coder-n32/checkpoint-<final>    # replace <final>
mkdir -p $REPRO_ROOT/eval/humaneval

cd $REPO

# 1. AR baseline (denominator for speedup)
python JacobiForcing/ar_inference_baseline.py \
    --model-path $CODER_CKPT \
    --benchmark humaneval \
    --output $REPRO_ROOT/eval/humaneval/ar_baseline.json \
    2>&1 | tee $REPRO_ROOT/eval/humaneval/ar_baseline.log

# 2. Plain Jacobi
python JacobiForcing/jacobi_forcing_inference_humaneval.py \
    --model-path $CODER_CKPT \
    --output $REPRO_ROOT/eval/humaneval/jacobi_plain.json \
    2>&1 | tee $REPRO_ROOT/eval/humaneval/jacobi_plain.log

# 3. Jacobi + MR (upstream defaults)
python JacobiForcing/jacobi_forcing_inference_MR_humaneval.py \
    --model-path $CODER_CKPT \
    --n 64 --K 2 --pool_size 4 --r 0.85 \
    --output $REPRO_ROOT/eval/humaneval/jacobi_mr.json \
    2>&1 | tee $REPRO_ROOT/eval/humaneval/jacobi_mr.log
```

> **Flag names** (`--n`, `--K`, `--pool_size`, `--r`) mirror the inference README; confirm against the script's `argparse` and update these commands if the upstream flag names differ. Log any such rename in Appendix A.

### Metrics to record

- pass@1 (from each JSON).
- tokens/forward (from the log — usually printed as `avg_tokens_per_forward`).
- wall-clock elapsed (from the log).
- Speedup = AR wall-clock / Jacobi-MR wall-clock. Expected paper-reported value: **4.0× with 83.5% pass@1** (Jacobi-MR vs AR on the same checkpoint).

### Out of scope

**MBPP** uses the upstream authors' `evalchemy` harness setup — not reproduced here. If needed, it gets its own plan.

### Reproducibility checklist

- [ ] `$REPRO_ROOT/eval/humaneval/ar_baseline.{json,log}`, `jacobi_plain.{json,log}`, `jacobi_mr.{json,log}` archived.
- [ ] pass@1, tokens/forward, wall-clock, speedup recorded in a summary table.

### Expected artifacts

- `$REPRO_ROOT/eval/humaneval/*.json`
- `$REPRO_ROOT/eval/humaneval/*.log`

## Stage 6 — MATH500 on Math checkpoint

### Scripts used

- `$REPO/JacobiForcing/jacobi_forcing_inference_MATH500.py` — Jacobi decoding on MATH500.
- `$REPO/JacobiForcing/ar_inference_baseline.py` — AR denominator.

### Commands

```bash
export MATH_CKPT=$REPRO_ROOT/runs/math-n64/checkpoint-<final>    # replace <final>
mkdir -p $REPRO_ROOT/eval/math500

cd $REPO

# 1. AR baseline
python JacobiForcing/ar_inference_baseline.py \
    --model-path $MATH_CKPT \
    --benchmark math500 \
    --output $REPRO_ROOT/eval/math500/ar_baseline.json \
    2>&1 | tee $REPRO_ROOT/eval/math500/ar_baseline.log

# 2. Jacobi inference on MATH500
python JacobiForcing/jacobi_forcing_inference_MATH500.py \
    --model-path $MATH_CKPT \
    --output $REPRO_ROOT/eval/math500/jacobi.json \
    2>&1 | tee $REPRO_ROOT/eval/math500/jacobi.log
```

> Confirm AR baseline's `--benchmark` flag accepts `math500`. If not, find the script's expected value and log the change in Appendix A.

### Metrics to record

- Accuracy on MATH500.
- Tokens/forward, wall-clock, speedup vs AR.

### Out of scope

**GSM8K** (paper-reported 91.4% / 3.7×) is not reproduced here — needs `evalchemy`. Flag as follow-up.

### Reproducibility checklist

- [ ] `$REPRO_ROOT/eval/math500/ar_baseline.{json,log}`, `jacobi.{json,log}` archived.
- [ ] MATH500 accuracy, tokens/forward, wall-clock, speedup recorded.

### Expected artifacts

- `$REPRO_ROOT/eval/math500/*.json`
- `$REPRO_ROOT/eval/math500/*.log`
````

- [ ] **Step 3: Verify Stage 5 and Stage 6**

Run: `grep -c "jacobi_forcing_inference_humaneval.py" docs/training_plan.md` — expect ≥ 1.
Run: `grep -c "jacobi_forcing_inference_MR_humaneval.py" docs/training_plan.md` — expect ≥ 1.
Run: `grep -c "jacobi_forcing_inference_MATH500.py" docs/training_plan.md` — expect ≥ 1.
Run: `grep -c "ar_inference_baseline.py" docs/training_plan.md` — expect ≥ 2.
Run: `grep -c "n=64, K=2\|n 64 --K 2" docs/training_plan.md` — expect ≥ 1.

- [ ] **Step 4: Commit**

```bash
git add docs/training_plan.md
git commit -m "docs(training_plan): add Stage 5 HumanEval and Stage 6 MATH500"
```

---

## Task 13: Doc 2 Stage 7 and Appendices A + B

**Files:**
- Modify: `docs/training_plan.md`

- [ ] **Step 1: Acceptance bullets**

**Stage 7 (Reporting)** must:
- Include a comparison table template with columns: Benchmark, Our pass@1/accuracy, Our speedup, Paper pass@1/accuracy, Paper speedup, Delta.
- Populate the "Paper" column for HumanEval (83.5% / 4.0×) and leave MATH500 paper value as "from paper Table X — fill at report time".
- State tolerance is a *reporting-time* user decision (no hard threshold in the plan).
- List provenance artifacts the paper must cite.

**Appendix A** must be a running-ledger template: table with columns (File, Change, Reason, Commit).

**Appendix B** must be a failure-mode table: Symptom, Likely cause, Fix. Cover OOM, NaN loss, diverging loss, DeepSpeed hang, flash-attn import error, HF download 429.

- [ ] **Step 2: Draft Stage 7 and Appendices**

Append:

````markdown

## Stage 7 — Reporting

### Results table (fill during execution)

| Benchmark | Our pass@1 / acc | Our speedup | Paper pass@1 / acc | Paper speedup | Δ (ours − paper) |
| --------- | ---------------- | ----------- | ------------------ | ------------- | ---------------- |
| HumanEval (Coder, MR) | _<fill>_  | _<fill>_  | 83.5% | 4.0× | _<fill>_ |
| HumanEval (Coder, plain Jacobi) | _<fill>_ | _<fill>_ | _(not primary paper comparison — use for internal diagnostic)_ | — | — |
| MATH500 (Math)        | _<fill>_  | _<fill>_  | _<from paper table — fill at report time>_ | _<from paper>_ | _<fill>_ |

### Out of scope — explicitly

- **GSM8K** (paper: 91.4% / 3.7×) — requires `evalchemy`. Not reproduced here.
- **MBPP** — same.

### Tolerance

The threshold for declaring "reproduced" is a reporting-time decision informed by run-to-run variance and the paper's stated error bars (if any). Common practice is *within ±0.5 to ±1.0 pp on accuracy and within ±5% on speedup*; justify whatever threshold you use in the paper.

### Provenance artifacts to cite

The paper's methods section should reference:

- `$REPRO_ROOT/upstream_commit.txt` — JacobiForcing repo commit used.
- `$REPRO_ROOT/base_model_revisions.txt` — base Qwen revisions used.
- `$REPRO_ROOT/dataset_revisions.txt` — HF training-shard commits used.
- `$REPRO_ROOT/versions.lock` — installed Python package versions.
- `$REPRO_ROOT/appendix_A_*.diff` — every upstream-script edit.
- wandb/tensorboard run IDs for Coder and Math training.
- Note that HF training shards were rolled from `JacobiForcing_Coder_7B_v0` per the HF dataset card (disclosable provenance).

## Appendix A — Deviations from upstream recipe

Running ledger. Every edit to any upstream file, every changed flag, every non-default choice goes here. Each row cross-references a git commit (in this repo) and a diff artifact.

| File                                                              | Change                                      | Reason                                          | Commit SHA |
| ----------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------- | ---------- |
| `JacobiForcing/scripts/train/train_jacobi_forcing_coder_n32.sh`   | dataset / base-model / output-dir paths    | point at local $REPRO_ROOT paths                | _<fill>_   |
| `JacobiForcing/scripts/train/train_clean_context_conditioned_cllm_openthinker2_n64.sh` | dataset / base-model / output-dir paths | same, for Math | _<fill>_   |
| _add rows as you go_ | _<fill>_ | _<fill>_ | _<fill>_ |

Store the raw diffs alongside this table at `$REPRO_ROOT/appendix_A_*.diff`.

## Appendix B — Failure-mode runbook

| Symptom                                      | Likely cause                                                         | Fix                                                                                      |
| -------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `torch.cuda.OutOfMemoryError` during training| Unexpected memory — a kernel or activation is larger than budgeted   | Enable gradient checkpointing in the trainer flag (`--gradient_checkpointing`); if still OOM, switch DeepSpeed config to `ds_config_cpu_offloading.json` and record in Appendix A. |
| NaN loss within first 50 steps               | Flash-attention / torch mismatch; numerically unstable kernel        | Reinstall flash-attn pinned to the version in `versions.lock`. Rerun smoke test.         |
| Loss diverges after warmup                   | Wrong dataset path (wrong `n/w` shard loaded) or mis-edited LR       | Verify `dataset_revisions.txt` matches the expected commit; `git diff` the script against upstream — any non-path edit is the cause. |
| DeepSpeed hangs at init, no NCCL traffic      | Interconnect / NCCL init misconfiguration                            | Export `NCCL_DEBUG=INFO`, rerun; check IFNAME and `NCCL_SOCKET_IFNAME`. For single-node, ensure no stale `NCCL_ASYNC_ERROR_HANDLING` override. |
| `ImportError: flash_attn`                    | flash-attention wheel not built against installed torch              | Reinstall flash-attn from prebuilt wheels matching torch version in `versions.lock`.     |
| HuggingFace 429 / rate-limit on download      | Dataset/model download throttled                                     | Wait and retry; pass `--max-workers 4` to `huggingface-cli download`.                    |
| Evaluation JSON shows 0 pass@1 with no errors| Wrong checkpoint path (loading base model by mistake)                | Confirm `$CODER_CKPT` points at the final training checkpoint, not `base-coder`.          |
````

- [ ] **Step 3: Verify Stage 7 + Appendices**

Run: `grep -c "^| " docs/training_plan.md` — expect ≥ 25 (headers + rows from both Stage 7 table and Appendix tables).
Run: `grep -c "Appendix A" docs/training_plan.md` — expect ≥ 3.
Run: `grep -c "Appendix B" docs/training_plan.md` — expect ≥ 2.
Run: `grep -cE "TBD|TODO|FIXME" docs/training_plan.md` — expect 0 (underscored `_<fill>_` placeholders are intentional — they're where the engineer records data during execution, not pre-execution TBDs).

- [ ] **Step 4: Commit**

```bash
git add docs/training_plan.md
git commit -m "docs(training_plan): add Stage 7 reporting and Appendices A+B"
```

---

## Task 14: Cross-doc review and finalization

**Files:**
- Modify: `docs/codebase_explanation.md`, `docs/training_plan.md` (final cleanup only)

- [ ] **Step 1: Acceptance bullets**

Final pass must verify:
- Cross-references: Doc 1 §5 mentions `docs/training_plan.md`; Doc 2's scaffold mentions `docs/codebase_explanation.md` for conceptual background (add a one-line cross-reference in Doc 2's scaffold if missing).
- Every upstream path cited across BOTH docs exists in the upstream tree.
- Markdown renders (TOC anchors match header slugs).
- No `TBD`/`TODO`/`FIXME` anywhere (only the intentional `_<fill>_` markers in Stage 7 and Appendix A, which are recording placeholders).

- [ ] **Step 2: Add cross-reference if missing**

In `docs/training_plan.md`, after the "Conventions" section of the scaffold, add this paragraph if not already present:

```markdown
For conceptual background on Jacobi Forcing (why it exists, what each code file does), read `docs/codebase_explanation.md` first.
```

- [ ] **Step 3: Cross-doc path audit**

Run this over both docs and confirm every upstream path is reachable:

```bash
for f in docs/codebase_explanation.md docs/training_plan.md; do
  echo "=== $f ==="
  grep -oE '`[A-Za-z0-9_/.-]+\.(py|sh|json)`' "$f" | sort -u | while read -r p; do
    stripped=${p//\`/}
    curl -sf -o /dev/null "https://raw.githubusercontent.com/hao-ai-lab/JacobiForcing/main/$stripped" \
      && echo "OK  $stripped" \
      || echo "MISS $stripped"
  done
done
```

Every line must be `OK`. Any `MISS` must be resolved (typo or an upstream path moved) before committing this task.

- [ ] **Step 4: Placeholder audit**

```bash
# Intentional recording placeholders are `_<fill>_` — allow those.
# Flag any TBD/TODO/FIXME and any bare `<foo>` that isn't `<fill>`.
for f in docs/codebase_explanation.md docs/training_plan.md; do
  echo "=== $f ==="
  grep -nE "TBD|TODO|FIXME" "$f" || echo "  clean"
  grep -nE '<[a-z][a-z _0-9-]+>' "$f" | grep -v 'fill' || echo "  no stray <placeholders>"
done
```

Expected: every section says `clean` / `no stray <placeholders>`.

- [ ] **Step 5: TOC anchor check (both docs)**

For each doc, confirm that every TOC link anchor matches an actual heading slug. In Markdown, `[Stage 0 — Environment](#stage-0--environment)` requires a heading slug `stage-0--environment`. Quick check:

```bash
for f in docs/codebase_explanation.md docs/training_plan.md; do
  echo "=== $f ==="
  python - <<EOF
import re, sys
text = open("$f").read()
toc_anchors = re.findall(r'\]\(#([^)]+)\)', text)
headings = re.findall(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
def slug(h):
    s = h.lower()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'\s+', '-', s).strip('-')
    return s
heading_slugs = {slug(h) for h in headings}
missing = [a for a in toc_anchors if a not in heading_slugs]
if missing:
    print("MISSING:", missing)
    sys.exit(1)
print("all TOC anchors match")
EOF
done
```

Expected: `all TOC anchors match` for both.

- [ ] **Step 6: Commit final pass**

```bash
git add docs/codebase_explanation.md docs/training_plan.md
git commit -m "docs: cross-doc review — anchors, paths, placeholders clean"
```

- [ ] **Step 7: Human readability sanity check**

Open both files and skim them front-to-back. Ask: could a collaborator handed these two files with no other context actually (a) understand Jacobi Forcing and (b) reproduce a Coder and Math checkpoint? If any section fails this test, fix inline and add a commit.

---

## Self-review (plan author, pre-handoff)

After writing every task, I ran this checklist against the spec:

### Spec coverage

| Spec section                                           | Plan task(s)     |
| ------------------------------------------------------ | ---------------- |
| §1 Goal (two docs)                                     | Tasks 1–5 (Doc 1), Tasks 6–13 (Doc 2), Task 14 (cross-doc) |
| §2 Why this design                                     | Reflected in plan framing and acceptance bullets |
| §3 Dataset choice (Option A)                           | Tasks 8, 9 (source + data download)             |
| §3 v0-trajectory provenance note                       | Task 5 (quirks) + Task 13 (Stage 7 provenance artifacts) |
| §4 Doc 1 outline (§1–§5)                               | Tasks 1–5 (one per section)                     |
| §5 Doc 2 outline (Stages 0–7 + Appendices)             | Tasks 6–13 (one or two per stage)               |
| §6 Success criteria                                    | Task 14 Step 7 (readability sanity check)       |
| §7 Out of scope (MBPP, GSM8K, hparam search, multinode default) | Stage 5, Stage 6, Stage 3/4 discipline note |
| §8 Risks (paths, legacy naming, unpinned revisions, version drift, v0 data) | Task 5 (quirks), Task 13 (provenance artifacts) |
| §9 File layout                                         | Task 1 (Doc 1 scaffold), Task 6 (Doc 2 scaffold) |

No uncovered spec sections.

### Placeholder scan

Checked the plan for `TBD`, `TODO`, `FIXME`, "fill in later", "similar to Task N", "implement appropriate error handling". None present. The intentional `_<fill>_` markers inside Stage 7 and Appendix A are template cells to be filled during *execution*, not plan-authoring placeholders; their presence is called out in Task 13 Step 3 and Task 14 Step 4.

### Type / naming consistency

- `$REPRO_ROOT` and `$REPO` introduced once in Task 6 and used consistently across Tasks 7–13.
- Dataset names used consistently: `OpenCodeInstruct_training_data_n32w16` (Coder), `OpenThoughts_Math_training_data_n64w32` (Math). No variants.
- Model path conventions: `$REPRO_ROOT/models/base-coder`, `$REPRO_ROOT/models/base-math`; output dirs `$REPRO_ROOT/runs/coder-n32`, `$REPRO_ROOT/runs/math-n64`. Consistent across Tasks 8, 10, 11, 12.
- Artifact filenames used across tasks: `upstream_commit.txt`, `base_model_revisions.txt`, `dataset_revisions.txt`, `versions.lock`, `appendix_A_coder_script.diff`, `appendix_A_math_script.diff`. Each introduced once and referred to by the same name afterward.

No inconsistencies found.
