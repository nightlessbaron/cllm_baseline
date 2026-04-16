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

## 2. The five method stages

### §2.1 Data bucketing

Before any Jacobi-specific supervision is created, the raw prompt-and-completion examples are reorganized by sequence length. Length bucketing means grouping training examples whose tokenized lengths are similar, then packing examples from the same bucket together instead of mixing very short and very long sequences indiscriminately. That matters here because this pipeline relies on efficient packed training: if a batch contains highly uneven lengths, the shorter sequences contribute mostly padding tokens, which waste compute, memory bandwidth, and attention work. Keeping lengths close reduces that waste and lets the downstream FlexAttention path and DeepSpeed packing strategy spend more of each forward pass on real tokens. In other words, bucketing is not an optional data-cleanliness step; it is part of making the packed multisequence recipe practical at scale.

The repo keeps the two upstream corpora separate at this stage because they are different sources with different prompt distributions and formatting conventions. The coding corpus is bucketed through the OpenCodeInstruct-specific script, while the math corpus is bucketed through the OpenThoughts2-specific script. That separation makes the preprocessing explicit and avoids pretending there is one universal bucketer for both domains. The bucket boundaries themselves are a pipeline hyperparameter: they define how coarsely or finely lengths are grouped before packing. For reproduction, though, they are not the knob you are expected to retune. They are part of the fixed data-preparation recipe that feeds the later stages.

**Code:**
- `generate_trajectory/data/0_bucketing_opencodeinstruct.py` — Coder bucketing.
- `generate_trajectory/data/0_bucketing_openthought2.py` — Math bucketing.

### §2.2 Trajectory generation

The second stage is where Jacobi Forcing gets its supervision signal. Instead of generating ordinary autoregressive completions and treating the final answer as the only target, the base AR model is run in a Jacobi rollout mode over real prompts sampled from OpenCodeInstruct or OpenThoughts2. Operationally, that means the model performs fixed-point iteration over blocks of future tokens: start from a provisional future block, update the block with the model's predictions, then repeat. Each iteration produces an intermediate state, and the eventual converged block provides the clean target state. The important training asset is therefore the full trajectory, not just the endpoint. Intermediate noisy states are paired with later clean states so the model can later learn to denoise its own imperfect future-token guesses.

This is also where the repo exposes greedy and nongreedy variants. The greedy path uses deterministic token updates during the Jacobi rollout, which makes the generated trajectories more standardized. The nongreedy path relaxes that assumption and produces trajectories with different exploratory behavior, which is useful for studying how supervision changes when the rollout is less deterministic. Both variants still serve the same purpose: convert real prompts into aligned noisy-to-clean training pairs for the noise-conditioned trainer in §2.4. The release used for reproduction does not require rerunning this expensive stage because the published dataset shards already contain its output.

> **Option A note:** This stage is not executed in the reproduction — the HF dataset shards ship its output.

**Code:**
- `generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.py` (+ `.sh` wrapper).
- `generate_trajectory/generation/qwen2_modeling_jacobi_forcing_greedy.py` — greedy Jacobi rollout on Qwen2.
- `generate_trajectory/generation/qwen2_modeling_jacobi_forcing_nongreedy_blk32.py` — nongreedy variant.

### §2.3 Training-sequence prep with progressive noise window

After trajectories exist, they still need to be reshaped into the exact tensor format consumed by training. The window-preparation script does that by deciding how many future tokens are predicted in parallel, how much noisy future context is shown around each position, and how that noise varies across examples. The first critical knob is `n_token_seq_length`: this is the block size the model is asked to predict in parallel, so it defines the basic granularity of Jacobi-style supervision. The second is `window_size`, which sets how much future context is exposed around each training position inside the noised window. A larger window gives the model broader future context, but it also changes the input format the trainer expects.

The noise schedule is controlled by `min_noisy_ratio` and `max_noisy_ratio`, which bound how corrupted the visible future block can be. With `strategy progressive`, the script does not lock the whole dataset to one fixed noise level. Instead, it ramps noise across training examples, exposing the model to an organized range from cleaner to noisier supervision. That design is central to the recipe, because the model needs to learn both near-converged cleanup and harder denoising cases. In reproduction, these are not free-form preprocessing options once you choose the released shards. The Hugging Face shard name, such as `n32w16`, encodes the block and window settings, and the produced columns must match the trainer contract exactly. If the shard recipe and trainer assumptions disagree, training still runs but you are no longer reproducing the intended method.

> **Option A note:** The HF shards already encode this step's output (columns `prompt_ids`, `complete_training_sequence_ids`, `traj_position_indices`).

**Code:**
- `generate_trajectory/data/2_prepare_efficient_cllm_training_data_progressive_noise_window.py`

**Key arguments:** `--n_token_seq_length`, `--window_size`, `--min_noisy_ratio`, `--max_noisy_ratio`, `--strategy progressive`.

### §2.4 Noise-conditioned training

The trainer turns those prepared noisy-window examples into a finetuned causal model that can clean up future-token guesses. Conceptually, each example contains a prompt prefix plus a noised view of the continuation, and the model is optimized to recover the clean continuation. The loss is still ordinary next-token cross-entropy, but it is applied over the positions corresponding to the noised region rather than treating the entire sequence as a standard left-to-right language-model target. That is the core Jacobi Forcing move: keep the causal architecture and token-level objective, but redefine the training distribution so the model becomes robust to imperfect future blocks.

There are several trainer implementations in the repo, reflecting different experiments and data formats, so the file choice matters. For the released windowed-progressive shards, the primary pairing is `soft_flexattn_cllm_trainer_multiblock_window.py`, called from `soft_flexattn_train_cllm_multiblock_window.py`, with `train_cllm.py` acting as the top-level wrapper that launches the recipe. This is the trainer path whose assumptions line up with the shard format described in §2.3. On the systems side, `ds_config.json` is the normal H200-oriented DeepSpeed configuration and assumes standard multi-GPU training without CPU offload. The repository also includes `ds_config_cpu_offloading.json` as a ZeRO-3 fallback for lower-VRAM environments. That fallback is useful for portability, but it is not part of the intended H200 reproduction path here.

**Code:**
- `JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock_window.py` — primary trainer for the windowed-progressive recipe.
- `JacobiForcing/train/soft_flexattn_train_cllm_multiblock_window.py` — train loop.
- `JacobiForcing/train/train_cllm.py` — top-level entry point.
- `JacobiForcing/scripts/train/ds_config.json` — DeepSpeed config for standard (no-offload) multi-GPU training.
- `JacobiForcing/scripts/train/ds_config_cpu_offloading.json` — ZeRO-3 CPU offload config for lower-VRAM setups (not needed on H200).

### §2.5 Inference with rejection recycling

At inference time, the trained model is used in Jacobi mode rather than plain token-by-token decoding. In ordinary Jacobi inference, the decoder proposes a future block, iteratively updates it, and commits whatever prefix has converged after each pass. Multi-block Rejection Recycling (MR) extends that idea by running several candidate blocks in flight at once instead of treating one block as the only active hypothesis. Concretely, MR launches `K` parallel blocks, checks them against a convergence threshold, rejects the blocks that are not progressing well enough, and immediately recycles the freed compute budget into fresh candidate blocks drawn from a pool. The result is a more opportunistic decoding process that tries to keep GPU work concentrated on blocks likely to yield committable tokens soon.

The aggressiveness of that filtering is controlled by the activation ratio `r`: higher or lower values change how readily the system discards underperforming blocks and replaces them. The user-facing tradeoff is therefore not just "Jacobi on or off" but how aggressively MR manages parallel block speculation. The repo exposes both plain and MR-oriented inference entry points. Plain HumanEval evaluation goes through the non-MR script, while MR HumanEval uses the dedicated recycling entry point. The inference knobs the user is expected to tune live here, not in the earlier data pipeline: block size `n` for inference, block count `K`, pool size, and activation ratio `r`. The defaults documented in this repo are `n=64`, `K=2`, pool size `4`, and `r=0.85`.

**Code:**
- `inference_engine/engine/jacobi_decoding.py` — greedy Jacobi decoding.
- `inference_engine/engine/jacobi_decoding_nongreedy.py` — nongreedy variant.
- `JacobiForcing/jacobi_forcing_inference_humaneval.py` — plain HumanEval inference entry.
- `JacobiForcing/jacobi_forcing_inference_MR_humaneval.py` — HumanEval with MR.

**Hyperparameters:** block size `n` (default 64 for inference), block count `K` (default 2), pool size (default 4), activation ratio `r` (default 0.85).

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

## 5. Quirks and gotchas for reproduction

These are the details that bit us (or would have) during reproduction. Each matters because it affects either which command you run, which numbers you compare against, or what you disclose in the methods section of a paper that cites this work as a baseline.

- **Nested `JacobiForcing/JacobiForcing/`.** The repo root `JacobiForcing/` contains a subdirectory *also* named `JacobiForcing/` that houses the training scripts, trainer classes, inference entry points, and DeepSpeed configs. Absolute paths in this doc and in `docs/training_plan.md` use the correct nested form (e.g. `JacobiForcing/JacobiForcing/scripts/train/...`).
- **README path error.** The upstream README shows `scripts/train/train_jacobi_forcing_coder_n32.sh`. The actual path after clone is `JacobiForcing/scripts/train/train_jacobi_forcing_coder_n32.sh`. Copying the README command verbatim from within the repo root will `No such file`.
- **Legacy math script name.** The math training script is `train_clean_context_conditioned_cllm_openthinker2_n64.sh`. The `openthinker2` naming is a carryover from an earlier method; this *is* the script to use with `OpenThoughts_Math_training_data_n64w32`. Don't assume unrelated.
- **Python 3.12 pin.** `requirements.txt` is pinned against Python 3.12. Newer versions (3.13+) are untested; older versions will fail to resolve some deps.
- **Unpinned HF revisions upstream.** Neither the base Qwen models nor the HF training datasets have pinned revisions in the upstream README. Our reproduction pins both at download time — see Stage 1 and Stage 2 in `docs/training_plan.md`.
- **flash-attention version drift.** Flash-attention, torch, and transformers version changes can shift training loss at the 4th decimal across otherwise-identical runs. Stage 0's `versions.lock` captures the installed set.
- **Training-data provenance (disclosable).** The HF training shards were rolled from `JacobiForcing_Coder_7B_v0` — an intermediate checkpoint, not the base Qwen and not the published `_v1`. A paper that uses this as a baseline should disclose that the reproduction re-used the authors' v0-generated trajectories rather than re-rolling from the base model.
