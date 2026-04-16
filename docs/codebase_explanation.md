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
