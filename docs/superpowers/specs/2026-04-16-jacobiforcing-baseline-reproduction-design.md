# JacobiForcing Baseline Reproduction — Documentation Design

**Date:** 2026-04-16
**Working directory:** `/mnt/weka/home/varad.pimpalkhute/diffusion/`
**Upstream:** [hao-ai-lab/JacobiForcing](https://github.com/hao-ai-lab/JacobiForcing), arXiv 2512.14681
**Target hardware:** single-node 8× H200 (multinode available but not default)

---

## 1. Goal

Produce two documentation artifacts that let the user reproduce the JacobiForcing Coder 7B and Math 7B checkpoints from upstream, with enough rigor that the results can be cited as a baseline in the user's own research paper.

The two artifacts are:

1. **`docs/codebase_explanation.md`** — a method-first walkthrough of the JacobiForcing repository. Explains what Jacobi Forcing is as a technique and maps each stage of the method to the code that implements it.
2. **`docs/training_plan.md`** — a stage-oriented reproduction runbook. Takes an empty environment on the target hardware and produces both checkpoints plus their in-repo evaluation numbers (HumanEval for Coder, MATH500 for Math).

This spec captures the design of those two documents. The documents themselves are produced in the implementation phase that follows.

## 2. Why this design

The user's purpose is **baseline reproduction for a paper**, not personal onboarding or a team handoff. That reframes every design choice:

- Doc 1 is concept-first (not directory-first) because the user must be able to *explain* the method in their paper's Related Work, not just navigate the repo.
- Doc 2 is stage-oriented (not linear) because when a reproduction diverges from upstream, the user needs to say *"we diverged at stage X because of Y"* — stage boundaries make that easy.
- Every stage records provenance (HF commit hashes, base-model revisions, installed library versions). Upstream does not pin these; the user must, otherwise reviewers will question comparability.

## 3. Dataset choice (Option A, both variants)

The user selected **Option A** from the upstream README: download HuggingFace-hosted preprocessed training shards rather than generating Jacobi trajectories locally. This applies to both variants:

| Variant | Base model                              | Preprocessed dataset                                       | Notes                         |
| ------- | --------------------------------------- | ---------------------------------------------------------- | ----------------------------- |
| Coder   | `Qwen/Qwen2.5-Coder-7B-Instruct`        | `JacobiForcing/OpenCodeInstruct_training_data_n32w16`      | n=32, w=16, 2.16M samples     |
| Math    | `Qwen/Qwen2.5-Math-7B-Instruct`         | `JacobiForcing/OpenThoughts_Math_training_data_n64w32`     | n=64, w=32, 230k samples      |

Option A removes data-generation variance from the reproduction. Only training-run variance remains.

## 4. Doc 1 — `codebase_explanation.md` outline

Concept-first structure with code pointers in each section.

1. **What Jacobi Forcing is** — AR-to-diffusion mismatch, Jacobi fixed-point iteration, noise-conditioned causal training, the meaning of "causal parallel decoder."
2. **The five method stages** — each subsection gives *what it does*, *key hyperparameters*, *where it lives in code*:
   1. **Data bucketing** — `generate_trajectory/data/0_bucketing_opencodeinstruct.py`, `0_bucketing_openthought2.py`.
   2. **Trajectory generation** — `generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.py` and its shell wrapper; Qwen2 Jacobi rollout models at `generate_trajectory/generation/qwen2_modeling_jacobi_forcing_{greedy,nongreedy_blk32}.py`. *Under Option A this stage is not run, but the doc must explain it so the reader understands what went into the preprocessed HuggingFace shards.*
   3. **Training-sequence prep with progressive noise window** — `generate_trajectory/data/2_prepare_efficient_cllm_training_data_progressive_noise_window.py` (explain `n_token_seq_length`, `window_size`, `min_noisy_ratio`, `max_noisy_ratio`, `strategy=progressive`).
   4. **Noise-conditioned training** — `JacobiForcing/train/soft_flexattn_cllm_trainer{_multiblock,_multiblock_window}.py`, `train_cllm.py`, `soft_flexattn_train_cllm*.py`. DeepSpeed configs at `JacobiForcing/scripts/train/ds_config{,_cpu_offloading}.json`.
   5. **Inference with rejection recycling** — `inference_engine/engine/jacobi_decoding{,_nongreedy,_nongreedy_on_policy}.py`; entrypoints `JacobiForcing/jacobi_forcing_inference_humaneval.py` and `jacobi_forcing_inference_MR_humaneval.py`. Hyperparameters: block size `n`, block count `K`, pool size, activation ratio `r`.
3. **Model-code layer** — `modeling/cllm2_qwen2_modeling_kv_terminate_on_eos_improved.py` (standard KV + EOS termination) and `..._multiblock_lookahead_unified.py` (multiblock lookahead, used with MR inference). When each is used.
4. **Repo map (one-screen table)** — every top-level directory with a one-line purpose. Lookup reference.
5. **Quirks and gotchas for reproduction** — nested `JacobiForcing/JacobiForcing/` path; upstream README's incorrect script path (`scripts/train/...` vs. actual `JacobiForcing/scripts/train/...`); `openthinker2` naming carryover in the math training script; Python 3.12 pin; HF revision pinning advice.

## 5. Doc 2 — `training_plan.md` outline

Stage-oriented. Each stage carries: **prereqs · exact commands · expected artifacts · reproducibility checklist · known failure modes**.

- **Stage 0 — Environment.** Conda env (`python=3.12`), `pip install -r requirements.txt`, CUDA/driver version check, `git lfs install`, HF auth (`huggingface-cli login`), deterministic-mode env vars (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, CUDA deterministic flags). Capture `pip freeze` into `versions.lock`.
- **Stage 1 — Source and base models.** Clone repo into `diffusion/JacobiForcing/`. Record repo commit hash. Download `Qwen/Qwen2.5-Coder-7B-Instruct` and `Qwen/Qwen2.5-Math-7B-Instruct`; record both HF revision hashes.
- **Stage 2 — Option A data download.** `git lfs clone` both preprocessed datasets; record HF commit hashes.
- **Stage 2.5 — Smoke test.** 10-step training run on a 100-sample slice of the Coder dataset to verify the pipeline end-to-end before committing the full run. Validates: data loader, DeepSpeed launch, checkpoint write, loss decreases.
- **Stage 3 — Coder training (7B, n32w16).** `JacobiForcing/JacobiForcing/scripts/train/train_jacobi_forcing_coder_n32.sh` on 8× H200 with `ds_config.json`. Record every hparam from the script (LR, schedule, epochs, micro-bs, grad-accum, seed), NCCL settings, wandb/tensorboard run IDs. Artifact: final checkpoint + loss curve.
- **Stage 4 — Math training (7B, n64w32).** `train_clean_context_conditioned_cllm_openthinker2_n64.sh` with math dataset and `Qwen2.5-Math-7B-Instruct` base. Any required edits (paths, base-model arg, dataset arg) are recorded as a diff in Appendix A.
- **Stage 5 — HumanEval on Coder checkpoint.** Run both `jacobi_forcing_inference_humaneval.py` (plain Jacobi) and `jacobi_forcing_inference_MR_humaneval.py` (with rejection recycling, hparams n=64, K=2, pool=4, r=0.85) plus `ar_inference_baseline.py` for the AR-speedup denominator. Artifact: pass@1 + tokens/forward + wall-clock speedup.
- **Stage 6 — MATH500 on Math checkpoint.** `jacobi_forcing_inference_MATH500.py`. Artifact: accuracy + speedup.
- **Stage 7 — Reporting.** Comparison table: our Coder run vs. the paper's HumanEval number (83.5% / 4.0×), our Math run vs. the paper's MATH500 number. Note that GSM8K 91.4% / 3.7× and MBPP are *not* reproduced here — they need `evalchemy` and are flagged as follow-up. Tolerance threshold is left to the user at reporting time.

**Appendix A — Deviations from upstream recipe.** Running ledger of any edits (path fixes, arg corrections) required to make the upstream scripts run. Feeds the paper's methods section.

**Appendix B — Failure-mode runbook.** OOM → grad-checkpointing flag. Diverging loss → check data hash, LR, seed. NaN → check flash-attn version. DeepSpeed hangs → NCCL_DEBUG=INFO, check interconnect.

## 6. Success criteria

1. **Doc 1 acceptance** — a reader unfamiliar with JacobiForcing can, after reading `codebase_explanation.md`, point to where each of the five method stages lives in code and explain what each stage does.
2. **Doc 2 acceptance** — a reader following `training_plan.md` on an 8× H200 node produces a Coder checkpoint, a Math checkpoint, a HumanEval score, and a MATH500 score without consulting the upstream README.
3. **Reproducibility evidence** — HF commit hashes for both datasets and both base models are recorded before training starts; `versions.lock` captured in Stage 0; final loss curves and seed-configured training logs archived with checkpoints.

## 7. Out of scope

- MBPP and GSM8K evaluation — requires external `evalchemy` harness; follow-up doc.
- Hyperparameter search / ablations — this is a reproduction, hparams are frozen to upstream.
- Code changes beyond minimal path/arg fixes — those fixes get logged in Appendix A, nothing else is modified.
- Multinode training — default is single-node 8× H200; multinode appears only as an optional sidebar.
- Producing the JacobiForcing clone during brainstorm/spec phase — cloning is Stage 1 of the training plan itself.

## 8. Known risks

- **Upstream README path errors.** The `scripts/train/…` path in the README is actually `JacobiForcing/scripts/train/…` after cloning. Plan must use correct paths.
- **Math script legacy naming.** `train_clean_context_conditioned_cllm_openthinker2_n64.sh` uses a prior method's naming; args may not match the current data layout. Appendix A captures fixes.
- **Unpinned upstream versions.** HF datasets and base models are not pinned in the upstream README; our docs pin them at download time via commit hashes.
- **Library-version drift.** Flash-attention / torch / transformers versions can shift loss at the 4th decimal. `versions.lock` captures installed versions in Stage 0.
- **Single-node assumption.** Reproducibility depends on topology matching upstream as closely as possible. Multinode introduces NCCL/allreduce ordering variables that can shift curves.

## 9. File layout and versioning

```
diffusion/
├── docs/
│   ├── codebase_explanation.md                 (Doc 1 — produced in implementation)
│   ├── training_plan.md                        (Doc 2 — produced in implementation)
│   └── superpowers/
│       └── specs/
│           └── 2026-04-16-jacobiforcing-baseline-reproduction-design.md  (this file)
└── JacobiForcing/                              (created in Stage 1 of the training plan, not now)
```

`diffusion/` is initialized as a git repo. This spec is committed on creation.
