# JacobiForcing Reproduction Notes

> **What this is.** A living log of every non-obvious issue we hit while reproducing the JacobiForcing (arXiv 2512.14681) checkpoints, and the fix for each. Complements `codebase_explanation.md` (method-level) and `training_plan.md` (stage-level runbook). Read this before starting a fresh attempt — each section saves hours of debugging.
>
> **Last updated:** 2026-04-17 during active Phase 1 training.

---

## Table of contents

1. [Environment + tooling](#1-environment--tooling)
2. [Dependency quirks (pip / CUDA / flash-attn)](#2-dependency-quirks-pip--cuda--flash-attn)
3. [Base model + dataset download pitfalls](#3-base-model--dataset-download-pitfalls)
4. [Paper ↔ released code discrepancies](#4-paper--released-code-discrepancies)
5. [Training code patches required](#5-training-code-patches-required)
6. [Data subsetting / sampling decisions](#6-data-subsetting--sampling-decisions)
7. [Operational lessons](#7-operational-lessons)
8. [Appendix A consolidated deviations](#8-appendix-a-consolidated-deviations)
9. [What's still in progress](#9-whats-still-in-progress)

---

## 1. Environment + tooling

### 1.1 Hardware + allocation

- Target: 8× NVIDIA H200 (141 GB HBM3e each), CUDA 12.8, driver ≥ 570.
- Allocate an exclusive SLURM node before doing CUDA-heavy work (compiles, training). The login shell may be shared and will slow flash-attn kernel compilation dramatically.
  ```bash
  srun --partition=main --time=15-10:00:00 --nodes=1 --exclusive --gres=gpu:8 --pty bash
  ```
- Long-running training should be submitted as `sbatch`, not `nohup` inside a login shell — session refreshes / orphaned wrapper shells can kill nohup'd jobs. sbatch detaches cleanly.

### 1.2 Conda environment

- Python **3.12** is required (upstream's `requirements.txt` pins wheels to cp312 tags).
- We put determinism env vars in `env.sh` at the repo root instead of modifying `~/.bashrc`. That keeps them version-controlled and scoped per run:
  ```bash
  # env.sh
  export REPRO_ROOT=/mnt/weka/home/varad.pimpalkhute/diffusion
  export REPO=$REPRO_ROOT/JacobiForcing
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONHASHSEED=0
  export NCCL_DEBUG=WARN
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate jacobi_forcing
  ```

### 1.3 Codex / sandboxed subagent limitations

- Codex's sandbox (when used via `codex:codex-rescue`) defaults to **workspace-write inside cwd, read-only outside**. That blocks:
  - `conda create -n …` → can't write to `~/anaconda3/envs/`
  - `pip install` → can't write to conda env's `site-packages/`
  - `git commit` → `.git/index.lock` is read-only
- Workflow: delegate file edits + in-workspace reads to Codex, but **run install/commit from the main session**. For commits after Codex edits, stage and commit from the user shell.

---

## 2. Dependency quirks (pip / CUDA / flash-attn)

### 2.1 `torch==2.7.1+cu128` is not on vanilla PyPI

Upstream's `requirements.txt` pins `torch==2.7.1+cu128`. This is a PEP 440 *local version identifier*; PyTorch hosts cu-suffixed wheels only at their own index, not on pypi.org (and not on mirrors like TUNA).

**Symptom:** `ERROR: No matching distribution found for torch==2.7.1+cu128` after pip downloaded ~90 other packages.

**Fix:** add PyTorch's index to pip install:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
```

### 2.2 flash-attn 2.8.3 prebuilt wheel has broken torch ABI

The PyPI prebuilt `flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` was compiled against a different torch version than our pinned `torch==2.7.1+cu128`. Runtime import fails with:

```
ImportError: .../flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so:
undefined symbol: _ZN3c104cuda9SetDeviceEab
```

The missing symbol is `c10::cuda::SetDevice(DeviceIndex, bool)` — a torch-internal ABI change between versions.

**Fix options (either works):**

**(A) Source-rebuild from scratch against our torch:**
```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 \
pip install flash-attn==2.8.3 \
    --no-build-isolation --no-deps --force-reinstall \
    --no-binary flash_attn --no-cache-dir
```
- `--no-binary flash_attn` forbids prebuilt wheels (crucial — without this pip reuses the cached broken wheel)
- `--no-build-isolation` uses our installed torch during compile instead of pip's isolated build env
- `--no-deps --force-reinstall` rebuilds just this package without reinstalling torch
- `--no-cache-dir` + `pip cache remove flash_attn` first if you've got the bad wheel cached
- ~15–30 min compile time on 8 cores.

**(B) Prebuilt wheel from Dao-AILab GitHub releases** — much faster if you pick the right ABI tag:
```bash
python -c "import torch; print(torch.compiled_with_cxx11_abi())"   # True or False
pip install --no-deps --force-reinstall \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.7cxx11abi<TRUE|FALSE>-cp312-cp312-linux_x86_64.whl
```

Verification (should show no ImportError):
```python
import torch, flash_attn
from flash_attn import flash_attn_func
x = torch.randn(1, 8, 64, 128, device='cuda', dtype=torch.bfloat16)
print(flash_attn_func(x, x, x).shape)
```

### 2.3 `hf` CLI vs `huggingface-cli`

- Hugging Face is migrating from `huggingface-cli` to `hf` (newer versions of `huggingface_hub` ship both).
- Pinned `huggingface-hub==0.33.2` in upstream's requirements does **not** ship the `hf` entry point (it arrives in a later version).
- Use `huggingface-cli` (which still works and emits a deprecation warning).

---

## 3. Base model + dataset download pitfalls

### 3.1 Incomplete model shards look "done"

`huggingface-cli download … --local-dir <dir>` can exit **before finalizing a .incomplete staging file**, leaving the model silently missing one shard.

- Our Qwen2.5-Coder-7B-Instruct download appeared successful (15 GB on disk) but `model-00001-of-00004.safetensors` was missing. The `.cache/huggingface/download/<hash>.incomplete` staging file sat at 99.99% (4,877,608,061 / 4,877,660,776 bytes).
- Training crashed with `FileNotFoundError: .../model-00001-of-00004.safetensors`.

**Fix:** after download, verify all shards are present, not just total directory size:
```bash
ls models/base-coder/*.safetensors | wc -l   # should be 4 for Qwen 7B
```
If one's missing, re-download just that file:
```bash
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
    --include "model-00001-of-00004.safetensors" \
    --local-dir $REPRO_ROOT/models/base-coder
```

### 3.2 HF revision pinning after `--local-dir` download

`huggingface-cli scan-cache` doesn't record downloads done with `--local-dir` (it only tracks the `~/.cache/huggingface/hub/` cache). To pin revisions for reproducibility, use `HfApi` to query the hub:

```python
from huggingface_hub import HfApi
api = HfApi()
print(api.model_info('Qwen/Qwen2.5-Coder-7B-Instruct').sha)
print(api.dataset_info('JacobiForcing/OpenCodeInstruct_training_data_n32w16').sha)
```

### 3.3 Different dataset filename conventions per HF repo

Upstream's two Coder datasets ship with wildly different filename schemes:
- `OpenCodeInstruct_training_data_n32w16` — files like `merged-traj-data-oct-16-n32w16.jsonl`
- `OpenCodeInstruct_training_data_n16w16` — files like `merged_data_v2_8_30_opencodeinstruct_progressive_noise_cyclic_all.jsonl`

Don't assume file names. `ls <local-dir>` the dataset after download.

### 3.4 HF dataset viewer row count is misleading

The HF dataset viewer transcodes JSONL/CSV to parquet for preview, and its row count can reflect the *total across all files* in the repo — not the single file your training script loads.

Example: the `n32w16` Coder dataset viewer reports **2.16M rows**. But that's `n32w16.jsonl` (1.38M) **+** `n32w32.jsonl` (780k, a bonus variant in the same repo). The file you'll train on has 1.38M rows.

**Lesson:** once downloaded, run `wc -l *.jsonl` (or `datasets.load_dataset` row count) on the *actual* file to get the real count.

### 3.5 Dataset format differs Coder vs Math

- Coder datasets: **JSONL** (`load_dataset("json", …)`)
- Math datasets: **parquet** (`load_dataset("parquet", …)`)

Upstream's trainer code has `load_dataset("json", data_files=…)` hardcoded. For Math, either patch the trainer to accept parquet or pre-slice the parquet into a JSONL file. We did the latter (see §6.2).

### 3.6 Parquet shards are bucket-sorted by length

Math's 7-shard parquet (`train-00000-of-00007.parquet` → `train-00006-of-00007.parquet`) is **ordered by length bucket**:
- Shard 0: bucket_0000 → bucket_0007 (shortest sequences)
- Shard 6: bucket_0042 → bucket_0054 (longest)

Taking "the first N rows" of Math data = **only shortest sequences** → badly biased training. Use uniform random sampling with a fixed seed (§6.2).

---

## 4. Paper ↔ released code discrepancies

The paper's Section 4.1 (Evaluation Settings) describes a recipe that does **not match** the released training shell script. Trusting the released script blindly will produce different numbers than the paper reports.

### 4.1 Hyperparameters

| Parameter | Paper (arXiv 2512.14681) | Released `train_jacobi_forcing_coder_n32.sh` |
| --- | --- | --- |
| Learning rate | 1e-6 | 1e-5 (**10× higher**) |
| Max seq length | 2048 | 16384 (**8× longer**) |
| Step count | 10k per phase | `num_train_epochs=1` (~344k steps on n32w16) |
| Effective batch | 4 (4 GPUs × bs=1 × grad_accum=1) | Same — script uses 4 GPUs |

### 4.2 Two-phase curriculum (not encoded in any shell)

The paper describes a **two-round curriculum**:
> "initial block size at 16, window size at 16, and train for 10k steps, and a second round of training with block size at 32, window size at 8, and train for another 10k steps."

This is **completely absent** from the released `scripts/train/` folder — there's only a single n32 script. To reproduce the paper, you have to write your own Phase 1 (n=16, w=16) and Phase 2 scripts, then chain them with Phase 2 initialized from Phase 1's final checkpoint.

### 4.3 Missing intermediate / "v0" artifacts

- The released `train_jacobi_forcing_coder_n32.sh` has `model_path="/data/numa0/train-tests/models/progressive_noise_cllm2_mask_1m_steps"` — an intermediate checkpoint the authors never released.
- The Coder dataset README links `JacobiForcing_Coder_7B_v0` as the trajectory generator — this repo **404s** on HuggingFace (verified with authenticated `HfApi.list_models(author='JacobiForcing')` which returns only `_v1` models).

**Impact:** we cannot reproduce the exact 2-stage training chain upstream ran (their Stage A bootstrap is unpublished). Our reproduction starts from the public base Qwen2.5-{Coder,Math}-7B-Instruct instead. Expected consequence per the paper's reported error bars: sub-percentage pass@1 delta, likely on the low side.

### 4.4 The Phase-2 "n=32 w=8" data variant doesn't exist

The paper specifies `n=32, w=8` for Phase 2. HuggingFace has only:
- Coder: `n16w16`, `n32w16`, `n32w32` (no `n32w8`)
- Math: `n16w16`, `n64w32` (no `n32w8`, no `n32w16` either!)

**We substitute:**
- Coder Phase 2: `n32w16` (closest available)
- Math Phase 2: `n64w32` (only other option for Math)

### 4.5 Noise-schedule label mismatch

Paper says "linear progressive noise schedule". Dataset files for n16w16 are named `..._progressive_noise_cyclic_all.jsonl` — suggests **cyclic** progressive schedule, not linear. The shipped data is what it is; nothing we can change. Document in paper methods that our schedule matches the shipped data, not the paper's description.

---

## 5. Training code patches required

Upstream's `JacobiForcing/train/soft_flexattn_train_cllm_multiblock.py` has three breakages against our pinned dependencies. All three patches are in our `appendix_A_trainer_patch.diff`:

### 5.1 Dual-Accelerator-with-DS conflict

**Problem:** The script manually creates `Accelerator(deepspeed_plugin=ds_plugin)` (line 162 original). HF Trainer then internally creates its own Accelerator during `super().__init__()`. With accelerate ≥ 1.6, two Accelerators with DS plugins is explicitly blocked:

```
NotImplementedError: You cannot pass in a `deepspeed_plugin` when creating a
second `Accelerator`. Please make sure the first `Accelerator` is initialized
with all the plugins you want to use.
```

**Fix:** Drop DeepSpeed entirely — H200 has plenty of VRAM, plain DDP is mathematically identical to ZeRO-3 no-offload:
```python
training_args.deepspeed = None   # prevent HF Trainer from DS-initializing

accelerator = Accelerator(
    gradient_accumulation_steps=training_args.gradient_accumulation_steps or 1,
    mixed_precision="bf16" if training_args.bf16 else ...,
    log_with=training_args.report_to if training_args.report_to else None,
    # no deepspeed_plugin=...
)

# Also comment out: dschf = HfDeepSpeedConfig(training_args.deepspeed)
```

### 5.2 Hardcoded broken DS config path

The fallback `training_args.deepspeed = training_args.deepspeed or "scripts/ds_config_cpu_offloading.json"` references a relative path that doesn't resolve from any reasonable cwd (actual path is `scripts/train/ds_config_cpu_offloading.json`). Setting `training_args.deepspeed = None` (per §5.1) makes this moot.

### 5.3 Gradient checkpointing double-enable

**Problem:** User script calls `model.gradient_checkpointing_enable()` on the raw model (line 219) **before** `accelerator.prepare()` wraps it in DDP. HF Trainer then *also* tries to enable it in `_inner_training_loop` (transformers 4.53's `trainer.py:2331`) — but the model is now DDP-wrapped and doesn't expose that method:

```
AttributeError: 'DistributedDataParallel' object has no attribute
'gradient_checkpointing_enable'
```

**Fix:** flip the flag after manual enable so HF Trainer doesn't re-attempt:
```python
if getattr(training_args, "gradient_checkpointing", False):
    model.gradient_checkpointing_enable()
    training_args.gradient_checkpointing = False   # prevent HF Trainer re-enable
```

### 5.4 bf16 NaN on Math — dual guard required

**Problem (Math-specific):** Training Math from vanilla `Qwen2.5-Math-7B-Instruct` in bf16 on `n16w16` data produces non-finite losses and/or gradients at unpredictable steps. We first hit it at step ~530 (LR 5.88e-7, ~60% of peak 1e-6). Without a guard, one bad sample produces NaN gradients → optimizer applies NaN updates → weights corrupted → all subsequent losses read as 0.0 with grad_norm=NaN indefinitely.

**Why Math and not Coder:** Qwen2.5-Math has sharper output distributions (math answers are often deterministic tokens like `\boxed{D}`) than Qwen2.5-Coder. Combined with bf16's 7-mantissa precision and the Jacobi-forcing task's noise-conditioned OOD inputs at noised positions, occasional logit spikes overflow bf16. Upstream likely didn't hit this because they trained from an intermediate (unreleased) checkpoint that had already adapted to Jacobi-noise inputs; starting from vanilla base is more fragile.

**A surprise we found the hard way:** the NaN is produced in the **backward pass, not forward**. A forward-only guard (check `total_loss` is finite before backward) fails silently — it never triggers, because the forward loss is always finite. The backward produces NaN gradients from some numerically-delicate activation. clip_grad_norm_ on NaN returns NaN, optimizer applies NaN, weights NaN'd.

**Further surprise:** the NaN is **non-deterministic across identical runs.** Same code, same data, same seed, same node: run 1 hit NaN at step 530 loss=7.89, run 3 didn't hit NaN at all at step 530 loss=7.84. Root cause: GPU reduction kernels use atomic adds whose ordering depends on thread scheduling — 4th-decimal jitter most steps, but at the bf16 overflow boundary this jitter decides NaN-vs-finite. **This makes "it worked last run" not a guarantee; the guard must always be there.**

**Fix — dual guard in `CllmTrainer._one_pass_losses_step`:**

```python
# --- Guard A: forward produced a non-finite loss ---
if not torch.isfinite(total_loss):
    if self.args.local_rank == 0:
        self._nan_skip_count = getattr(self, "_nan_skip_count", 0) + 1
        wandb.log({
            "nan_skip_count": self._nan_skip_count,
            "nan_skip_reason": "loss_nonfinite",
            "nan_skip_loss_ar_finite": bool(torch.isfinite(loss_ar)),
            "nan_skip_loss_consistency_finite": bool(torch.isfinite(loss_consistency)),
        })
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    return torch.zeros_like(total_loss).detach()

with self.accelerator.accumulate(model):
    self.accelerator.backward(total_loss)

# --- Guard B: backward produced non-finite gradients ---
any_nan_grad = False
for p in model.parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        any_nan_grad = True
        break
if any_nan_grad:
    if self.args.local_rank == 0:
        self._nan_skip_count = getattr(self, "_nan_skip_count", 0) + 1
        wandb.log({
            "nan_skip_count": self._nan_skip_count,
            "nan_skip_reason": "grad_nonfinite",
            "nan_skip_step": self.train_step_cnt,
        })
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    # loss itself was finite → no need to return zero, real loss is fine for logging
```

**Verification pattern:** if only the forward-only guard is present, the failure mode is subtle — `loss=<finite value>, grad_norm=0.0` at the failing step, then `loss=0.0, grad_norm=0.0` for every subsequent step. The loss being 0.0 (rather than NaN) in the log is a red flag that the trainer kept running on corrupted weights. Dual guard keeps losses in the normal band (~5-8) and keeps `nan_skip_count` on wandb for tracking how often it fires.

**Why this isn't in the released upstream code:** Math-finetune literature routinely includes NaN-skip guards. Upstream probably had them internally but stripped them before release, or simply didn't need them because their training started from a Jacobi-pre-adapted intermediate. For any reproduction from vanilla Qwen2.5-Math, this patch is mandatory.

---

## 6. Data subsetting / sampling decisions

### 6.1 Coder Phase 1: upstream shipped the right subset

The `JacobiForcing/OpenCodeInstruct_training_data_n16w16` dataset ships **4 files**:
- `merged_data_v2_8_30_..._all.jsonl` (474k rows, full)
- `100k_samples_....jsonl`
- **`40k_samples_....jsonl`** ← use this for Phase 1
- `10k_samples_....jsonl`

The 40k file is a pre-computed **non-contiguous random sample** across the full dataset (we verified — 40k unique ids scattered at positions 26…474057 in the main file, spanning buckets 0018–0065). Upstream clearly prepared these precisely because `10k_steps × bs=4 = 40k_samples` exactly matches one epoch over a 40k subset. Using the 40k file gives the cleanest paper-match for Coder Phase 1.

Length check: only 32 rows / 0.08% exceed 2048 tokens — upstream effectively pre-filtered.

### 6.2 Math Phase 1: we had to build our own subset

The `JacobiForcing/OpenThoughts_Math_training_data_n16w16` dataset ships **only one format** (7-shard parquet, 250k rows total). No pre-carved `40k_samples` equivalent.

**Two problems:**
1. 10.4% of rows exceed paper's 2048 max_seq_len (26,162 of 250,619) — median 1091, max 6822.
2. Parquet shards are bucket-sorted by length (§3.6) — "first 40k" = only shortest sequences.

**Solution** — construct an unbiased, length-filtered 40k subset:
```python
from datasets import load_dataset
ds = load_dataset('parquet', data_files='…/Math_n16w16/data/*.parquet', split='train')
filtered = ds.filter(lambda x: len(x['complete_training_sequence_ids']) <= 2048)
sub = filtered.shuffle(seed=42).select(range(40000))
sub.to_json('.../40k_samples_seed42_maxlen2048.jsonl', lines=True)
```

Result: 40k rows drawn from 224k ≤2048 rows, covering all 54 buckets (310–1040 rows per bucket). Seed 42 for reproducibility.

### 6.3 Why "first N rows" is never right on bucketed data

Bucket-sorted or length-ordered parquet datasets are extremely common in LLM training (speeds up padding-efficient batching). "First N rows" ≡ "only the shortest / easiest samples" → biases training toward one length class.

Always verify ordering by sampling `data_id` from start, middle, end of the file before slicing:
```python
ds = load_dataset('parquet', data_files='…', split='train')
for i in [0, len(ds)//2, len(ds)-1]:
    print(ds[i]['data_id'], ds[i]['prompt_ids_len'])
```

---

## 7. Operational lessons

### 7.1 Long-running jobs need sbatch

Claude Code's `run_in_background` Bash tool stores PIDs in `/tmp/claude-*/tasks/` — these can be **orphaned by session refreshes** (we lost the first pip install this way). `nohup … & disown` reparents to init (PPID=1) and survives, but only until the node itself reboots or the user logs out.

For multi-hour training, always use `sbatch`:
```bash
#SBATCH --job-name=jf-coder-phase1
#SBATCH --partition=main
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --output=/…/runs/…/slurm-%j.out
#SBATCH --error=/…/runs/…/slurm-%j.err

source /…/diffusion/env.sh
bash /…/scripts/train/train_coder_phase1_n16w16.sh
```

### 7.2 wandb online for monitoring

Just unset `WANDB_MODE=offline` and ensure `wandb login` has been run once. Script must have:
```bash
export WANDB_PROJECT=cllm2_training          # project namespace
export WANDB_RUN_NAME="coder-phase1-…"       # useful run name
# Leave WANDB_MODE unset -> online.
report_to="wandb"                            # in the torchrun args
```

For offline-capture + later-push workflow, keep `WANDB_MODE=offline` and run `wandb sync <wandb/offline-run-*>` after training.

### 7.3 Avoid `torchrun` port collisions when running multiple jobs

If two torchrun invocations ever share a node, their `--master_port` values must differ. We set:
- Coder: `--master_port 10000 --rdzv_id 101 --rdzv_endpoint localhost:5666`
- Math: `--master_port 20000 --rdzv_id 102 --rdzv_endpoint localhost:5667`

### 7.4 Revision hashes are your reproducibility backbone

For every downloaded artifact, record the HF commit SHA **before training starts** into plain-text files the paper's methods section can cite:

```
upstream_commit.txt        → 028dbd6283fdc6fa356f0b109ce04c5ca538e49f
base_model_revisions.txt   → coder c03e6d3582…, math ef9926d75a…
dataset_revisions.txt      → coder-n32w16 319070367b…, math-n64w32 679d883031…,
                             n16w16 3f492cef21…, math-n16w16 0e60ee8db7…
versions.lock              → (pip freeze output, 108 packages)
```

### 7.5 Monitor's shell-glob gotcha

`grep -c "pattern" file` exits **1** when zero matches exist (not an error — just the count-is-zero signal). If you use it as a stop-on-failure gate in a bash script, you'll falsely think your file has placeholders when it doesn't. Wrap with `|| true` or check the printed count directly:
```bash
n=$(grep -cE "TBD|TODO" file || true)
test "$n" = "0"
```

---

## 8. Appendix A consolidated deviations

From our running `docs/training_plan.md` Appendix A — every change we've made to upstream's recipe, with reasons:

| # | File / setting | Change | Reason | Where |
|---|---|---|---|---|
| 1 | `env.sh` (new file) | Replaces `~/.bashrc` determinism edits | Version-controlled, scoped per-run | Stage 0 |
| 2 | `pip install` | Added `--extra-index-url https://download.pytorch.org/whl/cu128` | `torch==2.7.1+cu128` only on PyTorch index | Stage 1 |
| 3 | Coder row-count expectation | Plan said ~2.16M; actual n32w16 is **1,376,368** | HF viewer combined n32w16 + n32w32 counts | Stage 2 |
| 4 | `huggingface-cli` (not `hf`) | Use `huggingface-cli download` | `hf` not shipped in `huggingface-hub==0.33.2` | Stage 2 |
| 5 | Coder `model_path=` | Upstream intermediate → `$REPRO_ROOT/models/base-coder` | `progressive_noise_cllm2_mask_1m_steps` not released (404) | Stage 3 |
| 6 | flash-attn install | Source-rebuild (or GitHub-release wheel) | PyPI prebuilt ABI-incompatible with torch 2.7.1+cu128 | Stage 2.5 |
| 7 | `--deepspeed` flag / DS plugin | **Dropped DS entirely** — plain DDP | `training_args.deepspeed=None` + removed manual `deepspeed_plugin`. Two Accelerators with DS plugins blocked by accelerate 1.8.1. H200 VRAM doesn't need ZeRO-3 offload. Math is identical to DS no-offload. | Stage 2.5 |
| 8 | `gradient_checkpointing` | Flipped to False after manual enable on raw model | Prevents HF Trainer from re-enabling on DDP-wrapped model | Stage 2.5 |
| 9 | Paper-vs-script hyperparams | LR 1e-5→1e-6, max_len 16384→2048, epochs=1→max_steps=10000 | Match paper, not released shell script | Stage 3 Phase 1 |
| 10 | Two-phase training (not in any released shell) | Wrote `train_{coder,math}_phase1_n16w16.sh`; Phase 2 scripts pending | Paper specifies 2-round curriculum the released script doesn't encode | Stage 3 |
| 11 | Coder Phase 2 data (paper n=32 w=8) | Substituted n32w16 | w=8 variant never released | Stage 3 Phase 2 |
| 12 | Math Phase 2 data (paper n=32 w=8) | Substituted n64w32 | w=8 not released; n64w32 is closest Math option | Stage 3 Phase 2 |
| 13 | Math Phase 1 data subset | Constructed locally: filter ≤2048, then uniform random 40k at seed=42 | Upstream shipped no `40k_samples` file for Math; full parquet is bucket-sorted by length + 10.4% overlength | Stage 3 Phase 1 |
| 14 | `save_steps` | 5000 → 1000 | More granular checkpointing on a 10k-step run (doesn't change weights/gradients) | Stage 3 |
| 15 | `CllmTrainer._one_pass_losses_step` | Added dual NaN guard (pre-backward loss check + post-backward gradient check) | Math bf16 training from vanilla Qwen2.5-Math-7B-Instruct produces sporadic NaN gradients (§5.4). Without the guard, one bad step corrupts all weights permanently. Backward-NaN is non-deterministic across runs — guard must always be present. | Stage 3 Math Phase 1 |

All diffs archived at `$REPRO_ROOT/appendix_A_*.diff` / `appendix_A_*.sh`.

---

## 9. Results & progress

### 9.1 Phase 1 completion (Coder + Math)

Both Phase 1 runs completed cleanly:

| Run | SLURM job | Duration | Final mean loss | Last step loss | Final LR |
| --- | --- | --- | --- | --- | --- |
| Coder Phase 1 (n16w16) | 1561965 | 3h 56m (14,158.76 s) | 5.334 | 4.98 | 7.78e-7 |
| Math Phase 1 (n16w16) | 1562006 | 4h 01m (14,455.75 s) | 5.166 | 5.18 | 7.78e-7 |

Both trained for exactly 10,000 optimizer steps (1 full epoch over 40k samples at bs=4), matching the paper's Phase 1 recipe. Math converges slightly faster than Coder (lower mean loss) thanks to longer sequences and denser supervision per sample.

Final checkpoints (top-level `pytorch_model-0000{1..4}-of-00004.bin` + tokenizer + generation config) are in:
- `runs/coder-phase1-n16w16/` and `runs/coder-phase1-n16w16/checkpoint-10000/`
- `runs/math-phase1-n16w16/` and `runs/math-phase1-n16w16/checkpoint-10000/`

Periodic checkpoints from steps 6000, 7000, 8000, 9000 are retained (`save_total_limit=5`); earlier ones were pruned.

**NaN-guard triggers during Math Phase 1:** 0 (non-determinism worked in our favor this run). The dual guard stays in the trainer patch as a safety net — a rerun of the same data could easily hit NaN again.

### 9.2 Dataset length statistics (what we measured)

These numbers informed the Phase 1 subset construction (§6.1–§6.2):

| Dataset | Total rows | Min | Median | p95 | p99 | Max | % > 2048 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Coder 40k_samples (shipped) | 40,000 | 206 | 682 | 1,175 | 1,506 | 4,550 | **0.08%** (32 rows) |
| Math n16w16 full 250k (parquet) | 250,619 | 76 | 1,091 | 4,369 | 5,584 | 6,822 | **10.44%** (26,162 rows) |
| Math n16w16 40k (ours, seed=42, filtered) | 40,000 | 98 | — | — | — | 2,048 | 0.00% (enforced) |

Observations:
- Upstream's Coder `40k_samples_...jsonl` is effectively pre-filtered for paper's max_seq_len=2048.
- Upstream shipped **no length-filtered Math subset** — we had to build one. Uniform random from the ≤2048 subset at seed=42 preserved bucket distribution (54 distinct buckets, 310–1040 rows per bucket).
- The Math max of 6,822 tokens (reasoning chains with `\boxed{...}` + LaTeX) is 1.5× Coder's max — Math has a fundamentally longer tail.

### 9.3 Still in progress

- [ ] **Coder Phase 2** — resume from `runs/coder-phase1-n16w16/checkpoint-10000/`, 10k more steps on `n32w16` (paper specifies `n32w8`, unreleased — closest available).
- [ ] **Math Phase 2** — resume from `runs/math-phase1-n16w16/checkpoint-10000/`, 10k more steps on `n64w32` (only other released Math option).
- [ ] **Coder Phase 2 / Math Phase 2 data subsets** — both datasets are full parquet/JSONL without shipped 40k-samples files; build our own using the same recipe as §6.2 (filter ≤2048, uniform random 40k at seed=42).
- [ ] **Eval: Coder HumanEval** — plain + MR (n=64, K=2, pool=4, r=0.85). Compare to paper's 83.5% / 4.0×.
- [ ] **Eval: Math MATH500** — ditto. Fill paper's reported number at report-time.
- [ ] **Report: `Stage 7` table fill-in** in `training_plan.md`.

### 9.4 Out of scope (documented in `training_plan.md` §7)

- MBPP (Coder) and GSM8K (Math) evals — require `evalchemy` harness, not in this reproduction.
- Hyperparameter search / ablations — paper's settings are frozen.
- Multinode training — not needed at paper's effective batch=4.
