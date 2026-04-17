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

- [Stage 0 — Environment](#stage-0-environment)
- [Stage 1 — Source and base models](#stage-1-source-and-base-models)
- [Stage 2 — Option A data download](#stage-2-option-a-data-download)
- [Stage 2.5 — Smoke test](#stage-25-smoke-test)
- [Stage 3 — Coder training (7B, n32w16)](#stage-3-coder-training-7b-n32w16)
- [Stage 4 — Math training (7B, n64w32)](#stage-4-math-training-7b-n64w32)
- [Stage 5 — HumanEval on Coder checkpoint](#stage-5-humaneval-on-coder-checkpoint)
- [Stage 6 — MATH500 on Math checkpoint](#stage-6-math500-on-math-checkpoint)
- [Stage 7 — Reporting](#stage-7-reporting)
- [Appendix A — Deviations from upstream recipe](#appendix-a-deviations-from-upstream-recipe)
- [Appendix B — Failure-mode runbook](#appendix-b-failure-mode-runbook)

## Conventions

- `$REPRO_ROOT` = `/mnt/weka/home/varad.pimpalkhute/diffusion` (this plan's working directory).
- `$REPO` = `$REPRO_ROOT/JacobiForcing` (created in Stage 1).
- All `bash` blocks are copy-pasteable. Each stage ends with a **Reproducibility checklist** (what to record) and **Expected artifacts** (what the stage produces).

For conceptual background on Jacobi Forcing (why it exists, what each code file does), read `docs/codebase_explanation.md` first.

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

# HF stores a commit hash inside the snapshot dir — capture it via scan-cache.
huggingface-cli scan-cache | grep -E "Qwen2.5-(Coder|Math)-7B-Instruct" \
    > $REPRO_ROOT/base_model_revisions.txt
cat $REPRO_ROOT/base_model_revisions.txt
```

> **Note on HF revision capture.** If `huggingface-cli scan-cache` output does not include the commit hash column you need, re-download with `--revision main` pinned and record the resolved hash from the CLI output. Either way, the 40-char SHA must be saved before training starts.

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
import os
from datasets import load_dataset
root = os.environ['REPRO_ROOT']
for name, approx in [
    ('OpenCodeInstruct_training_data_n32w16', 2_160_000),
    ('OpenThoughts_Math_training_data_n64w32', 230_000),
]:
    p = f"{root}/data/{name}"
    ds = load_dataset('parquet', data_files=f"{p}/**/*.parquet", split='train')
    print(f"{name}: {len(ds):,} rows (expected ~{approx:,})")
EOF
```

Expected:
- `OpenCodeInstruct_training_data_n32w16: ~2,160,000 rows`
- `OpenThoughts_Math_training_data_n64w32: ~230,000 rows`

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
import os
from datasets import load_dataset
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

# Apply the override edits (read the script, identify its dataset arg and
# training-args env vars, then edit smoke_coder_n32.sh):
#   - point dataset path at $REPRO_ROOT/data/coder_smoke_100
#   - add `--max_steps 10 --save_steps 5 --logging_steps 1`
#   - set output_dir to $REPRO_ROOT/tmp_smoke
# These edits are script-specific; see Stage 3 for the full arg surface.

# 3. Run on all 8 GPUs (single-GPU requires DeepSpeed config changes, which
#    add variance — safer to keep the 8-GPU topology).
mkdir -p $REPRO_ROOT/tmp_smoke
cd $REPO
bash JacobiForcing/scripts/train/smoke_coder_n32.sh \
    2>&1 | tee $REPRO_ROOT/tmp_smoke/smoke.log
```

### Pass criteria

- Exit code 0.
- `$REPRO_ROOT/tmp_smoke/checkpoint-5/` exists and contains `pytorch_model*` or `model.safetensors`.
- Loss at step 10 < loss at step 1 (by any amount):

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
- `$REPRO_ROOT/runs/coder-n32/checkpoint-FINAL/` (final Coder checkpoint)
- `$REPRO_ROOT/appendix_A_coder_script.diff`

## Stage 4 — Math training (7B, n64w32)

### Script used

`$REPO/JacobiForcing/scripts/train/train_clean_context_conditioned_cllm_openthinker2_n64.sh` — pairs with `OpenThoughts_Math_training_data_n64w32`. The `openthinker2` in the filename is a legacy naming carryover; this *is* the current math training script. Same DeepSpeed config as Stage 3 (`JacobiForcing/scripts/train/ds_config.json`).

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
- `$REPRO_ROOT/runs/math-n64/checkpoint-FINAL/` (final Math checkpoint)
- `$REPRO_ROOT/appendix_A_math_script.diff`

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
export CODER_CKPT=$REPRO_ROOT/runs/coder-n32/checkpoint-FINAL    # replace FINAL with your final checkpoint step or symlink
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
export MATH_CKPT=$REPRO_ROOT/runs/math-n64/checkpoint-FINAL    # replace FINAL with your final checkpoint step or symlink
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
| `env.sh` (new file, `$REPRO_ROOT`)                                | Created in place of modifying `~/.bashrc`   | Keeps determinism env vars version-controlled alongside the docs; no pollution of user shell state | (Stage 0) |
| `pip install -r requirements.txt` invocation                      | Added `--extra-index-url https://download.pytorch.org/whl/cu128` | `torch==2.7.1+cu128` is a PEP 440 local-version wheel hosted only at PyTorch's own index, not on vanilla PyPI / the TUNA mirror. Without this flag, pip resolution fails with `No matching distribution found for torch==2.7.1+cu128`. Upstream's README does not document this index URL requirement. | (Stage 1) |
| Stage 2 Coder row-count expectation                               | Corrected from ~2,160,000 to **1,376,368** | HF viewer reported 2.16M because it was summing n32w16.jsonl + n32w32.jsonl (the dataset ships both window-size variants in one repo). Only `merged-traj-data-oct-16-n32w16.jsonl` (1.38M rows) pairs with `train_jacobi_forcing_coder_n32.sh`. Dataset format is JSONL (not parquet as the viewer suggests — it transcodes for preview). | (Stage 2) |
| `huggingface-cli` used instead of `hf`                            | No behavioral change                        | The `hf` CLI entry point is not shipped in `huggingface-hub==0.33.2` (it arrives in a later version). `huggingface-cli download` remains fully supported and is what's present in the pinned env. | (Stage 2) |
| Math Phase 1 data subset                                          | Constructed locally: filter `complete_training_sequence_ids <= 2048`, uniform random 40k at seed=42 | The paper's "max 2048 seq len" + "10k steps @ bs=4" means training sees 40k samples, all ≤2048 tokens. Upstream shipped a pre-carved `40k_samples` file for **Coder** (median 682, max 4550, only 0.08% over 2048 — effectively pre-filtered). For **Math**, upstream shipped only the full 250k parquet which is bucket-sorted by length AND has 10.4% of rows above 2048 tokens. Naive "first 40k" would select only the shortest-bucket rows (heavy length bias). Uniform random 40k from the length-filtered 224k subset at fixed seed matches the intent of upstream's Coder protocol while being deterministic and reproducible. File: `data/OpenThoughts_Math_training_data_n16w16/40k_samples_seed42_maxlen2048.jsonl`. | (Stage 3 Math Phase 1) |
| `train_jacobi_forcing_coder_n32.sh` `model_path=`                 | Upstream value `/data/numa0/train-tests/models/progressive_noise_cllm2_mask_1m_steps` (unreleased intermediate) → `$REPRO_ROOT/models/base-coder` (Qwen2.5-Coder-7B-Instruct) | Upstream's recipe is two-stage: (A) base Qwen → intermediate progressive-noise checkpoint, (B) this script trains intermediate → final. The Stage A output is not published (authenticated `list_models(author='JacobiForcing')` returns only `_v1`; `_v0` and `progressive_noise_cllm2_mask_1m_steps` both 404). We initialize from base Qwen and run a single-stage fine-tune with the released n32w16 trajectory shards. Tokenizer is identical so shard data is compatible. Expected consequence: our pass@1 numbers may land slightly below the paper's two-stage results, typically sub-percent — document this explicitly in the paper's methods section. | (Stage 3) |
| `flash-attn==2.8.3` prebuilt PyPI wheel                           | Replaced with wheel from Dao-AILab/flash-attention GitHub releases (or source-rebuild against our torch) | The prebuilt PyPI wheel for `flash-attn==2.8.3` was compiled against a different torch version and fails at runtime with `undefined symbol: _ZN3c104cuda9SetDeviceEab` (ABI incompat with torch 2.7.1+cu128). Correct install: use `FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation --no-deps --force-reinstall --no-binary flash_attn --no-cache-dir`, or download the matching wheel from the Dao-AILab releases page (`flash_attn-2.8.3+cu128torch2.7cxx11abi<TRUE|FALSE>-cp312...`). | (Stage 2.5) |
| Training script `--deepspeed` flag                                | Added explicit `--deepspeed <absolute-path>/JacobiForcing/scripts/train/ds_config.json`; upstream default was implicit and broken | The trainer code hardcodes a *relative* fallback path (`scripts/ds_config_cpu_offloading.json`) that doesn't resolve from any reasonable cwd in the cloned repo (the actual config files live at `scripts/train/`, not `scripts/`). Passing `--deepspeed` with an absolute path bypasses the broken default. Switched from `ds_config_cpu_offloading.json` to `ds_config.json` (no CPU offload) because H200's 141 GB VRAM makes offload unnecessary — training math is identical (same ZeRO-3 sharding), only wall-clock differs. | (Stage 2.5) |
| `JacobiForcing/scripts/train/train_jacobi_forcing_coder_n32.sh`   | dataset / output-dir paths                  | point at local `$REPRO_ROOT` paths              | _<fill>_   |
| `JacobiForcing/scripts/train/train_clean_context_conditioned_cllm_openthinker2_n64.sh` | dataset / base-model / output-dir paths | same, for Math | _<fill>_   |
| _add rows as you go_ | _<fill>_ | _<fill>_ | _<fill>_ |

Store the raw diffs alongside this table at `$REPRO_ROOT/appendix_A_*.diff`.

## Appendix B — Failure-mode runbook

| Symptom                                      | Likely cause                                                         | Fix                                                                                      |
| -------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `torch.cuda.OutOfMemoryError` during training| Unexpected memory — a kernel or activation is larger than budgeted   | Enable gradient checkpointing in the trainer flag (`--gradient_checkpointing`); if still OOM, switch DeepSpeed config to `JacobiForcing/scripts/train/ds_config_cpu_offloading.json` and record in Appendix A. |
| NaN loss within first 50 steps               | Flash-attention / torch mismatch; numerically unstable kernel        | Reinstall flash-attn pinned to the version in `versions.lock`. Rerun smoke test.         |
| Loss diverges after warmup                   | Wrong dataset path (wrong `n/w` shard loaded) or mis-edited LR       | Verify `dataset_revisions.txt` matches the expected commit; `git diff` the script against upstream — any non-path edit is the cause. |
| DeepSpeed hangs at init, no NCCL traffic     | Interconnect / NCCL init misconfiguration                            | Export `NCCL_DEBUG=INFO`, rerun; check IFNAME and `NCCL_SOCKET_IFNAME`. For single-node, ensure no stale `NCCL_ASYNC_ERROR_HANDLING` override. |
| `ImportError: flash_attn`                    | flash-attention wheel not built against installed torch              | Reinstall flash-attn from prebuilt wheels matching torch version in `versions.lock`.     |
| HuggingFace 429 / rate-limit on download     | Dataset/model download throttled                                     | Wait and retry; pass `--max-workers 4` to `huggingface-cli download`.                    |
| Evaluation JSON shows 0 pass@1 with no errors| Wrong checkpoint path (loading base model by mistake)                | Confirm `$CODER_CKPT` points at the final training checkpoint, not `base-coder`.         |
