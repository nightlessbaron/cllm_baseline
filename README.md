# cllm_baseline

Reproduction of **Jacobi Forcing** (Hu et al., arXiv 2512.14681) as a baseline for our paper.

Upstream: [hao-ai-lab/JacobiForcing](https://github.com/hao-ai-lab/JacobiForcing), pinned at commit `028dbd6283fdc6fa356f0b109ce04c5ca538e49f` (included as a submodule at `upstream/`).

## What's in here

This repo is a **reproduction artifact** — the scripts, patches, provenance, and notes needed to run a paper-faithful fine-tune of JacobiForcing's Coder-7B and Math-7B baselines on 8× NVIDIA H200. It does **not** ship model weights, training data, or checkpoints (those are downloaded on-demand from HuggingFace; paths pinned in `docs/reproduction_notes.md` §7.4).

Structure:

```
cllm_baseline/
├── upstream/                        # git submodule: hao-ai-lab/JacobiForcing @ 028dbd6...
├── docs/
│   ├── codebase_explanation.md      # Method-level walkthrough of the JacobiForcing repo
│   ├── training_plan.md             # Stage-oriented reproduction runbook (Stage 0 → eval)
│   └── reproduction_notes.md        # Living log of every non-obvious issue + fix
├── appendix_A_trainer_patch.diff    # Our edits to upstream trainer (DS removal + grad-ckpt + NaN guard)
├── appendix_A_cllm_trainer_multiblock_patched.py  # Full patched trainer file (for reference / copy-paste)
├── appendix_A_coder_phase1_script.sh              # Our Phase 1 shell script (Coder)
├── appendix_A_math_phase1_script.sh               # Our Phase 1 shell script (Math)
├── sbatch_coder_phase1.sh           # SLURM submit wrapper (Coder Phase 1)
├── sbatch_math_phase1.sh            # SLURM submit wrapper (Math Phase 1)
├── env.sh                           # Conda activation + determinism env vars
└── .gitignore
```

## Quick reproduction

Expected hardware: 1× node with 8× H200 (141 GB each). Adjust `CUDA_VISIBLE_DEVICES` / sbatch if your topology differs. The paper's effective batch size is 4, so only 4 GPUs are actively used; the other 4 can be idle (exclusive allocation) or run a parallel Math job.

### 1. Clone with submodule

```bash
git clone --recursive https://github.com/LLM360/cllm_baseline.git
cd cllm_baseline
```

If you already cloned without `--recursive`:
```bash
git submodule update --init
```

### 2. Apply our patches to the upstream trainer

The upstream trainer has three breakages against the library versions pinned in `requirements.txt` (detailed in `docs/reproduction_notes.md` §5). All fixes are in `appendix_A_trainer_patch.diff`:

- DeepSpeed removed — plain DDP suffices on H200 and avoids the "second Accelerator with DS plugin" error in `accelerate ≥1.6`
- Gradient-checkpointing flag cleared after manual enable (prevents HF Trainer from re-attempting on a DDP-wrapped model)
- **Dual NaN guard** in `CllmTrainer._one_pass_losses_step` — guards both forward loss AND post-backward gradients. **Required for Math training** (bf16 + sharp Qwen2.5-Math output distributions → sporadic NaN gradients).

Apply:

```bash
cd upstream
git apply ../appendix_A_trainer_patch.diff
cd ..
```

Verify:
```bash
cd upstream
git diff --stat
# expect: JacobiForcing/train/soft_flexattn_train_cllm_multiblock.py changed
#         JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock.py changed
cd ..
```

If the patch fails to apply cleanly (e.g., you're using a different upstream commit than the pinned one), fall back to copy-pasting `appendix_A_cllm_trainer_multiblock_patched.py` over `upstream/JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock.py` — it contains the full patched file as of our reproduction.

### 3. Environment + dependencies

See `docs/training_plan.md` Stage 0–1 for full details. Quick version:

```bash
# Create conda env
conda create -n jacobi_forcing python=3.12 -y
source env.sh

# Install upstream deps (note the --extra-index-url for torch+cu128)
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r upstream/requirements.txt

# flash-attn may need a source rebuild if the PyPI wheel's ABI doesn't match torch 2.7.1+cu128:
# FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 \
# pip install flash-attn==2.8.3 --no-build-isolation --no-deps \
#     --force-reinstall --no-binary flash_attn --no-cache-dir
# See docs/reproduction_notes.md §2.2 for details.
```

### 4. Download models + data

Pinned revision hashes (record these — they're the paper-cite-able provenance):

| Artifact | HF repo | Pinned SHA |
| --- | --- | --- |
| Upstream code | `hao-ai-lab/JacobiForcing` | `028dbd6283fdc6fa356f0b109ce04c5ca538e49f` |
| Base model: Coder | `Qwen/Qwen2.5-Coder-7B-Instruct` | `c03e6d358207e414f1eca0bb1891e29f1db0e242` |
| Base model: Math | `Qwen/Qwen2.5-Math-7B-Instruct` | `ef9926d75ab1d54532f6a30dd5e760355eb9aa4d` |
| Coder data (Phase 1) | `JacobiForcing/OpenCodeInstruct_training_data_n16w16` | `3f492cef21fc0b7b31a914934188f5c803af5227` |
| Coder data (Phase 2) | `JacobiForcing/OpenCodeInstruct_training_data_n32w16` | `319070367bd30aff42a469c994770c6470b5544a` |
| Math data (Phase 1)  | `JacobiForcing/OpenThoughts_Math_training_data_n16w16` | `0e60ee8db7671e4299002797f7eec4572a66d33f` |
| Math data (Phase 2)  | `JacobiForcing/OpenThoughts_Math_training_data_n64w32` | `679d8830314d2b25bedb141437a040792dab198a` |

Download with `huggingface-cli download <repo> --local-dir models/... ` (see `docs/training_plan.md` Stage 1–2 for full commands). Verify all safetensors shards land complete — see §3.1 of reproduction notes for a trap we hit.

### 5. Build data subsets

The paper trains 10k steps × batch size 4 = 40k samples per phase. For **Coder Phase 1**, upstream ships a `40k_samples_...jsonl` file that's already filtered and random-sampled. For **Math** and **Phase 2** for both variants, we build our own 40k subsets — filter `len ≤ 2048` and uniform random sample at `seed=42`:

```python
from datasets import load_dataset
ds = load_dataset('parquet', data_files='data/<dataset>/data/*.parquet', split='train')
filtered = ds.filter(lambda x: len(x['complete_training_sequence_ids']) <= 2048)
sub = filtered.shuffle(seed=42).select(range(40000))
sub.to_json('data/<dataset>/40k_samples_seed42_maxlen2048.jsonl', lines=True)
```

See `docs/reproduction_notes.md` §6 for rationale (why "first N rows" is wrong on bucket-sorted data).

### 6. Launch training

```bash
# Coder Phase 1 (~4 hours on 4× H200)
sbatch sbatch_coder_phase1.sh

# Math Phase 1 (~4 hours, parallel to Coder on a separate node)
sbatch sbatch_math_phase1.sh
```

Monitor via `squeue -u $USER` and your wandb project (`cllm2_training` — run names `coder-phase1-n16w16-paperhparams` / `math-phase1-n16w16-paperhparams`).

### 7. Phase 2 + eval

Phase 2 scripts and eval (HumanEval + MATH500) are TBD — see `docs/reproduction_notes.md` §9.3. Phase 2 resumes from each Phase 1's `checkpoint-10000/` with a different block/window recipe.

## Results so far

See `docs/reproduction_notes.md` §9.1 for the Phase 1 completion table.

| Phase | Steps | Wall-clock | Final mean loss | Last-step loss |
| --- | --- | --- | --- | --- |
| Coder Phase 1 (n16w16) | 10,000 | 3 h 56 m | 5.334 | 4.98 |
| Math Phase 1 (n16w16)  | 10,000 | 4 h 01 m | 5.166 | 5.18 |

## Deviations from upstream

Every change we made is logged in `docs/training_plan.md` Appendix A (15 entries). The big ones:

1. **Paper vs released-script hyperparameters** — the released `train_jacobi_forcing_coder_n32.sh` uses LR=1e-5, max_len=16384, num_train_epochs=1; the paper specifies LR=1e-6, max_len=2048, max_steps=10000 per phase. We follow the paper.
2. **Two-phase curriculum** — described in the paper but not encoded in any released shell. We wrote our own phase scripts.
3. **Starting model** — the released script starts from an unreleased intermediate checkpoint (`progressive_noise_cllm2_mask_1m_steps`, 404 on HF). We initialize from base `Qwen2.5-Coder-7B-Instruct` / `Qwen2.5-Math-7B-Instruct`. Expected consequence: sub-percent pass@1 delta vs the paper's two-stage.
4. **Trainer patches** — DS removal, gradient-checkpointing flag, dual NaN guard. All in `appendix_A_trainer_patch.diff`.

## Citation

If you use this reproduction, please cite the original paper:

```bibtex
@article{hu2025jacobiforcing,
  title={Jacobi Forcing: Fast and Accurate Causal Parallel Decoding},
  author={Hu, Lanxiang and others},
  journal={arXiv preprint arXiv:2512.14681},
  year={2025}
}
```

And link back to this reproduction: `https://github.com/LLM360/cllm_baseline`.

## License

Follows the upstream Apache 2.0 license (see `upstream/LICENSE` after submodule init).
