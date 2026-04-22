#!/usr/bin/env python
"""
Normalize a Phase-1 JacobiForcing checkpoint for Hugging Face release.

Mirrors `normalize_ckpt_keys` from
JacobiForcing/generate_trajectory/data/tool_merge_ds_ckpts.py but operates on
HF-saved safetensors shards instead of DeepSpeed ZeRO output. Strips the
`module.` (and `_fsdp_wrapped_module.`) prefix added by the DDP-wrapped
CllmTrainer, rewrites the shards + index, and keeps only the files needed
for inference (model shards, tokenizer, configs).

Usage:
    python normalize_ckpt.py --src <checkpoint_dir> --dst <output_dir>
"""
import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def normalize_key(k):
    changed = False
    while True:
        if k.startswith("module."):
            k = k[len("module."):]
            changed = True
            continue
        if k.startswith("_fsdp_wrapped_module."):
            k = k[len("_fsdp_wrapped_module."):]
            changed = True
            continue
        break
    return k, changed


KEEP_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "trainer_state.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--verify", action="store_true",
                    help="Load with Qwen2ForCausalLM.from_pretrained after rewrite.")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    idx = json.load(open(src / "model.safetensors.index.json"))
    old_map = idx["weight_map"]
    new_map = {}
    shards = sorted(set(old_map.values()))

    total_changed = 0
    for shard in shards:
        print(f"[rewrite] {shard}")
        tensors = load_file(str(src / shard))
        new_tensors = {}
        for k, v in tensors.items():
            nk, ch = normalize_key(k)
            total_changed += int(ch)
            new_tensors[nk] = v
            new_map[nk] = shard
        save_file(new_tensors, str(dst / shard), metadata={"format": "pt"})
    print(f"[normalize] stripped prefix on {total_changed} keys "
          f"across {len(shards)} shards")

    new_idx = {"metadata": idx.get("metadata", {}), "weight_map": new_map}
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(new_idx, f, indent=2)

    for fn in KEEP_FILES:
        sp = src / fn
        if sp.exists():
            shutil.copy2(sp, dst / fn)
            print(f"[copy] {fn}")

    if args.verify:
        import torch
        from transformers import Qwen2ForCausalLM
        print("[verify] loading via Qwen2ForCausalLM.from_pretrained ...")
        m = Qwen2ForCausalLM.from_pretrained(str(dst), torch_dtype=torch.bfloat16)
        print(f"[verify] ok, num_parameters={m.num_parameters():,}")


if __name__ == "__main__":
    main()
