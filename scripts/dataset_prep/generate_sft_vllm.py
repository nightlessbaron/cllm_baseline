#!/usr/bin/env python
"""
Fast SFT data generation: vLLM greedy batched decoding + correctness filter.

Reads the filtered OpenThoughts3 math parquets (produced by
filter_openthoughts3_math.py), generates a greedy response per prompt with
Qwen2.5-Math-7B-Instruct via vLLM, checks `\\boxed{...}` against gold via
math_verify, and writes only the correct (prompt, response) pairs as JSONL.

Output format (per line):
    {
      "messages": [
        {"role": "user", "content": "<problem>"},
        {"role": "assistant", "content": "<full response up to EOS>"}
      ],
      "problem": "...",
      "response": "...",
      "gold_answer": "...",
      "pred_answer": "...",
      "source": "...",
    }

Sharding: pass --shard_idx k --num_shards N for array-task parallelism; the
script picks rows k, k+N, k+2N, ...  Output filename includes the shard so
parallel processes do not collide.

Usage (single GPU, quick test):
    python generate_sft_vllm.py \\
        --input_dir  /mnt/weka/.../openthoughts3_math_boxed \\
        --output_dir /mnt/weka/.../openthoughts3_math_sft \\
        --shard_idx 0 --num_shards 1 --limit 100
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("ERROR: vllm not installed. Run `pip install vllm` first.", file=sys.stderr)
    sys.exit(1)

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False

BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def ts():
    return datetime.now().strftime("%H:%M:%S")


def extract_last_boxed(text):
    m = BOXED_RE.findall(text or "")
    return m[-1].strip() if m else ""


def normalise(s):
    s = (s or "").strip().strip("$")
    s = re.sub(r"\s+", "", s)
    return s.replace(",", "").replace("\\!", "")


def is_equivalent(pred, gold):
    if not pred or not gold:
        return False
    if normalise(pred) == normalise(gold):
        return True
    if HAS_MATH_VERIFY:
        try:
            if mv_verify(mv_parse(f"\\boxed{{{gold}}}"),
                         mv_parse(f"\\boxed{{{pred}}}")):
                return True
        except Exception:
            pass
    return False


def load_items(input_dir):
    parquets = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not parquets:
        raise FileNotFoundError(f"No parquet files in {input_dir}")
    items = []
    for p in parquets:
        t = pq.read_table(p, columns=["problem", "gold_answer", "source"])
        for pr, gd, src in zip(t["problem"].to_pylist(),
                               t["gold_answer"].to_pylist(),
                               t["source"].to_pylist()):
            if gd:
                items.append({"problem": pr, "gold_answer": gd, "source": src})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model",
                    default="/mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline/models/base-math")
    ap.add_argument("--max_total_len", type=int, default=4096,
                    help="prompt + response <= this (sets vLLM max_model_len)")
    ap.add_argument("--max_new_tokens", type=int, default=3584,
                    help="response cap; set so prompt_max + this <= max_total_len")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = Path(args.output_dir) / f"sft_shard_{args.shard_idx:04d}_of_{args.num_shards:04d}.jsonl"
    stats_path = out_path.with_suffix(".stats.json")

    if out_path.exists() and not args.overwrite:
        print(f"[{ts()}] Output already exists, skipping: {out_path}", flush=True)
        return

    items = load_items(args.input_dir)
    items = items[args.shard_idx::args.num_shards]
    if args.limit:
        items = items[:args.limit]
    n_total = len(items)
    print(f"[{ts()}] shard {args.shard_idx}/{args.num_shards}: {n_total} prompts "
          f"(math_verify={HAS_MATH_VERIFY})", flush=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_total_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": it["problem"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for it in items
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t_gen = time.time() - t0
    print(f"[{ts()}] shard {args.shard_idx}: generation done in {t_gen / 60:.1f} min "
          f"({t_gen / max(1, n_total):.2f} s/prompt)", flush=True)

    n_correct = n_no_boxed = n_mismatch = 0
    kept_lines = []
    for item, out in zip(items, outputs):
        response = out.outputs[0].text
        pred = extract_last_boxed(response)
        gold = item["gold_answer"]
        if not pred:
            n_no_boxed += 1
            continue
        if is_equivalent(pred, gold):
            n_correct += 1
            rec = {
                "messages": [
                    {"role": "user", "content": item["problem"]},
                    {"role": "assistant", "content": response},
                ],
                "problem": item["problem"],
                "response": response,
                "gold_answer": gold,
                "pred_answer": pred,
                "source": item["source"],
            }
            kept_lines.append(json.dumps(rec, ensure_ascii=False))
        else:
            n_mismatch += 1

    with open(out_path, "w") as f:
        for line in kept_lines:
            f.write(line + "\n")

    stats = {
        "shard": args.shard_idx,
        "num_shards": args.num_shards,
        "total": n_total,
        "correct": n_correct,
        "no_boxed": n_no_boxed,
        "mismatch": n_mismatch,
        "accept_rate": round(n_correct / max(1, n_total), 4),
        "generation_seconds": round(t_gen, 1),
        "generation_prompts_per_sec": round(n_total / max(1, t_gen), 2),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[{ts()}] shard {args.shard_idx}: kept {n_correct}/{n_total} "
          f"({100 * n_correct / max(1, n_total):.1f}%) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
