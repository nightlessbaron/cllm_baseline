#!/usr/bin/env python
"""
Filter open-thoughts/OpenThoughts3-1.2M to math-only rows, extract the
gold \\boxed{...} answer from the gpt turn, and write out per-shard parquet.

Usage:
    # Pilot (single shard):
    python filter_openthoughts3_math.py --start 40 --end 41 \\
        --out /mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline/data/openthoughts3_math_pilot

    # Full math range (shards 25..109 inclusive => 85 shards, 850k rows):
    python filter_openthoughts3_math.py --start 25 --end 110 \\
        --out /mnt/weka/shrd/k2m/varad.pimpalkhute/cllm_baseline/data/openthoughts3_math

Then:
    python filter_openthoughts3_math.py --upload \\
        --out ... --repo_id melhoushi/OpenThoughts3_math --token ...

OpenThoughts3 math section fields (all from ai2-adapt-dev/openmath-2-math):
    difficulty (int64, often None)
    source      (str)
    domain      ("math")
    conversations (list[{from: human|gpt, value: str}])

Output schema (one parquet per shard):
    problem            str    -- the human turn's value
    gold_response_full str    -- the gpt turn's value (CoT + final answer)
    gold_answer        str    -- the last \\boxed{...} content, or "" when absent
    source             str
    shard_id           int
    row_id             int
"""
import argparse
import os
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, upload_folder

# Match \boxed{...} with one level of nested braces (common in \boxed{\frac{1}{2}}).
BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_last_boxed(text: str) -> str:
    matches = BOXED_RE.findall(text or "")
    return matches[-1].strip() if matches else ""


def process_shard(shard_id: int, out_dir: Path, boxed_only: bool = False) -> dict:
    fname = f"data/train-{shard_id:05d}-of-00120.parquet"
    local = hf_hub_download(
        repo_id="open-thoughts/OpenThoughts3-1.2M",
        repo_type="dataset",
        filename=fname,
    )
    table = pq.read_table(local)
    domains = table["domain"].to_pylist()
    if not all(d == "math" for d in domains):
        non_math = sum(1 for d in domains if d != "math")
        print(f"[shard {shard_id}] warning: {non_math}/{len(domains)} non-math rows (kept math only)")

    sources = table["source"].to_pylist()
    convs = table["conversations"].to_pylist()

    problems, golds_full, golds_ans, src_out, shard_out, row_out = [], [], [], [], [], []
    for i, (src, conv, dom) in enumerate(zip(sources, convs, domains)):
        if dom != "math":
            continue
        human = next((t["value"] for t in conv if t["from"] == "human"), None)
        gpt = next((t["value"] for t in conv if t["from"] == "gpt"), None)
        if not human or not gpt:
            continue
        boxed = extract_last_boxed(gpt)
        if boxed_only and not boxed:
            continue
        problems.append(human)
        golds_full.append(gpt)
        golds_ans.append(boxed)
        src_out.append(src)
        shard_out.append(shard_id)
        row_out.append(i)

    out_table = pa.table({
        "problem": problems,
        "gold_response_full": golds_full,
        "gold_answer": golds_ans,
        "source": src_out,
        "shard_id": shard_out,
        "row_id": row_out,
    })
    out_path = out_dir / f"math-{shard_id:05d}.parquet"
    pq.write_table(out_table, out_path, compression="snappy")

    n = out_table.num_rows
    n_boxed = sum(1 for x in golds_ans if x)
    return {
        "shard": shard_id,
        "rows": n,
        "with_boxed": n_boxed,
        "boxed_pct": round(100 * n_boxed / n, 2) if n else 0.0,
        "out_path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=25, help="inclusive")
    ap.add_argument("--end", type=int, default=110, help="exclusive")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--upload", action="store_true",
                    help="Skip processing; upload --out dir to --repo_id.")
    ap.add_argument("--repo_id", type=str, default=None)
    ap.add_argument("--token", type=str, default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--boxed_only", action="store_true",
                    help="Drop rows whose gpt turn has no \\\\boxed{...} answer.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.upload:
        total_rows = 0
        total_boxed = 0
        for shard_id in range(args.start, args.end):
            info = process_shard(shard_id, out_dir, boxed_only=args.boxed_only)
            print(f"[shard {info['shard']:3d}] rows={info['rows']} "
                  f"with_boxed={info['with_boxed']} ({info['boxed_pct']}%) -> {info['out_path']}")
            total_rows += info["rows"]
            total_boxed += info["with_boxed"]
        pct = round(100 * total_boxed / total_rows, 2) if total_rows else 0.0
        print(f"\n[total] rows={total_rows} with_boxed={total_boxed} ({pct}%) -> {out_dir}")
        return

    if not args.repo_id or not args.token:
        raise SystemExit("--upload requires --repo_id and --token")

    api = HfApi()
    try:
        api.repo_info(repo_id=args.repo_id, repo_type="dataset", token=args.token)
        print(f"Dataset repo '{args.repo_id}' exists.")
    except Exception:
        print(f"Creating dataset repo '{args.repo_id}' (private={args.private})...")
        api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                        token=args.token, private=args.private, exist_ok=True)

    print(f"Uploading {out_dir} ...")
    upload_folder(
        folder_path=str(out_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        token=args.token,
        commit_message="upload math-only filtered subset",
    )
    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
