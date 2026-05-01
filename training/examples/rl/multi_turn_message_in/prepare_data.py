#!/usr/bin/env python3
"""Download GSM8K from HuggingFace and convert to JSONL for the recipe.

Pulls the real ``openai/gsm8k`` dataset (config ``main``) — both ``train`` and
``test`` splits in one call — and writes ``train.jsonl`` and ``test.jsonl``
next to this script (or wherever ``--output-dir`` points).

Output format per row::

    {"id": "gsm8k-train-0",
     "messages": [{"role": "user",
                   "content": "<question>\\nPlease put your final answer within \\boxed{}."}],
     "answer": "<full GSM8K answer string ending in '#### N'>"}

Mirrors AReaL's ``examples/multi_turn_math/`` data shape so the reward function
can verify ``\\boxed{...}`` against the GSM8K ground-truth ``#### N`` token.

Usage::

    python prepare_data.py                       # writes train.jsonl + test.jsonl
    python prepare_data.py --split train         # only train
    python prepare_data.py --max-rows 100        # cap each split
"""

from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset

PROMPT_SUFFIX = "\nPlease put your final answer within \\boxed{}."

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))


def _write_split(split_name: str, split_data, out_path: str, max_rows: int | None) -> int:
    n = 0
    with open(out_path, "w") as f:
        for idx, row in enumerate(split_data):
            if max_rows is not None and n >= max_rows:
                break
            entry = {
                "id": f"gsm8k-{split_name}-{idx}",
                "messages": [
                    {"role": "user", "content": row["question"] + PROMPT_SUFFIX},
                ],
                "answer": row["answer"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Prepare GSM8K JSONL")
    parser.add_argument("--split", default="all", choices=["all", "train", "test"],
                        help="Which split to write (default: both)")
    parser.add_argument("--output-dir", default=DEFAULT_DIR,
                        help="Directory for {split}.jsonl files")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Optional cap on rows per split")
    args = parser.parse_args()

    ds = load_dataset("openai/gsm8k", "main")
    print(f"Loaded openai/gsm8k(main): "
          f"train={len(ds['train'])} rows, test={len(ds['test'])} rows")

    splits = ["train", "test"] if args.split == "all" else [args.split]
    os.makedirs(args.output_dir, exist_ok=True)
    for s in splits:
        out_path = os.path.join(args.output_dir, f"{s}.jsonl")
        n = _write_split(s, ds[s], out_path, args.max_rows)
        print(f"Wrote {n} rows to {out_path}")


if __name__ == "__main__":
    main()
