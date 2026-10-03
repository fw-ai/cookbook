#!/usr/bin/env python3
"""Download DeepMath-103K and convert to JSONL for rl_loop.

Output format per row:
  {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
   "ground_truth": "<final_answer>"}

Usage from the cookbook directory:
    python -m training.examples.rl.deepmath.prepare_data --output ./deepmath_103k.jsonl
"""

import argparse
import os
import json
from pathlib import Path

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "showing your reasoning. Put your final answer inside \\boxed{}."
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "dataset.jsonl")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(OUTPUT_PATH),
        help="Destination JSONL file (default: deepmath/dataset.jsonl)",
    )
    args = parser.parse_args(argv)
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    print(f"Loaded {len(ds)} rows from DeepMath-103K")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in ds:
            entry = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                ],
                "ground_truth": row["final_answer"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
