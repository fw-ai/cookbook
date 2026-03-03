#!/usr/bin/env python3
"""Download DeepMath-103K and convert to JSONL for rl_loop.

Output format per row:
  {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
   "ground_truth": "<final_answer>"}

Usage:
    python prepare_data.py
"""

import os
import json

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "showing your reasoning. Put your final answer inside \\boxed{}."
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "deepmath_103k.jsonl")


def main():
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    print(f"Loaded {len(ds)} rows from DeepMath-103K")

    count = 0
    with open(OUTPUT_PATH, "w") as f:
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

    print(f"Wrote {count} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
