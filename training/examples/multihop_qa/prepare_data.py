#!/usr/bin/env python3
"""Download HotpotQA and convert to JSONL for multi-hop QA IGPO training.

Output format per row:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}],
   "ground_truth": "<answer>",
   "context": {"titles": [...], "paragraphs": [["sent1", ...], ...]}}

Usage:
    pip install datasets
    python prepare_data.py [--max-rows 500] [--split validation]
"""

import argparse
import json
import os

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question by searching for "
    "relevant information. You have access to two tools:\n"
    "- search(query): Search for information about a topic. Returns "
    "relevant paragraphs.\n"
    "- submit_answer(answer): Submit your final answer. Use this once you "
    "are confident.\n\n"
    "Search as many times as needed, then submit your answer."
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "dataset.jsonl")


def format_row(row: dict) -> dict:
    """Convert a single HotpotQA dataset row to the JSONL training format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ],
        "ground_truth": row["answer"],
        "context": {
            "titles": row["context"]["title"],
            "paragraphs": row["context"]["sentences"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare HotpotQA dataset for IGPO training")
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    ds = load_dataset("hotpot_qa", "distractor", split=args.split)
    print(f"Loaded {len(ds)} rows from HotpotQA ({args.split})")

    count = 0
    with open(args.output, "w") as f:
        for row in ds:
            if count >= args.max_rows:
                break
            entry = format_row(row)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
