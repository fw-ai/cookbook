#!/usr/bin/env python3
"""Write a tiny arithmetic dataset for the remote-rollout GRPO example."""

from __future__ import annotations

import json
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "train.jsonl")

SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem and put the final "
    "numeric answer in \\boxed{}."
)

ROWS = [
    ("remote-grpo-1", "What is 17 + 25?", "42"),
    ("remote-grpo-2", "What is 9 * 8?", "72"),
    ("remote-grpo-3", "What is 144 / 12?", "12"),
    ("remote-grpo-4", "What is 31 - 14?", "17"),
]


def main() -> None:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        for row_id, question, answer in ROWS:
            row = {
                "id": row_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "answer": answer,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(ROWS)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
