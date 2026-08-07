#!/usr/bin/env python3
"""Download UltraFeedback preferences and convert to JSONL for DPO training.

Mirrors Tinker's UltraFeedback DPO builder (single-turn instruction +
chosen/rejected responses). Output format per row:

  {"chosen":   {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]},
   "rejected": {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}}

Usage:
    python prepare_data.py                      # writes data/preference_train.jsonl
    python prepare_data.py --max-rows 2000 --output /tmp/prefs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random

HF_DATASET = "argilla/ultrafeedback-binarized-preferences"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "preference_train.jsonl")

# Shuffle seed for the train file; deterministic so re-runs reproduce.
SHUFFLE_SEED = 0


def row_to_preference(example: dict) -> dict | None:
    """Convert one UltraFeedback row to the chosen/rejected messages schema.

    Returns None for rows with empty fields, which the writer skips.
    """
    instruction = (example.get("instruction") or "").strip()
    chosen = (example.get("chosen_response") or "").strip()
    rejected = (example.get("rejected_response") or "").strip()
    if not instruction or not chosen or not rejected or chosen == rejected:
        return None
    prompt = [{"role": "user", "content": instruction}]
    return {
        "chosen": {"messages": [*prompt, {"role": "assistant", "content": chosen}]},
        "rejected": {"messages": [*prompt, {"role": "assistant", "content": rejected}]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    indices = list(range(len(ds)))
    random.Random(SHUFFLE_SEED).shuffle(indices)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    written = 0
    with open(args.output, "w") as f:
        for i in indices:
            if written >= args.max_rows:
                break
            pref = row_to_preference(ds[i])
            if pref is None:
                continue
            f.write(json.dumps(pref, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
