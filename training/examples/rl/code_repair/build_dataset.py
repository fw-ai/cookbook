#!/usr/bin/env python3
"""Keep the training candidates GRPO can learn from and write ``dataset.jsonl``.

Reads ``train_candidates.jsonl`` and the base model's per-row samples
(``sampled_rows.jsonl`` from ``train_code_repair_serverless --sampling-only``).
A row is kept when its sampled rewards differ (a zero-spread group gives no
gradient and the loop would drop it anyway) and the base model is not already
exact on every sample. Output: the eval rows first, then the kept rows, so
``--eval-rows`` holds out exactly ``eval.jsonl``.

    python -m training.examples.rl.code_repair.build_dataset \\
        --sampled-rows .../runs/baseline_train/sampled_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from training.examples.rl.code_repair.prepare_data import DATA_DIR


def keep(sampled: dict) -> bool:
    rewards = sampled["rewards"]
    return len(rewards) >= 2 and max(rewards) > min(rewards) and sampled["exact"] < sampled["samples"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sampled-rows", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=DATA_DIR / "train_candidates.jsonl")
    parser.add_argument("--eval", type=Path, default=DATA_DIR / "eval.jsonl")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "dataset.jsonl")
    args = parser.parse_args()

    sampled = {r["id"]: r for r in map(json.loads, args.sampled_rows.open()) if r.get("id")}
    candidates = [json.loads(line) for line in args.candidates.open() if line.strip()]
    eval_rows = [json.loads(line) for line in args.eval.open() if line.strip()]
    missing = [row["id"] for row in candidates if row["id"] not in sampled]
    if missing:
        raise SystemExit(f"{len(missing)} candidates have no samples, e.g. {missing[:3]}")

    kept = [row for row in candidates if keep(sampled[row["id"]])]
    exact_counts = Counter(sampled[row["id"]]["exact"] for row in candidates)
    with args.output.open("w") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in eval_rows + kept)
    print("exact-match count per candidate (of samples):", dict(sorted(exact_counts.items())))
    print(f"candidates={len(candidates)} kept={len(kept)} eval={len(eval_rows)} -> {args.output}")


if __name__ == "__main__":
    main()
