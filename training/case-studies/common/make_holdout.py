#!/usr/bin/env python3
"""Split a JSONL dataset into train + holdout with a seeded shuffle.

Guarantees the holdout is disjoint from the training set, so you can evaluate
base vs fine-tuned models on data the trainer never saw.

Usage:
    python make_holdout.py \
        --in food_reasoning.jsonl \
        --train-out food_train.jsonl \
        --eval-out food_holdout.jsonl \
        --eval-fraction 0.1 --seed 42

Then point cord_receipt_sft.ipynb at food_train.jsonl and
eval_before_after.ipynb at food_holdout.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True, help="Input JSONL path.")
    p.add_argument("--train-out", required=True, help="Output train JSONL path.")
    p.add_argument("--eval-out", required=True, help="Output holdout JSONL path.")
    p.add_argument("--eval-fraction", type=float, default=0.1,
                   help="Fraction of rows to hold out for eval (0..1, default 0.1).")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed (default 42).")
    args = p.parse_args()

    if not 0 < args.eval_fraction < 1:
        sys.exit("--eval-fraction must be in (0, 1).")

    in_path = Path(args.inp)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    rows = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(rows)} rows from {in_path}")

    random.Random(args.seed).shuffle(rows)
    n_eval = int(len(rows) * args.eval_fraction)
    n_train = len(rows) - n_eval
    holdout, train = rows[:n_eval], rows[n_eval:]
    print(f"Split (seed={args.seed}): train={n_train} holdout={n_eval}")

    for out_path, split_rows, label in [
        (Path(args.train_out), train, "train"),
        (Path(args.eval_out), holdout, "holdout"),
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for row in split_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[{label}] wrote {len(split_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
