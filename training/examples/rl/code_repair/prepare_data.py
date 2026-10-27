#!/usr/bin/env python3
"""Build code-repair RL rows from ``archiki/UTGenDebug`` and validate them.

Every row is graded in the sandbox twice: the canonical solution must pass
every case (that count becomes ``num_cases``) and the buggy code must fail at
least one. Rows that fail either check, or whose test loop cannot be
rewritten, are dropped and listed. Output, in the ``{"messages",
"ground_truth"}`` format, with eval rows drawn proportionally from each split:
``data/eval.jsonl`` and ``data/train_candidates.jsonl``.

    python -m training.examples.rl.code_repair.prepare_data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

from training.examples.rl.code_repair.sandbox import grade

DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET = "archiki/UTGenDebug"
SPLITS = ("he_plus_fix", "mbpp_plus_fix", "mbpp_plus_fix_hard")

SYSTEM_PROMPT = (
    "You fix bugs in Python functions. You get a task description and a buggy implementation. "
    "Reply with the complete corrected implementation, including any imports and helper functions it needs, "
    "in a single ```python code block. Keep the same function name and parameters. "
    "Do not include tests, example calls or input reading."
)


def buggy_program(split: str, row: dict[str, Any]) -> tuple[str, str]:
    """``(task description, buggy code)``; HumanEval rows store only the body."""
    if split == "he_plus_fix":
        return row["prompt"].strip(), row["prompt"] + row["code"]
    return row["prompt"].strip(), row["code"]


def canonical_program(split: str, row: dict[str, Any]) -> str:
    return row["prompt"] + row["canonical_solution"] if split == "he_plus_fix" else row["canonical_solution"]


def user_message(description: str, signature: str, buggy: str) -> str:
    return (
        f"## Task\n{description}\n\n## Function signature\n{signature}\n\n"
        f"## Buggy implementation\n```python\n{buggy.strip()}\n```"
    )


async def build_row(split: str, row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    entry, test = row["entry_point"], row["test"]
    try:
        canonical = await grade(canonical_program(split, row), test, entry)
    except ValueError as err:
        return None, f"test loop: {err}"
    if canonical.status != "ok" or canonical.seen == 0 or canonical.passed != canonical.seen:
        return None, f"canonical {canonical}"
    description, buggy = buggy_program(split, row)
    broken = await grade(buggy, test, entry)
    if broken.passed >= canonical.seen:
        return None, "buggy code passes every case"
    return {
        "id": f"{split}/{row['task_id']}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(description, row["signature"], buggy)},
        ],
        "ground_truth": {
            "entry_point": entry,
            "test": test,
            "num_cases": canonical.seen,
            "buggy_pass_rate": round(broken.passed / canonical.seen, 4),
        },
    }, ""


async def build_all(eval_rows: int, seed: int) -> tuple[list[dict], list[dict], list[str]]:
    from datasets import load_dataset

    ds = load_dataset(DATASET)
    built: dict[str, list[dict]] = {}
    dropped: list[str] = []
    for split in SPLITS:
        results = await asyncio.gather(*(build_row(split, row) for row in ds[split]))
        built[split] = []
        for row, (out, reason) in zip(ds[split], results):
            if out is None:
                dropped.append(f"{split}/{row['task_id']}: {reason}")
            else:
                built[split].append(out)
    total = sum(len(rows) for rows in built.values())
    rng = random.Random(seed)
    eval_out, train_out = [], []
    for split in SPLITS:
        rows = built[split]
        rng.shuffle(rows)
        k = round(eval_rows * len(rows) / total)
        eval_out += rows[:k]
        train_out += rows[k:]
    rng.shuffle(train_out)
    return eval_out, train_out, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--eval-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    eval_rows, train_rows, dropped = asyncio.run(build_all(args.eval_rows, args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("eval.jsonl", eval_rows), ("train_candidates.jsonl", train_rows)):
        with (args.output_dir / name).open("w") as f:
            f.writelines(json.dumps(row) + "\n" for row in rows)
    for line in dropped:
        print("dropped", line)
    print(f"eval={len(eval_rows)} train_candidates={len(train_rows)} dropped={len(dropped)}")


if __name__ == "__main__":
    main()
