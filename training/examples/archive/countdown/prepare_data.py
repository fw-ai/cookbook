#!/usr/bin/env python3
"""Write the TinyZero Countdown tasks as chat-format RL training JSONL.

Deprecated: archived example, covered by ``rl/deepmath/train_deepmath_serverless.py``. See ``examples/archive/README.md``.

    python -m training.examples.archive.countdown.prepare_data
    python -m training.examples.archive.countdown.prepare_data --output /tmp/countdown.jsonl --num-rows 2000

Downloads ``Jiayi-Pan/Countdown-Tasks-3to4`` from Hugging Face, shuffles it with
``--seed``, and writes ``--num-rows`` rows shaped as::

    {"messages": [system, user], "ground_truth": "{\\"numbers\\": [...], \\"target\\": N}"}

Needs only ``datasets``; it does not import the training stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HF_DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"
DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "countdown_3to4_train.jsonl"

SYSTEM_PROMPT = (
    "You are a math puzzle solver. Given a target number and a set of available "
    "numbers, find an arithmetic expression using each number exactly once with "
    "operations +, -, *, / to reach the target.\n\n"
    "Show your reasoning inside <think>...</think> tags, then put your final "
    "equation inside <answer>...</answer> tags.\n\n"
    "Example:\n"
    "Target: 24, Numbers: [1, 2, 3, 4]\n"
    "<think>I need to reach 24. Let me try 1 * 2 * 3 * 4 = 24.</think>\n"
    "<answer>1 * 2 * 3 * 4</answer>"
)

USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use +, -, *, / and each number must be used exactly once."
)


def countdown_row(numbers: list[int], target: int) -> dict[str, Any]:
    """One training row; ``ground_truth`` is a JSON string, as the reward expects."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(numbers=list(numbers), target=target)},
        ],
        "ground_truth": json.dumps({"numbers": list(numbers), "target": int(target)}),
    }


def prepare_dataset(output: Path, num_rows: int = 20000, seed: int = 0) -> None:
    """Download the Countdown tasks and write ``num_rows`` shuffled rows to ``output``."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, split="train")
    if num_rows and num_rows < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for rec in ds:
            f.write(json.dumps(countdown_row(list(rec["nums"]), int(rec["target"]))) + "\n")
    print(f"wrote {len(ds)} rows -> {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--num-rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    prepare_dataset(args.output, args.num_rows, args.seed)


if __name__ == "__main__":
    main()
