"""Thin helpers for the Countdown serverless GRPO teaching notebook.

Reward helpers and ``prepare_dataset`` import without ``tinker`` — a
lightweight prereq (``pip install fireworks-ai datasets``) covers data prep.
The GRPO loop / ``Config`` / ``ServerlessCountdownRL`` live in
``training.examples.serverless_rl.countdown_rl`` and load lazily.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from training.examples.serverless_rl.countdown_rewards import (
    accuracy_reward,
    composite_reward,
    format_reward,
)

__all__ = [
    "accuracy_reward",
    "composite_reward",
    "format_reward",
    "group_relative_advantages",
    "load_countdown_rl",
    "prepare_dataset",
    "smoke_config",
]


def load_countdown_rl() -> Any:
    """Import ``countdown_rl`` (needs ``tinker`` from the cookbook training install)."""
    # lazy: countdown_rl imports tinker at module import time
    from training.examples.serverless_rl import countdown_rl as mod

    return mod


def group_relative_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Delegate to ``countdown_rl._group_relative_advantages`` — do not re-derive."""
    return load_countdown_rl()._group_relative_advantages(rewards, eps=eps)


def prepare_dataset(output: Path, num_rows: int, seed: int) -> None:
    """Download TinyZero Countdown rows and write them as training JSONL.

    Needs only ``datasets`` — no ``tinker`` import, so the notebook's
    ``PREPARE_FULL_DATASET`` path works on the lightweight prereq. Mirrors
    ``countdown_rl.prepare_dataset`` (dataset id + prompts + row shape);
    keep the two in sync.
    """
    from datasets import load_dataset

    ds = load_dataset(_HF_DATASET_ID, split="train")
    if num_rows and num_rows < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for rec in ds:
            row = {
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _USER_TEMPLATE.format(
                            numbers=list(rec["nums"]), target=rec["target"]
                        ),
                    },
                ],
                "ground_truth": json.dumps(
                    {"numbers": list(rec["nums"]), "target": int(rec["target"])}
                ),
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(ds)} rows -> {output}")


# Mirror countdown_rl's constants (countdown_rl imports tinker at module load,
# so importing them from there would reintroduce the tinker dependency).
_HF_DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"
_SYSTEM_PROMPT = (
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
_USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use +, -, *, / and each number must be used exactly once."
)


def smoke_config(
    *,
    dataset: str | Path,
    run_dir: str | Path,
    api_key: str | None = None,
    base_model: str = "accounts/fireworks/models/qwen3-8b",
    tokenizer_model: str = "Qwen/Qwen3-8B",
    renderer_name: str = "",
) -> Any:
    """Smoke-sized ``countdown_rl.Config`` — cheap enough to learn the loop."""
    Config = load_countdown_rl().Config
    key = api_key if api_key is not None else os.environ.get("FIREWORKS_API_KEY", "")
    return replace(
        Config(),
        base_model=base_model,
        tokenizer_model=tokenizer_model,
        renderer_name=renderer_name,
        dataset=str(dataset),
        run_dir=str(run_dir),
        api_key=key,
        steps=2,
        prompt_groups_per_step=2,
        group_size=4,
        prompt_concurrency=2,
        max_sample_tokens=512,
        eval_prompt_groups=4,
        eval_group_size=2,
        eval_interval=1,
        eval_at_start=True,
        eval_at_end=True,
        dcp_save_interval=0,
        plot_reward_curve=True,
        router_replay=True,
    )
