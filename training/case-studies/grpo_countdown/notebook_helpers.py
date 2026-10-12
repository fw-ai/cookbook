"""Thin helpers for the Countdown serverless GRPO teaching notebook.

Reward helpers and ``prepare_dataset`` import without the training stack — a
lightweight prereq (``pip install fireworks-ai datasets``) covers data prep.
The live run uses ``training.examples.archive.countdown.countdown_rl`` (Countdown
task code handed to the serverless async-RL recipe) and loads lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.examples.archive.countdown.countdown_rewards import (
    accuracy_reward,
    composite_reward,
    format_reward,
)
from training.examples.archive.countdown.prepare_data import prepare_dataset

__all__ = [
    "SMOKE_EVAL_ROWS",
    "accuracy_reward",
    "composite_reward",
    "format_reward",
    "group_relative_advantages",
    "load_countdown_rl",
    "prepare_dataset",
    "smoke_config",
]

SMOKE_EVAL_ROWS = 4
"""Leading dataset rows held out for the smoke run's evaluation."""


def load_countdown_rl() -> Any:
    """Import ``countdown_rl`` (needs ``tinker`` from the cookbook training install)."""
    from training.examples.archive.countdown import countdown_rl as mod

    return mod


def group_relative_advantages(rewards: list[float]) -> list[float]:
    """The recipe's GRPO advantage (``training.utils.data.compute_advantages``)."""
    from training.utils.data import compute_advantages

    return compute_advantages(rewards)


def smoke_config(
    *,
    dataset: str | Path,
    run_dir: str | Path,
    base_model: str = "accounts/fireworks/models/qwen3-8b",
    tokenizer_model: str = "Qwen/Qwen3-8B",
) -> Any:
    """Smoke-sized serverless recipe ``Config``: two steps of two prompts."""
    from training.recipes.experiment.async_rl_loop_serverless import Config

    return Config(
        base_model=base_model,
        tokenizer_model=tokenizer_model,
        dataset=str(dataset),
        max_rows=4,
        prompt_groups_per_step=2,
        completions_per_prompt=4,
        min_group_size=2,
        max_concurrency_rollout_sample=8,
        max_completion_tokens=512,
        max_seq_len=32768,
        learning_rate=1e-4,
        lora_rank=32,
        weight_decay=0.0,
        shuffle=False,
        snapshot_prefix="countdown-smoke",
        metrics_file=str(Path(run_dir) / "metrics.jsonl"),
    )
