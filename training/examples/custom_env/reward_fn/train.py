#!/usr/bin/env python3
"""Single-turn async RL example: bring your own ``reward_fn``.

Matches the sync-loop ergonomics -- one function, no envs to wire up.
The async loop internally wraps the reward into a one-step env via
``SingleTurnEnv``.

Usage::

    export FIREWORKS_API_KEY=...
    python -m training.examples.custom_env.reward_fn.train
"""

from __future__ import annotations

import re

from training.recipes.rl_loop_async import AsyncConfig, Config, main


def _extract_number(text: str) -> str | None:
    match = re.search(r"<answer>\s*(-?\d+)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else None


def reward_fn(completion: str, row: dict) -> float:
    """Return 1.0 iff the model's extracted number matches ``row['answer']``."""
    predicted = _extract_number(completion)
    if predicted is None:
        return 0.0
    truth = str(row.get("answer", "")).strip()
    return 1.0 if predicted == truth else 0.0


if __name__ == "__main__":
    cfg = Config(
        log_path="/tmp/rl-async-reward-fn",
        base_model="accounts/fireworks/models/qwen3-8b",
        dataset="https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl",
        async_config=AsyncConfig(max_steps_off_policy=1, groups_per_batch=4),
        completions_per_prompt=4,
        max_rows=32,
    )
    main(cfg, reward_fn=reward_fn)
