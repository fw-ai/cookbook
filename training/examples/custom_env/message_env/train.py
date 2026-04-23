#!/usr/bin/env python3
"""Multi-turn async RL example: custom ``MessageEnv``.

Implements a toy "guessing" task where the env replies with "higher" or
"lower" feedback until the model guesses the target number or runs out of
turns.  Real multi-turn tasks (tool use, code agents, simulators) follow
the same shape: subclass :class:`MessageEnv` and keep state on ``self``.

Usage::

    export FIREWORKS_API_KEY=...
    python -m training.examples.custom_env.message_env.train
"""

from __future__ import annotations

import re

from training.recipes.rl_loop_async import AsyncConfig, Config, main
from training.utils.rl import MessageEnv, MessageStepResult


def _extract_guess(text: str) -> int | None:
    match = re.search(r"<guess>\s*(-?\d+)\s*</guess>", text)
    return int(match.group(1)) if match else None


class GuessingEnv(MessageEnv):
    """Env replies with higher/lower feedback until the model nails the target."""

    SYSTEM_PROMPT = (
        "You are playing a number-guessing game. Reply with your guess in "
        "<guess>...</guess> tags. The user will tell you whether the target is "
        "higher or lower than your guess."
    )

    def __init__(self, row: dict):
        self.target = int(row["target"])
        self.turns_left = int(row.get("max_turns", 6))

    async def initial_messages(self):
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": "Guess a number between 1 and 100."},
        ]

    async def step(self, assistant_message):
        self.turns_left -= 1
        guess = _extract_guess(assistant_message.get("content", "") or "")
        if guess is None:
            # Unparseable -- penalise and bail.
            return MessageStepResult(reward=-0.1, episode_done=True)

        if guess == self.target:
            return MessageStepResult(reward=1.0, episode_done=True)

        if self.turns_left <= 0:
            return MessageStepResult(reward=0.0, episode_done=True)

        hint = "higher" if guess < self.target else "lower"
        return MessageStepResult(
            reward=0.0,
            episode_done=False,
            next_messages=[{"role": "user", "content": f"The target is {hint}."}],
        )


def env_builder(row: dict) -> GuessingEnv:
    return GuessingEnv(row)


if __name__ == "__main__":
    import random

    random.seed(0)
    rows = [{"target": random.randint(1, 100), "max_turns": 6} for _ in range(64)]

    cfg = Config(
        log_path="/tmp/rl-async-message-env",
        base_model="accounts/fireworks/models/qwen3-8b",
        async_config=AsyncConfig(max_steps_off_policy=1, groups_per_batch=4),
        completions_per_prompt=4,
        max_turns=6,
        max_rows=len(rows),
    )
    main(cfg, env_builder=env_builder, rows=rows)
