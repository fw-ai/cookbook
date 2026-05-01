#!/usr/bin/env python3
"""Minimal async RL wiring for the single-turn token-in rollout."""

from __future__ import annotations

from training.examples.rl.single_turn_token_in.rollout import make_rollout_fn
from training.recipes.async_rl_loop import Config, main
from training.utils import DeployConfig


if __name__ == "__main__":
    cfg = Config(
        log_path="/tmp/rl-single-turn-token-in",
        base_model="accounts/fireworks/models/example",
        prompt_groups_per_step=1,
        completions_per_prompt=2,
        deployment=DeployConfig(tokenizer_model="example-tokenizer"),
    )
    rows = [{"id": "tok-1", "prompt_token_ids": [1, 2, 3], "reward": 1.0}]
    main(cfg, rollout_fn_factory=make_rollout_fn, rows=rows)
