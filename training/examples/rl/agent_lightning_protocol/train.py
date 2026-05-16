#!/usr/bin/env python3
"""Minimal async RL wiring for the Agent Lightning protocol adapter.

Rows are pre-tokenized here to keep the example focused on the integration
boundary.  The rollout provider wraps one sampled completion as an
Agent-Lightning-like triplet, then ``make_agent_lightning_rollout_fn`` converts
that triplet into the cookbook's token-native ``RolloutSample``.
"""

from __future__ import annotations

from training.examples.rl.agent_lightning_protocol.rollout import make_rollout_fn
from training.recipes.async_rl_loop import Config, main
from training.utils import DeployConfig


if __name__ == "__main__":
    cfg = Config(
        log_path="/tmp/rl-agent-lightning-protocol",
        base_model="accounts/fireworks/models/example",
        prompt_groups_per_step=1,
        completions_per_prompt=2,
        deployment=DeployConfig(tokenizer_model="example-tokenizer"),
    )
    rows = [
        {"id": "agl-protocol-1", "prompt_token_ids": [1, 2, 3], "reward": 1.0},
    ]
    main(cfg, rollout_fn_factory=make_rollout_fn, rows=rows)
