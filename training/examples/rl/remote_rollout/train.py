#!/usr/bin/env python3
"""``RolloutService``-driven generic remote-rollout example.

Wires :func:`make_remote_rollout_fn` (re-exported from
``training.utils.rl.rollout``) with a deterministic mock
``RolloutService`` so users can see the token-native contract end to end
without standing up a real inference service.  The renderer is applied
service-side; the cookbook helper itself is renderer-name-agnostic.

Run::

    python -m training.examples.rl.remote_rollout.train

The ``mock_service`` returns a deterministic two-completion group whose
rewards have variance (so the GRPO ``dynamic_filter`` filter passes).
Swap ``MockRolloutService`` for any class implementing the
``RolloutService`` protocol to drive real training.
"""

from __future__ import annotations

import logging

from training.examples.rl.remote_rollout.mock_service import MockRolloutService
from training.recipes.async_rl_loop import Config, main
from training.utils import DeployConfig
from training.utils.rl.losses import PromptGroup
from training.utils.rl.rollout import make_remote_rollout_fn


logger = logging.getLogger(__name__)


def dynamic_filter(pg: PromptGroup) -> bool:
    return len(set(pg.rewards)) > 1


if __name__ == "__main__":
    cfg = Config(
        log_path="/tmp/rl-remote-rollout-mock",
        base_model="accounts/fireworks/models/qwen3-8b",
        prompt_groups_per_step=1,
        completions_per_prompt=2,
        max_head_offpolicy_versions=0,
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
    )

    rows = [
        {"messages": [{"role": "user", "content": "Hello, world."}]},
        {"messages": [{"role": "user", "content": "What is 2+2?"}]},
    ]

    service = MockRolloutService(tokenizer_id=cfg.deployment.tokenizer_model)
    rollout_fn = make_remote_rollout_fn(service)
    main(cfg, rollout_fn=rollout_fn, dynamic_filter_fn=dynamic_filter, rows=rows)
