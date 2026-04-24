#!/usr/bin/env python3
"""Remote-grader async RL -- thin wiring.

Every piece lives somewhere reusable:

  * :mod:`training.utils.rl.rollout_service` -- service-agnostic
    protocol and dataclasses (no EP dep).
  * :mod:`training.utils.rl.text_rollout` -- generic packer that turns
    service payloads into :class:`~training.utils.rl.rollout.Rollout`,
    handling tokenization and echo re-score.
  * :mod:`.ep_service` -- the only file that imports ``eval_protocol``.
  * :mod:`.grader` -- EP-decorated scoring function.
  * :mod:`.mock_agent` -- stand-in remote completion service.

To swap the backend (agent framework, RAG pipeline, LLM judge): write a
new class that satisfies :class:`RolloutService` and replace
``EPService()`` below.  Nothing else changes.

Run::

    export FIREWORKS_API_KEY=...
    python -m training.examples.ep_remote_grader.train
"""

from __future__ import annotations

from training.examples.ep_remote_grader.ep_service import EPService
from training.recipes.async_rl_loop import Config, main
from training.utils.rl.losses import PromptGroup
from training.utils.rl.text_rollout import make_text_rollout_fn


def should_accept(pg: PromptGroup) -> bool:
    """Reject zero-variance groups (GRPO assigns zero advantage on ties)."""
    return len(set(pg.rewards)) > 1


if __name__ == "__main__":
    cfg = Config(
        log_path="/tmp/rl-async-ep",
        base_model="accounts/fireworks/models/qwen3-8b",
        dataset=(
            "https://raw.githubusercontent.com/eval-protocol/python-sdk/"
            "main/development/gsm8k_sample.jsonl"
        ),
        prompt_groups_per_step=2,
        max_head_offpolicy_versions=1,
        completions_per_prompt=4,
    )
    rollout_fn = make_text_rollout_fn(EPService())
    main(cfg, rollout_fn=rollout_fn, dynamic_filter_fn=should_accept)
