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

Reward plumbing supports both conventions (text_rollout checks
``payload.total_reward`` first, falls back to ``reward_fn`` when
``None``):

  * **Server-graded** (default here): ``EPService(grade=True)`` runs
    the EP grader inside the service and fills ``total_reward``.  Use
    when grading is cheap on the service side and doesn't need
    trainer-side state.

  * **Trainer-graded**: ``EPService(grade=False)`` returns
    ``total_reward=None`` and ``make_text_rollout_fn`` is handed a
    ``reward_fn(row, payload) -> float``.  Use when the reward needs
    trainer-side state (reference model, local reward model, etc.) or
    when grading is expensive and you want to batch it.

To swap the completion backend (agent framework, RAG pipeline, LLM
judge): write a new class that satisfies :class:`RolloutService` and
replace ``EPService()`` below.  Nothing else changes.

Run::

    export FIREWORKS_API_KEY=...
    python -m training.examples.ep_remote_grader.train
"""

from __future__ import annotations

from training.examples.ep_remote_grader.ep_service import EPService, _grade
from training.recipes.async_rl_loop import Config, main
from training.utils.rl.losses import PromptGroup
from training.utils.rl.rollout_service import RolloutPayload
from training.utils.rl.text_rollout import make_text_rollout_fn


def should_accept(pg: PromptGroup) -> bool:
    """Reject zero-variance groups (GRPO assigns zero advantage on ties)."""
    return len(set(pg.rewards)) > 1


async def trainer_grade(row: dict, payload: RolloutPayload) -> float:
    """Trainer-side reward: defers to the same EP grader, but runs here
    rather than on the service.  Swap for a reference-model scorer, a
    local reward model, a metric-join against another dataset, etc."""
    last = next(
        (t for t in reversed(payload.turns) if t.role == "assistant"), None,
    )
    completion = last.text if last is not None else ""
    return await _grade(
        prompt_messages=row.get("messages") or [],
        completion_text=completion,
        ground_truth=str(row.get("ground_truth", "")),
    )


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

    # Server-graded: EPService grades internally and fills total_reward.
    rollout_fn = make_text_rollout_fn(EPService())

    # Trainer-graded alternative -- uncomment to use; swap the two lines:
    #
    #   rollout_fn = make_text_rollout_fn(
    #       EPService(grade=False),
    #       reward_fn=trainer_grade,
    #   )

    main(cfg, rollout_fn=rollout_fn, dynamic_filter_fn=should_accept)
