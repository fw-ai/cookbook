#!/usr/bin/env python3
"""Train a multi-turn rollout using ``TrajectoryAssembler``.

Mirrors :mod:`training.examples.rl.ep_remote_grader.train` but the rollout
service is a plain Fireworks Completions client driven by the assembler
-- no EP dependency.  The interesting code lives in :mod:`.rollout`.

Run::

    export FIREWORKS_API_KEY=...
    python -m training.examples.rl.multi_turn_minimal.train
"""

from __future__ import annotations

from training.examples.rl.multi_turn_minimal.rollout import MultiTurnService
from training.recipes.async_rl_loop import Config, RolloutContext, main
from training.utils import DeployConfig
from training.utils.rl.losses import PromptGroup
from training.utils.rl.rollout import Rollout
from training.utils.rl.rollout_service import RolloutPayload
from training.utils.rl.text_rollout import make_text_rollout_fn


def should_accept(pg: PromptGroup) -> bool:
    """Reject zero-variance groups (GRPO assigns zero advantage on ties)."""
    return len(set(pg.rewards)) > 1


async def length_reward(messages: list[dict], asm) -> float:
    """Toy reward: prefer rollouts whose final assistant turn is non-empty.

    Replace with the real grading logic for your task -- a verifier,
    a reward model, an EP test, a unit-test runner.  ``asm`` is the
    populated :class:`TrajectoryAssembler`; reach into its turns for
    structured access to per-call outputs.
    """
    payload: RolloutPayload = asm.to_payload(total_reward=None)
    last_assistant = next(
        (t for t in reversed(payload.turns) if t.role == "assistant"), None,
    )
    if last_assistant is None or not last_assistant.token_ids:
        return 0.0
    return min(1.0, len(last_assistant.token_ids) / 64.0)


if __name__ == "__main__":
    cfg = Config(
        log_path="/tmp/rl-multi-turn-minimal",
        base_model="accounts/fireworks/models/qwen3-8b",
        prompt_groups_per_step=2,
        max_head_offpolicy_versions=1,
        completions_per_prompt=4,
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
    )

    cached_rollout_fn = []

    async def rollout_fn(row: dict, ctx: RolloutContext) -> Rollout | None:
        if not cached_rollout_fn:
            service = MultiTurnService(
                base_url=ctx.inference_base_url,
                api_key=ctx.api_key,
                model=ctx.model,
                tokenizer=ctx.tokenizer,
                n_turns=2,
                followup_text="Please refine your answer.",
                reward_fn=length_reward,
            )
            cached_rollout_fn.append(make_text_rollout_fn(service))
        return await cached_rollout_fn[0](row, ctx)

    rows = [
        {"messages": [{"role": "user", "content": "Explain Bayes' theorem briefly."}]},
        {"messages": [{"role": "user", "content": "What is gradient descent?"}]},
    ]
    main(cfg, rollout_fn=rollout_fn, dynamic_filter_fn=should_accept, rows=rows)
