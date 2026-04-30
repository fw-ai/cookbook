#!/usr/bin/env python3
"""Single-turn renderer-backed async RL example.

Wires :func:`single_turn_renderer_rollout` (from
``training.utils.rl.rollout``) into the async RL recipe.  The
example uses the Tinker-compatible renderer registered for the configured
model (built via :func:`training.utils.supervised.build_renderer`) plus
the SDK's pre-tokenized sampling primitive
:meth:`DeploymentSampler.sample_with_prompt_tokens`.  No chat-template
re-rendering happens client-side.

Run::

    export FIREWORKS_API_KEY=...
    python -m training.examples.rl.single_turn_async.train
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import transformers

from fireworks.training.sdk.deployment import DeploymentSampler

from training.recipes.async_rl_loop import Config, RolloutContext, main
from training.utils import DeployConfig
from training.utils.rl.losses import PromptGroup
from training.utils.rl.rollout import single_turn_renderer_rollout
from training.utils.rl.rollout import Rollout
from training.utils.supervised import build_renderer


logger = logging.getLogger(__name__)


def _extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    digits = re.search(r"(-?\d+)", match.group(1))
    return digits.group(1) if digits else None


async def _message_builder(row: dict, ctx: RolloutContext) -> list[dict]:
    """Seed messages for the single-turn rollout.

    Falls back to a permissive shape if the row already includes messages.
    """
    msgs = row.get("messages")
    if msgs:
        return list(msgs)
    return [
        {"role": "user", "content": str(row.get("prompt", row.get("question", "")))},
    ]


async def _reward_fn_factory(row, parsed_message, parse_success):
    """User-owned parse-failure handling: DROP on parse failure.

    This is the inline DROP pattern documented in the canonical examples;
    swap ``return None`` for ``return 0.0`` to switch to zero-reward.
    """
    if not parse_success:
        return None
    truth = str(row.get("ground_truth", "")).strip() or None
    text = getattr(parsed_message, "content", None) or str(parsed_message)
    pred = _extract_answer(text)
    if pred is None or truth is None:
        return 0.0
    return 1.0 if pred == truth else 0.0


def should_accept(pg: PromptGroup) -> bool:
    """Reject zero-variance groups (GRPO assigns zero advantage on ties)."""
    return len(set(pg.rewards)) > 1


def _build_rollout_fn(cfg: Config):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.deployment.tokenizer_model, trust_remote_code=True,
    )
    renderer = build_renderer(tokenizer, cfg.deployment.tokenizer_model)
    sampler: DeploymentSampler | None = None

    async def rollout_fn(row: dict, ctx: RolloutContext) -> Rollout | None:
        nonlocal sampler
        if sampler is None:
            sampler = DeploymentSampler(
                inference_url=ctx.inference_base_url,
                model=ctx.model,
                api_key=ctx.api_key,
                tokenizer=tokenizer,
            )
        return await single_turn_renderer_rollout(
            row,
            ctx,
            renderer=renderer,
            sample_with_prompt_tokens=sampler.sample_with_prompt_tokens,
            message_builder=_message_builder,
            reward_fn=_reward_fn_factory,
        )

    return rollout_fn


if __name__ == "__main__":
    cfg = Config(
        log_path="/tmp/rl-async-single-turn-renderer",
        base_model="accounts/fireworks/models/qwen3-8b",
        dataset=(
            "https://raw.githubusercontent.com/eval-protocol/python-sdk/"
            "main/development/gsm8k_sample.jsonl"
        ),
        prompt_groups_per_step=1,
        max_head_offpolicy_versions=0,
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
    )
    main(cfg, rollout_fn=_build_rollout_fn(cfg), dynamic_filter_fn=should_accept)
