#!/usr/bin/env python3
"""Single-turn renderer-backed synchronous on-policy RL example.

Wires :func:`single_turn_renderer_rollout` (from
``training.utils.rl.rollout``) into the **synchronous** RL
recipe ``training.recipes.rl_loop``.  The recipe accepts an optional
``rollout_fn(row, ctx) -> Rollout | None`` keyword argument (mirroring the
async recipe's contract) so renderer-backed rollouts plug in directly.
The example deliberately does NOT import ``async_rl_loop``.

Run::

    export FIREWORKS_API_KEY=...
    python -m training.examples.rl.single_turn_sync_on_policy.train
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import transformers

from fireworks.training.sdk.deployment import DeploymentSampler

from training.recipes.rl_loop import Config, RolloutContext, main
from training.utils import DeployConfig
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
    msgs = row.get("messages")
    if msgs:
        return list(msgs)
    return [
        {"role": "user", "content": str(row.get("prompt", row.get("question", "")))},
    ]


async def _reward_fn(row, parsed_message, parse_success):
    """Inline DROP-on-parse-failure pattern.

    Returning ``None`` drops the completion (no sample emitted); returning
    ``0.0`` would emit a zero-reward sample instead.  The framework
    deliberately does not bake a parse-failure-policy enum.
    """
    if not parse_success:
        return None
    truth = str(row.get("ground_truth", "")).strip() or None
    text = getattr(parsed_message, "content", None) or str(parsed_message)
    pred = _extract_answer(text)
    if pred is None or truth is None:
        return 0.0
    return 1.0 if pred == truth else 0.0


def _build_rollout_fn(cfg: Config):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.deployment.tokenizer_model, trust_remote_code=True,
    )
    renderer = build_renderer(tokenizer, cfg.deployment.tokenizer_model)

    async def rollout_fn(row: dict, ctx: RolloutContext) -> Rollout | None:
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
            reward_fn=_reward_fn,
        )

    return rollout_fn


if __name__ == "__main__":
    cfg = Config(
        log_path="/tmp/rl-sync-on-policy-single-turn-renderer",
        base_model="accounts/fireworks/models/qwen3-8b",
        dataset=(
            "https://raw.githubusercontent.com/eval-protocol/python-sdk/"
            "main/development/gsm8k_sample.jsonl"
        ),
        prompt_groups_per_step=1,
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
    )
    main(cfg, rollout_fn=_build_rollout_fn(cfg))
