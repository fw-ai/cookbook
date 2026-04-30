#!/usr/bin/env python3
"""GSM8K single-turn async RL — renderer-backed wiring.

Migrated to the renderer-backed surface introduced by the RL renderer-reuse
change set: the rollout function is built from
:func:`single_turn_renderer_rollout` with the SDK's pre-tokenized
:meth:`DeploymentSampler.sample_with_prompt_tokens` primitive.  No
client-side ``apply_chat_template`` calls; no hand-packed
``RolloutSample``; the helper packs tokens / logprobs / loss-mask for us.

Trajectory logging stays in user-owned callbacks via the rollout closure
so existing dashboards / inspection scripts keep working.

Run::

    export FIREWORKS_API_KEY=...
    python -m training.examples.rl.gsm8k_async.train
"""

from __future__ import annotations

import json
import logging
import os
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


TRAJECTORY_DIR: str | None = os.environ.get("TRAJECTORY_DIR")


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    digits = re.search(r"(-?\d+)", match.group(1))
    return digits.group(1) if digits else None


def _grade(parsed_text: str, ground_truth: str) -> float:
    predicted = extract_answer(parsed_text)
    truth = extract_answer(ground_truth)
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


def should_accept(pg: PromptGroup) -> bool:
    """Reject zero-variance groups (GRPO assigns zero advantage on ties)."""
    return len(set(pg.rewards)) > 1


async def _message_builder(row: dict, ctx: RolloutContext) -> list[dict]:
    msgs = row.get("messages")
    if msgs:
        return list(msgs)
    return [
        {"role": "user", "content": str(row.get("prompt", row.get("question", "")))},
    ]


def _make_reward_fn():
    """Wrap the GSM8K grader as the (row, parsed_message, parse_success) reward_fn.

    On parse failure, returns 0.0 (zero-reward pattern) — keeping the
    completion in the group lets GRPO see the negative signal.  Switch to
    ``return None`` to DROP instead.
    """
    async def reward_fn(row, parsed_message, parse_success):
        text = getattr(parsed_message, "content", None) or str(parsed_message)
        if not parse_success:
            return 0.0
        return _grade(text, str(row.get("ground_truth", "")))

    return reward_fn


def _dump_rollout(rollout: Rollout, version: int) -> None:
    if not TRAJECTORY_DIR:
        return
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    path = os.path.join(TRAJECTORY_DIR, f"version_{version:04d}.jsonl")
    with open(path, "a") as f:
        for ci, s in enumerate(rollout.samples):
            f.write(json.dumps({
                "version": version,
                "completion_index": ci,
                "reward": s.reward,
                "completion_text": s.text,
                "finish_reason": s.finish_reason,
                "ground_truth": (rollout.row_meta or {}).get("ground_truth"),
            }) + "\n")


def _build_rollout_fn(cfg: Config):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.deployment.tokenizer_model, trust_remote_code=True,
    )
    renderer = build_renderer(tokenizer, cfg.deployment.tokenizer_model)
    reward_fn = _make_reward_fn()
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
        rollout = await single_turn_renderer_rollout(
            row,
            ctx,
            renderer=renderer,
            sample_with_prompt_tokens=sampler.sample_with_prompt_tokens,
            message_builder=_message_builder,
            reward_fn=reward_fn,
        )
        if rollout is None:
            return None
        # User-owned trajectory logging callback (replaces the legacy in-line dump).
        if rollout.row_meta is None:
            rollout.row_meta = {"ground_truth": row.get("ground_truth", "")}
        else:
            rollout.row_meta.setdefault("ground_truth", row.get("ground_truth", ""))
        _dump_rollout(rollout, ctx.current_version())
        return rollout

    return rollout_fn


if __name__ == "__main__":
    cfg = Config(
        log_path="/tmp/rl-async-gsm8k",
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
