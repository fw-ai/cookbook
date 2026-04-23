#!/usr/bin/env python3
"""GSM8K single-turn async RL -- canonical ``rollout_fn`` example.

Shows the flat ``Rollout`` contract: one :class:`RolloutSample` per
completion, three parallel lists (``tokens``, ``logprobs``,
``loss_mask``) plus a scalar reward.  For single-turn rollouts the
loss_mask is just ``[0]*prompt_len + [1]*completion_len``.

Run::

    export FIREWORKS_API_KEY=...
    python -m training.examples.gsm8k_async.train
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from training.recipes.async_rl_loop import Config, RolloutContext, main
from training.utils.rl.losses import PromptGroup
from training.utils.rl.rollout import Rollout, RolloutSample

logger = logging.getLogger(__name__)


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    digits = re.search(r"(-?\d+)", match.group(1))
    return digits.group(1) if digits else None


def reward_fn(completion: str, row: dict) -> float:
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


def should_accept(pg: PromptGroup) -> bool:
    """Reject zero-variance groups (GRPO assigns zero advantage on ties)."""
    return len(set(pg.rewards)) > 1


TRAJECTORY_DIR: str | None = os.environ.get("TRAJECTORY_DIR")


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


async def rollout_fn(row: dict, ctx: RolloutContext) -> Rollout | None:
    messages = row.get("messages") or []
    if not messages:
        return None

    try:
        sampled = await ctx.sampler.sample_with_tokens(
            messages=messages, n=ctx.completions_per_prompt, **ctx.sample_kwargs,
        )
    except Exception as exc:
        logger.warning("sample_with_tokens failed: %s", exc)
        return None

    if not sampled or len(sampled) < ctx.completions_per_prompt:
        return None

    version = ctx.current_version()
    samples: list[RolloutSample] = []
    for s in sampled:
        tokens = list(s.full_tokens)
        prompt_len = s.prompt_len
        comp_len = len(tokens) - prompt_len
        if comp_len <= 0:
            return None

        inf_lp = list(s.inference_logprobs or [])
        if len(inf_lp) != comp_len:
            logger.warning(
                "logprob length %d != completion length %d",
                len(inf_lp), comp_len,
            )
            return None

        # Flat contract: tokens[:prompt_len] are prompt (mask 0, logprob 0.0);
        # tokens[prompt_len:] are assistant-generated (mask 1, logprob from sampler).
        logprobs = [0.0] * prompt_len + inf_lp
        loss_mask = [0] * prompt_len + [1] * comp_len

        samples.append(RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            loss_mask=loss_mask,
            reward=reward_fn(s.text, row),
            versions=[version] * len(tokens),
            finish_reason=s.finish_reason,
            text=s.text,
        ))

    rollout = Rollout(
        samples=samples,
        row_meta={"ground_truth": row.get("ground_truth", "")},
    )
    _dump_rollout(rollout, version)
    return rollout


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
    )
    main(cfg, rollout_fn=rollout_fn, dynamic_filter_fn=should_accept)
