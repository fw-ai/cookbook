#!/usr/bin/env python3
"""GSM8K single-turn async RL -- canonical ``rollout_fn`` example.

Shows the common pattern: a pure ``reward_fn(completion, row) -> float``
paired with a single-turn rollout closure that calls the cookbook's
``DeploymentSampler`` and hands back a :class:`Trajectory`.  All three of
the things the recipe itself doesn't know about -- grading, zero-variance
filtering, and per-step trajectory dumping -- live in this file.

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
from training.utils.rl.trajectory import CompletionSegment, Trajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward + filter -- user-owned, not the cookbook's concern
# ---------------------------------------------------------------------------


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
    """Reject zero-variance groups (GRPO would assign zero advantage)."""
    return len(set(pg.rewards)) > 1


# ---------------------------------------------------------------------------
# Optional trajectory dumping -- also user-owned
# ---------------------------------------------------------------------------

TRAJECTORY_DIR: str | None = os.environ.get("TRAJECTORY_DIR")


def _dump_trajectory(traj: Trajectory, version: int) -> None:
    if not TRAJECTORY_DIR:
        return
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    path = os.path.join(TRAJECTORY_DIR, f"version_{version:04d}.jsonl")
    with open(path, "a") as f:
        for ci, segments in enumerate(traj.completions):
            f.write(json.dumps({
                "version": version,
                "completion_index": ci,
                "reward": traj.rewards[ci],
                "completion_text": "".join(s.text for s in segments),
                "finish_reason": segments[-1].finish_reason,
                "ground_truth": (traj.row_meta or {}).get("ground_truth"),
            }) + "\n")


# ---------------------------------------------------------------------------
# The one extension point
# ---------------------------------------------------------------------------


async def rollout_fn(row: dict, ctx: RolloutContext) -> Trajectory | None:
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
    prompt_len = sampled[0].prompt_len
    prompt_tokens = list(sampled[0].full_tokens[:prompt_len])

    completions: list[list[CompletionSegment]] = []
    rewards: list[float] = []
    for s in sampled:
        completion_tokens = list(s.full_tokens[prompt_len:])
        inf_lp = list(s.inference_logprobs or [])
        if len(inf_lp) != len(completion_tokens):
            logger.warning(
                "logprob length mismatch (got %d, expected %d); dropping row",
                len(inf_lp), len(completion_tokens),
            )
            return None
        completions.append([CompletionSegment(
            tokens=completion_tokens,
            inference_logprobs=inf_lp,
            version=version,
            finish_reason=s.finish_reason,
            text=s.text,
        )])
        rewards.append(reward_fn(s.text, row))

    traj = Trajectory(
        prompt_tokens=prompt_tokens,
        completions=completions,
        rewards=rewards,
        prompt_messages=messages,
        row_meta={"ground_truth": row.get("ground_truth", "")},
    )
    _dump_trajectory(traj, version)
    return traj


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
