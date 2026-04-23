#!/usr/bin/env python3
"""Async RL example: bring your own ``rollout_source``.

Use this path when the cookbook's sampler is not the one producing the
completions -- typical cases are remote agent frameworks, pre-recorded
trajectories, or LLM-as-judge flows.  The rollout source hands back a
list of :class:`Trajectory` objects; the cookbook packs them into a
:class:`PromptGroup` and (if ``inference_logprobs`` was not provided)
recovers per-token logprobs via an ``echo=True`` prefill call.

For the example below the "remote agent" is just a stub -- replace
``_call_remote_agent`` with your real integration.

Usage::

    export FIREWORKS_API_KEY=...
    python -m training.examples.custom_env.rollout_source.train
"""

from __future__ import annotations

import os
from typing import Any

import transformers

from training.recipes.rl_loop_async import AsyncConfig, Config, main
from training.utils.rl import Trajectory, Transition, tokenize_chat_turn


def _call_remote_agent(prompt_messages: list[dict]) -> list[dict[str, Any]]:
    """Stub remote agent: returns a fixed completion + placeholder reward.

    Replace with HTTP call to your agent framework.  Return one dict per
    completion with keys ``text`` and ``reward``.
    """
    return [
        {"text": "<answer>42</answer>", "reward": 1.0},
        {"text": "<answer>7</answer>", "reward": 0.0},
        {"text": "<answer>100</answer>", "reward": 0.0},
        {"text": "<answer>42</answer>", "reward": 1.0},
    ]


def _build_rollout_source(tokenizer: Any):
    """Bind the tokenizer so we can build :class:`Transition` objects."""

    async def rollout_source(row: dict, *, n: int) -> list[Trajectory]:
        prompt_messages = row.get("messages") or [
            {"role": "user", "content": row.get("prompt", "")},
        ]
        completions = _call_remote_agent(prompt_messages)[:n]

        trajectories: list[Trajectory] = []
        for comp in completions:
            assistant_message = {"role": "assistant", "content": comp["text"]}
            prompt_tokens, completion_tokens = tokenize_chat_turn(
                prompt_messages, assistant_message, tokenizer,
            )
            transition = Transition(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                completion_text=comp["text"],
                inference_logprobs=None,  # cookbook will prefill-recover.
                assistant_message=assistant_message,
                reward=float(comp["reward"]),
                episode_done=True,
            )
            trajectories.append(Trajectory(transitions=[transition]))
        return trajectories

    return rollout_source


if __name__ == "__main__":
    rows = [
        {"messages": [{"role": "user", "content": "What's the answer?"}]}
        for _ in range(32)
    ]

    cfg = Config(
        log_path="/tmp/rl-async-rollout-source",
        base_model="accounts/fireworks/models/qwen3-8b",
        async_config=AsyncConfig(max_steps_off_policy=1, groups_per_batch=4),
        completions_per_prompt=4,
        max_rows=len(rows),
    )

    tokenizer_model = cfg.deployment.tokenizer_model or os.environ.get(
        "COOKBOOK_TOKENIZER_MODEL", "Qwen/Qwen3-1.7B",
    )
    cfg.deployment.tokenizer_model = tokenizer_model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_model, trust_remote_code=True,
    )

    main(cfg, rollout_source=_build_rollout_source(tokenizer), rows=rows)
