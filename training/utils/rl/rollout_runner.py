"""Run :class:`MessageEnv` groups against a sampler to produce trajectories.

The first turn is issued as a single ``sample_with_tokens(n=N)`` call so the
group-batched API shape matches the sync loop.  On later turns each env's
conversation has diverged, so the runner falls back to per-env sampling fired
concurrently via ``asyncio.gather``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from training.utils.rl.env import (
    Message,
    MessageEnv,
    MessageStepResult,
    Trajectory,
    Transition,
)

logger = logging.getLogger(__name__)

__all__ = [
    "run_env_group_to_trajectories",
    "run_env_to_trajectory",
]


def _completion_to_assistant_message(text: str) -> Message:
    return {"role": "assistant", "content": text}


def _transition_from_sample(
    sampled: Any,
    step_result: MessageStepResult,
    *,
    assistant_message: Message,
) -> Transition:
    full_tokens = list(sampled.full_tokens)
    prompt_len = int(sampled.prompt_len)
    prompt_tokens = full_tokens[:prompt_len]
    completion_tokens = full_tokens[prompt_len:]

    inf_lp = getattr(sampled, "inference_logprobs", None)
    if inf_lp is not None:
        # DeploymentSampler returns a flat list aligned with the generated
        # tokens when echo=False; pass through unchanged.
        inf_lp = list(inf_lp)

    return Transition(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        completion_text=getattr(sampled, "text", "") or "",
        inference_logprobs=inf_lp,
        assistant_message=assistant_message,
        reward=float(step_result.reward),
        episode_done=bool(step_result.episode_done),
        finish_reason=getattr(sampled, "finish_reason", "stop") or "stop",
        is_reward_valid=bool(step_result.is_reward_valid),
        metrics=dict(step_result.metrics),
        routing_matrices=getattr(sampled, "routing_matrices", None),
    )


async def run_env_to_trajectory(
    env: MessageEnv,
    sampler: Any,
    *,
    max_turns: int = 1,
    sample_kwargs: dict[str, Any] | None = None,
) -> Trajectory | None:
    """Run a single env to completion, one turn at a time.

    Returns ``None`` if the sampler returns no completions at any point.
    """
    sample_kwargs = dict(sample_kwargs or {})
    messages = list(await env.initial_messages())
    if not messages:
        return None

    trajectory = Trajectory()
    for _ in range(max_turns):
        sampled = await sampler.sample_with_tokens(messages=messages, n=1, **sample_kwargs)
        if not sampled:
            return None
        s = sampled[0]
        assistant_message = _completion_to_assistant_message(getattr(s, "text", "") or "")
        step_result = await env.step(assistant_message)
        trajectory.transitions.append(
            _transition_from_sample(s, step_result, assistant_message=assistant_message)
        )
        if step_result.episode_done:
            return trajectory
        messages = messages + [assistant_message, *step_result.next_messages]

    return trajectory


async def run_env_group_to_trajectories(
    envs: list[MessageEnv],
    sampler: Any,
    *,
    completions_per_prompt: int,
    max_turns: int = 1,
    sample_kwargs: dict[str, Any] | None = None,
) -> list[Trajectory] | None:
    """Run ``len(envs)`` parallel rollouts and return one trajectory each.

    Preserves the sync loop's call shape: turn 1 is a single
    ``sample_with_tokens(n=N)`` call, so the deployment sees one batched request
    for N divergent completions.  Turn 2+ uses concurrent per-env calls because
    each env's conversation has diverged.

    Returns ``None`` if the first-turn sampler call produces fewer than ``N``
    completions (match sync loop drop-on-short behaviour).
    """
    if completions_per_prompt != len(envs):
        raise ValueError(
            f"completions_per_prompt ({completions_per_prompt}) must match len(envs) "
            f"({len(envs)})"
        )
    if completions_per_prompt < 1:
        raise ValueError("completions_per_prompt must be >= 1")

    sample_kwargs = dict(sample_kwargs or {})

    # All envs in a group share the same row and therefore the same
    # initial_messages -- use the first env's view as the group prompt.
    initial = list(await envs[0].initial_messages())
    if not initial:
        return None

    sampled = await sampler.sample_with_tokens(
        messages=initial, n=completions_per_prompt, **sample_kwargs,
    )
    if not sampled or len(sampled) < completions_per_prompt:
        return None

    trajectories: list[Trajectory] = [Trajectory() for _ in envs]
    per_env_messages: list[list[Message]] = [list(initial) for _ in envs]
    alive: list[bool] = [True] * len(envs)

    # -- Turn 1 (group-batched) ---------------------------------------------
    step_results = await asyncio.gather(
        *(
            env.step(_completion_to_assistant_message(getattr(s, "text", "") or ""))
            for env, s in zip(envs, sampled)
        )
    )
    for i, (env, s, step_result) in enumerate(zip(envs, sampled, step_results)):
        assistant_message = _completion_to_assistant_message(getattr(s, "text", "") or "")
        trajectories[i].transitions.append(
            _transition_from_sample(s, step_result, assistant_message=assistant_message)
        )
        if step_result.episode_done:
            alive[i] = False
        else:
            per_env_messages[i] = per_env_messages[i] + [
                assistant_message,
                *step_result.next_messages,
            ]

    if max_turns <= 1 or not any(alive):
        return trajectories

    # -- Turn 2+ (per-env, concurrent) --------------------------------------
    for _turn in range(1, max_turns):
        live_indices = [i for i, a in enumerate(alive) if a]
        if not live_indices:
            break

        samples = await asyncio.gather(
            *(
                sampler.sample_with_tokens(
                    messages=per_env_messages[i], n=1, **sample_kwargs,
                )
                for i in live_indices
            )
        )

        step_coros: list[Any] = []
        carry: list[tuple[int, Any, Message]] = []
        for idx, sampled_list in zip(live_indices, samples):
            if not sampled_list:
                alive[idx] = False
                continue
            s = sampled_list[0]
            assistant_message = _completion_to_assistant_message(getattr(s, "text", "") or "")
            step_coros.append(envs[idx].step(assistant_message))
            carry.append((idx, s, assistant_message))

        if not step_coros:
            break

        results = await asyncio.gather(*step_coros)
        for (idx, s, assistant_message), step_result in zip(carry, results):
            trajectories[idx].transitions.append(
                _transition_from_sample(s, step_result, assistant_message=assistant_message)
            )
            if step_result.episode_done:
                alive[idx] = False
            else:
                per_env_messages[idx] = per_env_messages[idx] + [
                    assistant_message,
                    *step_result.next_messages,
                ]

    return trajectories
