"""Builds the ``sample_fn`` the async loop consumes.

Dispatches among the three rollout regimes exposed by
:mod:`training.recipes.rl_loop_async`:

1. ``reward_fn(completion, row) -> float`` — single-turn shortcut.  The
   factory wraps it into a :class:`SingleTurnEnv` so the env path is used
   under the hood.
2. ``env_builder(row) -> MessageEnv`` — any single- or multi-turn env.
3. ``rollout_source(row, *, n) -> list[Trajectory] | None`` — user brings
   their own rollouts (remote agent, pre-recorded data, LLM judge output).

All three paths converge on :func:`trajectories_to_prompt_group`, which
computes advantages and packs the training datums.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Coroutine

from training.utils.rl.env import Trajectory
from training.utils.rl.env_adapters import wrap_reward_fn
from training.utils.rl.losses import PromptGroup
from training.utils.rl.rollout_builder import trajectories_to_prompt_group
from training.utils.rl.rollout_runner import run_env_group_to_trajectories

logger = logging.getLogger(__name__)

__all__ = [
    "RewardFn",
    "EnvBuilder",
    "RolloutSource",
    "build_sample_fn",
    "validate_rollout_regime",
]


RewardFn = Callable[[str, dict], "float | Awaitable[float]"]
EnvBuilder = Callable[[dict], Any]
RolloutSource = Callable[..., "list[Trajectory] | Awaitable[list[Trajectory] | None] | None"]


def validate_rollout_regime(
    *,
    reward_fn: RewardFn | None,
    env_builder: EnvBuilder | None,
    rollout_source: RolloutSource | None,
) -> None:
    """Ensure exactly one of the three rollout entry points is set."""
    provided = [
        name for name, val in (
            ("reward_fn", reward_fn),
            ("env_builder", env_builder),
            ("rollout_source", rollout_source),
        ) if val is not None
    ]
    if len(provided) == 0:
        raise ValueError(
            "rl_loop_async requires one of: reward_fn, env_builder, rollout_source."
        )
    if len(provided) > 1:
        raise ValueError(
            f"rl_loop_async accepts exactly one of reward_fn / env_builder / "
            f"rollout_source, got: {', '.join(provided)}."
        )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def build_sample_fn(
    *,
    reward_fn: RewardFn | None = None,
    env_builder: EnvBuilder | None = None,
    rollout_source: RolloutSource | None = None,
    group_reward_fn: Callable[..., Any] | None = None,
    sampler: Any,
    completions_per_prompt: int,
    sample_kwargs: dict[str, Any],
    max_turns: int = 1,
    tokenizer: Any = None,
    inference_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    router_replay: bool = False,
    router_replay_completion_only: bool = False,
    keep_trajectory_logs: bool = False,
) -> Callable[[dict], Coroutine[Any, Any, PromptGroup | None]]:
    """Return ``async def sample_fn(row) -> PromptGroup | None``."""
    validate_rollout_regime(
        reward_fn=reward_fn, env_builder=env_builder, rollout_source=rollout_source,
    )

    if reward_fn is not None:
        effective_env_builder: EnvBuilder | None = wrap_reward_fn(reward_fn)
    else:
        effective_env_builder = env_builder

    async def _sample_via_envs(row: dict) -> PromptGroup | None:
        assert effective_env_builder is not None
        envs = [effective_env_builder(row) for _ in range(completions_per_prompt)]
        trajectories = await run_env_group_to_trajectories(
            envs,
            sampler,
            completions_per_prompt=completions_per_prompt,
            max_turns=max_turns,
            sample_kwargs=sample_kwargs,
        )
        if trajectories is None:
            return None
        if group_reward_fn is not None:
            deltas = await _maybe_await(group_reward_fn(trajectories, row))
            if deltas is not None:
                for traj, delta in zip(trajectories, deltas):
                    traj.add_turn_reward(float(delta))
        return trajectories_to_prompt_group(
            trajectories,
            need_reference=True,
            tokenizer=tokenizer,
            inference_url=inference_url,
            api_key=api_key,
            model=model,
            router_replay=router_replay,
            router_replay_completion_only=router_replay_completion_only,
            keep_trajectory_logs=keep_trajectory_logs,
            row_meta={"ground_truth": row.get("ground_truth", "")} if keep_trajectory_logs else None,
        )

    async def _sample_via_rollout_source(row: dict) -> PromptGroup | None:
        assert rollout_source is not None
        try:
            raw = rollout_source(row, n=completions_per_prompt)
        except TypeError:
            # Rollout source signatures that don't accept `n` keyword.
            raw = rollout_source(row)
        trajectories = await _maybe_await(raw)
        if not trajectories:
            return None
        if group_reward_fn is not None:
            deltas = await _maybe_await(group_reward_fn(trajectories, row))
            if deltas is not None:
                for traj, delta in zip(trajectories, deltas):
                    traj.add_turn_reward(float(delta))
        return trajectories_to_prompt_group(
            trajectories,
            need_reference=True,
            tokenizer=tokenizer,
            inference_url=inference_url,
            api_key=api_key,
            model=model,
            router_replay=router_replay,
            router_replay_completion_only=router_replay_completion_only,
            keep_trajectory_logs=keep_trajectory_logs,
            row_meta={"ground_truth": row.get("ground_truth", "")} if keep_trajectory_logs else None,
        )

    if rollout_source is not None:
        return _sample_via_rollout_source
    return _sample_via_envs
