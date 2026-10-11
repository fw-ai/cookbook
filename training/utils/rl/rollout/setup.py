"""The rollout seam shared by the RL recipes.

A recipe builds one :class:`RolloutSetup` at startup and calls the user's
``rollout_fn_factory(setup) -> rollout_fn`` once. ``rollout_fn(row)`` plays
one trajectory and returns a :class:`RolloutRun`, or ``None`` to drop it.
Task settings belong to the factory, not the setup: close over a typed config
(``functools.partial(make_rollout_fn, task=task_config)``). ``recipes/rl_loop.py``
and ``recipes/async_rl_loop.py`` both take this seam; see
``skills/fireworks-training/references/rl-async.md``.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from training.utils.rl.losses import PromptGroup
from training.utils.rl.rollout.types import RolloutRun

__all__ = [
    "DynamicFilterFn",
    "RolloutFn",
    "RolloutFnFactory",
    "RolloutSetup",
    "call_rollout_fn",
    "default_sample_kwargs",
]


@dataclass
class RolloutSetup:
    """Dependencies the recipe hands the rollout factory once at startup.

    ``sampler`` is borrowed from the recipe: share it across trajectories and
    never close it. ``inference_base_url``, ``api_key`` and ``model`` describe
    the same deployment for rollouts that talk to it over HTTP themselves
    (Eval Protocol processors, Harbor sidecars); ``model`` changes when a
    serverless recipe publishes a new sampler snapshot.
    """

    sampler: Any
    tokenizer: Any
    tokenizer_id: str
    sample_kwargs: dict[str, Any]
    completions_per_prompt: int
    inference_base_url: str
    api_key: str
    model: str


RolloutFn = Callable[..., Awaitable[RolloutRun | None]]
RolloutFnFactory = Callable[[RolloutSetup], RolloutFn]
DynamicFilterFn = Callable[[PromptGroup], bool]
"""Return ``True`` to train on a prompt group (e.g. ``has_reward_variance``)."""


def default_sample_kwargs(
    *,
    max_tokens: int,
    temperature: float,
    max_seq_len: int | None,
    http_timeout: float,
    router_replay: bool,
) -> dict[str, Any]:
    """Sampling kwargs every RL recipe uses for policy rollouts.

    ``top_p``/``top_k`` are explicit: otherwise serving applies the model's
    ``generation_config.json`` defaults (e.g. Qwen3.5: ``top_k=20``,
    ``top_p=0.95``), which truncate rollouts while the trainer normalizes over
    the full vocabulary. ``max_seq_len`` is the single prompt-plus-output
    limit. MoE policies (``router_replay``) also return the expert routes of
    the generated tokens for Router Replay (R3).
    """
    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": 0,
        "max_seq_len": max_seq_len,
        "http_timeout": http_timeout,
        "logprobs": True,
    }
    if router_replay:
        kwargs.update(include_routing_matrix=True, echo=False)
    return kwargs


ROLLOUT_CONTEXT_KWARGS = frozenset(
    {
        "cursor_index",
        "row_index",
        "epoch",
        "rollout_idx",
        "sample_index",
        "end_of_epoch",
        "evaluation",
    }
)
"""Context a recipe offers each ``rollout_fn`` call; it receives the names it accepts."""


def rollout_context_names(rollout_fn: RolloutFn) -> frozenset[str]:
    """Context keyword names ``rollout_fn`` accepts (all of them for ``**kwargs``)."""
    parameters = inspect.signature(rollout_fn).parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return ROLLOUT_CONTEXT_KWARGS
    return ROLLOUT_CONTEXT_KWARGS & {param.name for param in parameters}


def call_rollout_fn(
    rollout_fn: RolloutFn, row: dict, context: dict[str, Any], names: frozenset[str]
) -> Awaitable[RolloutRun | None]:
    """Call ``rollout_fn(row)`` with the subset of ``context`` it accepts."""
    return rollout_fn(row, **{key: context[key] for key in names if key in context})
