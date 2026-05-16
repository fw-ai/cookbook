"""Agent Lightning protocol-level rollout example.

This example shows the intended integration boundary:

    agent code / tracer -> triplets -> cookbook RolloutSample

It does not import Agent Lightning or start its store/runners.  In a real
integration, replace ``make_triplet_provider`` with a function that executes
your Agent Lightning rollout and returns its trace triplets.  The triplets only
need the protocol fields consumed by ``make_agent_lightning_rollout_fn``:

* ``prompt.token_ids``
* ``response.token_ids``
* ``response.logprobs``
* ``reward``
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

from training.examples.rl.vanilla_sampler import build_deployment_sampler
from training.utils.rl.rollout import (
    InferenceCall,
    RolloutSample,
    TrajectoryAssembler,
    pack_payload_to_sample,
)
from training.utils.rl.rollout.service import Role

if TYPE_CHECKING:
    from training.recipes.async_rl_loop import RolloutFn, RolloutSetup

logger = logging.getLogger(__name__)

TripletProvider = Callable[[Any], Sequence[Any] | Awaitable[Sequence[Any]]]


def make_rollout_fn(setup: "RolloutSetup") -> "RolloutFn":
    return make_agent_lightning_rollout_fn(
        make_triplet_provider(setup),
        tokenizer_id=setup.tokenizer_id,
    )


def make_triplet_provider(setup: "RolloutSetup"):
    """Return a provider shaped like an Agent Lightning trace adapter.

    The implementation below is deliberately tiny: it samples once from the
    cookbook deployment and wraps that call as one Agent-Lightning-like triplet.
    If your agent already runs under Agent Lightning, keep the returned triplet
    shape and swap this function's internals for your trace/span adapter.
    """
    sampler = build_deployment_sampler(setup)
    sample_kwargs = dict(setup.sample_kwargs)

    async def provider(sample_prompt: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_token_ids = list(sample_prompt.get("prompt_token_ids") or [])
        if not prompt_token_ids:
            return []

        completions = await sampler.sample_with_prompt_tokens(
            prompt_token_ids,
            n=1,
            **sample_kwargs,
        )
        if not completions:
            return []

        completion = completions[0]
        prompt_len = int(completion.prompt_len)
        full_tokens = list(completion.full_tokens)
        output_tokens = full_tokens[prompt_len:]
        output_logprobs = list(completion.inference_logprobs or [])

        if getattr(completion, "logprobs_echoed", False) and len(output_logprobs) == len(full_tokens):
            output_logprobs = output_logprobs[prompt_len:]
        if not output_tokens or len(output_logprobs) != len(output_tokens):
            return []

        return [
            {
                "prompt": {"token_ids": prompt_token_ids},
                "response": {
                    "token_ids": output_tokens,
                    "logprobs": output_logprobs,
                    "finish_reason": getattr(completion, "finish_reason", "stop"),
                },
                # Real Agent Lightning integrations usually infer this from
                # the final reward span.  This toy row carries it directly.
                "reward": float(sample_prompt.get("reward", 0.0)),
            }
        ]

    return provider


def make_agent_lightning_rollout_fn(
    triplet_provider: TripletProvider,
    *,
    tokenizer_id: str | None = None,
    role_before: Role = "user",
    swallow_exceptions: bool = True,
):
    """Build ``rollout_fn(row) -> RolloutSample | None`` from triplets.

    This helper lives in the example on purpose: it demonstrates the protocol
    shape without adding a supported cookbook API surface.
    """

    async def rollout_fn(row: Any) -> RolloutSample | None:
        try:
            triplets = await _maybe_await(triplet_provider(row))
            return await agent_lightning_triplets_to_sample(
                list(triplets),
                tokenizer_id=tokenizer_id,
                role_before=role_before,
            )
        except Exception:
            if not swallow_exceptions:
                raise
            logger.exception("Agent Lightning protocol example failed; dropping sample")
            return None

    return rollout_fn


async def agent_lightning_triplets_to_sample(
    triplets: Sequence[Any],
    *,
    tokenizer_id: str | None = None,
    total_reward: float | None = None,
    role_before: Role = "user",
) -> RolloutSample:
    payload = agent_lightning_triplets_to_payload(
        triplets,
        tokenizer_id=tokenizer_id,
        total_reward=total_reward,
        role_before=role_before,
    )
    return await pack_payload_to_sample(payload, tokenizer_id=tokenizer_id)


def agent_lightning_triplets_to_payload(
    triplets: Sequence[Any],
    *,
    tokenizer_id: str | None = None,
    total_reward: float | None = None,
    role_before: Role = "user",
):
    if not triplets:
        raise ValueError("Agent Lightning triplet list is empty")

    assembler = TrajectoryAssembler(tokenizer_id=tokenizer_id)
    inferred_reward: float | None = None

    for triplet in triplets:
        prompt = _mapping(_field(triplet, "prompt"), "triplet.prompt")
        response = _mapping(_field(triplet, "response"), "triplet.response")
        assembler.add_call(
            InferenceCall(
                input_tokens=_token_ids(prompt, "triplet.prompt"),
                output_tokens=_token_ids(response, "triplet.response"),
                output_logprobs=_response_logprobs(response),
                finish_reason=str(_field(response, "finish_reason", default="stop") or "stop"),
            ),
            role_before=role_before,
        )
        reward = _field(triplet, "reward", default=None)
        if reward is not None:
            inferred_reward = float(reward)

    return assembler.to_payload(
        total_reward=float(total_reward) if total_reward is not None else inferred_reward,
    )


def _mapping(value: Any, name: str) -> Any:
    if not isinstance(value, dict) and not hasattr(value, "__dict__"):
        raise TypeError(f"{name} must be dict-like or object-like, got {type(value).__name__}")
    return value


def _field(value: Any, name: str, *, default: Any = ...):
    if isinstance(value, dict):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)

    if default is not ...:
        return default
    raise ValueError(f"missing required field {name!r} on {type(value).__name__}")


def _token_ids(value: Any, name: str) -> list[int]:
    token_ids = _field(value, "token_ids", default=None)
    if token_ids is None:
        token_ids = _field(value, "tokens", default=None)
    if not token_ids:
        raise ValueError(f"{name} is missing non-empty token_ids")
    return [int(t) for t in token_ids]


def _response_logprobs(response: Any) -> list[float]:
    raw = _field(response, "logprobs", default=None)
    if raw is None:
        raw = _field(response, "token_logprobs", default=None)
    if raw is None:
        raise ValueError("triplet.response is missing logprobs")

    if isinstance(raw, dict):
        raw = raw.get("token_logprobs") or raw.get("content") or raw.get("logprobs")

    values = []
    for item in raw or []:
        if isinstance(item, dict):
            item = item.get("logprob")
        elif hasattr(item, "logprob"):
            item = item.logprob
        values.append(float(item) if item is not None else 0.0)
    return values


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
