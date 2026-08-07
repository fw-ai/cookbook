"""Sampling and packing helpers shared by token-exact agent rollouts."""

from __future__ import annotations

from typing import Any

from training.utils.rl.agent.trajectory import TokenSegment
from training.utils.rl.rollout.types import RolloutSample


def completion_values(
    completion: Any,
    *,
    attribute: str,
    output_len: int,
) -> list[float] | None:
    """Return completion-only numeric values from an SDK completion.

    Sampling and raw inference logprobs may be completion-only or echoed over
    the full prompt and completion. A missing optional field returns ``None``;
    a present but misaligned field is invalid trajectory data.
    """
    raw_values = getattr(completion, attribute, None)
    if raw_values is None:
        return None

    values = list(raw_values)
    prompt_len = int(completion.prompt_len)
    full_len = len(completion.full_tokens)
    if getattr(completion, "logprobs_echoed", False):
        if len(values) == full_len:
            values = values[prompt_len:]
        elif len(values) == max(0, full_len - 1):
            values = values[max(0, prompt_len - 1) :]

    if len(values) != output_len or any(value is None for value in values):
        raise ValueError(
            f"completion {attribute} are misaligned "
            f"({len(values)} values for {output_len} output tokens)"
        )
    return [float(value) for value in values]


def completion_routes(
    completion: Any,
    *,
    output_len: int,
) -> list[str] | None:
    """Return completion-only routing matrices from an SDK completion."""
    raw_routes = getattr(completion, "routing_matrices", None)
    if raw_routes is None:
        return None

    routes = list(raw_routes)
    prompt_len = int(completion.prompt_len)
    full_len = len(completion.full_tokens)
    if getattr(completion, "logprobs_echoed", False):
        if len(routes) == full_len:
            routes = routes[prompt_len:]
        elif len(routes) == max(0, full_len - 1):
            routes = routes[max(0, prompt_len - 1) :]

    if len(routes) != output_len or any(route is None for route in routes):
        raise ValueError(
            "completion routing matrices are misaligned "
            f"({len(routes)} values for {output_len} output tokens)"
        )
    return [str(route) for route in routes]


def token_segment_to_sample(
    segment: TokenSegment,
    *,
    reward: float,
) -> RolloutSample:
    """Pack one exact agent segment into the async RL rollout contract."""
    response_len = len(segment.response_ids)
    if not (len(segment.loss_mask) == len(segment.rollout_log_probs) == response_len):
        raise ValueError("agent segment token, mask, and logprob lengths must match")
    if (
        segment.rollout_raw_log_probs is not None
        and len(segment.rollout_raw_log_probs) != response_len
    ):
        raise ValueError("agent segment raw logprobs are misaligned")
    if (
        segment.routing_matrices is not None
        and len(segment.routing_matrices) != response_len
    ):
        raise ValueError("agent segment routing matrices are misaligned")

    prompt_len = len(segment.prompt_ids)
    raw_logprobs = (
        [0.0] * prompt_len + segment.rollout_raw_log_probs
        if segment.rollout_raw_log_probs is not None
        else None
    )
    routing_matrices = (
        [""] * max(0, prompt_len - 1) + segment.routing_matrices
        if segment.routing_matrices is not None
        else None
    )
    return RolloutSample(
        tokens=[*segment.prompt_ids, *segment.response_ids],
        logprobs=[0.0] * prompt_len + segment.rollout_log_probs,
        loss_mask=[0] * prompt_len + segment.loss_mask,
        reward=float(reward),
        finish_reason=str(segment.metadata.get("finish_reason") or "stop"),
        text=str(segment.metadata.get("text") or ""),
        raw_logprobs=raw_logprobs,
        # Use trainer input coordinates directly. Inferring a legacy
        # completion suffix from the first trainable token is ambiguous after
        # a short response rewrite masks an earlier generated span.
        routing_matrices=routing_matrices,
    )


__all__ = [
    "completion_routes",
    "completion_values",
    "token_segment_to_sample",
]
