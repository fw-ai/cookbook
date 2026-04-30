"""Shared per-turn renderer-backed sampling step for the multi-turn examples.

PRIVATE — leading underscore in the module name signals "do not import
from outside the examples tree."  This is NOT a `utils/rl/` framework
helper and is NOT a Protocol export.  The two canonical example rollouts
(``multi_turn_minimal_renderer/rollout.py`` and
``multi_turn_tool/rollout.py``) call into this module so the renderer +
sampler + assembler core loop is single-sourced (per AC-4 sub-rule).

Each call performs ONE engine round trip:

    extension-property guard (when turns_seen==1)
        -> renderer.build_generation_prompt
        -> model_input_to_token_ids
        -> sample_with_prompt_tokens
        -> TrajectoryAssembler.add_call(InferenceCall(...))
        -> renderer.parse_response

Tool execution and env stepping (the parts that *differ* between the two
example rollouts) stay in the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List

from training.utils.rl.rollout import (
    InferenceCall,
    RolloutSample,
    TrajectoryAssembler,
    model_input_to_token_ids,
)


logger = logging.getLogger(__name__)


__all__ = [
    "TurnOutcome",
    "ExtensionPropertyError",
    "pack_assembled_to_sample",
    "renderer_turn_step",
]


class ExtensionPropertyError(RuntimeError):
    """AC-3 guard trip — multi-turn flatten not supported in current renderer mode.

    Intentionally NOT exported from ``utils/rl/`` as a typed error class.
    Lives here as a private detail of the example tree.
    """


@dataclass(frozen=True)
class TurnOutcome:
    """One renderer-backed turn's outcome handed back to the example caller."""

    parsed_message: Any
    parse_success: bool
    finish_reason: str
    out_tokens: List[int]
    """Assistant tokens for THIS turn (already assembled)."""
    out_logprobs: List[float]
    """Per-token assistant logprobs for THIS turn."""
    dropped: bool = False
    """When True, the caller should ``return None`` from its rollout_fn:
    the sampler returned no completions, the assistant tokens were
    empty, or the inference logprobs were missing / misaligned.  Note
    that ``PrefixMismatch`` from the assembler is NOT caught and is
    NOT a dropped state — it propagates so renderer / env drift fails
    loud rather than masquerading as a benign sample miss.  The other
    fields are unset / empty when ``dropped`` is True."""


async def renderer_turn_step(
    *,
    messages: List[Any],
    renderer: Any,
    sample_with_prompt_tokens: Callable[..., Awaitable[List[Any]]],
    assembler: TrajectoryAssembler,
    turns_seen: int,
    sample_kwargs: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    turn_version: int | None = None,
) -> TurnOutcome:
    """Run one renderer-backed turn through the shared inner loop.

    Returns a :class:`TurnOutcome`.  When ``outcome.dropped is True`` the
    caller's ``rollout_fn`` should ``return None`` (no sample emitted).
    Otherwise the caller decides what to do with ``parsed_message`` /
    ``parse_success`` (env.step, env.execute for tool calls, etc.).

    Raises :class:`ExtensionPropertyError` when ``turns_seen == 1`` and
    ``renderer.has_extension_property`` is False — the AC-3 guard.
    """
    if turns_seen == 1 and not renderer.has_extension_property:
        raise ExtensionPropertyError(
            f"Renderer {type(renderer).__name__} (name="
            f"{getattr(renderer, 'name', None)!r}) has "
            "has_extension_property=False in its current mode "
            "(e.g. Qwen3 with strip_thinking_from_history=True). "
            "Multi-turn flattening is not supported for this renderer "
            "configuration. Reconfigure the renderer or pick one whose "
            "current mode preserves the extension property."
        )

    model_input = renderer.build_generation_prompt(messages)
    prompt_token_ids = model_input_to_token_ids(model_input)

    call_kwargs: dict[str, Any] = dict(sample_kwargs or {})
    call_kwargs["n"] = 1
    call_kwargs["stop"] = renderer.get_stop_sequences()
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens

    completions = await sample_with_prompt_tokens(prompt_token_ids, **call_kwargs)
    if not completions:
        return _dropped()
    c = completions[0]
    prompt_len = int(c.prompt_len)
    out_tokens = list(c.full_tokens[prompt_len:])
    if not out_tokens:
        return _dropped()
    # Reject turns whose sampler did not return per-token
    # ``inference_logprobs``.  Fabricating zeros here would silently
    # corrupt PPO/GRPO: every assistant token would look like behavior
    # probability ``exp(0) = 1``, breaking importance ratios and KL.
    # Mirrors the strict validation already enforced by
    # ``single_turn_renderer_rollout`` and ``pack_payload_to_sample``.
    out_logprobs_raw = getattr(c, "inference_logprobs", None)
    if out_logprobs_raw is None:
        logger.warning(
            "renderer_turn_step: dropping turn with no inference_logprobs. "
            "Configure the sampler with logprobs=True so PPO/GRPO sees real "
            "behavior-policy probabilities."
        )
        return _dropped()
    out_logprobs = list(out_logprobs_raw)
    # Mirror ``single_turn_renderer_rollout`` and ``rl_loop``: when
    # the caller sets ``echo=True`` in ``sample_kwargs`` the sampler
    # returns logprobs for the full ``prompt + completion`` span.
    # Slice off the prompt prefix so the per-token alignment matches
    # the assistant tokens.  Without this, every multi-turn turn
    # under echoed sampling was dropped as "misaligned" and the
    # rollout yielded no trainable samples.
    if getattr(c, "logprobs_echoed", False) and len(out_logprobs) == prompt_len + len(out_tokens):
        out_logprobs = out_logprobs[prompt_len:]
    if len(out_logprobs) != len(out_tokens):
        logger.warning(
            "renderer_turn_step: dropping turn with misaligned logprobs "
            "(got %d, expected %d for assistant tokens).",
            len(out_logprobs), len(out_tokens),
        )
        return _dropped()

    finish_reason = getattr(c, "finish_reason", "stop")
    output_versions: List[int] | None = None
    if turn_version is not None:
        output_versions = [int(turn_version)] * len(out_tokens)
    # NOTE: do NOT catch ``PrefixMismatch`` here.  ``TrajectoryAssembler``
    # raises it when the engine's ``input_tokens`` for a new turn diverge
    # from the previously-assembled prefix — that signals a real
    # integration bug (renderer / env re-rendered history differently
    # between turns, or environment tokens were mis-injected).  Wrapping
    # it in ``_dropped()`` would convert a hard token-native invariant
    # violation into an ordinary sample miss and let training churn
    # through every prompt with no actionable signal.  Letting the
    # exception propagate fails loud so the integration gets fixed.
    assembler.add_call(InferenceCall(
        input_tokens=prompt_token_ids,
        output_tokens=out_tokens,
        output_logprobs=out_logprobs,
        finish_reason=finish_reason,
        output_versions=output_versions,
    ))

    parsed_message, parse_success = renderer.parse_response(out_tokens)
    return TurnOutcome(
        parsed_message=parsed_message,
        parse_success=bool(parse_success),
        finish_reason=finish_reason,
        out_tokens=out_tokens,
        out_logprobs=out_logprobs,
    )


def pack_assembled_to_sample(
    assembler: TrajectoryAssembler,
    *,
    total_reward: float,
    finish_reason: str | None = None,
    text: str = "",
) -> RolloutSample:
    """Pack a multi-turn assembled trajectory into a :class:`RolloutSample`.

    Reads ``(tokens, logprobs, loss_mask, versions)`` from
    :meth:`TrajectoryAssembler.to_flat`, preserving per-call
    ``output_versions`` on assistant tokens (and ``-1`` on non-assistant
    gap tokens) so decoupled-IS / per-token version-aware losses see the
    real call-time deployment version.

    Distinct from :func:`pack_payload_to_sample` (which overwrites
    ``versions`` with one terminal scalar).  Use this packer for
    multi-turn renderer-backed examples where each engine call may have
    happened on a different deployment version.
    """
    tokens, logprobs, loss_mask, versions = assembler.to_flat()
    return RolloutSample(
        tokens=tokens,
        logprobs=logprobs,
        loss_mask=loss_mask,
        reward=float(total_reward),
        versions=versions,
        finish_reason=finish_reason or "stop",
        text=text,
    )


def _dropped() -> TurnOutcome:
    return TurnOutcome(
        parsed_message=None,
        parse_success=False,
        finish_reason="dropped",
        out_tokens=[],
        out_logprobs=[],
        dropped=True,
    )
