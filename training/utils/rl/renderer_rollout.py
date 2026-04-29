"""Renderer-backed RL rollout primitives.

This module exposes the two renderer-backed framework helpers and the
``ModelInput`` flattening adapter:

* :func:`single_turn_renderer_rollout` — single-turn helper that turns a
  renderer + a pre-tokenized sampling primitive into a flat
  :class:`~training.utils.rl.rollout.Rollout` whose tokens, logprobs, and loss
  mask are derived end-to-end from the renderer-built prompt and the
  sampler-returned assistant tokens.
* :func:`make_remote_rollout_fn` — drives a ``RolloutService.rollout(...)`` and
  packs payloads via the existing ``pack_payload_to_sample`` validator
  (token-native; rejects text-only payloads, missing assistant logprobs, or
  mismatched tokenizer ids).  Re-exported from
  :mod:`training.utils.rl.text_rollout`; the renderer is applied service-side,
  so this helper is renderer-name-agnostic by design.
* :func:`model_input_to_token_ids` — flatten a Tinker ``ModelInput`` from a
  renderer's ``build_generation_prompt(...)`` into ``list[int]``.  Multimodal
  chunks are rejected with :class:`MultimodalRenderingNotSupported`.

Multi-turn / tool flows are NOT framework helpers — they live in
``cookbook/training/examples/rl/multi_turn_minimal_renderer/rollout.py`` and
``cookbook/training/examples/rl/multi_turn_tool/rollout.py`` as concrete
``async def rollout_fn(row, ctx)`` rollout functions that users read and copy.

Boundary
--------

The renderer is consumed inside the rollout; it is not the trainer's data
contract.  The trainer remains slime-style: ``rollout_fn(row, ctx) -> Rollout
| None``.  This module is renderer-name-agnostic and never re-renders chat
templates client-side.

Parse-failure / truncation handling
-----------------------------------

Parse-failure handling is *not* a framework primitive.  This helper hands
``(parsed_message, parse_success)`` back to the user-supplied ``reward_fn``;
the caller chooses what to do (drop by returning ``None``, score zero by
returning ``0.0``, or branch on ``parse_success`` for custom behavior).
``finish_reason='length'`` flows through unchanged unless the user branches
on it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Protocol

import tinker

from training.utils.rl.rollout import Rollout, RolloutSample
from training.utils.rl.text_rollout import make_text_rollout_fn as make_remote_rollout_fn


logger = logging.getLogger(__name__)


__all__ = [
    "MultimodalRenderingNotSupported",
    "RolloutHelperInfo",
    "make_remote_rollout_fn",
    "model_input_to_token_ids",
    "renderer_helper_info",
    "single_turn_renderer_rollout",
]


class MultimodalRenderingNotSupported(RuntimeError):
    """Raised when a ``ModelInput`` carries non-text chunks.

    Renderer-backed RL rollouts are text-only in this iteration.  A multimodal
    chunk (image asset pointer, image bytes) reaching this adapter indicates a
    multimodal rollout flow that has not yet been scoped for RL training.
    """


def model_input_to_token_ids(model_input: tinker.ModelInput) -> List[int]:
    """Flatten a renderer ``ModelInput`` to ``list[int]``.

    Accepts only :class:`tinker.EncodedTextChunk` chunks.  Any other chunk
    type — image asset pointers or raw image bytes — raises
    :class:`MultimodalRenderingNotSupported`.
    """
    out: List[int] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            raise MultimodalRenderingNotSupported(
                f"chunk type {type(chunk).__name__} is not supported by "
                "renderer-backed RL rollouts"
            )
    return out


# A renderer-shaped protocol.  Avoids a hard import of the Tinker base class
# in this module's signature surface so tests can pass simple stubs without
# subclassing the heavy upstream class.  We do not export this — the helper
# accepts any object that responds to these methods.
class _RendererLike(Protocol):
    def build_generation_prompt(self, messages: List[Any]) -> tinker.ModelInput: ...
    def parse_response(self, tokens: List[int]) -> Any: ...
    def get_stop_sequences(self) -> List[Any]: ...


SampleWithPromptTokens = Callable[..., Awaitable[List[Any]]]
"""Callable matching :meth:`DeploymentSampler.sample_with_prompt_tokens`."""


MessageBuilder = Callable[[Any, Any], Awaitable[List[Any]]]
"""``async (row, ctx) -> messages`` — builds the seed conversation."""


RewardFn = Callable[[Any, Any, bool], Awaitable[Optional[float]]]
"""``async (row, parsed_message, parse_success) -> float | None``.

Return ``None`` to drop the completion (no sample emitted).  Return a float
to emit a sample with that reward.  ``parse_success`` is the second element
returned by the renderer's ``parse_response``.
"""


@dataclass(frozen=True)
class RolloutHelperInfo:
    """Read-only triage metadata for a renderer-backed rollout helper.

    Public, frozen dataclass exposed for logging / debugging.  Returned by
    :func:`renderer_helper_info` and attached to the
    ``single_turn_renderer_rollout.helper_info(...)`` accessor (when the
    helper is constructed with explicit ``tokenizer_id`` / ``max_tokens``
    metadata).  Never participates in computation; correctness lives in the
    helper's pipeline, not in this record.
    """

    tokenizer_id: str | None
    renderer_name: str | None
    stop_condition: List[Any] | None
    max_tokens: int | None


async def single_turn_renderer_rollout(
    row: Any,
    ctx: Any,
    *,
    renderer: _RendererLike,
    sample_with_prompt_tokens: SampleWithPromptTokens,
    message_builder: MessageBuilder,
    reward_fn: RewardFn,
    max_tokens: int | None = None,
    stop: List[str] | List[int] | None = None,
) -> Rollout | None:
    """Single-turn renderer-backed rollout.

    Builds messages via ``message_builder``, calls
    ``renderer.build_generation_prompt(...)``, flattens the resulting
    ``ModelInput`` via :func:`model_input_to_token_ids`, samples ``n``
    completions through ``sample_with_prompt_tokens`` (the SDK's
    pre-tokenized sampling primitive), and packs each completion into a
    :class:`RolloutSample` whose tokens / logprobs / loss-mask come straight
    from the renderer + sampler.  No chat-template re-rendering, no
    re-tokenization of decoded assistant text.

    ``stop`` defaults to ``renderer.get_stop_sequences()`` and preserves its
    ``list[str] | list[int]`` shape.  The user-supplied ``reward_fn``
    receives the renderer's parsed message and parse-success flag and
    returns ``None`` (drop), ``0.0`` (zero-reward sample), or any other
    float.  Multimodal prompts raise :class:`MultimodalRenderingNotSupported`
    via the adapter; the helper does not catch it.
    """
    messages = await message_builder(row, ctx)
    model_input = renderer.build_generation_prompt(messages)
    prompt_token_ids = model_input_to_token_ids(model_input)

    if stop is None:
        stop = renderer.get_stop_sequences()

    sample_kwargs: dict[str, Any] = dict(getattr(ctx, "sample_kwargs", {}) or {})
    n = int(getattr(ctx, "completions_per_prompt", 1))

    call_kwargs: dict[str, Any] = dict(sample_kwargs)
    call_kwargs["n"] = n
    call_kwargs["stop"] = stop
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens

    completions = await sample_with_prompt_tokens(prompt_token_ids, **call_kwargs)

    samples: List[RolloutSample] = []
    for c in completions:
        prompt_len = int(c.prompt_len)
        out_tokens: List[int] = list(c.full_tokens[prompt_len:])
        if not out_tokens:
            continue
        out_logprobs_raw = getattr(c, "inference_logprobs", None)
        # Reject completions whose sampler did not return per-token
        # ``inference_logprobs`` (e.g. an integration forgot to request
        # them).  Fabricating zeros here would silently corrupt PPO/GRPO:
        # the trainer would see every assistant token as having behavior
        # probability ``exp(0) = 1``, which throws off importance ratios
        # and KL terms.  Mirrors the strict validation in
        # ``extract_completion`` and ``pack_payload_to_sample`` — fail
        # loud at the rollout boundary rather than ship bogus data.
        if out_logprobs_raw is None:
            logger.warning(
                "single_turn_renderer_rollout: dropping completion with "
                "no inference_logprobs (got None).  Configure the sampler "
                "with logprobs=True so PPO/GRPO ratio/KL math sees real "
                "behavior-policy probabilities."
            )
            continue
        out_logprobs: List[float] = list(out_logprobs_raw)
        # When the caller passes ``echo=True`` in ``ctx.sample_kwargs``
        # the sampler returns logprobs for the full ``prompt + completion``
        # span, not just the assistant tokens.  Mirror the main RL loop
        # (``rl_loop.py``: ``echoed = getattr(s, "logprobs_echoed", False)``)
        # and slice off the prompt prefix instead of treating the
        # different length as a misalignment.
        if getattr(c, "logprobs_echoed", False) and len(out_logprobs) == prompt_len + len(out_tokens):
            out_logprobs = out_logprobs[prompt_len:]
        if len(out_logprobs) != len(out_tokens):
            logger.warning(
                "single_turn_renderer_rollout: dropping completion with "
                "misaligned logprobs (got %d, expected %d for assistant tokens).",
                len(out_logprobs), len(out_tokens),
            )
            continue

        parsed_message, parse_success = renderer.parse_response(out_tokens)
        reward = await reward_fn(row, parsed_message, bool(parse_success))
        if reward is None:
            continue

        samples.append(
            RolloutSample(
                tokens=list(prompt_token_ids) + out_tokens,
                logprobs=[0.0] * len(prompt_token_ids) + out_logprobs,
                loss_mask=[0] * len(prompt_token_ids) + [1] * len(out_tokens),
                reward=float(reward),
                finish_reason=getattr(c, "finish_reason", "stop"),
                text=getattr(c, "text", ""),
            )
        )

    return Rollout(samples=samples) if samples else None


def renderer_helper_info(
    renderer: _RendererLike,
    *,
    tokenizer_id: str | None = None,
    max_tokens: int | None = None,
) -> RolloutHelperInfo:
    """Build a public read-only triage record for a renderer-backed helper.

    Returns the four AC-8 triage fields — ``tokenizer_id``, ``renderer_name``
    (resolved from ``renderer.name`` or its class), ``stop_condition``
    (snapshot of ``renderer.get_stop_sequences()``), ``max_tokens`` — as a
    frozen :class:`RolloutHelperInfo`.  Pure data; no side effects.

    Use this from wiring code that wants to log helper configuration before
    a training run starts, or from triage scripts that need a deterministic
    record of the renderer + sampler shape that drove a rollout batch.
    """
    return RolloutHelperInfo(
        tokenizer_id=tokenizer_id,
        renderer_name=getattr(renderer, "name", None) or type(renderer).__name__,
        stop_condition=list(renderer.get_stop_sequences()),
        max_tokens=max_tokens,
    )


# Backwards-compat alias for callers that imported the private name during
# Round 0; the public name is :func:`renderer_helper_info`.
helper_info = renderer_helper_info


def _attach_helper_info(
    helper: Any,
    renderer: _RendererLike,
    *,
    tokenizer_id: str | None,
    max_tokens: int | None,
) -> None:
    """Attach a ``helper_info`` accessor to a helper coroutine function.

    Called from the example wiring layer (see ``single_turn_renderer_rollout``
    docstring) so runtime triage code can do
    ``single_turn_renderer_rollout.helper_info(renderer, ...)`` without
    re-deriving the metadata each time.  Idempotent.
    """
    helper.helper_info = lambda: renderer_helper_info(  # type: ignore[attr-defined]
        renderer, tokenizer_id=tokenizer_id, max_tokens=max_tokens,
    )


# Expose a default ``helper_info`` accessor on the single-turn helper so
# triage code can call ``single_turn_renderer_rollout.helper_info(renderer,
# tokenizer_id=..., max_tokens=...)`` directly.  This keeps the AC-8 triage
# surface discoverable from the helper itself.
single_turn_renderer_rollout.helper_info = renderer_helper_info  # type: ignore[attr-defined]
