"""Renderer-backed RL rollout primitives.

This module exposes the renderer-backed single-turn helper and the
``ModelInput`` flattening adapter:

* :func:`single_turn_renderer_rollout` — single-turn helper that turns a
  renderer + a pre-tokenized sampling primitive into a flat
  :class:`~training.utils.rl.rollout.types.Rollout` whose tokens, logprobs,
  and loss mask are derived end-to-end from the renderer-built prompt and
  the sampler-returned assistant tokens.
* :func:`model_input_to_token_ids` — flatten a Tinker ``ModelInput`` from a
  renderer's ``build_generation_prompt(...)`` into ``list[int]``.  Multimodal
  chunks are rejected with :class:`MultimodalRenderingNotSupported`.

Multi-turn flows are shown as concrete ``async def
rollout_fn(sample_prompt) -> RolloutSample | None`` examples under
``cookbook/training/examples/rl/``.  Per-rollout context (sampler,
tokenizer, sample kwargs, custom state) is closed over via
:class:`RolloutSetup` at factory time -- the framework no longer threads
a ``ctx`` argument through.  Keep environment/tool policy in user code;
this helper only covers single-turn renderer packing.

Boundary
--------

The renderer is consumed inside the rollout; it is not the trainer's
data contract.  The trainer's contract is per-sample:
``rollout_fn(sample_prompt) -> RolloutSample | None``.  This module is
renderer-name-agnostic and never re-renders chat templates client-side.

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
from typing import Any, Awaitable, Callable, List, Optional, Protocol

import tinker

from training.utils.rl.rollout.types import RolloutSample


logger = logging.getLogger(__name__)


__all__ = [
    "MultimodalRenderingNotSupported",
    "model_input_to_token_ids",
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


MessageBuilder = Callable[[Any], Awaitable[List[Any]]]
"""``async (row) -> messages`` — builds the seed conversation."""


RewardFn = Callable[[Any, Any, bool], Awaitable[Optional[float]]]
"""``async (row, parsed_message, parse_success) -> float | None``.

Return ``None`` to drop the completion (no sample emitted).  Return a float
to emit a sample with that reward.  ``parse_success`` is the second element
returned by the renderer's ``parse_response``.
"""


async def single_turn_renderer_rollout(
    row: Any,
    *,
    renderer: _RendererLike,
    sample_with_prompt_tokens: SampleWithPromptTokens,
    message_builder: MessageBuilder,
    reward_fn: RewardFn,
    sample_kwargs: dict[str, Any] | None = None,
    tokenizer: Any | None = None,
    max_tokens: int | None = None,
    stop: List[str] | List[int] | None = None,
) -> RolloutSample | None:
    """Single-turn renderer-backed rollout (per-sample).

    Builds messages via ``message_builder(row)``, calls
    ``renderer.build_generation_prompt(...)``, flattens the resulting
    ``ModelInput`` via :func:`model_input_to_token_ids`, samples ONE
    completion through ``sample_with_prompt_tokens`` (the SDK's
    pre-tokenized sampling primitive), and packs the completion into a
    :class:`RolloutSample` whose tokens / logprobs / loss-mask come straight
    from the renderer + sampler.  No chat-template re-rendering, no
    re-tokenization of decoded assistant text.

    The framework fans each row out to ``completions_per_prompt`` parallel
    calls; this helper requests ``n=1`` per call.

    ``stop`` defaults to ``renderer.get_stop_sequences()`` and preserves its
    ``list[str] | list[int]`` shape.  The user-supplied ``reward_fn``
    receives the renderer's parsed message and parse-success flag and
    returns ``None`` (drop) or a float.  ``tokenizer`` is required only
    when the renderer returns integer stop token IDs that need decoding
    to strings for the inference API.  Multimodal prompts raise
    :class:`MultimodalRenderingNotSupported` via the adapter.
    """
    messages = await message_builder(row)
    model_input = renderer.build_generation_prompt(messages)
    prompt_token_ids = model_input_to_token_ids(model_input)

    if stop is None:
        stop = renderer.get_stop_sequences()

    # The Fireworks inference completions API rejects integer stop token
    # IDs; it requires string stop sequences. Renderers like Qwen's return
    # ``[151645]`` (an int token id) from ``get_stop_sequences()``. Decode
    # via the tokenizer when one is supplied.
    if stop and all(isinstance(s, int) for s in stop):
        if tokenizer is None:
            raise ValueError(
                "Renderer returned integer stop token IDs but no tokenizer "
                "was passed to decode them; the inference API only accepts "
                "string stop sequences."
            )
        stop = [tokenizer.decode([s], skip_special_tokens=False) for s in stop]

    sk: dict[str, Any] = dict(sample_kwargs or {})
    sk["n"] = 1
    sk["stop"] = stop
    if max_tokens is not None:
        sk["max_tokens"] = max_tokens

    completions = await sample_with_prompt_tokens(prompt_token_ids, **sk)
    if not completions:
        return None

    c = completions[0]
    prompt_len = int(c.prompt_len)
    out_tokens: List[int] = list(c.full_tokens[prompt_len:])
    if not out_tokens:
        return None
    out_logprobs_raw = getattr(c, "inference_logprobs", None)
    if out_logprobs_raw is None:
        logger.warning(
            "single_turn_renderer_rollout: dropping completion with "
            "no inference_logprobs (got None).  Configure the sampler "
            "with logprobs=True so PPO/GRPO ratio/KL math sees real "
            "behavior-policy probabilities."
        )
        return None
    out_logprobs: List[float] = list(out_logprobs_raw)
    if getattr(c, "logprobs_echoed", False) and len(out_logprobs) == prompt_len + len(out_tokens):
        out_logprobs = out_logprobs[prompt_len:]
    if len(out_logprobs) != len(out_tokens):
        logger.warning(
            "single_turn_renderer_rollout: dropping completion with "
            "misaligned logprobs (got %d, expected %d for assistant tokens).",
            len(out_logprobs), len(out_tokens),
        )
        return None

    parsed_message, parse_success = renderer.parse_response(out_tokens)
    reward = await reward_fn(row, parsed_message, bool(parse_success))
    if reward is None:
        return None

    return RolloutSample(
        tokens=list(prompt_token_ids) + out_tokens,
        logprobs=[0.0] * len(prompt_token_ids) + out_logprobs,
        loss_mask=[0] * len(prompt_token_ids) + [1] * len(out_tokens),
        reward=float(reward),
        finish_reason=getattr(c, "finish_reason", "stop"),
        text=getattr(c, "text", ""),
    )
