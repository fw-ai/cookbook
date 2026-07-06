"""Renderer-backed RL rollout primitives.

This module exposes the renderer-backed single-turn helper and adapters for
text and multimodal prompts:

* :func:`single_turn_renderer_rollout` — single-turn helper that turns a
  renderer + a sampling primitive into a flat
  :class:`~training.utils.rl.rollout.types.RolloutRun` whose tokens, logprobs,
  and loss mask are derived end-to-end from the renderer-built prompt and
  the sampler-returned assistant tokens.
* :func:`model_input_to_token_ids` — flatten a text-only ``ModelInput`` from a
  renderer's ``build_generation_prompt(...)`` into ``list[int]``.
* Multimodal prompts use token-in completions (unexpanded ``list[int]`` prompt
  with one image-pad token per image + base64 ``images``); the server expands
  pads and returns expanded ``prompt_token_ids``. Training data is packed via
  :func:`training.utils.supervised.build_multimodal_policy_datum` at group time).

Multi-turn flows are shown as concrete ``async def
rollout_fn(sample_prompt) -> RolloutRun | None`` examples under
``cookbook/training/examples/rl/``.  Per-rollout context (sampler,
tokenizer, sample kwargs, custom state) is closed over via
:class:`RolloutSetup` at factory time -- the framework no longer threads
a ``ctx`` argument through.  Keep environment/tool policy in user code;
this helper only covers single-turn renderer packing.

Boundary
--------

The renderer is consumed inside the rollout; it is not the trainer's
data contract.  The trainer's contract is per-sample:
``rollout_fn(sample_prompt) -> RolloutRun | None``.  This module is
renderer-name-agnostic and never re-renders chat templates client-side.

Parse-failure / truncation handling
-----------------------------------

Parse-failure handling is *not* a framework primitive.  This helper hands
``(parsed_message, parse_success)`` back to the user-supplied ``reward_fn``;
the caller chooses what to do (drop by returning ``None``, score zero by
returning ``0.0``, or branch on ``parse_success`` for custom behavior.
``finish_reason='length'`` flows through unchanged unless the user branches
on it.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, Protocol, Union

import httpx
import tinker

from training.utils.rl.rollout.types import RolloutRun, RolloutSample
from training.utils.supervised import (
    has_non_text_chunks,
    normalize_messages,
    _extract_text_only_token_ids,
)

logger = logging.getLogger(__name__)


async def _maybe_await(value: Any) -> Any:
    """Await coroutines; pass through plain values (sync reward/message builders)."""
    if inspect.isawaitable(value):
        return await value
    return value


__all__ = [
    "MultimodalRenderingNotSupported",
    "VisionCompletionsResult",
    "build_multimodal_completions_prompt_token_ids",
    "build_multimodal_completions_request",
    "model_input_to_token_ids",
    "sample_vision_completion",
    "single_turn_renderer_rollout",
]


class MultimodalRenderingNotSupported(RuntimeError):
    """Raised when a multimodal prompt cannot be sampled for RL.

    Renderer-backed RL rollouts with image (or other non-text) chunks require
    a ``tokenizer`` and either token-in completions (``list[int]`` prompt +
    base64 ``images`` via ``sample_with_prompt_tokens``) or string-prompt
    vision completions via ``sample_with_vision``.
    """


@dataclass(frozen=True)
class VisionCompletionsResult:
    """Normalized vision completions response for multimodal RL rollouts."""

    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    completion_logprobs: List[float]
    finish_reason: str
    text: str


# Qwen3-VL vision span for string-prompt completions when ``apply_chat_template``
# is unavailable (``sample_with_vision`` / legacy inference).
_QWEN_VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def model_input_to_token_ids(model_input: tinker.ModelInput) -> List[int]:
    """Flatten a text-only renderer ``ModelInput`` to ``list[int]``.

    Accepts only :class:`tinker.EncodedTextChunk` chunks.      For multimodal inputs, use :func:`build_multimodal_completions_prompt_token_ids`
    and :func:`single_turn_renderer_rollout` instead of calling this directly.
    """
    out: List[int] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            raise MultimodalRenderingNotSupported(
                f"chunk type {type(chunk).__name__} is not supported by "
                "model_input_to_token_ids; use vision-completions sampling "
                "for multimodal renderer-backed RL rollouts."
            )
    return out


def _messages_for_hf_chat_template(messages: List[Any]) -> List[dict[str, Any]]:
    """Convert normalized Tinker messages into HF ``apply_chat_template`` input."""
    normalized = normalize_messages(messages)
    out: List[dict[str, Any]] = []
    for msg in normalized:
        content = msg["content"]
        if isinstance(content, list):
            hf_parts: List[dict[str, Any]] = []
            for part in content:
                if part.get("type") == "image":
                    hf_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": str(part["image"])},
                        }
                    )
                else:
                    hf_parts.append(part)
            content = hf_parts
        row: dict[str, Any] = {"role": msg["role"], "content": content}
        if msg.get("tool_calls") is not None:
            row["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id") is not None:
            row["tool_call_id"] = msg["tool_call_id"]
        if msg.get("name") is not None:
            row["name"] = msg["name"]
        out.append(row)
    return out


def _is_base64_image_payload(value: str) -> bool:
    """Return True when *value* is a Fireworks completions ``images`` data URL."""
    text = str(value).strip()
    if ";base64," not in text:
        return False
    _prefix, _sep, payload = text.partition(";base64,")
    return bool(payload.strip())


def _validate_base64_image_payload(value: str, *, source: str) -> str:
    """Require a ``data:<mime>;base64,...`` (or ``<mime>;base64,...``) payload."""
    text = str(value).strip()
    if not _is_base64_image_payload(text):
        raise MultimodalRenderingNotSupported(
            f"{source} must be a base64-encoded image data URL "
            f"(e.g. data:image/jpeg;base64,...); got {text[:64]!r}"
        )
    return text


def _collect_base64_images(
    messages: List[Any],
    model_input: tinker.ModelInput,
) -> List[str]:
    """Collect base64 image payloads for the completions ``images`` field."""
    seen: set[str] = set()
    images: List[str] = []

    def _add(value: str, *, source: str) -> None:
        payload = _validate_base64_image_payload(value, source=source)
        if payload not in seen:
            seen.add(payload)
            images.append(payload)

    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.types.ImageAssetPointerChunk):
            _add(
                str(chunk.location),
                source="ImageAssetPointerChunk.location",
            )

    for msg in normalize_messages(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image" and part.get("image") is not None:
                _add(str(part["image"]), source="message image")
                continue
            image_url = part.get("image_url")
            if isinstance(image_url, str):
                _add(image_url, source="message image_url")
            elif isinstance(image_url, dict) and image_url.get("url") is not None:
                _add(str(image_url["url"]), source="message image_url.url")
    return images


def _image_placeholder_token_id(tokenizer: Any) -> int | None:
    """Return the model's image-placeholder token id when exposed by the tokenizer."""
    special_ids = getattr(tokenizer, "special_ids", None)
    if special_ids is None:
        return None
    image_id = getattr(special_ids, "image", None)
    if image_id is None:
        return None
    return int(image_id)


def build_multimodal_completions_prompt_token_ids(
    messages: List[Any],
    model_input: tinker.ModelInput,
    tokenizer: Any,
) -> tuple[List[int], List[str]]:
    """Build ``(prompt_token_ids, images)`` for token-in vision completions.

    Prompt token IDs come from ``tokenizer.apply_chat_template(..., tokenize=True)``
    so placeholder tokens match the model chat template. They must be
    **unexpanded** (one image-placeholder id per image). Each image must be a
    base64 data URL (``data:<mime>;base64,...``).
    """
    images = _collect_base64_images(messages, model_input)
    if not images:
        raise MultimodalRenderingNotSupported(
            "multimodal ModelInput has no base64 images; cannot call completions with images"
        )

    try:
        tokenized = tokenizer.apply_chat_template(
            _messages_for_hf_chat_template(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    except Exception as exc:
        raise MultimodalRenderingNotSupported(
            "tokenizer.apply_chat_template(..., tokenize=True) is required for "
            "multimodal RL completions; it failed for this message format. "
            "Use a VL tokenizer with a multimodal chat template, or fix the "
            "dataset message shape."
        ) from exc

    if hasattr(tokenized, "tolist"):
        prompt_token_ids = tokenized.tolist()
    else:
        prompt_token_ids = list(tokenized)

    if not prompt_token_ids:
        raise MultimodalRenderingNotSupported(
            "multimodal completions prompt tokenization produced an empty prompt"
        )

    image_placeholder_id = _image_placeholder_token_id(tokenizer)
    if image_placeholder_id is not None:
        pad_count = sum(1 for tid in prompt_token_ids if int(tid) == image_placeholder_id)
        if pad_count != len(images):
            raise MultimodalRenderingNotSupported(
                f"image placeholder count in prompt ({pad_count}) != images count "
                f"({len(images)}); chat template and image list are misaligned"
            )
    return [int(t) for t in prompt_token_ids], images


def _model_input_to_completions_prompt_text(
    model_input: tinker.ModelInput,
    tokenizer: Any,
) -> str:
    """Stitch a completions prompt string from renderer ``ModelInput`` chunks."""
    parts: List[str] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            parts.append(
                tokenizer.decode(
                    list(chunk.tokens),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        elif isinstance(chunk, tinker.types.ImageAssetPointerChunk):
            parts.append(_QWEN_VISION_PLACEHOLDER)
        else:
            raise MultimodalRenderingNotSupported(
                f"cannot build completions prompt from chunk type {type(chunk).__name__}"
            )
    return "".join(parts)


def build_multimodal_completions_request(
    messages: List[Any],
    model_input: tinker.ModelInput,
    tokenizer: Any,
) -> tuple[str, List[str]]:
    """Build ``(prompt_text, images)`` for string-prompt vision completions.

    Each image must be a base64 data URL (``data:<mime>;base64,...``).
    Remote HTTP(S) image URLs are not supported on this path.
    """
    images = _collect_base64_images(messages, model_input)
    if not images:
        raise MultimodalRenderingNotSupported(
            "multimodal ModelInput has no base64 images; cannot call completions with images"
        )

    try:
        prompt_text = str(
            tokenizer.apply_chat_template(
                _messages_for_hf_chat_template(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    except Exception:
        logger.debug(
            "apply_chat_template failed for multimodal rollout; "
            "falling back to ModelInput string stitch",
            exc_info=True,
        )
        prompt_text = _model_input_to_completions_prompt_text(model_input, tokenizer)
    return prompt_text, images


def _normalize_completions_api_base(inference_base_url: str) -> str:
    base = inference_base_url.rstrip("/")
    is_direct_route = ".direct.fireworks.ai" in base
    if base.endswith("/inference/v1") or (is_direct_route and base.endswith("/v1")):
        return base
    if base.endswith("/inference"):
        return f"{base}/v1"
    if base.endswith("/v1"):
        return base
    if is_direct_route:
        return f"{base}/v1"
    return f"{base}/inference/v1"


def _extract_structured_choice_logprobs(choice: dict[str, Any]) -> List[float] | None:
    lp_data = choice.get("logprobs")
    if not lp_data or not isinstance(lp_data, dict):
        return None
    content = lp_data.get("content")
    if isinstance(content, list) and content:
        return [float(tok.get("logprob", 0.0)) for tok in content]
    return None


def _extract_completions_choice_logprobs(choice: dict[str, Any]) -> List[float] | None:
    """Extract completion logprobs from a ``/v1/completions`` choice."""
    lp_data = choice.get("logprobs")
    if not isinstance(lp_data, dict):
        return None
    token_logprobs = lp_data.get("token_logprobs")
    if isinstance(token_logprobs, list) and token_logprobs:
        return [float(lp) if lp is not None else 0.0 for lp in token_logprobs]
    return _extract_structured_choice_logprobs(choice)


def _parse_vision_completions_payload(payload: dict[str, Any]) -> VisionCompletionsResult:
    """Parse a ``/v1/completions`` body with ``return_token_ids`` / ``raw_output``."""
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("completions response missing choices")

    choice = choices[0]
    finish_reason = str(choice.get("finish_reason") or "stop")
    text = str(choice.get("text") or "")
    raw_output = choice.get("raw_output") if isinstance(choice.get("raw_output"), dict) else {}

    prompt_ids = (
        choice.get("prompt_token_ids")
        or raw_output.get("prompt_token_ids")
        or []
    )
    completion_ids = (
        choice.get("token_ids")
        or raw_output.get("completion_token_ids")
        or raw_output.get("token_ids")
        or []
    )
    completion_logprobs = _extract_completions_choice_logprobs(choice)

    if not prompt_ids:
        raise RuntimeError(
            "completions response missing prompt_token_ids; "
            f"payload keys={list(payload)}"
        )
    if not completion_ids:
        raise RuntimeError("completions response missing completion token_ids")
    if completion_logprobs is None:
        raise RuntimeError(
            "completions response missing per-token logprobs; pass logprobs=True"
        )

    prompt_token_ids = [int(x) for x in prompt_ids]
    completion_token_ids = [int(x) for x in completion_ids]
    completion_logprobs = list(completion_logprobs)
    if len(completion_logprobs) > len(completion_token_ids):
        completion_logprobs = completion_logprobs[-len(completion_token_ids) :]
    if len(completion_logprobs) != len(completion_token_ids):
        raise RuntimeError(
            "completions logprobs misaligned with completion_token_ids "
            f"(got {len(completion_logprobs)} logprobs, "
            f"{len(completion_token_ids)} tokens)"
        )

    return VisionCompletionsResult(
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        completion_logprobs=completion_logprobs,
        finish_reason=finish_reason,
        text=text,
    )


async def sample_vision_completion(
    *,
    prompt_text: str,
    images: List[str],
    inference_base_url: str,
    api_key: str,
    deployment_model: str,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    stop: List[str] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> VisionCompletionsResult:
    """Sample one assistant turn via ``/v1/completions`` (string prompt + images).

    Each entry in ``images`` must be a base64 data URL.
    """
    validated_images = [
        _validate_base64_image_payload(img, source=f"images[{idx}]")
        for idx, img in enumerate(images)
    ]
    url = f"{_normalize_completions_api_base(inference_base_url)}/completions"
    payload: dict[str, Any] = {
        "model": deployment_model,
        "prompt": prompt_text,
        "images": validated_images,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": True,
        "return_token_ids": True,
        "raw_output": True,
        "n": 1,
    }
    if stop:
        payload["stop"] = stop
    if extra_kwargs:
        payload.update(extra_kwargs)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return _parse_vision_completions_payload(resp.json())


def _build_text_only_rollout_sample(
    *,
    prompt_token_ids: List[int],
    completion_tokens: List[int],
    completion_logprobs: List[float],
    logprobs_echoed: bool,
    reward: float,
    finish_reason: str,
    text: str,
) -> RolloutRun:
    out_logprobs = list(completion_logprobs)
    if logprobs_echoed and len(out_logprobs) == len(prompt_token_ids) + len(completion_tokens):
        out_logprobs = out_logprobs[len(prompt_token_ids) :]
    sample = RolloutSample(
        tokens=list(prompt_token_ids) + completion_tokens,
        logprobs=[0.0] * len(prompt_token_ids) + out_logprobs,
        loss_mask=[0] * len(prompt_token_ids) + [1] * len(completion_tokens),
        reward=float(reward),
        finish_reason=finish_reason,
        text=text,
    )
    return RolloutRun(segments=[sample])


def _build_multimodal_rollout_sample(
    *,
    prompt_model_input: tinker.ModelInput,
    completion_tokens: List[int],
    completion_logprobs: List[float],
    reward: float,
    finish_reason: str,
    text: str,
) -> RolloutRun:
    prompt_text_ids = _extract_text_only_token_ids(prompt_model_input)
    completion = [int(t) for t in completion_tokens]
    text_tokens = list(prompt_text_ids) + completion
    if len(completion_logprobs) != len(completion):
        raise ValueError(
            "multimodal completion logprobs misaligned "
            f"(got {len(completion_logprobs)}, expected {len(completion)})"
        )
    sample = RolloutSample(
        tokens=text_tokens,
        logprobs=[0.0] * len(prompt_text_ids) + list(completion_logprobs),
        loss_mask=[0] * len(prompt_text_ids) + [1] * len(completion),
        reward=float(reward),
        finish_reason=finish_reason,
        text=text,
        prompt_model_input=prompt_model_input,
    )
    return RolloutRun(segments=[sample])


# A renderer-shaped protocol.  Avoids a hard import of the Tinker base class
# in this module's signature surface so tests can pass simple stubs without
# subclassing the heavy upstream class.  We do not export this — the helper
# accepts any object that responds to these methods.
class _RendererLike(Protocol):
    def build_generation_prompt(self, messages: List[Any], **kwargs: Any) -> tinker.ModelInput: ...
    def parse_response(self, tokens: List[int]) -> Any: ...
    def get_stop_sequences(self) -> List[Any]: ...


SampleWithPromptTokens = Callable[..., Awaitable[List[Any]]]
"""Callable matching :meth:`DeploymentSampler.sample_with_prompt_tokens`."""

SampleWithVision = Callable[..., Awaitable[VisionCompletionsResult]]
"""Callable matching :func:`sample_vision_completion`."""


MessageBuilder = Callable[[Any], Union[List[Any], Awaitable[List[Any]]]]
"""``(row) -> messages`` — builds the seed conversation (sync or async)."""


RewardFn = Callable[[Any, Any, bool], Union[Optional[float], Awaitable[Optional[float]]]]
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
    inference_base_url: str | None = None,
    api_key: str | None = None,
    deployment_model: str | None = None,
    sample_with_vision: SampleWithVision | None = None,
) -> RolloutRun | None:
    """Single-turn renderer-backed rollout (per-run).

    Builds messages via ``message_builder(row)``, calls
    ``renderer.build_generation_prompt(...)``, then samples ONE completion.

    Text-only and multimodal prompts default to ``sample_with_prompt_tokens``
    (token-in completions API).  Multimodal rollouts also pass base64
    ``images`` (one unexpanded image-pad token per image).  Pass a
    ``tokenizer`` for multimodal prompts.

    For inference that only supports string ``prompt`` + ``images``, pass
    ``sample_with_vision`` (e.g. :func:`sample_vision_completion`).

    The framework fans each row out to ``completions_per_prompt`` parallel
    calls; this helper requests ``n=1`` per call.
    """
    messages = await _maybe_await(message_builder(row))
    try:
        model_input = renderer.build_generation_prompt(messages)
    except TypeError:
        model_input = renderer.build_generation_prompt(messages, role="assistant")

    multimodal = has_non_text_chunks(model_input)

    if stop is None:
        stop = renderer.get_stop_sequences()

    stop_strings: List[str] | None = None
    if stop:
        if all(isinstance(s, int) for s in stop):
            if tokenizer is None:
                raise ValueError(
                    "Renderer returned integer stop token IDs but no tokenizer "
                    "was passed to decode them."
                )
            stop_strings = [
                tokenizer.decode([s], skip_special_tokens=False) for s in stop
            ]
        else:
            stop_strings = [str(s) for s in stop]

    sk: dict[str, Any] = dict(sample_kwargs or {})
    sk["n"] = 1
    if max_tokens is not None:
        sk["max_tokens"] = max_tokens

    if multimodal:
        if tokenizer is None:
            raise ValueError(
                "Multimodal renderer-backed RL requires a tokenizer to "
                "build the completions prompt."
            )

        if sample_with_vision is not None:
            prompt_text, images = build_multimodal_completions_request(
                messages, model_input, tokenizer
            )
            extra = {
                k: v
                for k, v in sk.items()
                if k not in {"n", "max_tokens", "temperature", "stop"}
            }
            extra.setdefault("logprobs", True)
            vision_result = await sample_with_vision(
                prompt_text=prompt_text,
                images=images,
                max_tokens=sk.get("max_tokens", 1024),
                temperature=sk.get("temperature", 1.0),
                stop=stop_strings,
                **extra,
            )
            completion_tokens = vision_result.completion_token_ids
            if not completion_tokens:
                return None
            parsed_message, parse_success = renderer.parse_response(completion_tokens)
            reward = await _maybe_await(
                reward_fn(row, parsed_message, bool(parse_success))
            )
            if reward is None:
                return None
            return _build_multimodal_rollout_sample(
                prompt_model_input=model_input,
                completion_tokens=completion_tokens,
                completion_logprobs=vision_result.completion_logprobs,
                reward=reward,
                finish_reason=vision_result.finish_reason,
                text=vision_result.text,
            )

        prompt_token_ids, images = build_multimodal_completions_prompt_token_ids(
            messages, model_input, tokenizer
        )
        sk["stop"] = stop_strings if stop_strings is not None else stop
        sk.setdefault("logprobs", True)
        sk.setdefault("return_token_ids", True)
        sk["images"] = images

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
                "single_turn_renderer_rollout: dropping multimodal completion "
                "with no inference_logprobs (got None). Configure logprobs=True."
            )
            return None

        parsed_message, parse_success = renderer.parse_response(out_tokens)
        reward = await _maybe_await(reward_fn(row, parsed_message, bool(parse_success)))
        if reward is None:
            return None

        out_logprobs = list(out_logprobs_raw)
        if bool(getattr(c, "logprobs_echoed", False)) and len(out_logprobs) == (
            prompt_len + len(out_tokens)
        ):
            out_logprobs = out_logprobs[prompt_len:]
        if len(out_logprobs) != len(out_tokens):
            logger.warning(
                "single_turn_renderer_rollout: dropping multimodal completion with "
                "misaligned logprobs (got %d, expected %d for assistant tokens).",
                len(out_logprobs), len(out_tokens),
            )
            return None

        return _build_multimodal_rollout_sample(
            prompt_model_input=model_input,
            completion_tokens=out_tokens,
            completion_logprobs=out_logprobs,
            reward=reward,
            finish_reason=getattr(c, "finish_reason", "stop"),
            text=getattr(c, "text", ""),
        )

    prompt_token_ids = model_input_to_token_ids(model_input)
    sk["stop"] = stop_strings if stop_strings is not None else stop

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

    parsed_message, parse_success = renderer.parse_response(out_tokens)
    reward = await _maybe_await(reward_fn(row, parsed_message, bool(parse_success)))
    if reward is None:
        return None

    out_logprobs = list(out_logprobs_raw)
    if bool(getattr(c, "logprobs_echoed", False)) and len(out_logprobs) == (
        prompt_len + len(out_tokens)
    ):
        out_logprobs = out_logprobs[prompt_len:]
    if len(out_logprobs) != len(out_tokens):
        logger.warning(
            "single_turn_renderer_rollout: dropping completion with "
            "misaligned logprobs (got %d, expected %d for assistant tokens).",
            len(out_logprobs), len(out_tokens),
        )
        return None

    return _build_text_only_rollout_sample(
        prompt_token_ids=prompt_token_ids,
        completion_tokens=out_tokens,
        completion_logprobs=out_logprobs,
        logprobs_echoed=False,
        reward=reward,
        finish_reason=getattr(c, "finish_reason", "stop"),
        text=getattr(c, "text", ""),
    )
