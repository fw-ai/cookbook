"""Shared supervised-rendering helpers.

This module gives cookbook training code one token-level representation for
supervised data:

- text/tool conversations rendered via a Tinker renderer
- eval-protocol-style token trajectories with a per-token mask

Both paths end in the same ``tinker.Datum`` schema with ``target_tokens`` and
token-level ``weights`` so training uses the same spans that the UI shows.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import tinker
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import (
    Message,
    Renderer,
    ToolCall,
    TrainOnWhat,
    get_renderer,
)

from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.supervised.common import datum_from_model_input_weights
import training.renderer.nemotron as _nemotron_renderer  # noqa: F401 — triggers register_renderer
import training.renderer.minimax_m2 as _minimax_m2_renderer  # noqa: F401 — triggers register_renderer


@dataclass(frozen=True)
class RenderedSupervisedDatum:
    """Rendered sequence ids, token-level weights, and the final training datum."""

    token_ids: list[int]
    token_weights: list[float]
    datum: tinker.Datum


@dataclass(frozen=True)
class RenderedPreferencePair:
    """Rendered chosen/rejected preference pair with a shared response boundary."""

    chosen_tokens: list[int]
    rejected_tokens: list[int]
    response_start: int
    chosen_datum: tinker.Datum
    rejected_datum: tinker.Datum


def parse_train_on_what(value: str | TrainOnWhat) -> TrainOnWhat:
    """Normalize ``train_on_what`` config into Tinker's enum."""
    if isinstance(value, TrainOnWhat):
        return value
    return TrainOnWhat(value.lower())


def resolve_renderer_name(
    tokenizer_model: str,
    renderer_name: str = "",
) -> str:
    """Choose the renderer used for message -> token rendering."""
    if renderer_name:
        return renderer_name
    normalized_model_name = tokenizer_model.lower()
    if "moonshotai/kimi-k2.5" in normalized_model_name:
        return "kimi_k25"
    if "nemotron" in normalized_model_name:
        return "nemotron"
    if "minimax-m2" in normalized_model_name or "minimax_m2" in normalized_model_name:
        return "minimax_m2"
    if "qwen3-vl" in normalized_model_name:
        return "qwen3_vl_instruct"
    try:
        return get_recommended_renderer_name(tokenizer_model)
    except Exception as exc:  # pragma: no cover - message only
        raise ValueError(
            f"Could not infer a renderer for tokenizer_model={tokenizer_model!r}. "
            "Set Config.renderer_name explicitly."
        ) from exc


def build_renderer(
    tokenizer: Any,
    tokenizer_model: str,
    renderer_name: str = "",
) -> Renderer:
    """Construct the Tinker renderer used for supervised formatting."""
    resolved_name = resolve_renderer_name(tokenizer_model, renderer_name)
    if get_image_processor is not None and _renderer_uses_images(resolved_name):
        return get_renderer(
            resolved_name,
            tokenizer,
            image_processor=get_image_processor(tokenizer_model),
        )
    return get_renderer(resolved_name, tokenizer)


def _renderer_uses_images(renderer_name: str) -> bool:
    return any(
        marker in renderer_name
        for marker in (
            "_vl",
            "qwen3_5",
            "kimi_k25",
        )
    )


logger = logging.getLogger(__name__)


def _truncate_model_input(model_input: tinker.ModelInput) -> tinker.ModelInput:
    """Return a copy of *model_input* with the last text token removed.

    This is the multimodal equivalent of ``tokens[:-1]`` used for the standard
    next-token prediction shift.
    """
    chunks = list(model_input.chunks)
    for i in range(len(chunks) - 1, -1, -1):
        chunk = chunks[i]
        if isinstance(chunk, tinker.types.EncodedTextChunk) and len(chunk.tokens) > 0:
            remaining = list(chunk.tokens)[:-1]
            if remaining:
                chunks[i] = tinker.types.EncodedTextChunk(tokens=remaining)
            else:
                chunks.pop(i)
            result = tinker.ModelInput.empty()
            for c in chunks:
                result = result.append(c)
            return result
    raise ValueError("ModelInput has no text tokens to truncate")


def _normalize_tool_calls(tool_calls: Any) -> list[ToolCall]:
    """Normalize common tool-call shapes into Tinker's structured ToolCall form."""
    normalized: list[ToolCall] = []
    for tool_call in tool_calls or []:
        if not isinstance(tool_call, Mapping):
            raise TypeError(f"Unsupported tool call type: {type(tool_call)!r}")

        if isinstance(tool_call.get("name"), str) and isinstance(
            tool_call.get("args"), Mapping
        ):
            normalized.append(
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name=tool_call["name"],
                        arguments=json.dumps(dict(tool_call["args"])),
                    ),
                    id=tool_call.get("id"),
                )
            )
            continue

        function = tool_call.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            raw_args = function.get("arguments", {})
            if isinstance(raw_args, str):
                parsed_args = json.loads(raw_args) if raw_args else {}
            elif isinstance(raw_args, Mapping):
                parsed_args = dict(raw_args)
            else:
                raise TypeError(
                    f"Unsupported tool call arguments type: {type(raw_args)!r}"
                )
            normalized.append(
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name=function["name"],
                        arguments=json.dumps(parsed_args),
                    ),
                    id=tool_call.get("id"),
                )
            )
            continue

        raise ValueError(f"Unsupported tool call shape: {tool_call}")
    return normalized


def _normalize_image_part(part: Mapping[str, Any]) -> dict[str, Any]:
    image_value = part.get("image")
    if image_value is not None:
        return {"type": "image", "image": image_value}

    image_url = part.get("image_url")
    if isinstance(image_url, str):
        return {"type": "image", "image": image_url}
    if isinstance(image_url, Mapping) and isinstance(image_url.get("url"), str):
        return {"type": "image", "image": image_url["url"]}
    raise TypeError(f"Unsupported image content part: {part}")


def _normalize_content(content: Any) -> str | list[dict[str, Any]]:
    """Convert OpenAI-style message content into Tinker's text or structured format."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        part_type = content.get("type")
        if part_type in {"image", "image_url"}:
            return [_normalize_image_part(content)]
        if part_type == "thinking" and isinstance(content.get("thinking"), str):
            return [{"type": "thinking", "thinking": content["thinking"]}]
        if isinstance(content.get("text"), str):
            return content["text"]
        raise TypeError(f"Unsupported message content mapping: {content}")
    if isinstance(content, Sequence):
        normalized_parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                normalized_parts.append({"type": "text", "text": part})
                continue
            if not isinstance(part, Mapping):
                raise TypeError(f"Unsupported message content part: {part!r}")
            part_type = part.get("type")
            if part_type == "text" and isinstance(part.get("text"), str):
                normalized_parts.append({"type": "text", "text": part["text"]})
                continue
            if part_type in {"image", "image_url"}:
                normalized_parts.append(_normalize_image_part(part))
                continue
            if part_type == "thinking" and isinstance(part.get("thinking"), str):
                normalized_parts.append(
                    {"type": "thinking", "thinking": part["thinking"]}
                )
                continue
            raise TypeError(f"Unsupported message content part: {part!r}")
        if normalized_parts and all(
            part["type"] == "text" for part in normalized_parts
        ):
            return "".join(str(part["text"]) for part in normalized_parts)
        return normalized_parts
    raise TypeError(f"Unsupported message content type: {type(content)!r}")


def _ensure_content_parts(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def normalize_messages(messages: Iterable[Mapping[str, Any]]) -> list[Message]:
    """Normalize cookbook/eval-style messages into Tinker's message schema."""
    normalized: list[Message] = []
    for message in messages:
        role = message.get("role")
        if not isinstance(role, str):
            raise ValueError(f"Message is missing a string role: {message}")

        normalized_message: Message = {
            "role": role,
            "content": _normalize_content(message.get("content")),
        }

        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            normalized_message["tool_calls"] = _normalize_tool_calls(tool_calls)

        thinking = message.get("thinking")
        if thinking is not None:
            if not isinstance(thinking, str):
                raise TypeError(f"Unsupported thinking value type: {type(thinking)!r}")
            normalized_message["content"] = [
                {"type": "thinking", "thinking": thinking},
                *_ensure_content_parts(normalized_message["content"]),
            ]

        trainable = message.get("trainable")
        if trainable is not None:
            normalized_message["trainable"] = bool(trainable)

        tool_call_id = message.get("tool_call_id")
        if tool_call_id is not None:
            normalized_message["tool_call_id"] = str(tool_call_id)

        name = message.get("name")
        if name is not None:
            normalized_message["name"] = str(name)

        normalized.append(normalized_message)

    return normalized


def _stable_chunk_sentinel(chunk: Any) -> int:
    if isinstance(chunk, tinker.types.ImageAssetPointerChunk):
        payload = f"{chunk.type}:{chunk.location}:{chunk.format}:{chunk.expected_tokens}".encode()
    elif isinstance(chunk, tinker.types.ImageChunk):
        payload = b"|".join(
            [
                chunk.type.encode(),
                chunk.format.encode(),
                str(chunk.expected_tokens).encode(),
                bytes(chunk.data),
            ]
        )
    else:  # pragma: no cover - defensive branch for future chunk types
        payload = repr(chunk).encode()

    digest = hashlib.sha1(payload).digest()
    return -(int.from_bytes(digest[:8], "big") + 1)


def _flatten_model_input_sequence_ids(model_input: tinker.ModelInput) -> list[int]:
    sequence_ids: list[int] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            sequence_ids.extend(int(token) for token in chunk.tokens)
            continue

        sentinel = _stable_chunk_sentinel(chunk)
        sequence_ids.extend([sentinel] * int(chunk.length))
    return sequence_ids


def _rendered_sequence_ids_from_datum(datum: tinker.Datum) -> list[int]:
    sequence_ids = _flatten_model_input_sequence_ids(datum.model_input)
    target_tokens = [int(x) for x in datum.loss_fn_inputs["target_tokens"].data]
    if not target_tokens:
        raise ValueError(
            "Need at least one target token to reconstruct the rendered sequence."
        )
    return sequence_ids + [target_tokens[-1]]


def build_datum_from_tokens_and_weights(
    token_ids: Sequence[int],
    token_weights: Sequence[float],
    *,
    max_seq_len: int | None = None,
    include_loss_mask: bool = False,
) -> RenderedSupervisedDatum:
    """Build a weighted ``tinker.Datum`` from full tokens and per-token weights."""
    tokens = [int(x) for x in token_ids]
    weights = [float(x) for x in token_weights]
    if len(tokens) != len(weights):
        raise ValueError(
            f"tokens/weights length mismatch: {len(tokens)} != {len(weights)}"
        )
    if len(tokens) < 2:
        raise ValueError("Need at least 2 tokens to build a supervised datum.")

    if max_seq_len is not None:
        tokens = tokens[:max_seq_len]
        weights = weights[:max_seq_len]
        if len(tokens) < 2:
            raise ValueError("Truncation left fewer than 2 tokens.")

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    datum = datum_from_model_input_weights(
        tinker.ModelInput.from_ints(tokens),
        weight_tensor,
        max_length=max_seq_len,
    )

    if include_loss_mask:
        shifted_weights = [float(x) for x in datum.loss_fn_inputs["weights"].data]
        datum.loss_fn_inputs["loss_mask"] = tinker.TensorData(
            data=shifted_weights,
            dtype="float32",
            shape=[len(shifted_weights)],
        )

    return RenderedSupervisedDatum(
        token_ids=[int(x) for x in datum.model_input.to_ints()]
        + [int(datum.loss_fn_inputs["target_tokens"].data[-1])],
        token_weights=[0.0] + [float(x) for x in datum.loss_fn_inputs["weights"].data],
        datum=datum,
    )


def _extract_token_ids(model_input: tinker.ModelInput) -> list[int]:
    """Extract token IDs from a ModelInput, handling multimodal chunks.

    Text chunks contribute their token IDs directly.  Non-text chunks
    (e.g. ``ImageAssetPointerChunk``) contribute placeholder zeros
    matching their ``expected_tokens`` count.
    """
    ids: list[int] = []
    for chunk in model_input.chunks:
        if hasattr(chunk, "tokens"):
            ids.extend(chunk.tokens)
        elif hasattr(chunk, "expected_tokens"):
            ids.extend([0] * chunk.expected_tokens)
    return ids


def _extract_text_only_token_ids(model_input: tinker.ModelInput) -> list[int]:
    """Extract only text token IDs from a ModelInput, skipping image chunks.

    Non-text chunks are silently skipped.  The returned list contains only
    actual text token IDs in sequence order, with no placeholders for images.
    This is required because the server corrupts logprobs when target_tokens
    contains zeros at image chunk positions.
    """
    ids: list[int] = []
    for chunk in model_input.chunks:
        if hasattr(chunk, "tokens"):
            ids.extend(chunk.tokens)
    return ids


def _has_non_text_chunks(model_input: tinker.ModelInput) -> bool:
    return any(
        not isinstance(c, tinker.types.EncodedTextChunk) for c in model_input.chunks
    )


def _build_multimodal_datum(
    model_input: tinker.ModelInput,
    weights: list[float],
    max_seq_len: int | None = None,
) -> tinker.Datum:
    """Build a next-token-prediction datum that preserves image chunks.

    ``target_tokens`` contains only text token IDs (no image placeholders).
    The server uses these for logprob gathering; including zeros at image
    positions corrupts the logprob computation.
    """
    token_ids = _extract_token_ids(model_input)

    if max_seq_len is not None and len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]
        weights = weights[:max_seq_len]

    if len(token_ids) < 2:
        raise ValueError("Need at least 2 tokens to build a supervised datum.")

    input_mi = _truncate_model_input(model_input)
    shifted_weights = weights[1:]

    text_target_tokens = _extract_text_only_token_ids(model_input)[1:]

    return tinker.Datum(
        model_input=input_mi,
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=shifted_weights,
                dtype="float32",
                shape=[len(shifted_weights)],
            ),
            "target_tokens": tinker.TensorData(
                data=[int(x) for x in text_target_tokens],
                dtype="int64",
                shape=[len(text_target_tokens)],
            ),
        },
    )


def build_datum_from_model_input_and_weights(
    model_input: tinker.ModelInput,
    token_weights: Sequence[int | float],
    *,
    max_seq_len: int | None = None,
    include_loss_mask: bool = False,
) -> RenderedSupervisedDatum:
    """Build a weighted datum from a multimodal-capable ``ModelInput``."""
    weights = [float(x) for x in token_weights]

    if _has_non_text_chunks(model_input):
        # Multimodal path: use our datum builder which produces text-only
        # target_tokens.  The upstream datum_from_model_input_weights includes
        # image-position zeros in target_tokens that corrupt the server's
        # logprob computation.
        datum = _build_multimodal_datum(model_input, weights, max_seq_len)
    elif datum_from_model_input_weights is not None:
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        if weight_tensor.numel() != model_input.length:
            raise ValueError(
                f"model_input/weights length mismatch: {model_input.length} != {weight_tensor.numel()}"
            )
        datum = datum_from_model_input_weights(
            model_input, weight_tensor, max_length=max_seq_len
        )
    else:
        token_ids = _extract_token_ids(model_input)
        return build_datum_from_tokens_and_weights(
            token_ids,
            token_weights,
            max_seq_len=max_seq_len,
            include_loss_mask=include_loss_mask,
        )

    if include_loss_mask:
        shifted_weights = [float(x) for x in datum.loss_fn_inputs["weights"].data]
        datum.loss_fn_inputs["loss_mask"] = tinker.TensorData(
            data=shifted_weights,
            dtype="float32",
            shape=[len(shifted_weights)],
        )

    return RenderedSupervisedDatum(
        token_ids=_rendered_sequence_ids_from_datum(datum),
        token_weights=[0.0] + [float(x) for x in datum.loss_fn_inputs["weights"].data],
        datum=datum,
    )


def build_next_token_datum(token_ids: Sequence[int]) -> tinker.Datum:
    """Build a standard next-token prediction datum without token weights."""
    tokens = [int(x) for x in token_ids]
    if len(tokens) < 2:
        raise ValueError("Need at least 2 tokens to build a next-token datum.")
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=tokens[1:],
                dtype="int64",
                shape=[len(tokens) - 1],
            )
        },
    )


def build_datum_from_token_mask(
    token_ids: Sequence[int],
    token_mask: Sequence[int | float],
    *,
    max_seq_len: int | None = None,
    include_loss_mask: bool = False,
) -> RenderedSupervisedDatum:
    """Build a weighted datum from an eval-protocol-style per-token mask."""
    token_weights = [1.0 if float(mask_value) > 0 else 0.0 for mask_value in token_mask]
    return build_datum_from_tokens_and_weights(
        token_ids,
        token_weights,
        max_seq_len=max_seq_len,
        include_loss_mask=include_loss_mask,
    )


def render_messages_to_datum(
    messages: Sequence[Mapping[str, Any]],
    *,
    renderer: Renderer,
    train_on_what: str | TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    max_seq_len: int | None = None,
    include_loss_mask: bool = False,
) -> RenderedSupervisedDatum:
    """Render a multi-turn conversation into the shared weighted datum format."""
    normalized_messages = normalize_messages(messages)
    rendered_input, weights = renderer.build_supervised_example(
        normalized_messages,
        train_on_what=parse_train_on_what(train_on_what),
    )
    weight_values = weights.tolist() if hasattr(weights, "tolist") else list(weights)
    if isinstance(rendered_input, tinker.ModelInput):
        return build_datum_from_model_input_and_weights(
            rendered_input,
            weight_values,
            max_seq_len=max_seq_len,
            include_loss_mask=include_loss_mask,
        )
    token_values = (
        rendered_input.tolist()
        if hasattr(rendered_input, "tolist")
        else list(rendered_input)
    )
    return build_datum_from_tokens_and_weights(
        token_values,
        weight_values,
        max_seq_len=max_seq_len,
        include_loss_mask=include_loss_mask,
    )


def _common_prefix_length(tokens_a: Sequence[int], tokens_b: Sequence[int]) -> int:
    min_len = min(len(tokens_a), len(tokens_b))
    for idx in range(min_len):
        if int(tokens_a[idx]) != int(tokens_b[idx]):
            return idx
    return min_len


def _render_preference_item_tokens(
    item: Mapping[str, Any],
    *,
    renderer: Renderer,
    tokenizer: Any,
) -> tuple[list[int], tinker.Datum] | None:
    if "messages" in item:
        messages = item.get("messages") or []
        if not messages:
            return None
        rendered = render_messages_to_datum(messages, renderer=renderer)
        return rendered.token_ids, rendered.datum
    if isinstance(item.get("text"), str):
        token_ids = [int(x) for x in tokenizer.encode(item["text"])]
        return token_ids, build_next_token_datum(token_ids)
    return None


def render_preference_pair(
    chosen: Mapping[str, Any],
    rejected: Mapping[str, Any],
    *,
    renderer: Renderer,
    tokenizer: Any,
    max_seq_len: int | None = None,
) -> RenderedPreferencePair | None:
    """Render a chosen/rejected pair through the shared tokenizer path."""
    chosen_rendered = _render_preference_item_tokens(
        chosen, renderer=renderer, tokenizer=tokenizer
    )
    rejected_rendered = _render_preference_item_tokens(
        rejected, renderer=renderer, tokenizer=tokenizer
    )
    if chosen_rendered is None or rejected_rendered is None:
        return None
    chosen_tokens, chosen_datum = chosen_rendered
    rejected_tokens, rejected_datum = rejected_rendered
    if len(chosen_tokens) < 2 or len(rejected_tokens) < 2:
        return None
    if max_seq_len is not None and (
        len(chosen_tokens) > max_seq_len or len(rejected_tokens) > max_seq_len
    ):
        return None

    return RenderedPreferencePair(
        chosen_tokens=chosen_tokens,
        rejected_tokens=rejected_tokens,
        response_start=_common_prefix_length(chosen_tokens, rejected_tokens),
        chosen_datum=chosen_datum,
        rejected_datum=rejected_datum,
    )
