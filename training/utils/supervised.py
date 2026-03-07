"""Shared supervised-rendering helpers.

This module gives cookbook training code one token-level representation for
supervised data:

- text/tool conversations rendered via a Tinker renderer
- eval-protocol-style token trajectories with a per-token mask

Both paths end in the same ``tinker.Datum`` schema with ``target_tokens`` and
token-level ``weights`` so training uses the same spans that the UI shows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import tinker
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Message, Renderer, TrainOnWhat, get_renderer
from tinker_cookbook.supervised.common import datum_from_tokens_weights


@dataclass(frozen=True)
class RenderedSupervisedDatum:
    """Rendered tokens, token-level weights, and the final training datum."""

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
    return get_renderer(resolve_renderer_name(tokenizer_model, renderer_name), tokenizer)


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Normalize common tool-call shapes into Tinker's ``{\"name\", \"args\"}`` form."""
    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls or []:
        if not isinstance(tool_call, Mapping):
            raise TypeError(f"Unsupported tool call type: {type(tool_call)!r}")

        if isinstance(tool_call.get("name"), str) and isinstance(tool_call.get("args"), Mapping):
            normalized.append({
                "name": tool_call["name"],
                "args": dict(tool_call["args"]),
            })
            continue

        function = tool_call.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            raw_args = function.get("arguments", {})
            if isinstance(raw_args, str):
                parsed_args = json.loads(raw_args) if raw_args else {}
            elif isinstance(raw_args, Mapping):
                parsed_args = dict(raw_args)
            else:
                raise TypeError(f"Unsupported tool call arguments type: {type(raw_args)!r}")
            normalized.append({
                "name": function["name"],
                "args": parsed_args,
            })
            continue

        raise ValueError(f"Unsupported tool call shape: {tool_call}")
    return normalized


def _normalize_content(content: Any) -> str:
    """Convert OpenAI-style message content into plain text for text renderers."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        if isinstance(content.get("text"), str):
            return content["text"]
        raise TypeError(f"Unsupported message content mapping: {content}")
    if isinstance(content, Sequence):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, Mapping):
                raise TypeError(f"Unsupported message content part: {part!r}")
            part_type = part.get("type")
            if part_type == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
                continue
            raise ValueError(
                "Multimodal content is not supported by the shared text renderer. "
                "Use the token_ids + mask path for eval-protocol visual trajectories."
            )
        return "".join(text_parts)
    raise TypeError(f"Unsupported message content type: {type(content)!r}")


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
            normalized_message["thinking"] = thinking

        trainable = message.get("trainable")
        if trainable is not None:
            normalized_message["trainable"] = bool(trainable)

        normalized.append(normalized_message)

    return normalized


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
        raise ValueError(f"tokens/weights length mismatch: {len(tokens)} != {len(weights)}")
    if len(tokens) < 2:
        raise ValueError("Need at least 2 tokens to build a supervised datum.")

    if max_seq_len is not None:
        tokens = tokens[:max_seq_len]
        weights = weights[:max_seq_len]
        if len(tokens) < 2:
            raise ValueError("Truncation left fewer than 2 tokens.")

    token_tensor = torch.tensor(tokens, dtype=torch.int64)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    datum = datum_from_tokens_weights(token_tensor, weight_tensor)

    if include_loss_mask:
        shifted_weights = [float(x) for x in weights[1:]]
        datum.loss_fn_inputs["loss_mask"] = tinker.TensorData(
            data=shifted_weights,
            dtype="float32",
            shape=[len(shifted_weights)],
        )

    return RenderedSupervisedDatum(
        token_ids=tokens,
        token_weights=weights,
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
    tokens, weights = renderer.build_supervised_example(
        normalized_messages,
        train_on_what=parse_train_on_what(train_on_what),
    )
    return build_datum_from_tokens_and_weights(
        tokens.tolist(),
        weights.tolist(),
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
) -> list[int]:
    if "messages" in item:
        messages = item.get("messages") or []
        if not messages:
            return []
        return render_messages_to_datum(messages, renderer=renderer).token_ids
    if isinstance(item.get("text"), str):
        return [int(x) for x in tokenizer.encode(item["text"])]
    return []


def render_preference_pair(
    chosen: Mapping[str, Any],
    rejected: Mapping[str, Any],
    *,
    renderer: Renderer,
    tokenizer: Any,
    max_seq_len: int | None = None,
) -> RenderedPreferencePair | None:
    """Render a chosen/rejected pair through the shared tokenizer path."""
    chosen_tokens = _render_preference_item_tokens(chosen, renderer=renderer, tokenizer=tokenizer)
    rejected_tokens = _render_preference_item_tokens(rejected, renderer=renderer, tokenizer=tokenizer)
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
        chosen_datum=build_next_token_datum(chosen_tokens),
        rejected_datum=build_next_token_datum(rejected_tokens),
    )
