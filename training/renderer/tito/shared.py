"""Shared TITO renderer primitives: tokenizer pinning and template normalization.

Prompt construction delegates to each pinned tokenizer's authoritative chat
template. This module owns only protocol normalization shared by the
per-model renderers; importing it must not load Tinker or Torch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fireworks.training.sdk import (
    TITOError,
    normalize_openai_tool_arguments,
)

_DYNAMIC_TEMPLATE_FIELDS = frozenset(
    {
        "chat_template_kwargs",
        "clear_thinking",
        "drop_thinking",
        "enable_thinking",
        "preserve_thinking",
        "reasoning_effort",
        "response_format",
        "thinking",
    }
)


@dataclass(frozen=True)
class TITORendererCertification:
    """Reviewed full-history capability for one renderer/tokenizer contract.

    This base certification does not certify the experimental incremental
    method; renderer authors own that additional model-specific contract.
    """

    certification_id: str
    renderer_names: frozenset[str]
    tokenizer_fingerprint: str
    renderer_factory: Callable[[Any, "TITORendererCertification"], Any]


def load_sidecar_tokenizer(path: str | Path) -> Any:
    """Load the pinned tokenizer and its bundled authoritative chat template."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        Path(path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("TITO sidecar tokenizer has no bundled chat template")
    return tokenizer


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        raise ValueError(
            "production TITO certification requires a fast tokenizer with a "
            "serializable backend"
        )
    contract = {
        "backend": json.loads(backend.to_str()),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_no_dynamic_template_fields(request: Any) -> None:
    """Reject per-request template options; the certified contract fixes them."""
    fields = sorted(_DYNAMIC_TEMPLATE_FIELDS.intersection(request.sampling_fields))
    if fields:
        raise TITOError(
            "tito_invalid_request",
            400,
            "TITO renderer/template options are fixed by the certified "
            "renderer contract; unsupported per-request fields: "
            + ", ".join(fields),
        )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def _normalize_template_tools(
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_tool in tools:
        tool = _plain(raw_tool)
        function = tool.get("function")
        if isinstance(function, dict):
            # Match ChatCompletionTool.model_dump() field order at Fireworks
            # chat admission. Templates render this function object directly,
            # so its envelope order is prompt-visible too.
            parameters = function.get("parameters") or {}
            tool["function"] = {
                "name": function.get("name"),
                "description": function.get("description"),
                # ChatCompletionFunction treats parameters as an opaque
                # mapping, so request admission preserves its key order.
                "parameters": _plain(parameters),
            }
            normalized.append(tool)
            continue
        normalized.append(tool)
    return normalized


def _normalize_template_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = [_plain(message) for message in messages]
    for message in normalized:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                # Templates iterate historical tool arguments as a mapping;
                # string arguments must round-trip to one.
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise TITOError(
                        "tito_invalid_request",
                        400,
                        "historical assistant tool arguments are not valid JSON",
                    ) from exc
    return normalized


def _ensure_tool_call_ids(
    message: Mapping[str, Any],
    completion_ids: Sequence[int],
) -> dict[str, Any]:
    calls = message.get("tool_calls") or []
    if not calls:
        return dict(message)
    normalized_calls: list[dict[str, Any]] = []
    for index, raw_call in enumerate(calls):
        call = dict(raw_call)
        function = dict(call.get("function") or {})
        if not call.get("id"):
            identity_function = {
                **function,
                "arguments": normalize_openai_tool_arguments(
                    function.get("arguments", "")
                ),
            }
            identity = {
                "completion_ids": [int(token) for token in completion_ids],
                "index": index,
                "function": identity_function,
            }
            digest = hashlib.sha256(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            call["id"] = f"call_{digest[:24]}"
        normalized_calls.append(call)
    return {**message, "tool_calls": normalized_calls}
