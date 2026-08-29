"""Renderer bridge for black-box agents that speak OpenAI chat completions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from training.renderer import Renderer, get_renderer
from training._vendor.tinker_cookbook_0_4_3.renderers.base import (
    Message,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
)

import training.renderer  # noqa: F401 - register cookbook renderer extensions
from training.renderer.reasoning_fields import ORIGINAL_REASONING_CONTENT


def flatten_content(content: Any) -> str:
    """Flatten OpenAI string or typed-part content to visible text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
    return "".join(parts)


class TurnRenderer(Protocol):
    def prompt_tokens(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        system_prompt: str,
    ) -> list[int]: ...

    def stop_sequences(self) -> list[Any]: ...

    def parse_completion(self, tokens: Sequence[int]) -> dict[str, Any]: ...


class CookbookTurnRenderer:
    """Adapt a tinker-cookbook renderer to OpenAI requests and responses."""

    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer
        self._stop = list(renderer.get_stop_sequences() or [])
        self._stop_tokens = [value for value in self._stop if isinstance(value, int)]

    def prompt_tokens(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        system_prompt: str,
    ) -> list[int]:
        specs = _to_tool_specs(tools)
        if specs:
            prefix = self._renderer.create_conversation_prefix_with_tools(
                specs,
                system_prompt,
            )
        elif system_prompt:
            prefix = [{"role": "system", "content": system_prompt}]
        else:
            prefix = []
        preserve_protocol_fields = bool(getattr(self._renderer, "_preserves_openai_protocol_fields", False))
        rendered = list(prefix) + [
            _to_renderer_message(
                message,
                preserve_protocol_fields=preserve_protocol_fields,
            )
            for message in messages
        ]
        return list(self._renderer.build_generation_prompt(rendered).to_ints())

    def stop_sequences(self) -> list[Any]:
        return list(self._stop)

    def parse_completion(self, tokens: Sequence[int]) -> dict[str, Any]:
        parsed, termination = self._renderer.parse_response(list(tokens))
        if not getattr(termination, "is_clean", True) and self._stop_tokens:
            retried, retried_termination = self._renderer.parse_response(
                [*tokens, self._stop_tokens[0]]
            )
            if getattr(retried_termination, "is_clean", False):
                parsed = retried
        return self._renderer.to_openai_message(parsed)


def build_turn_renderer(tokenizer: Any, renderer_name: str) -> TurnRenderer:
    return CookbookTurnRenderer(get_renderer(renderer_name, tokenizer))


def _to_tool_specs(
    tools: Sequence[dict[str, Any]] | None,
) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for tool in tools or []:
        function = tool.get("function") or tool
        specs.append(
            ToolSpec(
                name=str(function.get("name", "")),
                description=str(function.get("description", "") or ""),
                parameters=function.get("parameters")
                or {"type": "object", "properties": {}},
            )
        )
    return specs


def _to_renderer_message(
    raw: dict[str, Any],
    *,
    preserve_protocol_fields: bool = False,
) -> Message:
    visible_content = flatten_content(raw.get("content"))
    reasoning_content = raw.get("reasoning_content")
    content: str | list[TextPart | ThinkingPart] = visible_content
    if reasoning_content is not None:
        content = [
            ThinkingPart(
                type="thinking",
                thinking=str(reasoning_content),
            ),
            TextPart(
                type="text",
                text=visible_content,
            ),
        ]
    message: Message = {
        "role": raw.get("role", "user"),
        "content": content,
    }
    if preserve_protocol_fields and reasoning_content is not None:
        # Preserve the source field as well as the normalized ThinkingPart.
        # Opted-in renderers consume the former for exact field semantics.
        message[ORIGINAL_REASONING_CONTENT] = str(reasoning_content)  # type: ignore[typeddict-unknown-key]
    if preserve_protocol_fields and "recipient" in raw:
        message["recipient"] = raw.get("recipient")  # type: ignore[typeddict-unknown-key]
    if preserve_protocol_fields and "end_turn" in raw:
        message["end_turn"] = raw.get("end_turn")  # type: ignore[typeddict-unknown-key]
    tool_calls = raw.get("tool_calls") or []
    if tool_calls:
        message["tool_calls"] = [
            ToolCall(
                id=call.get("id"),
                function=ToolCall.FunctionBody(
                    name=str((call.get("function") or {}).get("name", "")),
                    arguments=(call.get("function") or {}).get(
                        "arguments",
                        "",
                    )
                    or "{}",
                ),
            )
            for call in tool_calls
        ]
    if message["role"] == "tool":
        if raw.get("tool_call_id"):
            message["tool_call_id"] = raw["tool_call_id"]
        if raw.get("name"):
            message["name"] = raw["name"]
    return message


__all__ = [
    "CookbookTurnRenderer",
    "TurnRenderer",
    "build_turn_renderer",
    "flatten_content",
]
