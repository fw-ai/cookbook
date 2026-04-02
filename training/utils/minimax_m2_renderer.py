"""Renderer for MiniMax M2 chat templates.

This matches the HuggingFace ``MiniMaxAI/MiniMax-M2`` tokenizer template:

- Always starts with the MiniMax BOS token and a system message
- Renders assistant turns with the ``ai`` role prefix
- Preserves thinking only for assistant turns after the last user message
- Uses MiniMax's XML-style tool-call format
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import tinker
import torch
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    TrainOnWhat,
    UnparsedToolCall,
    parse_response_for_stop_token,
    parse_think_blocks,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

_DEFAULT_SYSTEM_PROMPT = (
    "You are MiniMax-M2, a helpful AI assistant built by MiniMax. "
    "Knowledge cutoff: 2025-06."
)
_BOS_TEXT = "]~!b["
_ROLE_PREFIX = "]~b]"
_END_MESSAGE_TEXT = "[e~["
_TOOL_CALL_BEGIN = "<minimax:tool_call>"
_TOOL_CALL_END = "</minimax:tool_call>"
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<minimax:tool_call>\n?(.*?)</minimax:tool_call>",
    re.DOTALL,
)
_TOOL_INVOKE_RE = re.compile(
    r'<invoke name="([^"]+)">\n?(.*?)</invoke>\n?',
    re.DOTALL,
)
_TOOL_PARAMETER_RE = re.compile(
    r'<parameter name="([^"]+)">(.*?)</parameter>',
    re.DOTALL,
)


def _visible_text(content: Any) -> str:
    """Render visible text from string or structured content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                rendered_parts.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                rendered_parts.append(item["text"])
                continue
            if isinstance(item.get("output"), str):
                rendered_parts.append(item["output"])
        return "".join(rendered_parts)
    return str(content)


def _extract_assistant_reasoning_and_text(content: Any) -> tuple[str, str]:
    """Split assistant content into reasoning and visible text."""
    if isinstance(content, str):
        if "</think>" not in content:
            return "", content
        reasoning = content.split("</think>")[0].split("<think>")[-1].strip("\n")
        text = content.split("</think>")[-1].strip("\n")
        return reasoning, text

    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "thinking" and isinstance(part.get("thinking"), str):
                reasoning_parts.append(part["thinking"])
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
    return "".join(reasoning_parts), "".join(text_parts)


def _truncate_assistant_history(content: Any) -> str:
    """Drop thinking content from assistant history before the last user turn."""
    _, visible = _extract_assistant_reasoning_and_text(content)
    return visible


def _format_parameter_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _parse_parameter_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    parsed = json.loads(raw_arguments) if raw_arguments else {}
    if not isinstance(parsed, dict):
        raise TypeError(
            f"MiniMax tool arguments must be a JSON object, got {type(parsed)!r}"
        )
    return parsed


def _format_tool_call(tool_call: ToolCall) -> str:
    arguments = _normalize_tool_arguments(tool_call.function.arguments)
    parts = [f'<invoke name="{tool_call.function.name}">\n']
    for key, value in arguments.items():
        parts.append(
            f'<parameter name="{key}">{_format_parameter_value(value)}</parameter>\n'
        )
    parts.append("</invoke>\n")
    return "".join(parts)


def _format_tool_calls(tool_calls: list[ToolCall]) -> str:
    rendered_calls = "".join(_format_tool_call(tool_call) for tool_call in tool_calls)
    return f"{_TOOL_CALL_BEGIN}\n{rendered_calls}{_TOOL_CALL_END}"


def _parse_tool_call_block(
    raw_block: str,
    raw_text: str,
) -> tuple[list[ToolCall], list[UnparsedToolCall]]:
    invoke_matches = list(_TOOL_INVOKE_RE.finditer(raw_block))
    if not invoke_matches:
        return [], [
            UnparsedToolCall(raw_text=raw_text, error="No <invoke> block found")
        ]

    tool_calls: list[ToolCall] = []
    unparsed_tool_calls: list[UnparsedToolCall] = []
    for match in invoke_matches:
        name = match.group(1)
        body = match.group(2)
        arguments = {
            param_match.group(1): _parse_parameter_value(param_match.group(2))
            for param_match in _TOOL_PARAMETER_RE.finditer(body)
        }
        tool_calls.append(
            ToolCall(
                function=ToolCall.FunctionBody(
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
            )
        )

        stripped_body = _TOOL_PARAMETER_RE.sub("", body).strip()
        if stripped_body:
            unparsed_tool_calls.append(
                UnparsedToolCall(
                    raw_text=match.group(0),
                    error=f"Unexpected content inside <invoke>: {stripped_body!r}",
                )
            )

    return tool_calls, unparsed_tool_calls


def _extract_tool_calls_from_content(
    content: str,
) -> tuple[str, list[ToolCall], list[UnparsedToolCall]]:
    """Strip MiniMax tool-call blocks and parse them into structured calls."""
    pos = 0
    cleaned_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    unparsed_tool_calls: list[UnparsedToolCall] = []

    for match in _TOOL_CALL_BLOCK_RE.finditer(content):
        text_before = content[pos : match.start()]
        if text_before.endswith("\n"):
            text_before = text_before[:-1]
        cleaned_parts.append(text_before)
        parsed_calls, parsed_unparsed = _parse_tool_call_block(
            raw_block=match.group(1),
            raw_text=match.group(0),
        )
        tool_calls.extend(parsed_calls)
        unparsed_tool_calls.extend(parsed_unparsed)
        pos = match.end()

    cleaned_parts.append(content[pos:])
    return "".join(cleaned_parts), tool_calls, unparsed_tool_calls


class MiniMaxM2Renderer(Renderer):
    """Renderer for ``MiniMaxAI/MiniMax-M2`` with optional history truncation."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        strip_thinking_from_history: bool = True,
    ) -> None:
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history
        self.default_system_prompt = self._detect_default_system_prompt()

    @property
    def has_extension_property(self) -> bool:
        return not self.strip_thinking_from_history

    @property
    def _bos_tokens(self) -> list[int]:
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_token_id is not None:
            return [int(bos_token_id)]
        return self.tokenizer.encode(_BOS_TEXT, add_special_tokens=False)

    @property
    def _end_message_token(self) -> int:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            return int(eos_token_id)
        tokens = self.tokenizer.encode(_END_MESSAGE_TEXT, add_special_tokens=False)
        assert (
            len(tokens) == 1
        ), f"Expected single token for {_END_MESSAGE_TEXT!r}, got {tokens}"
        return int(tokens[0])

    def _detect_default_system_prompt(self) -> str:
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_chat_template is None:
            return _DEFAULT_SYSTEM_PROMPT

        try:
            rendered = apply_chat_template(
                [],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            return _DEFAULT_SYSTEM_PROMPT

        prefix = f"{_BOS_TEXT}{_ROLE_PREFIX}system\n"
        suffix = f"{_END_MESSAGE_TEXT}\n"
        if rendered.startswith(prefix) and rendered.endswith(suffix):
            extracted = rendered[len(prefix) : -len(suffix)]
            if extracted:
                return extracted
        return _DEFAULT_SYSTEM_PROMPT

    @staticmethod
    def _ensure_system_message(messages: list[Message]) -> list[Message]:
        if not messages or messages[0]["role"] != "system":
            return [Message(role="system", content=""), *messages]
        return list(messages)

    @staticmethod
    def _group_tool_messages(messages: list[Message]) -> list[Message]:
        grouped: list[Message] = []
        idx = 0
        while idx < len(messages):
            message = messages[idx]
            if message["role"] != "tool":
                grouped.append(message)
                idx += 1
                continue

            tool_outputs: list[dict[str, str]] = []
            while idx < len(messages) and messages[idx]["role"] == "tool":
                tool_outputs.append({"output": _visible_text(messages[idx]["content"])})
                idx += 1
            if len(tool_outputs) == 1:
                grouped.append({**message, "content": tool_outputs[0]["output"]})
            else:
                grouped.append({**message, "content": tool_outputs})
        return grouped

    def _preprocess_messages(self, messages: list[Message]) -> list[Message]:
        return self._group_tool_messages(self._ensure_system_message(messages))

    def _role_for_message(self, message: Message) -> str:
        if message["role"] == "assistant":
            return "ai"
        return message["role"]

    def _render_system_message(self, message: Message) -> str:
        content = _visible_text(message["content"])
        if content:
            return content
        return self.default_system_prompt

    def _render_assistant_message(self, message: Message, ctx: RenderContext) -> str:
        should_truncate = (
            self.strip_thinking_from_history
            and ctx.last_user_index >= 0
            and ctx.idx < ctx.last_user_index
        )
        if should_truncate:
            rendered = _truncate_assistant_history(message["content"])
        else:
            reasoning, visible = _extract_assistant_reasoning_and_text(
                message["content"]
            )
            rendered = visible
            if reasoning:
                rendered = f"<think>\n{reasoning}\n</think>\n\n{visible}"

        if "tool_calls" in message and message["tool_calls"]:
            rendered += "\n" + _format_tool_calls(message["tool_calls"])
        return rendered

    def _assistant_uses_blank_line_before_tool_calls(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> bool:
        if message["role"] != "assistant" or not message.get("tool_calls"):
            return False

        should_truncate = (
            self.strip_thinking_from_history
            and ctx.last_user_index >= 0
            and ctx.idx < ctx.last_user_index
        )
        if should_truncate:
            visible = _truncate_assistant_history(message["content"])
            return visible == ""

        reasoning, visible = _extract_assistant_reasoning_and_text(message["content"])
        return reasoning == "" and visible == ""

    def _render_tool_message(self, message: Message) -> RenderedMessage:
        header_str = f"{_ROLE_PREFIX}tool\n"

        content = message["content"]
        if isinstance(content, str):
            body = f"<response>{content}</response>"
        else:
            body_parts: list[str] = []
            for idx, item in enumerate(content):
                if isinstance(item, Mapping) and isinstance(item.get("output"), str):
                    rendered_item = item["output"]
                elif (
                    isinstance(item, Mapping)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ):
                    rendered_item = item["text"]
                else:
                    rendered_item = str(item)
                prefix = "" if idx == 0 else "\n"
                body_parts.append(f"{prefix}<response>{rendered_item}</response>")
            body = "".join(body_parts)
        body += f"{_END_MESSAGE_TEXT}\n"

        header = (
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(header_str, add_special_tokens=False),
            )
            if header_str
            else None
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(body, add_special_tokens=False),
            )
        ]
        return RenderedMessage(header=header, output=output)

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        if message["role"] == "tool":
            return self._render_tool_message(message)

        role = self._role_for_message(message)
        header_str = f"{_ROLE_PREFIX}{role}\n"

        if message["role"] == "assistant":
            output_content = self._render_assistant_message(message, ctx)
            if self._assistant_uses_blank_line_before_tool_calls(message, ctx):
                header_str += "\n"
                output_content = output_content.removeprefix("\n")
        elif message["role"] == "system":
            output_content = self._render_system_message(message)
        else:
            output_content = _visible_text(message["content"])

        output_content += f"{_END_MESSAGE_TEXT}\n"
        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False),
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_content, add_special_tokens=False),
            )
        ]
        return RenderedMessage(header=header, output=output)

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        del ctx
        render_role = "ai" if role == "assistant" else role
        suffix = f"{_ROLE_PREFIX}{render_role}\n"
        if render_role == "ai":
            suffix += "<think>\n"
        return self.tokenizer.encode(suffix, add_special_tokens=False)

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        return super().build_generation_prompt(
            self._preprocess_messages(messages),
            role=role,
            prefill=prefill,
        )

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        return super().build_supervised_example(
            self._preprocess_messages(messages),
            train_on_what=train_on_what,
        )

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response=response,
            tokenizer=self.tokenizer,
            stop_token=self._end_message_token,
        )
        if not parse_success:
            return assistant_message, False

        assert isinstance(assistant_message["content"], str)
        rendered_content, tool_calls, unparsed_tool_calls = (
            _extract_tool_calls_from_content(assistant_message["content"])
        )
        content_parts = parse_think_blocks(rendered_content)
        assistant_message["content"] = (
            content_parts if content_parts is not None else rendered_content
        )
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if unparsed_tool_calls:
            assistant_message["unparsed_tool_calls"] = unparsed_tool_calls
        return assistant_message, True

    def to_openai_message(self, message: Message) -> dict[str, Any]:
        result: dict[str, Any] = {"role": message["role"]}

        content = message["content"]
        if isinstance(content, str):
            reasoning, visible = _extract_assistant_reasoning_and_text(content)
            result["content"] = visible if reasoning else content
            if reasoning:
                result["reasoning_content"] = reasoning
        else:
            thinking_parts: list[str] = []
            text_parts: list[str] = []
            for part in content:
                if part["type"] == "thinking":
                    thinking_parts.append(part["thinking"])
                elif part["type"] == "text":
                    text_parts.append(part["text"])
            result["content"] = "".join(text_parts)
            if thinking_parts:
                result["reasoning_content"] = "".join(thinking_parts)

        if "tool_calls" in message and message["tool_calls"]:
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": _normalize_tool_arguments(
                            tool_call.function.arguments
                        ),
                    },
                }
                for tool_call in message["tool_calls"]
            ]

        if message["role"] == "tool":
            if "tool_call_id" in message:
                result["tool_call_id"] = message["tool_call_id"]
            if "name" in message:
                result["name"] = message["name"]

        return result


def _minimax_m2_factory(
    tokenizer: Tokenizer, image_processor=None
) -> MiniMaxM2Renderer:
    return MiniMaxM2Renderer(tokenizer)


register_renderer("minimax_m2", _minimax_m2_factory)
