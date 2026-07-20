"""Renderer for the MiniMax M3 chat protocol."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any, Literal

import tinker
import torch
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
    parse_response_for_stop_token,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

ThinkingMode = Literal["enabled", "disabled", "adaptive"]

_NS = "]<]minimax[>["
_BOD = "]~!b["
_BOS = "]~b]"
_EOS = "[e~["
_THINK_BEGIN = "<mm:think>"
_THINK_END = "</mm:think>"
_TOOL_CALL_BEGIN = _NS + "<tool_call>"
_TOOL_CALL_END = _NS + "</tool_call>"
_IMAGE = "]<]image[>["
_VIDEO = "]<]video[>["

_DEFAULT_SYSTEM = (
    "Your model version is MiniMax-M3, developed by MiniMax. Knowledge cutoff: January 2026. "
    "Founded in early 2022, MiniMax is a global AI foundation model company committed to "
    "advancing the frontiers of AI towards AGI."
)
_DEFAULT_DEVELOPER = "You are a helpful assistant."
_DEFAULT_ROOT_SENTINEL = "__MINIMAX_M3_DEFAULT_ROOT__"
_DEFAULT_DEVELOPER_SENTINEL = "__MINIMAX_M3_DEFAULT_DEVELOPER__"

_TOOL_CALL_RE = re.compile(
    re.escape(_TOOL_CALL_BEGIN) + r"\n?(.*?)" + re.escape(_TOOL_CALL_END),
    re.DOTALL,
)


def _visible_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, Mapping) and item.get("type") == "image":
                parts.append(_IMAGE)
            elif isinstance(item, Mapping) and item.get("type") == "video":
                parts.append(_VIDEO)
            elif isinstance(item, Mapping) and isinstance(item.get("output"), str):
                parts.append(item["output"])
        return "".join(parts)
    return str(content)


def _assistant_parts(content: Any) -> tuple[str, str]:
    if isinstance(content, str):
        for begin, end in ((_THINK_BEGIN, _THINK_END), ("<think>", "</think>")):
            if end in content:
                reasoning = content.split(end, 1)[0].split(begin)[-1].strip("\n")
                visible = content.split(end, 1)[1].strip("\n")
                return reasoning, visible
        return "", content
    reasoning: list[str] = []
    visible: list[str] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "thinking":
                reasoning.append(str(part.get("thinking", "")))
            elif part.get("type") == "text":
                visible.append(str(part.get("text", "")))
    return "".join(reasoning), "".join(visible)


def _xml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return json.dumps(value)
    return str(value)


def _to_xml(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return "".join(
            f"{_NS}<{key}>{_to_xml(item)}{_NS}</{key}>"
            for key, item in value.items()
            if item is not None
        )
    if isinstance(value, list):
        return "".join(f"{_NS}<item>{_to_xml(item)}{_NS}</item>" for item in value)
    return _xml_scalar(value)


def _tool_arguments(tool_call: ToolCall) -> dict[str, Any]:
    arguments = json.loads(tool_call.function.arguments or "{}")
    if not isinstance(arguments, dict):
        raise TypeError("MiniMax M3 tool arguments must be a JSON object.")
    return arguments


def _format_tool_calls(tool_calls: list[ToolCall]) -> str:
    calls: list[str] = []
    for tool_call in tool_calls:
        body = [f'{_NS}<invoke name="{tool_call.function.name}">']
        for key, value in _tool_arguments(tool_call).items():
            if value is None:
                continue
            body.append(f"{_NS}<{key}>{_to_xml(value)}{_NS}</{key}>")
        body.append(f"{_NS}</invoke>\n")
        calls.append("".join(body))
    return f"{_TOOL_CALL_BEGIN}\n{''.join(calls)}{_TOOL_CALL_END}"


def _tool_function(tool: ToolSpec | Mapping[str, Any]) -> Mapping[str, Any]:
    raw = dict(tool)
    function = raw.get("function")
    return function if isinstance(function, Mapping) else raw


def _format_tools(tools: list[ToolSpec | Mapping[str, Any]]) -> str:
    rendered = "".join(
        f"<tool>{json.dumps(_tool_function(tool), ensure_ascii=False)}</tool>\n"
        for tool in tools
    )
    example = (
        f"{_TOOL_CALL_BEGIN}\n"
        f'{_NS}<invoke name="tool-name-1">'
        f"{_NS}<param-1>value-1{_NS}</param-1>"
        f"{_NS}<param-2>{_NS}<item>{_NS}<key-a>val-a{_NS}</key-a>"
        f"{_NS}<key-b>val-b{_NS}</key-b>{_NS}</item>{_NS}</param-2>"
        f"{_NS}</invoke>\n"
        f'{_NS}<invoke name="tool-name-2">'
        f"{_NS}<param-1>value-1{_NS}</param-1>{_NS}</invoke>\n"
        f"{_TOOL_CALL_END}"
    )
    return (
        "\n\n# Tools\nYou may call one or more tools to assist with the user query.\n"
        "Here are the tools available in JSONSchema format:\n\n<tools>\n"
        f"{rendered}</tools>\n\n"
        f"To call tools, wrap all invocations in a single {_TOOL_CALL_BEGIN}{_TOOL_CALL_END} block. "
        "Parameter values containing nested objects or arrays are recursively expanded into XML elements. "
        f"Example:\n\n{example}"
    )


def _thinking_instructions(mode: ThinkingMode) -> str:
    mode_text = {
        "enabled": (
            "Current thinking mode: enabled. You MUST think step by step before every response, "
            "including after receiving function/tool results.\n"
        ),
        "disabled": "Current thinking mode: disabled. Do not output any thinking process.\n",
        "adaptive": (
            "Current thinking mode: adaptive. You are encouraged to think for complex decision-making, "
            "multi-step reasoning, or when analyzing function/tool results.\n"
        ),
    }[mode]
    return (
        "\n\n<thinking_instructions>\n"
        f"You have a thinking capability that allows you to reason step by step before responding. "
        f"When thinking is enabled, wrap your reasoning in {_THINK_BEGIN}{_THINK_END} tags before your response. "
        f"When thinking is disabled, begin your response directly after the {_THINK_END} prefix. "
        "When thinking is adaptive, decide on your own whether to think for the current turn.\n"
        f"{mode_text}</thinking_instructions>"
    )


def _element_value(element: ET.Element) -> Any:
    children = list(element)
    if not children:
        text = element.text or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    if all(child.tag == "item" for child in children):
        return [_element_value(child) for child in children]
    return {child.tag: _element_value(child) for child in children}


def _parse_tool_block(
    raw: str, raw_text: str
) -> tuple[list[ToolCall], list[UnparsedToolCall]]:
    try:
        root = ET.fromstring(f"<root>{raw.replace(_NS, '')}</root>")
    except ET.ParseError as exc:
        return [], [UnparsedToolCall(raw_text=raw_text, error=str(exc))]
    calls: list[ToolCall] = []
    for invoke in root.findall("invoke"):
        name = invoke.attrib.get("name")
        if not name:
            continue
        arguments = {child.tag: _element_value(child) for child in invoke}
        calls.append(
            ToolCall(
                function=ToolCall.FunctionBody(
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
            )
        )
    if calls:
        return calls, []
    return [], [UnparsedToolCall(raw_text=raw_text, error="No <invoke> element found")]


class MiniMaxM3Renderer(Renderer):
    def __init__(
        self, tokenizer: Tokenizer, thinking_mode: ThinkingMode = "adaptive"
    ) -> None:
        super().__init__(tokenizer)
        if thinking_mode not in {"enabled", "disabled", "adaptive"}:
            raise ValueError(f"Unsupported MiniMax M3 thinking mode: {thinking_mode!r}")
        self.thinking_mode = thinking_mode
        self._pending_tools: list[ToolSpec | Mapping[str, Any]] | None = None

    @property
    def has_extension_property(self) -> bool:
        return True

    @property
    def _bos_tokens(self) -> list[int]:
        # M3 has separate BOD (conversation start) and BOS (role prefix)
        # tokens. ``tokenizer.bos_token_id`` is the latter, so encode BOD here.
        return list(self.tokenizer.encode(_BOD, add_special_tokens=False))

    @property
    def _end_message_token(self) -> int:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            return int(eos_token_id)
        tokens = self.tokenizer.encode(_EOS, add_special_tokens=False)
        if len(tokens) != 1:
            raise RuntimeError(
                f"Expected {_EOS!r} to encode as one token, got {tokens}"
            )
        return int(tokens[0])

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    @staticmethod
    def _group_tool_messages(messages: list[Message]) -> list[Message]:
        grouped: list[Message] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message["role"] != "tool":
                grouped.append(message)
                index += 1
                continue
            outputs: list[dict[str, str]] = []
            tool_messages: list[Message] = []
            while index < len(messages) and messages[index]["role"] == "tool":
                tool_message = messages[index]
                tool_messages.append(tool_message)
                outputs.append({"output": _visible_text(tool_message["content"])})
                index += 1
            grouped_tool = Message(role="tool", content=outputs)
            if any("trainable" in tool_message for tool_message in tool_messages):
                grouped_tool["trainable"] = any(
                    bool(tool_message.get("trainable", False))
                    for tool_message in tool_messages
                )
            grouped.append(grouped_tool)
        return grouped

    def _preprocess_messages(self, messages: list[Message]) -> list[Message]:
        remaining = list(messages)
        uses_customized_weights = any("trainable" in message for message in remaining)
        root_message = Message(role="root", content=_DEFAULT_ROOT_SENTINEL)
        developer_message = Message(
            role="developer",
            content=_DEFAULT_DEVELOPER_SENTINEL,
        )
        if remaining and remaining[0]["role"] == "root":
            source = remaining.pop(0)
            root_message["content"] = source["content"]
            if "trainable" in source:
                root_message["trainable"] = source["trainable"]
            if remaining and remaining[0]["role"] in {"system", "developer"}:
                source = remaining.pop(0)
                developer_message["content"] = source["content"]
                if "trainable" in source:
                    developer_message["trainable"] = source["trainable"]
        elif remaining and remaining[0]["role"] in {"system", "developer"}:
            source = remaining.pop(0)
            developer_message["content"] = source["content"]
            if "trainable" in source:
                developer_message["trainable"] = source["trainable"]
        if uses_customized_weights:
            root_message.setdefault("trainable", False)
            developer_message.setdefault("trainable", False)
        prefix = [root_message, developer_message]
        return prefix + self._group_tool_messages(remaining)

    def _encode_chunk(self, text: str) -> tinker.types.EncodedTextChunk | None:
        tokens = list(self.tokenizer.encode(text, add_special_tokens=False))
        return tinker.types.EncodedTextChunk(tokens=tokens) if tokens else None

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        del ctx
        role = message["role"]
        if role == "root":
            content = message["content"]
            body = (
                _DEFAULT_SYSTEM
                if content == _DEFAULT_ROOT_SENTINEL or not _visible_text(content)
                else _visible_text(content)
            )
            header_text = f"{_BOS}system\n"
            output_text = (
                body + _thinking_instructions(self.thinking_mode) + f"{_EOS}\n"
            )
        elif role == "developer":
            content = message["content"]
            body = (
                _DEFAULT_DEVELOPER
                if content == _DEFAULT_DEVELOPER_SENTINEL or not _visible_text(content)
                else _visible_text(content)
            )
            if self._pending_tools:
                body += _format_tools(self._pending_tools)
                self._pending_tools = None
            header_text = f"{_BOS}developer\n"
            output_text = body + f"{_EOS}\n"
        elif role == "assistant":
            reasoning, visible = _assistant_parts(message["content"])
            body = f"{_THINK_BEGIN}{reasoning}{_THINK_END}" if reasoning else _THINK_END
            body += visible
            if message.get("tool_calls"):
                body += _format_tool_calls(message["tool_calls"])
            header_text = f"{_BOS}ai\n"
            output_text = body + f"{_EOS}\n"
        elif role == "user":
            header_text = f"{_BOS}user\n"
            output_text = _visible_text(message["content"]) + f"{_EOS}\n"
        elif role == "tool":
            header_text = f"{_BOS}tool"
            content = message["content"]
            entries = (
                content
                if isinstance(content, list)
                else [{"output": _visible_text(content)}]
            )
            responses = "".join(
                f"\n<response>{_visible_text(entry.get('output') if isinstance(entry, Mapping) else entry)}</response>"
                for entry in entries
            )
            output_text = responses + f"{_EOS}\n"
        else:
            raise ValueError(f"Unsupported MiniMax M3 role: {role!r}")

        output = self._encode_chunk(output_text)
        return RenderedMessage(
            header=self._encode_chunk(header_text),
            output=[output] if output is not None else [],
        )

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        del ctx
        rendered_role = "ai" if role == "assistant" else role
        suffix = f"{_BOS}{rendered_role}\n"
        if rendered_role == "ai":
            if self.thinking_mode == "enabled":
                suffix += _THINK_BEGIN
            elif self.thinking_mode == "disabled":
                suffix += _THINK_END
        return list(self.tokenizer.encode(suffix, add_special_tokens=False))

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

    def create_conversation_prefix_with_tools(
        self,
        tools: list[ToolSpec],
        system_prompt: str = "",
    ) -> list[Message]:
        self._pending_tools = list(tools)
        return [
            Message(
                role="system",
                content=system_prompt or _DEFAULT_DEVELOPER_SENTINEL,
            )
        ]

    def _normalize_response_tokens(self, response: list[int]) -> list[int]:
        if self.thinking_mode == "enabled":
            return (
                list(self.tokenizer.encode(_THINK_BEGIN, add_special_tokens=False))
                + response
            )
        if self.thinking_mode == "disabled":
            return (
                list(self.tokenizer.encode(_THINK_END, add_special_tokens=False))
                + response
            )
        return response

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        response = self._normalize_response_tokens(response)
        message, termination = parse_response_for_stop_token(
            response=response,
            tokenizer=self.tokenizer,
            stop_token=self._end_message_token,
        )
        if termination == ParseTermination.MALFORMED:
            return message, termination
        assert isinstance(message["content"], str)
        raw_content = message["content"]
        tool_calls: list[ToolCall] = []
        unparsed: list[UnparsedToolCall] = []
        clean_parts: list[str] = []
        offset = 0
        for match in _TOOL_CALL_RE.finditer(raw_content):
            clean_parts.append(raw_content[offset : match.start()])
            parsed, failed = _parse_tool_block(match.group(1), match.group(0))
            tool_calls.extend(parsed)
            unparsed.extend(failed)
            offset = match.end()
        clean_parts.append(raw_content[offset:])
        clean = "".join(clean_parts)
        reasoning, visible = _assistant_parts(clean)
        content: list[dict[str, str]] = []
        if reasoning:
            content.append({"type": "thinking", "thinking": reasoning})
        if visible:
            content.append({"type": "text", "text": visible})
        message["content"] = content if content else visible
        if tool_calls:
            message["tool_calls"] = tool_calls
        if unparsed:
            message["unparsed_tool_calls"] = unparsed
        return message, termination

    def to_openai_message(self, message: Message) -> dict[str, Any]:
        result: dict[str, Any] = {"role": message["role"]}
        reasoning, visible = _assistant_parts(message["content"])
        result["content"] = visible
        if reasoning:
            result["reasoning_content"] = reasoning
        if message.get("tool_calls"):
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": _tool_arguments(tool_call),
                    },
                }
                for tool_call in message["tool_calls"]
            ]
        return result


def _factory(tokenizer: Tokenizer, image_processor=None) -> MiniMaxM3Renderer:
    del image_processor
    return MiniMaxM3Renderer(tokenizer)


register_renderer("minimax_m3", _factory)
