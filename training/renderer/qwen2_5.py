"""Qwen2.5 renderer pinned to the legacy Fireworks V1 training contract.

The production Qwen2.5 32B artifact predates Managed Training V2 and carries a
small but observable tokenizer-template override relative to the public
Hugging Face revision: the tool-call example uses literal double braces.  V1
loaded that artifact from GCS and rendered the whole conversation through its
``tokenizer.chat_template``.

This renderer deliberately does not inherit from the Qwen3 renderer.  Qwen3's
thinking/history rules are unrelated, and its per-message tokenization loses
Qwen2.5 parity when BPE merges cross message boundaries.  Instead, this class
renders the complete V1 wire string once, tokenizes it once, and derives loss
weights from character offsets using the same conservative boundary rule as
the V1 trainer.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import tinker
import torch
from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
    parse_content_blocks,
    parse_response_for_stop_token,
)
from tinker_cookbook.tokenizer_utils import Tokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
PRODUCTION_EOS_TOKEN = "<|im_end|>"
PRODUCTION_EOS_TOKEN_ID = 151645

_IM_START = "<|im_start|>"
_IM_END = PRODUCTION_EOS_TOKEN
_TOOLS_MARKER_ROLE = "_qwen2_5_tools"
_TOOLS_MARKER_KEY = "_qwen2_5_tool_specs"

_TOOLS_PREAMBLE = """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>"""
_TOOLS_EPILOGUE = """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


@dataclass(frozen=True)
class _OutputSpan:
    message_index: int | None
    role: str
    start: int
    end: int


@dataclass(frozen=True)
class _RenderTrace:
    text: str
    messages: tuple[Message, ...]
    spans: tuple[_OutputSpan, ...]


def _visible_text(content: Any) -> str:
    """Match the V1 template's text-only content contract."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # The production Jinja concatenates ``message.content`` directly with
    # strings. Lists (including text-only multipart content) therefore fail
    # instead of being silently flattened into a different wire contract.
    raise TypeError(f"Qwen2.5 content must be text, got {type(content)!r}")


def _jinja_tojson(value: Any) -> str:
    """Serialize exactly like Transformers' Jinja ``tojson`` filter."""

    # Transformers replaces Jinja's HTML-safe filter with plain json.dumps,
    # retaining Unicode, input key order, and literal ``<>&'`` characters.
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def _tool_call_name_and_arguments(tool_call: Any) -> tuple[str, Any]:
    function = getattr(tool_call, "function", None)
    if function is not None:
        name = function.name
        arguments = function.arguments
    elif isinstance(tool_call, Mapping):
        nested = tool_call.get("function")
        payload = nested if isinstance(nested, Mapping) else tool_call
        name = payload.get("name")
        arguments = payload.get("arguments", {})
    else:
        raise TypeError(f"Unsupported Qwen2.5 tool call: {type(tool_call)!r}")

    if not isinstance(name, str):
        raise TypeError("Qwen2.5 tool-call name must be a string")
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments else {}
    return name, arguments


def _wrap_tool_specs(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    wrapped: list[dict[str, Any]] = []
    for tool in tools:
        nested = tool.get("function")
        if isinstance(nested, Mapping):
            wrapped.append(copy.deepcopy(dict(tool)))
        else:
            wrapped.append(
                {
                    "type": "function",
                    "function": copy.deepcopy(dict(tool)),
                }
            )
    return wrapped


class _TraceBuilder:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0
        self.spans: list[_OutputSpan] = []

    def append(self, text: str) -> None:
        self._parts.append(text)
        self._length += len(text)

    def message(
        self,
        *,
        message_index: int | None,
        role: str,
        header: str,
        output: str,
    ) -> None:
        self.append(header)
        start = self._length
        self.append(output)
        self.spans.append(
            _OutputSpan(
                message_index=message_index,
                role=role,
                start=start,
                end=self._length,
            )
        )

    def text(self) -> str:
        return "".join(self._parts)


class Qwen2_5Renderer(Renderer):
    """Whole-conversation renderer for the Fireworks Qwen2.5 V1 contract."""

    supports_per_message_rendering = False
    _preserves_explicit_empty_system_with_tools = True

    @property
    def has_extension_property(self) -> bool:
        # Qwen2.5 has no thinking/history rewrite.  Valid sequential tool
        # trajectories therefore extend monotonically as well.
        return True

    @property
    def _end_message_token(self) -> int:
        tokens = list(
            self.tokenizer.encode(PRODUCTION_EOS_TOKEN, add_special_tokens=False)
        )
        if len(tokens) != 1:
            raise ValueError(f"Qwen2.5 expected one {_IM_END} token, got {len(tokens)}")
        return int(tokens[0])

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        del message, ctx
        raise RendererError("Qwen2.5 requires whole-conversation rendering")

    def create_conversation_prefix_with_tools(
        self,
        tools: list[ToolSpec],
        system_prompt: str = "",
    ) -> list[Message]:
        marker = cast(Message, {"role": _TOOLS_MARKER_ROLE, "content": ""})
        marker[_TOOLS_MARKER_KEY] = copy.deepcopy(tools)  # type: ignore[typeddict-unknown-key]
        prefix = [marker]
        if system_prompt:
            prefix.append(Message(role="system", content=system_prompt))
        return prefix

    @staticmethod
    def _split_tool_marker(
        messages: Sequence[Message],
    ) -> tuple[list[Message], list[dict[str, Any]]]:
        normalized = list(messages)
        if not normalized or normalized[0].get("role") != _TOOLS_MARKER_ROLE:
            return normalized, []

        marker = normalized.pop(0)
        raw_tools = marker.get(_TOOLS_MARKER_KEY, [])  # type: ignore[typeddict-item]
        if not isinstance(raw_tools, Sequence) or isinstance(
            raw_tools, (str, bytes, bytearray)
        ):
            raise TypeError("Qwen2.5 tool marker must contain a tool sequence")
        tools: list[Mapping[str, Any]] = []
        for tool in raw_tools:
            if not isinstance(tool, Mapping):
                raise TypeError("Qwen2.5 tool specs must be mappings")
            tools.append(tool)
        return normalized, _wrap_tool_specs(tools)

    def _render_trace(
        self,
        messages: Sequence[Message],
        *,
        add_generation_prompt: bool,
    ) -> _RenderTrace:
        normalized, tools = self._split_tool_marker(messages)
        if not normalized:
            raise ValueError("Qwen2.5 conversations must contain at least one message")

        builder = _TraceBuilder()
        first_is_system = normalized[0].get("role") == "system"
        system_index = 0 if first_is_system else None
        system_content = (
            _visible_text(normalized[0].get("content"))
            if first_is_system
            else DEFAULT_SYSTEM_PROMPT
        )

        system_output = system_content
        if tools:
            tool_lines = "".join(f"\n{_jinja_tojson(tool)}" for tool in tools)
            system_output += _TOOLS_PREAMBLE + tool_lines + _TOOLS_EPILOGUE
        system_output += f"{_IM_END}\n"
        builder.message(
            message_index=system_index,
            role="system",
            header=f"{_IM_START}system\n",
            output=system_output,
        )

        index = 1 if first_is_system else 0
        while index < len(normalized):
            message = normalized[index]
            role = str(message.get("role", ""))

            if role == "tool":
                group_start = index
                builder.append(f"{_IM_START}user\n")
                while (
                    index < len(normalized) and normalized[index].get("role") == "tool"
                ):
                    if index > group_start:
                        builder.append("\n")
                    start = builder._length
                    builder.append(
                        "<tool_response>\n"
                        + _visible_text(normalized[index].get("content"))
                        + "\n</tool_response>"
                    )
                    builder.spans.append(
                        _OutputSpan(
                            message_index=index,
                            role="tool",
                            start=start,
                            end=builder._length,
                        )
                    )
                    index += 1
                builder.append(f"{_IM_END}\n")
                last = builder.spans[-1]
                builder.spans[-1] = _OutputSpan(
                    message_index=last.message_index,
                    role=last.role,
                    start=last.start,
                    end=builder._length,
                )
                continue

            if role in {"user", "system"}:
                content = _visible_text(message.get("content"))
                builder.message(
                    message_index=index,
                    role=role,
                    header=f"{_IM_START}{role}\n",
                    output=f"{content}{_IM_END}\n",
                )
            elif role == "assistant":
                content = _visible_text(message.get("content"))
                tool_calls = list(message.get("tool_calls") or [])
                output = content
                if tool_calls:
                    rendered_calls: list[str] = []
                    for tool_call in tool_calls:
                        name, arguments = _tool_call_name_and_arguments(tool_call)
                        rendered_calls.append(
                            "<tool_call>\n"
                            f'{{"name": "{name}", "arguments": '
                            f"{_jinja_tojson(arguments)}}}\n"
                            "</tool_call>"
                        )
                    if output:
                        output += "\n"
                    output += "\n".join(rendered_calls)
                builder.message(
                    message_index=index,
                    role=role,
                    header=f"{_IM_START}assistant\n",
                    output=f"{output}{_IM_END}\n",
                )
            # The V1 Jinja template silently omits unknown roles.
            index += 1

        if add_generation_prompt:
            builder.append(f"{_IM_START}assistant\n")

        return _RenderTrace(
            text=builder.text(),
            messages=tuple(normalized),
            spans=tuple(builder.spans),
        )

    def render_text(
        self,
        messages: Sequence[Message],
        *,
        add_generation_prompt: bool,
    ) -> str:
        """Expose the authoritative V1 wire text for parity/debug tests."""

        return self._render_trace(
            messages,
            add_generation_prompt=add_generation_prompt,
        ).text

    def _tokenize_with_offsets(
        self,
        text: str,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        if not callable(self.tokenizer):
            raise TypeError("Qwen2.5 requires a fast callable tokenizer")
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = [int(token) for token in encoded["input_ids"]]
        offsets = [tuple(map(int, offset)) for offset in encoded["offset_mapping"]]
        if len(token_ids) != len(offsets):
            raise ValueError("Qwen2.5 tokenizer returned mismatched offsets")
        return token_ids, offsets

    @staticmethod
    def _message_is_selected(
        *,
        message: Message,
        message_index: int,
        messages: Sequence[Message],
        train_on_what: TrainOnWhat,
    ) -> bool:
        role = message.get("role")
        if train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE:
            return message_index == len(messages) - 1 and role == "assistant"
        if train_on_what == TrainOnWhat.LAST_ASSISTANT_TURN:
            last_user = max(
                (
                    index
                    for index, candidate in enumerate(messages)
                    if candidate.get("role") == "user"
                ),
                default=-1,
            )
            return role == "assistant" and message_index > last_user
        if train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
            return role == "assistant"
        if train_on_what == TrainOnWhat.ALL_MESSAGES:
            return True
        if train_on_what == TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES:
            return role in {"user", "system"}
        if train_on_what == TrainOnWhat.CUSTOMIZED:
            if "trainable" not in message:
                raise ValueError(
                    "CUSTOMIZED Qwen2.5 rows require trainable on every message"
                )
            return bool(message["trainable"])
        if train_on_what == TrainOnWhat.ALL_TOKENS:
            return True
        raise RendererError(f"Unknown train_on_what: {train_on_what}")

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: str = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        if role != "assistant":
            raise ValueError("Qwen2.5 can generate assistant messages only")
        text = self.render_text(messages, add_generation_prompt=True)
        if prefill:
            text += prefill
        token_ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        return tinker.ModelInput(
            chunks=[tinker.types.EncodedTextChunk(tokens=token_ids)]
        )

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        trace = self._render_trace(messages, add_generation_prompt=False)
        token_ids, offsets = self._tokenize_with_offsets(trace.text)

        if train_on_what == TrainOnWhat.ALL_TOKENS:
            weights = [1.0] * len(token_ids)
        else:
            weights = [0.0] * len(token_ids)
            for span in trace.spans:
                if span.message_index is None:
                    continue
                message = trace.messages[span.message_index]
                if not self._message_is_selected(
                    message=message,
                    message_index=span.message_index,
                    messages=trace.messages,
                    train_on_what=train_on_what,
                ):
                    continue
                for token_index, (start, end) in enumerate(offsets):
                    if end > start and start >= span.start and start < span.end:
                        weights[token_index] = 1.0

        model_input = tinker.ModelInput(
            chunks=[tinker.types.EncodedTextChunk(tokens=token_ids)]
        )
        return model_input, torch.tensor(weights, dtype=torch.float32)

    def build_supervised_examples(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_TURN,
    ) -> list[tuple[tinker.ModelInput, torch.Tensor]]:
        return [
            self.build_supervised_example(
                messages,
                train_on_what=train_on_what,
            )
        ]

    def parse_response(
        self,
        response: list[int],
    ) -> tuple[Message, ParseTermination]:
        assistant_message, termination = parse_response_for_stop_token(
            response,
            self.tokenizer,
            self._end_message_token,
        )
        if not termination.is_clean:
            return assistant_message, termination

        content = assistant_message.get("content", "")
        if not isinstance(content, str):
            return assistant_message, termination
        parsed = parse_content_blocks(content)
        if parsed is None:
            return assistant_message, termination

        _parts, tool_results = parsed
        # Qwen2.5 emits all tool calls after the optional narration. The
        # generic XML parser returns the renderer's structural separator
        # newlines as TextParts, which would both violate this renderer's
        # text-only input contract and add an extra newline when the parsed
        # message is rendered into the next turn. Recover the narration from
        # the original wire text and remove exactly the separator immediately
        # before the first tool call.
        marker_index = content.find("<tool_call>")
        if tool_results and marker_index >= 0:
            narration = content[:marker_index]
            if narration.endswith("\n"):
                narration = narration[:-1]
            assistant_message["content"] = narration
        else:
            assistant_message["content"] = content
        tool_calls = [item for item in tool_results if isinstance(item, ToolCall)]
        malformed = [
            item for item in tool_results if isinstance(item, UnparsedToolCall)
        ]
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if malformed:
            assistant_message["unparsed_tool_calls"] = malformed
        return assistant_message, termination


def _qwen2_5_factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> Qwen2_5Renderer:
    del image_processor
    return Qwen2_5Renderer(tokenizer)


register_renderer("qwen2_5", _qwen2_5_factory)
