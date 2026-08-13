"""Renderer for Meta's Muse Glimmer ATEM chat protocol.

This is a Python port of ``meta-models/Muse-Glimmer-30B``'s
``chat_template.jinja`` at revision
``a4e59da52a7bc87ae7251dd5545c0dd437c44b68``.  Keep the explicit strings
and whitespace in this module aligned with that template: ATEM is parsed as a
text protocol and seemingly cosmetic newlines are part of the model contract.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import tinker
import torch
from jinja2.exceptions import TemplateError
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
)
from tinker_cookbook.tokenizer_utils import Tokenizer

from training.renderer.message_weights import untrained_synthesized_context
from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin
from training.renderer.reasoning_fields import original_reasoning_content

_BOS = "<|begin_of_text|>"
_START = "<|start|>"
_MESSAGE = "<|message|>"
_EOM = "<|eom|>"
_EOT = "<|eot|>"
_TOOLS_ROLE = "_muse_glimmer_tools"
_TOOLS_KEY = "_muse_glimmer_tool_specs"
_TOOL_ARGUMENT_ERROR = (
    "Muse Glimmer ATEM chat template requires tool_call.function.arguments "
    "to be a dict (mapping); a JSON string cannot be parsed in the HF jinja "
    "sandbox."
)

_ATEM_CALL_RE = re.compile(
    r'<atem:function_calls>\n<atem:invoke name="(?P<name>[^"]*)">\n'
    r"(?P<body>.*?)</atem:invoke>\n</atem:function_calls>",
    re.DOTALL,
)
_ATEM_ARG_RE = re.compile(
    r'<atem:parameter name="(?P<name>[^"]*)">(?P<value>.*?)' r"</atem:parameter>\n",
    re.DOTALL,
)


@dataclass(frozen=True)
class MuseGlimmerOptions:
    """Optional Jinja globals used by the upstream template."""

    reasoning_strength: str = "high"
    knowledge_cutoff: str = "2026-01-04"
    current_date: str | None = None
    include_current_date: bool = True
    tool_namespace_descriptions: Mapping[str, str] = field(default_factory=dict)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    result: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type")
        if part_type == "image":
            result.append("<|patch|>")
        elif part_type == "video":
            result.append("<|video|>")
        elif part_type == "text":
            result.append(str(part["text"]))
        # The Jinja template intentionally drops unknown content-part types,
        # including Tinker's normalized ``thinking`` part.
    return "".join(result)


def _function(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    wrapped = tool.get("function")
    return wrapped if isinstance(wrapped, Mapping) else tool


def _namespaces(tools: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: list[str] = []
    for tool in tools:
        namespace = str(_function(tool)["name"]).split(".")[0]
        if namespace not in seen:
            seen.append(namespace)
    return seen


def _render_tool_defs(
    tools: Sequence[Mapping[str, Any]],
    namespace_descriptions: Mapping[str, str],
) -> str:
    out = (
        "In this environment you have access to a set of tools you can use to "
        "answer the user's question.\n\n"
        'You can invoke a function by writing a "<atem:function_calls>" block '
        "like the following:\n"
        '<atem:function_calls>\n<atem:invoke name="$FUNCTION_NAME">\n'
        '<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE'
        "</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\n"
        "String and scalar parameters should be specified as is, while lists "
        "and objects should use JSON format. Note that spaces for string values "
        "are not stripped. The output is not expected to be valid XML and is "
        "parsed with regular expressions.\n"
        "Here are the functions available in JSONSchema format:\n"
        "// Tool metadata\n"
    )
    for namespace in _namespaces(tools):
        out += (
            '{"name": '
            + _json(namespace)
            + ', "description": '
            + _json(namespace_descriptions.get(namespace, ""))
            + "}\n"
        )
    out += "// Function schemas"
    for tool in tools:
        fn = _function(tool)
        out += (
            '\n{"name": '
            + _json(fn["name"])
            + ', "description": '
            + _json(fn["description"])
            + ', "parameters": '
            + _json(fn["parameters"])
            + "}"
        )
    out += (
        "\n\nHere's an example of how to call a function in the tool set:\n"
        "(If the tool namespace is not specified, invoke the function directly "
        "as `example_function_name` rather than "
        "`example_tool_name.example_function_name`)\n\n"
        "to=example_tool_name.example_function_name\n\n"
        '<atem:function_calls>\n<atem:invoke name="example_tool_name.'
        'example_function_name">\n'
        '<atem:parameter name="example_parameter_1">value_1</atem:parameter>\n'
        '<atem:parameter name="example_parameter_2">This is the value for the '
        'second parameter\nthat can span\n"multiple" lines\n</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    return out


def _render_system_meta(tools: Sequence[Mapping[str, Any]]) -> str:
    recipients = ['"self"']
    recipients.extend(f'"{namespace}.*"' for namespace in _namespaces(tools))
    recipients.append('"user"')
    return "# Valid recipients: " + ", ".join(recipients) + "."


def _tool_arguments(tool_call: Any) -> Mapping[str, Any]:
    function = getattr(tool_call, "function", None)
    if function is not None:
        raw = function.arguments
        # Tinker's ToolCall contract stores validated object arguments as JSON
        # text. Decode that trusted normalized representation. A raw HF-style
        # dict with a JSON string still follows the template's rejection path.
        if isinstance(tool_call, ToolCall) and isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TemplateError(_TOOL_ARGUMENT_ERROR) from exc
    elif isinstance(tool_call, Mapping):
        fn = tool_call.get("function")
        raw = fn.get("arguments") if isinstance(fn, Mapping) else None
    else:
        raw = None
    if not isinstance(raw, Mapping):
        raise TemplateError(_TOOL_ARGUMENT_ERROR)
    return raw


def _tool_name(tool_call: Any) -> str:
    function = getattr(tool_call, "function", None)
    if function is not None:
        return str(function.name)
    if isinstance(tool_call, Mapping) and isinstance(tool_call.get("function"), Mapping):
        return str(tool_call["function"]["name"])
    raise TypeError(f"Unsupported Muse Glimmer tool call: {tool_call!r}")


def _tool_id(tool_call: Any) -> str | None:
    value = getattr(tool_call, "id", None)
    if value is None and isinstance(tool_call, Mapping):
        value = tool_call.get("id")
    return str(value) if value is not None else None


def _format_atem_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, Mapping) or (isinstance(value, Iterable) and not isinstance(value, (str, bytes))):
        return _json(value)
    return str(value)


def _render_atem(tool_call: Any) -> str:
    name = _tool_name(tool_call)
    out = f'<atem:function_calls>\n<atem:invoke name="{name}">\n'
    for key, value in _tool_arguments(tool_call).items():
        out += f'<atem:parameter name="{key}">' + _format_atem_value(value) + "</atem:parameter>\n"
    return out + "</atem:invoke>\n</atem:function_calls>"


def _replace_reasoning_effort(text: str) -> str:
    return (
        text.replace("Reasoning effort", "Reasoning strength")
        .replace("Reasoning Effort", "Reasoning Strength")
        .replace("reasoning effort", "reasoning strength")
        .replace("REASONING EFFORT", "REASONING STRENGTH")
    )


class MuseGlimmerRenderer(DisaggregateMultiTurnMixin, Renderer):
    """Whole-conversation-aware port of the Muse Glimmer Jinja template."""

    supports_per_message_rendering = False
    _preserves_explicit_empty_system_with_tools = True
    _preserves_openai_protocol_fields = True

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        options: MuseGlimmerOptions | None = None,
    ) -> None:
        super().__init__(tokenizer)
        self.options = options or MuseGlimmerOptions()

    @property
    def _bos_tokens(self) -> list[int]:
        bos = getattr(self.tokenizer, "bos_token", None) or _BOS
        return list(self.tokenizer.encode(bos, add_special_tokens=False))

    @property
    def has_extension_property(self) -> bool:
        # The last ATEM call changes EOT→EOM when another assistant message is
        # appended, so legal consecutive-assistant histories are not prefixes.
        return False

    def _encode(self, text: str) -> tinker.EncodedTextChunk:
        return tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(text, add_special_tokens=False))

    def _reasoning(self) -> str:
        return "Reasoning strength: " + (self.options.reasoning_strength or "high") + "."

    def _default_system(self, tools: list[Mapping[str, Any]]) -> Message:
        cutoff = self.options.knowledge_cutoff or "2026-01-04"
        text = "You are a helpful AI assistant.\nKnowledge cutoff: " + cutoff + "."
        if self.options.include_current_date:
            text += "\nCurrent date: " + (self.options.current_date or date.today().isoformat()) + "."
        return Message(
            role="system",
            content=text,
            _muse_glimmer_default_system=True,
            **{_TOOLS_KEY: tools},
        )

    def _preprocess(self, messages: list[Message]) -> list[Message]:
        tools: list[Mapping[str, Any]] = []
        actual: list[Message] = []
        for message in messages:
            if message.get("role") == _TOOLS_ROLE:
                tools.extend(message.get(_TOOLS_KEY, []))
            else:
                actual.append(dict(message))

        if not any(message.get("role") == "system" for message in actual):
            actual.insert(0, self._default_system(tools))

        # The template makes tools global and scans the entire conversation to
        # resolve tool result names by call id. Attach that resolved context to
        # each message before handing it to the concatenation base class.
        for idx, message in enumerate(actual):
            message[_TOOLS_KEY] = tools
            if message.get("role") == "tool" and not message.get("name"):
                call_id = message.get("tool_call_id")
                resolved = str(call_id) if call_id else ""
                for candidate in actual:
                    for call in candidate.get("tool_calls") or []:
                        if call_id is not None and _tool_id(call) == call_id:
                            resolved = _tool_name(call)
                message["_muse_glimmer_resolved_tool_name"] = resolved
            message["_muse_glimmer_next_same_role"] = bool(
                idx + 1 < len(actual) and actual[idx + 1].get("role") == message.get("role")
            )
        return untrained_synthesized_context(actual)

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        return super().build_generation_prompt(self._preprocess(messages), role=role, prefill=prefill)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        return super().build_supervised_example(self._preprocess(messages), train_on_what=train_on_what)

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        del ctx
        role = message.get("role")
        tools = list(message.get(_TOOLS_KEY, []))
        if role == "system":
            body = _content(message.get("content"))
            if not message.get("_muse_glimmer_default_system"):
                body = _replace_reasoning_effort(body)
            if "reasoning strength" not in body.lower():
                body += "\n\n" + self._reasoning()
            if tools:
                body += "\n\n" + _render_tool_defs(tools, self.options.tool_namespace_descriptions)
            body += "\n\n" + _render_system_meta(tools) + _EOT
            return RenderedMessage(
                header=self._encode(_START + "system" + _MESSAGE),
                output=[self._encode(body)],
            )
        if role == "user":
            return RenderedMessage(
                header=self._encode(_START + "user" + _MESSAGE),
                output=[self._encode(_content(message.get("content")) + _EOT)],
            )
        if role == "tool":
            name = str(
                message.get("name")
                or message.get("_muse_glimmer_resolved_tool_name")
                or message.get("tool_call_id")
                or ""
            )
            header = _START + "tool " + name + _MESSAGE
            body = f'<tool_output name="{name}">\n' + _content(message.get("content")) + "\n</tool_output>" + _EOT
            return RenderedMessage(header=self._encode(header), output=[self._encode(body)])
        if role == "assistant":
            has_reasoning, reasoning = original_reasoning_content(message)
            tool_calls = message.get("tool_calls") or []
            prefix = ""
            if has_reasoning and reasoning:
                prefix = " to=self" + _MESSAGE + reasoning + _EOM + _START + "assistant"
            if tool_calls:
                pieces: list[str] = []
                for index, call in enumerate(tool_calls):
                    ending = (
                        _EOM if index < len(tool_calls) - 1 or message.get("_muse_glimmer_next_same_role") else _EOT
                    )
                    pieces.append(" to=" + _tool_name(call) + _MESSAGE + _render_atem(call) + ending)
                    if index < len(tool_calls) - 1:
                        pieces.append(_START + "assistant")
                body = prefix + "".join(pieces)
            else:
                recipient = message.get("recipient") or "user"
                end_turn = message.get("end_turn")
                if end_turn is None:
                    end_turn = recipient == "user"
                body = (
                    prefix
                    + " to="
                    + str(recipient)
                    + _MESSAGE
                    + _content(message.get("content"))
                    + (_EOT if end_turn else _EOM)
                )
            return RenderedMessage(
                header=self._encode(_START + "assistant"),
                output=[self._encode(body)],
            )

        # Unknown roles are silently omitted by the Jinja template.
        return RenderedMessage(header=None, output=[])

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        # ``add_generation_prompt`` is hard-coded to the assistant role in the
        # upstream template; the Renderer API's requested role is ignored.
        del role, ctx
        return list(self.tokenizer.encode(_START + "assistant", add_special_tokens=False))

    def create_conversation_prefix_with_tools(self, tools: list[ToolSpec], system_prompt: str = "") -> list[Message]:
        # A non-system marker preserves the template's distinction between no
        # system (synthesize the default) and an explicit empty system. The
        # shared assembly helper appends the latter when requested via
        # ``_preserves_explicit_empty_system_with_tools``.
        prefix: list[Message] = [Message(role=_TOOLS_ROLE, content="", **{_TOOLS_KEY: list(tools)})]
        if system_prompt:
            prefix.append(Message(role="system", content=system_prompt))
        return prefix

    def get_stop_sequences(self) -> list[int]:
        ids = self.tokenizer.encode(_EOT, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Muse Glimmer expected {_EOT!r} to be one token: {ids}")
        return [int(ids[0])]

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        text = str(self.tokenizer.decode(response))
        stop_at = text.find(_EOT)
        termination = ParseTermination.STOP_SEQUENCE if stop_at >= 0 else ParseTermination.MALFORMED
        if stop_at >= 0:
            text = text[:stop_at]
        # Sampling begins after ``<|start|>assistant``. Be liberal when tests
        # feed a full assistant segment back into the parser.
        text = text.removeprefix(_START + "assistant")
        segments = text.split(_EOM + _START + "assistant")
        reasoning = ""
        visible = ""
        calls: list[ToolCall] = []
        unparsed: list[UnparsedToolCall] = []
        for segment in segments:
            if _MESSAGE not in segment:
                continue
            recipient, payload = segment.split(_MESSAGE, 1)
            recipient = recipient.removeprefix(" to=")
            if recipient == "self":
                reasoning += payload
                continue
            matches = list(_ATEM_CALL_RE.finditer(payload))
            if matches:
                for match in matches:
                    name = match.group("name")
                    args: dict[str, Any] = {}
                    for arg in _ATEM_ARG_RE.finditer(match.group("body")):
                        raw = arg.group("value")
                        try:
                            args[arg.group("name")] = json.loads(raw)
                        except json.JSONDecodeError:
                            args[arg.group("name")] = raw
                    calls.append(
                        ToolCall(
                            id=f"{name}:{len(calls)}",
                            function=ToolCall.FunctionBody(
                                name=name,
                                arguments=_json(args),
                            ),
                        )
                    )
                continue
            if recipient == "user":
                visible += payload
            elif payload:
                unparsed.append(
                    UnparsedToolCall(
                        raw_text=payload,
                        error=f"Unknown Muse Glimmer recipient {recipient!r}",
                    )
                )
        content: str | list[dict[str, str]]
        if reasoning:
            content = [{"type": "thinking", "thinking": reasoning}]
            if visible:
                content.append({"type": "text", "text": visible})
        else:
            content = visible
        message = Message(role="assistant", content=content)
        if calls:
            message["tool_calls"] = calls
        if unparsed:
            message["unparsed_tool_calls"] = unparsed
        return message, termination

    def to_openai_message(self, message: Message) -> dict[str, Any]:
        """Expose ATEM reasoning on the OpenAI ``reasoning_content`` field."""
        result: dict[str, Any] = {"role": message["role"]}
        content = message.get("content")
        reasoning = ""
        visible = ""
        if isinstance(content, str):
            visible = content
        elif isinstance(content, list):
            reasoning = "".join(
                str(part.get("thinking", ""))
                for part in content
                if isinstance(part, Mapping) and part.get("type") == "thinking"
            )
            visible = "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, Mapping) and part.get("type") == "text"
            )
        result["content"] = visible
        if reasoning:
            result["reasoning_content"] = reasoning
        if message.get("tool_calls"):
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": call.id,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message["tool_calls"]
            ]
        if message.get("recipient") is not None:
            result["recipient"] = message.get("recipient")
        if message.get("end_turn") is not None:
            result["end_turn"] = message.get("end_turn")
        return result


def _factory(tokenizer: Tokenizer, image_processor=None) -> MuseGlimmerRenderer:
    del image_processor
    return MuseGlimmerRenderer(tokenizer)


register_renderer("muse_glimmer", _factory)
