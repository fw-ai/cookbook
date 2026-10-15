"""Training renderer for the Muse Spark 1.2 open ATEM chat protocol.

This is a training port of the pinned ``muse_spark_1_2_oss.jinja``
template from the open Muse Spark 1.2 checkpoint.  It intentionally differs from
the older Muse Glimmer renderer in four places that affect the training
distribution:

* the default system metadata is always synthesized and the first system or
  developer message is appended to it;
* reasoning effort selects the vendor's numeric budgets and defaults to xhigh;
* ``commentary`` is a first-class, user-visible non-terminal recipient; and
* assistant content on tool-call turns is retained as a commentary message.

Images are represented with ``ImageChunk`` objects using the checkpoint's
native NaViT token-counting contract. Video and audio remain unsupported.
``muse_spark`` is the selectable name and pins xhigh; ``muse_spark_<effort>``
pins one of low, medium, high, or xhigh.
"""

from __future__ import annotations

import html
import itertools
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

from jinja2.exceptions import TemplateError
from training.renderer import RendererError, register_renderer
from training.renderer.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Role,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
    image_to_chunk,
)
from training.renderer.message_weights import untrained_synthesized_context
from training.renderer.muse_glimmer import MuseGlimmerRenderer
from training.renderer.plugins import register_renderer_image_capability
from training.renderer.reasoning_fields import (
    original_reasoning,
    original_reasoning_content,
)
from training.renderer.tokenizer import Tokenizer

_START = "<|start|>"
_MESSAGE = "<|message|>"
_EOM = "<|eom|>"
_EOT = "<|eot|>"
_IMAGE_PLACEHOLDER = "<|patch|>"
_TOOLS_ROLE = "_muse_spark_tools"
_TOOLS_KEY = "_muse_spark_tool_specs"
_SYNTHETIC_SYSTEM = "_muse_spark_synthetic_system"
_INITIAL_INSTRUCTIONS = "_muse_spark_initial_instructions"
_RESOLVED_TOOL_NAME = "_muse_spark_resolved_tool_name"
_NEXT_SAME_ROLE = "_muse_spark_next_same_role"

_REASONING_EFFORT_MAP = {
    "low": 8,
    "medium": 32,
    "high": 256,
    "xhigh": 512,
}
_RESERVED_RECIPIENTS = frozenset({"self", "commentary", "user"})
_TOOL_ARGUMENT_ERROR = (
    "Muse Spark ATEM chat template requires tool_call.function.arguments "
    "to be a dict (mapping); a JSON string cannot be parsed in the HF jinja "
    "sandbox."
)

_ATEM_CALL_RE = re.compile(
    r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">' r"(?P<body>.*?)</atem:invoke>",
    re.DOTALL,
)
_ATEM_ARG_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<name>[^"]+)"[^>]*?>(?P<value>.*?)' r"</atem:parameter>",
    re.DOTALL,
)


@dataclass(frozen=True)
class MuseSparkOptions:
    """Template globals that select Muse Spark's text-generation mode."""

    reasoning_effort: str | None = "xhigh"
    required_tool: str | None = None
    cutoff_date: str = "2026-01-04"
    current_date: str | None = None
    include_current_date: bool = True

    def __post_init__(self) -> None:
        if self.required_tool is not None and (
            not self.required_tool
            or self.required_tool in _RESERVED_RECIPIENTS
            or _escape_header(self.required_tool) != self.required_tool
        ):
            raise ValueError("required tool must be a valid native tool name")
        # Match the HF chat template's ``default('xhigh', true)``: an explicit
        # None still resolves to the vendor default budget.
        if self.reasoning_effort is None:
            object.__setattr__(self, "reasoning_effort", "xhigh")
        if self.reasoning_effort not in _REASONING_EFFORT_MAP:
            choices = ", ".join(_REASONING_EFFORT_MAP)
            raise ValueError(
                f"Muse Spark reasoning_effort must be one of {choices}; " f"got {self.reasoning_effort!r}"
            )


def _least_distortion_fit(
    ideal_h: float,
    ideal_w: float,
    image_h: int,
    image_w: int,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    candidates = {
        (height, width)
        for height, width in itertools.product(
            (math.floor(ideal_h), math.ceil(ideal_h)),
            (math.floor(ideal_w), math.ceil(ideal_w)),
        )
        if height > 0 and width > 0 and minimum <= height * width <= maximum
    }
    if not candidates:
        raise ValueError("no rounded resize target satisfies the token budget")
    aspect = image_h / image_w
    return min(candidates, key=lambda item: abs(item[0] / item[1] - aspect))


def _least_distance_fit(
    ideal_h: float,
    ideal_w: float,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    best: tuple[float, int, int] | None = None
    for height in range(1, maximum + 1):
        lower = math.ceil(minimum / height)
        upper = math.floor(maximum / height)
        if lower > upper:
            continue
        width = min(max(round(ideal_w), lower), upper)
        distance = (height - ideal_h) ** 2 + (width - ideal_w) ** 2
        if best is None or distance < best[0]:
            best = (distance, height, width)
    if best is None:
        raise ValueError("no resize target satisfies the token budget")
    return best[1], best[2]


def _image_token_count(
    image_h: int,
    image_w: int,
    *,
    token_patch_size: int,
    minimum_tokens: int,
    maximum_tokens: int,
) -> int:
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"Image dimensions must be positive, got {image_h}x{image_w}")
    ideal_h = image_h / token_patch_size
    ideal_w = image_w / token_patch_size
    ideal_count = ideal_h * ideal_w
    aspect = ideal_w / ideal_h
    if ideal_count > maximum_tokens:
        ideal_h = math.sqrt(maximum_tokens / aspect)
        ideal_w = ideal_h * aspect
    elif ideal_count < minimum_tokens:
        ideal_h = math.sqrt(minimum_tokens / aspect)
        ideal_w = ideal_h * aspect
    try:
        patches_h, patches_w = _least_distortion_fit(
            ideal_h,
            ideal_w,
            image_h,
            image_w,
            minimum_tokens,
            maximum_tokens,
        )
    except ValueError:
        patches_h, patches_w = _least_distance_fit(
            ideal_h,
            ideal_w,
            minimum_tokens,
            maximum_tokens,
        )
    return patches_h * patches_w


class MuseSparkImageTokenCounter:
    """Checkpoint-config-backed image token counter for Tinker chunks."""

    def __init__(
        self,
        *,
        patch_size: int = 14,
        downsample_factor: int = 2,
        min_image_tokens: int = 1,
        max_image_tokens: int = 4096,
    ) -> None:
        self.patch_size = int(patch_size)
        self.downsample_factor = int(downsample_factor)
        self.min_image_tokens = int(min_image_tokens)
        self.max_image_tokens = int(max_image_tokens)
        if (
            min(
                self.patch_size,
                self.downsample_factor,
                self.min_image_tokens,
                self.max_image_tokens,
            )
            <= 0
        ):
            raise ValueError("Muse Spark image processor dimensions must be positive")
        if self.min_image_tokens > self.max_image_tokens:
            raise ValueError("Muse Spark minimum image tokens exceed the maximum")

    @classmethod
    def from_pretrained(cls, model_name: str) -> MuseSparkImageTokenCounter:
        from transformers.utils.hub import cached_file

        config_path = cached_file(model_name, "config.json")
        if config_path is None:
            raise FileNotFoundError(f"Muse Spark config.json not found at {model_name!r}")
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
        vision = config.get("vision_config") or {}
        if not isinstance(vision, Mapping):
            vision = {}
        navit = vision.get("navit_config") or vision
        if not isinstance(navit, Mapping):
            navit = {}
        return cls(
            patch_size=int(navit.get("patch_size", vision.get("patch_size", 14))),
            downsample_factor=int(
                navit.get(
                    "downsample_factor",
                    vision.get("downsample_factor", vision.get("merge_size", 2)),
                )
            ),
            min_image_tokens=int(navit.get("min_num_image_tokens", 1)),
            max_image_tokens=int(navit.get("max_num_image_tokens", 4096)),
        )

    def get_resize_config(self, image_item: Mapping[str, Any]) -> dict[str, int]:
        image = image_item.get("image")
        size = getattr(image, "size", None)
        if not isinstance(size, tuple) or len(size) != 2:
            raise TypeError("Muse Spark image token counting requires a PIL image")
        width, height = (int(size[0]), int(size[1]))
        return {
            "num_tokens": _image_token_count(
                height,
                width,
                token_patch_size=self.patch_size * self.downsample_factor,
                minimum_tokens=self.min_image_tokens,
                maximum_tokens=self.max_image_tokens,
            )
        }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _escape(value: Any) -> str:
    return (
        str(value)
        .replace("<|", "< |")
        .replace("<image>", "< image>")
        .replace("<video>", "< video>")
        .replace("<audio>", "< audio>")
    )


def _escape_header(value: Any) -> str:
    return _escape(value).replace("\r", " ").replace("\n", " ")


def _escape_attr(value: Any) -> str:
    return (
        _escape_header(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _replace_reasoning_effort(text: str) -> str:
    return (
        text.replace("Reasoning effort", "Reasoning strength")
        .replace("Reasoning Effort", "Reasoning Strength")
        .replace("reasoning effort", "reasoning strength")
        .replace("REASONING EFFORT", "REASONING STRENGTH")
    )


def _function(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    wrapped = tool.get("function")
    return wrapped if isinstance(wrapped, Mapping) else tool


def _tool_name(tool_call: Any) -> str:
    function = getattr(tool_call, "function", None)
    if function is not None:
        name = getattr(function, "name", None)
        if name:
            return str(name)
    if isinstance(tool_call, Mapping):
        function = tool_call.get("function")
        if isinstance(function, Mapping) and function.get("name"):
            return str(function["name"])
    raise TemplateError("tool_call is missing its function payload or name")


def _tool_id(tool_call: Any) -> str | None:
    value = getattr(tool_call, "id", None)
    if value is None and isinstance(tool_call, Mapping):
        value = tool_call.get("id")
    return str(value) if value is not None else None


def _tool_arguments(tool_call: Any) -> Mapping[str, Any]:
    function = getattr(tool_call, "function", None)
    if function is not None:
        raw = function.arguments
        # Tinker's ToolCall stores validated object arguments as JSON text.
        # Decode that trusted normalized representation. Raw HF-style mappings
        # with string arguments still follow the template's rejection path.
        if isinstance(tool_call, ToolCall) and isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TemplateError(_TOOL_ARGUMENT_ERROR) from exc
    elif isinstance(tool_call, Mapping):
        wrapped = tool_call.get("function")
        raw = wrapped.get("arguments") if isinstance(wrapped, Mapping) else None
    else:
        raw = None
    if not isinstance(raw, Mapping):
        raise TemplateError(_TOOL_ARGUMENT_ERROR)
    return raw


def _format_atem_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, Mapping) or (isinstance(value, Iterable) and not isinstance(value, (str, bytes))):
        return _escape(_json(value))
    return _escape(value)


def _render_atem(tool_call: Any) -> str:
    name = _tool_name(tool_call)
    out = f'<atem:function_calls>\n<atem:invoke name="{_escape_attr(name)}">\n'
    for key, value in _tool_arguments(tool_call).items():
        out += f'<atem:parameter name="{_escape_attr(key)}">' + _format_atem_value(value) + "</atem:parameter>\n"
    return out + "</atem:invoke>\n</atem:function_calls>"


def _validate_tools(tools: Sequence[Mapping[str, Any]]) -> None:
    for tool in tools:
        name = str(_function(tool).get("name") or "")
        if not name:
            raise RendererError("Muse Spark tool declaration requires a non-empty function name")
        if name in _RESERVED_RECIPIENTS:
            raise RendererError(f"Muse Spark tool name {name!r} conflicts with a reserved message recipient")


def _render_tool_defs(tools: Sequence[Mapping[str, Any]]) -> str:
    _validate_tools(tools)
    out = (
        "In this environment you have access to a set of tools you can use to "
        "answer the user's question.\n\n"
        'You can invoke a function by writing a "<atem:function_calls>" block '
        "like the following as part of your reply to the user:\n"
        '<atem:function_calls>\n<atem:invoke name="$FUNCTION_NAME">\n'
        '<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE'
        "</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\n"
        "String and scalar parameters should be specified as is, while lists "
        "and objects should use JSON format. Note that spaces for string values "
        "are not stripped. The output is not expected to be valid XML and is "
        "parsed with regular expressions.\n"
        "Here are the functions available in JSONSchema format:\n"
        '// Tool metadata\n{"name": "", "description": ""}\n'
        "// Function schemas"
    )
    for tool in tools:
        function = _function(tool)
        out += "\n" + _escape(
            _json(
                {
                    "name": function.get("name") or "",
                    "description": function.get("description") or "",
                    "parameters": function.get("parameters") or {},
                }
            )
        )
    out += (
        "\n\nHere are examples of how to call functions in the tool set. To make "
        "parallel tool calls, emit each call as its own message (one tool call "
        "per message) in consecutive messages, as shown below:\n"
        "(If the tool namespace is not specified, invoke the function directly "
        "as `example_function_name_1` rather than "
        "`example_tool_name_1.example_function_name_1`)\n\n"
        "to=example_tool_name_1.example_function_name_1\n\n"
        '<atem:function_calls>\n<atem:invoke name="example_tool_name_1.'
        'example_function_name_1">\n'
        '<atem:parameter name="example_parameter_1">value_1</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>\n\n"
        "to=example_tool_name_2.example_function_name_2\n\n"
        '<atem:function_calls>\n<atem:invoke name="example_tool_name_2.'
        'example_function_name_2">\n'
        '<atem:parameter name="example_parameter_2">This is the value for the '
        'second parameter\nthat can span\n"multiple" lines\n</atem:parameter>\n'
        '<atem:parameter name="example_parameter_3">value_3</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    return out


def _render_valid_recipients(tools: Sequence[Mapping[str, Any]]) -> str:
    recipients: list[str] = ['"self"']
    lines = ["Use the appropriate recipient for each message:\n"]
    lines.append('- "self": private reasoning and tool planning.')
    recipients.append('"commentary"')
    lines.append(
        '- "commentary": user-visible intermediate updates while the assistant '
        "will continue working, including messages sent to the user before or "
        "between tool calls."
    )
    if tools:
        first_name = _escape_header(_function(tools[0]).get("name") or "")
        recipients.append(f'"[func_name] (e.g., {first_name})"')
        lines.append('- "tool" or "[tool name]": the name of the tool to call.')
    recipients.append('"user"')
    lines.append(
        '- "user": messages that end the assistant turn, such as a completed '
        "response or clarification question that waits for the user's reply; do "
        "not use for pre-tool updates or partial responses."
    )
    lines.append("\n# Valid recipients: " + ", ".join(recipients) + ".")
    return "\n".join(lines)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return _escape(content)
    if content is None:
        return ""
    result: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type", "")
        if part_type in {"image", "input_image", "image_url"}:
            result.append("<image>")
        elif part_type in {"video", "input_video", "video_url"}:
            result.append("<video>")
        elif part_type in {"audio", "input_audio", "audio_url"}:
            result.append("<audio>")
        elif part_type in {"text", "input_text", "output_text"}:
            result.append(_escape(str(part.get("text", ""))))
        # Structured thinking is rendered from reasoning_content, not content.
    return "".join(result)


class MuseSparkRenderer(MuseGlimmerRenderer):
    """Whole-conversation Muse Spark renderer with ATEM loss boundaries."""

    supports_per_message_rendering = False
    _preserves_explicit_empty_system_with_tools = False
    _preserves_openai_protocol_fields = True

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        image_processor: Any | None = None,
        options: MuseSparkOptions | None = None,
    ) -> None:
        super().__init__(tokenizer, image_processor=image_processor)
        # The final 816B open checkpoint emits <|patch|> (image_token_id
        # 200092), replacing RC1's <image> -> <|image|> rewrite. Token-in
        # requests still carry one unexpanded placeholder per attached image;
        # ImageChunks retain the full visual-token count for training.
        image_ids = list(tokenizer.encode(_IMAGE_PLACEHOLDER, add_special_tokens=False))
        if len(image_ids) != 1:
            raise RuntimeError("Muse Spark expected '<|patch|>' to be one token: " f"{image_ids}")
        self.image_placeholder_token_id = int(image_ids[0])
        self.options = options or MuseSparkOptions()

    def with_required_tool(self, name: str) -> "MuseSparkRenderer":
        """Prefill one native tool recipient while sampling its body."""
        return type(self)(
            self.tokenizer,
            image_processor=self.image_processor,
            options=replace(self.options, required_tool=name),
        )

    @staticmethod
    def _image_payload(part: Mapping[str, Any]) -> Any:
        image = part.get("image")
        if image is not None:
            return image
        image_url = part.get("image_url")
        if isinstance(image_url, Mapping):
            return image_url.get("url")
        return image_url

    def _content_pieces(
        self,
        content: Any,
        *,
        allow_images: bool = False,
        role: str = "assistant",
        text_transform=None,
    ) -> list[Any]:
        if text_transform is not None:
            raise TypeError("Muse Spark content escaping is renderer-owned")
        if isinstance(content, str):
            text = _escape(content)
            return [text] if text else []
        if content is None:
            return []

        pieces: list[Any] = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            part_type = part.get("type", "")
            if role == "system" and part_type in {
                "image",
                "input_image",
                "image_url",
                "video",
                "input_video",
                "video_url",
                "audio",
                "input_audio",
                "audio_url",
            }:
                raise RendererError(
                    "Muse Spark folded system/developer instructions support "
                    "text inputs only; media belongs only in user or tool messages"
                )
            if part_type in {"image", "input_image", "image_url"}:
                if not allow_images:
                    raise RendererError(
                        f"Muse Spark images are supported only in user or tool messages, not {role!r} messages"
                    )
                if self.image_processor is None:
                    raise RendererError(
                        "Muse Spark image content requires an image processor; "
                        "build the renderer with image loading enabled."
                    )
                image = self._image_payload(part)
                if image is None:
                    raise RendererError("Muse Spark image content is missing the image payload")
                pieces.append(image_to_chunk(image, self.image_processor))
            elif part_type in {
                "video",
                "input_video",
                "video_url",
                "audio",
                "input_audio",
                "audio_url",
            }:
                raise RendererError(f"Muse Spark {part_type} content is not supported by training")
            elif part_type in {"text", "input_text", "output_text"}:
                self._append_text_piece(
                    pieces,
                    _escape(str(part.get("text", ""))),
                )
            # Match the template by dropping unknown content-part types.
        return pieces

    def _synthetic_system_text(
        self,
        *,
        instructions: Any,
        tools: list[Mapping[str, Any]],
    ) -> str:
        instruction_text = _replace_reasoning_effort(
            "".join(
                self._content_pieces(
                    instructions,
                    allow_images=False,
                    role="system",
                )
            )
        )
        parts = [
            "You are a helpful AI assistant.",
            "Knowledge cutoff: " + (self.options.cutoff_date or "2026-01-04") + ".",
        ]
        if self.options.include_current_date:
            parts.append("Current date: " + (self.options.current_date or date.today().isoformat()) + ".")
        if instruction_text:
            parts.append(instruction_text)
        if tools:
            parts.append(_render_tool_defs(tools).strip())
        budget = _REASONING_EFFORT_MAP.get(self.options.reasoning_effort or "")
        if budget is not None and "reasoning strength:" not in instruction_text.lower():
            parts.append(f"Reasoning strength: {budget}.")
        parts.append(_render_valid_recipients(tools).strip())
        return "\n\n".join(parts)

    def _preprocess(self, messages: list[Message]) -> list[Message]:
        tools: list[Mapping[str, Any]] = []
        actual: list[Message] = []
        for message in messages:
            if message.get("role") == _TOOLS_ROLE:
                tools.extend(message.get(_TOOLS_KEY, []))
            else:
                actual.append(dict(message))
        _validate_tools(tools)
        if self.options.required_tool is not None and not any(
            _function(tool).get("name") == self.options.required_tool for tool in tools
        ):
            raise ValueError("required tool is not present in the conversation tools")

        instructions: Any = None
        if actual and actual[0].get("role") in {"system", "developer"}:
            instructions = actual.pop(0).get("content")

        synthetic = Message(
            role="system",
            content="",
            **{
                _SYNTHETIC_SYSTEM: True,
                _INITIAL_INSTRUCTIONS: instructions,
                _TOOLS_KEY: tools,
            },
        )
        rendered = [synthetic, *actual]

        for index, message in enumerate(rendered):
            message[_TOOLS_KEY] = tools
            if message.get("role") == "tool" and not message.get("name"):
                call_id = message.get("tool_call_id")
                resolved = str(call_id) if call_id else ""
                for candidate in rendered:
                    for call in candidate.get("tool_calls") or []:
                        if call_id is not None and _tool_id(call) == call_id:
                            resolved = _tool_name(call)
                if not resolved:
                    raise TemplateError("tool message has no name and no matching tool_call_id")
                message[_RESOLVED_TOOL_NAME] = resolved
            message[_NEXT_SAME_ROLE] = bool(
                index + 1 < len(rendered) and rendered[index + 1].get("role") == message.get("role")
            )
        return untrained_synthesized_context(rendered)

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        del ctx
        role = message.get("role")
        if role == "system" and message.get(_SYNTHETIC_SYSTEM):
            body = self._synthetic_system_text(
                instructions=message.get(_INITIAL_INSTRUCTIONS),
                tools=list(message.get(_TOOLS_KEY, [])),
            )
            return RenderedMessage(
                header=self._encode(_START + "system" + _MESSAGE),
                output=[self._encode(body + _EOT)],
            )
        if role in {"system", "developer"}:
            pieces = self._content_pieces(
                message.get("content"),
                allow_images=False,
                role=str(role),
            )
            self._append_text_piece(pieces, _EOT)
            return RenderedMessage(
                header=self._encode(_START + str(role) + _MESSAGE),
                output=self._encode_pieces(pieces),
            )
        if role == "user":
            prefix = ""
            if message.get("render_timestamp") and message.get("timestamp"):
                timestamp = message["timestamp"]
                if isinstance(timestamp, str):
                    prefix = "[" + _escape_header(timestamp) + "] "
                elif hasattr(timestamp, "strftime"):
                    prefix = "[" + timestamp.strftime("%a %Y-%m-%d %H:%M %Z").strip() + "] "
            pieces = [prefix] if prefix else []
            for piece in self._content_pieces(
                message.get("content"),
                allow_images=True,
                role="user",
            ):
                if isinstance(piece, str):
                    self._append_text_piece(pieces, piece)
                else:
                    pieces.append(piece)
            self._append_text_piece(pieces, _EOT)
            return RenderedMessage(
                header=self._encode(_START + "user" + _MESSAGE),
                output=self._encode_pieces(pieces),
            )
        if role == "tool":
            name = str(message.get("name") or message.get(_RESOLVED_TOOL_NAME) or message.get("tool_call_id") or "")
            if not name:
                raise TemplateError("tool message has no name and no matching tool_call_id")
            pieces = [f'<tool_output name="{_escape_attr(name)}">\n']
            for piece in self._content_pieces(
                message.get("content"),
                allow_images=True,
                role="tool",
            ):
                if isinstance(piece, str):
                    self._append_text_piece(pieces, piece)
                else:
                    pieces.append(piece)
            self._append_text_piece(pieces, "\n</tool_output>" + _EOT)
            return RenderedMessage(
                header=self._encode(_START + "tool " + _escape_header(name) + _MESSAGE),
                output=self._encode_pieces(pieces),
            )
        if role == "assistant":
            has_reasoning_content, reasoning_content = original_reasoning_content(message)
            has_reasoning, fallback_reasoning = original_reasoning(message)
            reasoning = (
                reasoning_content
                if has_reasoning_content and reasoning_content
                else fallback_reasoning if has_reasoning and fallback_reasoning else ""
            )
            reasoning = _escape(reasoning) if reasoning else ""
            tool_calls = message.get("tool_calls") or []
            body_pieces = self._content_pieces(
                message.get("content"),
                allow_images=False,
                role="assistant",
            )
            pieces: list[str] = []

            if reasoning:
                pieces.append(" to=self" + _MESSAGE + reasoning + _EOM)

            emits_followup = bool(tool_calls) or bool(body_pieces) or not reasoning
            if reasoning and emits_followup:
                pieces.append(_START + "assistant")

            if tool_calls:
                if body_pieces:
                    pieces.append(" to=commentary" + _MESSAGE)
                    for piece in body_pieces:
                        self._append_text_piece(pieces, piece)
                    self._append_text_piece(pieces, _EOM + _START + "assistant")
                for index, call in enumerate(tool_calls):
                    name = _tool_name(call)
                    ending = _EOM if index < len(tool_calls) - 1 or message.get(_NEXT_SAME_ROLE) else _EOT
                    pieces.append(" to=" + _escape_header(name) + _MESSAGE + _render_atem(call) + ending)
                    if index < len(tool_calls) - 1:
                        pieces.append(_START + "assistant")
            elif emits_followup:
                recipient = str(message.get("recipient") or "user")
                if recipient == "user":
                    end_turn = True
                elif recipient in {"self", "commentary"}:
                    end_turn = False
                elif message.get("end_turn") is not None:
                    end_turn = bool(message.get("end_turn"))
                else:
                    end_turn = False
                pieces.append(" to=" + _escape_header(recipient) + _MESSAGE)
                for piece in body_pieces:
                    self._append_text_piece(pieces, piece)
                self._append_text_piece(pieces, _EOT if end_turn else _EOM)

            return RenderedMessage(
                header=None,
                output=[
                    self._encode(_START + "assistant"),
                    *self._encode_pieces(pieces),
                ],
            )

        # Match the vendor template by dropping unknown roles.
        return RenderedMessage(header=None, output=[])

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        del role, ctx
        suffix = _START + "assistant"
        if self.options.required_tool is not None:
            suffix += " to=" + _escape_header(self.options.required_tool) + _MESSAGE
        return list(self.tokenizer.encode(suffix, add_special_tokens=False))

    def create_conversation_prefix_with_tools(
        self,
        tools: list[ToolSpec],
        system_prompt: str = "",
    ) -> list[Message]:
        _validate_tools(tools)
        prefix: list[Message] = [Message(role=_TOOLS_ROLE, content="", **{_TOOLS_KEY: list(tools)})]
        if system_prompt:
            prefix.append(Message(role="system", content=system_prompt))
        return prefix

    def get_stop_sequences(self) -> list[int]:
        # <|eom|> is never a stop: it means another assistant message follows in the same turn.
        ids = self.tokenizer.encode(_EOT, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Muse Spark expected {_EOT!r} to be one token: {ids}")
        return [int(ids[0])]

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        text = str(self.tokenizer.decode(response))
        required_tool = self.options.required_tool
        if required_tool is not None:
            text = " to=" + _escape_header(required_tool) + _MESSAGE + text
        stop_at = text.find(_EOT)
        termination = ParseTermination.STOP_SEQUENCE if stop_at >= 0 else ParseTermination.MALFORMED
        if stop_at >= 0:
            text = text[:stop_at]
        if required_tool is not None and _EOM in text:
            # The template closes a single forced call with <|eot|>; an <|eom|> means the model
            # kept the turn open for more assistant messages, which the forced call does not allow.
            termination = ParseTermination.MALFORMED
        # A complete nonterminal message ends with EOM. It is framing, not
        # reasoning/commentary payload; retain the non-clean termination above.
        text = text.removesuffix(_EOM)
        text = text.removeprefix(_START + "assistant")

        reasoning = ""
        visible = ""
        visible_recipient: str | None = None
        calls: list[ToolCall] = []
        unparsed: list[UnparsedToolCall] = []
        segments = text.split(_EOM + _START + "assistant")
        for segment in segments:
            if _MESSAGE not in segment:
                if segment:
                    visible += segment
                    visible_recipient = "user"
                continue
            recipient, payload = segment.split(_MESSAGE, 1)
            recipient = recipient.removeprefix(" to=").removeprefix("to=")
            if recipient == "self":
                reasoning += payload
                continue
            matches = list(_ATEM_CALL_RE.finditer(payload))
            if matches:
                for match in matches:
                    name = html.unescape(match.group("name"))
                    args: dict[str, Any] = {}
                    for argument in _ATEM_ARG_RE.finditer(match.group("body")):
                        raw = argument.group("value")
                        try:
                            value = json.loads(raw)
                        except json.JSONDecodeError:
                            value = raw
                        args[html.unescape(argument.group("name"))] = value
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
            if recipient in {"commentary", "user"}:
                visible += payload
                visible_recipient = recipient
            elif payload:
                unparsed.append(
                    UnparsedToolCall(
                        raw_text=payload,
                        error=f"Unknown Muse Spark recipient {recipient!r}",
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
        elif visible_recipient == "commentary":
            message["recipient"] = "commentary"
            message["end_turn"] = False
        if unparsed:
            message["unparsed_tool_calls"] = unparsed
        return message, termination


def _factory(tokenizer: Tokenizer, image_processor=None) -> MuseSparkRenderer:
    return MuseSparkRenderer(tokenizer, image_processor=image_processor)


def _options_factory(**options: Any):
    def factory(tokenizer: Tokenizer, image_processor=None) -> MuseSparkRenderer:
        return MuseSparkRenderer(
            tokenizer,
            image_processor=image_processor,
            options=MuseSparkOptions(**options),
        )

    return factory


RENDERER_NAMES = (
    "muse_spark",
    *(f"muse_spark_{effort}" for effort in _REASONING_EFFORT_MAP),
)


register_renderer("muse_spark", _factory)
for _effort in _REASONING_EFFORT_MAP:
    register_renderer(
        f"muse_spark_{_effort}",
        _options_factory(reasoning_effort=_effort),
    )
for _renderer_name in RENDERER_NAMES:
    register_renderer_image_capability(
        _renderer_name,
        MuseSparkImageTokenCounter.from_pretrained,
        supports_images=True,
        supports_tool_images=True,
    )


__all__ = [
    "MuseSparkImageTokenCounter",
    "MuseSparkOptions",
    "MuseSparkRenderer",
    "RENDERER_NAMES",
]
