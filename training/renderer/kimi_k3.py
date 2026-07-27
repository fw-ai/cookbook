"""Renderer for the Moonshot Kimi K3 Python chat-template protocol.

Kimi K3 does not ship a Jinja chat template. Its tokenizer implements
``apply_chat_template`` in ``tokenization_kimi.py`` and delegates the actual
XTML construction to ``encoding_k3.py``. This renderer deliberately resolves
and calls those exact Python helpers from the supplied tokenizer; it does not
carry a second handwritten template.

The implementation keeps three representations distinct for vision inputs:

* the symbolic XTML image placeholder emitted by ``encoding_k3.py``;
* the one-``<|media_pad|>`` tokenizer prompt used by message/token-in parity;
* the Tinker ``ImageChunk`` whose pad expands to the processor's expected
  visual-token count.

For supervised examples, the tokenizer's trailing ``<|end_of_msg|>`` is kept
in the model input with weight zero. The model is trained through
``<|close|>message<|sep|>``, which is also the serving stop sequence, and is
not trained to emit the hard-appended history delimiter after that stop.
"""

from __future__ import annotations

import base64
import copy
import html
import io
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import tinker
import torch
from PIL import Image
from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.image_processing_utils import ImageProcessor
from tinker_cookbook.renderers import (
    Message,
    ParseTermination,
    RenderContext,
    Renderer,
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
    ToolCall,
    TrainOnWhat,
    register_renderer,
)
from tinker_cookbook.renderers.base import RenderedMessage, ToolSpec
from tinker_cookbook.tokenizer_utils import Tokenizer

from training.renderer.reasoning_fields import (
    original_reasoning,
    original_reasoning_content,
)

STOP_SEQUENCE = "<|close|>message<|sep|>"
IMAGE_PLACEHOLDER = "<|kimi_image_placeholder|>"
MEDIA_PAD_TOKEN = "<|media_pad|>"

_OPEN = "<|open|>"
_CLOSE = "<|close|>"
_SEP = "<|sep|>"
_TOOL_MARKER_KEY = "_fireworks_kimi_k3_tools"
_DATA_URL = re.compile(
    r"^data:image/(?P<format>png|jpeg|jpg);base64,(?P<data>.+)$",
    re.DOTALL,
)


def _deep_sort(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _deep_sort(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_deep_sort(item) for item in value]
    return value


def _normalize_tool_choice(value: Any) -> str | None:
    if value is None or value == "auto":
        return None
    if value == "none":
        return "none"
    if value == "required":
        return "required"
    if isinstance(value, Mapping):
        return "required"
    raise ValueError(f"Unsupported Kimi K3 tool_choice: {value!r}")


def _normalize_reasoning(value: Any) -> tuple[bool, str | None]:
    if isinstance(value, Mapping):
        value = value.get("effort", value.get("reasoning_effort"))
    if value is False or value == "none":
        return False, None
    aliases = {
        None: "max",
        "low": "low",
        "medium": "high",
        "high": "high",
        "xhigh": "max",
        "max": "max",
    }
    if value is True:
        value = None
    if value not in aliases:
        raise ValueError(f"Unsupported Kimi K3 reasoning effort: {value!r}")
    return True, aliases[value]


@dataclass(frozen=True)
class KimiK3RenderOptions:
    """Request controls understood by the release Python template."""

    thinking: bool = True
    thinking_effort: str | None = "max"
    tool_choice: str | None = None
    response_format: Any = None

    @classmethod
    def from_api(
        cls,
        *,
        reasoning_effort: Any = None,
        thinking: Any = None,
        tool_choice: Any = None,
        response_format: Any = None,
    ) -> "KimiK3RenderOptions":
        if isinstance(thinking, Mapping):
            thinking_type = thinking.get("type", "enabled")
            if thinking_type in {"disabled", "none"}:
                reasoning_value = False
            elif thinking_type == "enabled":
                reasoning_value = (
                    reasoning_effort
                    if reasoning_effort is not None
                    else thinking.get("effort", thinking.get("reasoning_effort"))
                )
            else:
                raise ValueError(
                    f"Unsupported Kimi K3 thinking type: {thinking_type!r}"
                )
        elif thinking is False:
            reasoning_value = False
        else:
            reasoning_value = reasoning_effort

        enabled, effort = _normalize_reasoning(reasoning_value)
        return cls(
            thinking=enabled,
            thinking_effort=effort,
            tool_choice=_normalize_tool_choice(tool_choice),
            response_format=_deep_sort(response_format),
        )

    def native_kwargs(self, *, include_request_tail: bool = True) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"thinking": self.thinking}
        if self.thinking and self.thinking_effort is not None:
            kwargs["thinking_effort"] = self.thinking_effort
        if include_request_tail:
            if self.tool_choice is not None:
                kwargs["tool_choice"] = self.tool_choice
            if self.response_format is not None:
                kwargs["response_format"] = copy.deepcopy(self.response_format)
        return kwargs


@dataclass(frozen=True)
class _KimiK3Request:
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] = ()
    options: KimiK3RenderOptions = field(default_factory=KimiK3RenderOptions)


class KimiK3Conversation(list):
    """List-compatible envelope for request controls outside Tinker's base API."""

    def __init__(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        tools: Iterable[Mapping[str, Any]] | None = None,
        options: KimiK3RenderOptions | None = None,
    ) -> None:
        super().__init__(copy.deepcopy([dict(message) for message in messages]))
        self.tools = tuple(copy.deepcopy([dict(tool) for tool in (tools or ())]))
        self.options = options or KimiK3RenderOptions()


@dataclass(frozen=True)
class _KimiK3Media:
    data: bytes
    format: str
    mode: str
    width: int
    height: int

    def open(self) -> Image.Image:
        image = Image.open(io.BytesIO(self.data))
        image.load()
        return image


@dataclass(frozen=True)
class _KimiK3ProcessedMedia:
    media: _KimiK3Media
    image_prompt: str
    expected_tokens: int


@dataclass(frozen=True)
class KimiK3RenderTrace:
    """Inspectable representations used by parity and multimodal tests."""

    symbolic_token_ids: tuple[int, ...]
    token_in_ids: tuple[int, ...]
    materialized_one_pad_ids: tuple[int, ...]
    expanded_token_ids: tuple[int, ...]
    model_input: tinker.ModelInput
    processed_media: tuple[_KimiK3ProcessedMedia, ...]
    segment_labels: tuple[str, ...]


def _canonical_image_bytes(
    image: Image.Image,
    format_name: str = "png",
) -> tuple[bytes, str]:
    output = io.BytesIO()
    normalized = "jpeg" if format_name.lower() in {"jpg", "jpeg"} else "png"
    if normalized == "jpeg" and image.mode not in {"RGB", "L"}:
        raise ValueError("JPEG Kimi K3 inputs must already be RGB or L")
    image.save(output, format="JPEG" if normalized == "jpeg" else "PNG")
    return output.getvalue(), normalized


def _decode_image_payload(payload: Any) -> tuple[bytes, str]:
    if isinstance(payload, Image.Image):
        return _canonical_image_bytes(payload)
    if isinstance(payload, bytes):
        return payload, "png"
    if isinstance(payload, Mapping):
        if "url" in payload:
            return _decode_image_payload(payload["url"])
        if "data" in payload:
            raw = payload["data"]
            if isinstance(raw, str):
                raw = base64.b64decode(raw, validate=True)
            return bytes(raw), str(payload.get("format", "png")).lower()
    if isinstance(payload, str):
        match = _DATA_URL.match(payload)
        if match:
            return (
                base64.b64decode(match.group("data"), validate=True),
                "jpeg" if match.group("format") in {"jpg", "jpeg"} else "png",
            )
        raise ValueError(
            "Kimi K3 images must be inline bytes/data URLs; remote URLs must "
            "be materialized before rendering"
        )
    raise TypeError(f"Unsupported Kimi K3 image payload: {type(payload).__name__}")


def _normalize_media(payload: Any) -> _KimiK3Media:
    data, format_name = _decode_image_payload(payload)
    image = Image.open(io.BytesIO(data))
    image.load()
    actual_format = (image.format or format_name).lower()
    if actual_format == "jpg":
        actual_format = "jpeg"
    if actual_format not in {"png", "jpeg"}:
        data, actual_format = _canonical_image_bytes(image, "png")
    return _KimiK3Media(
        data=data,
        format=actual_format,
        mode=image.mode,
        width=image.width,
        height=image.height,
    )


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, Mapping):
        result = copy.deepcopy(dict(tool_call))
        function = result.get("function")
        if isinstance(function, Mapping):
            result["function"] = dict(function)
        return result

    function = getattr(tool_call, "function", None)
    if function is None:
        raise TypeError(f"Unsupported Kimi K3 tool call: {type(tool_call)!r}")
    result: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": function.name,
            "arguments": function.arguments,
        },
    }
    call_id = getattr(tool_call, "id", None)
    if call_id is not None:
        result["id"] = call_id
    return result


def _native_message(message: Mapping[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        raise ValueError(f"Unsupported Kimi K3 message role: {role!r}")

    native: dict[str, Any] = {"role": role}
    for key in ("name", "tool", "tool_call_id", "id", "tools"):
        if message.get(key) is not None:
            native[key] = copy.deepcopy(message[key])
    # Preserve per-message SFT masks for TrainOnWhat.CUSTOMIZED. The release
    # chat template ignores unknown keys, so this does not change oracle tokens.
    if "trainable" in message:
        native["trainable"] = bool(message["trainable"])

    content = message.get("content", "")
    reasoning_parts: list[str] = []
    if isinstance(content, list):
        visible_parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, Mapping):
                raise TypeError("Kimi K3 multipart content entries must be objects")
            part_type = part.get("type")
            if part_type == "thinking":
                if role != "assistant":
                    raise ValueError(
                        "Kimi K3 thinking content is supported only in assistant messages"
                    )
                reasoning_parts.append(str(part.get("thinking", "")))
            elif part_type == "text":
                visible_parts.append(
                    {"type": "text", "text": str(part.get("text", ""))}
                )
            elif part_type in {"image", "image_url"}:
                if role != "user":
                    raise ValueError(
                        "Kimi K3 structured images are supported only in user messages"
                    )
                payload = part.get(part_type)
                if payload is None:
                    payload = part.get("image")
                visible_parts.append({"type": "image", "image": payload})
            else:
                raise ValueError(f"Unsupported Kimi K3 content part: {part_type!r}")
        native["content"] = visible_parts
    else:
        native["content"] = "" if content is None else str(content)

    if role == "assistant":
        has_reasoning_content, reasoning_content = original_reasoning_content(message)
        has_reasoning, reasoning = original_reasoning(message)
        selected_reasoning = (
            reasoning_content
            if has_reasoning_content and reasoning_content
            else reasoning
            if has_reasoning and reasoning
            else "".join(reasoning_parts)
        )
        if selected_reasoning:
            native["reasoning_content"] = selected_reasoning

    tool_calls = message.get("tool_calls")
    if tool_calls:
        native["tool_calls"] = [_tool_call_to_dict(call) for call in tool_calls]
    return native


def _normalize_tools(tools: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        copied = copy.deepcopy(dict(tool))
        if copied.get("type") == "function" and isinstance(
            copied.get("function"), Mapping
        ):
            normalized.append(copied)
        else:
            normalized.append({"type": "function", "function": copied})
    return tuple(normalized)


def _segments_equal(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return len(left) == len(right) and all(
        a.text == b.text and bool(a.allow_special) == bool(b.allow_special)
        for a, b in zip(left, right, strict=True)
    )


def _find_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        return start
    limit = len(haystack) - len(needle) + 1
    for index in range(start, max(start, limit)):
        if haystack[index : index + len(needle)] == needle:
            return index
    return -1


def _count_subsequence(haystack: list[int], needle: list[int]) -> int:
    count = 0
    start = 0
    while True:
        found = _find_subsequence(haystack, needle, start)
        if found < 0:
            return count
        count += 1
        start = found + len(needle)


def _decode(tokenizer: Any, tokens: list[int]) -> str:
    try:
        return str(tokenizer.decode(tokens))
    except TypeError:
        return str(tokenizer.decode(tokens, skip_special_tokens=False))


def _tokenize_control(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer._encode_text_piece(text, allow_special_tokens=True))


def _attribute(tag: str, name: str) -> str | None:
    match = re.search(rf'\b{re.escape(name)}="([^"]*)"', tag)
    return html.unescape(match.group(1)) if match else None


def _typed_argument(value: str, type_name: str | None) -> Any:
    if type_name in {"object", "array", "number", "boolean", "null"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


_CALL_PATTERN = re.compile(
    rf"{re.escape(_OPEN)}call(?P<header>.*?){re.escape(_SEP)}"
    rf"(?P<body>.*?){re.escape(_CLOSE)}call{re.escape(_SEP)}",
    re.DOTALL,
)
_ARG_PATTERN = re.compile(
    rf"{re.escape(_OPEN)}argument(?P<header>.*?){re.escape(_SEP)}"
    rf"(?P<body>.*?){re.escape(_CLOSE)}argument{re.escape(_SEP)}",
    re.DOTALL,
)
_JSON_PATTERN = re.compile(
    rf"{re.escape(_OPEN)}json(?P<header>.*?){re.escape(_SEP)}"
    rf"(?P<body>.*?){re.escape(_CLOSE)}json{re.escape(_SEP)}",
    re.DOTALL,
)


def _parse_tool_calls(
    tokenizer: Any,
    tokens: list[int],
) -> tuple[list[ToolCall], bool]:
    text = _decode(tokenizer, tokens)
    calls: list[ToolCall] = []
    call_matches = list(_CALL_PATTERN.finditer(text))
    well_formed = bool(call_matches)
    for call_match in call_matches:
        name = _attribute(call_match.group("header"), "tool")
        if not name:
            well_formed = False
            continue
        call_well_formed = True
        arguments: dict[str, Any] = {}
        json_match = _JSON_PATTERN.search(call_match.group("body"))
        if json_match:
            try:
                parsed = json.loads(json_match.group("body"))
                if isinstance(parsed, dict):
                    arguments = parsed
                else:
                    call_well_formed = False
            except json.JSONDecodeError:
                call_well_formed = False
        else:
            for argument_match in _ARG_PATTERN.finditer(call_match.group("body")):
                key = _attribute(argument_match.group("header"), "key")
                if key is None:
                    call_well_formed = False
                    continue
                type_name = _attribute(argument_match.group("header"), "type")
                arguments[key] = _typed_argument(
                    argument_match.group("body"),
                    type_name,
                )
        if not call_well_formed:
            well_formed = False
            continue
        calls.append(
            ToolCall(
                id=f"{name}:{len(calls)}",
                function=ToolCall.FunctionBody(
                    name=name,
                    arguments=json.dumps(
                        arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            )
        )
    return calls, well_formed


@dataclass(frozen=True)
class _ParsedResponse:
    message: Message
    termination: ParseTermination
    reasoning: str
    content: str


def _parse_response(tokenizer: Any, response: Iterable[int]) -> _ParsedResponse:
    """Parse only structural token IDs, so literal control-looking text is safe."""

    tokens = [int(token) for token in response]
    stop = _tokenize_control(tokenizer, f"{_CLOSE}message{_SEP}")
    stop_count = _count_subsequence(tokens, stop)
    stop_at = _find_subsequence(tokens, stop)
    termination = (
        ParseTermination.STOP_SEQUENCE
        if stop_count == 1
        else ParseTermination.MALFORMED
    )
    body = tokens[:stop_at] if stop_at >= 0 else tokens

    close_think_open_response = _tokenize_control(
        tokenizer,
        f"{_CLOSE}think{_SEP}{_OPEN}response{_SEP}",
    )
    close_response = _tokenize_control(tokenizer, f"{_CLOSE}response{_SEP}")
    open_tools = _tokenize_control(tokenizer, f"{_OPEN}tools{_SEP}")
    close_tools = _tokenize_control(tokenizer, f"{_CLOSE}tools{_SEP}")

    channel_at = _find_subsequence(body, close_think_open_response)
    if channel_at >= 0:
        reasoning_tokens = body[:channel_at]
        response_start = channel_at + len(close_think_open_response)
    else:
        reasoning_tokens = []
        response_start = 0

    response_end = _find_subsequence(body, close_response, response_start)
    malformed = response_end < 0
    if malformed:
        response_end = len(body)
    content_tokens = body[response_start:response_end]
    remainder_start = min(len(body), response_end + len(close_response))

    tools: list[ToolCall] = []
    tools_at = _find_subsequence(body, open_tools, remainder_start)
    if tools_at >= 0:
        tools_end = _find_subsequence(
            body,
            close_tools,
            tools_at + len(open_tools),
        )
        if tools_end < 0:
            malformed = True
            tools_end = len(body)
        tools, tools_well_formed = _parse_tool_calls(
            tokenizer,
            body[tools_at + len(open_tools) : tools_end],
        )
        malformed = malformed or not tools_well_formed

    reasoning = _decode(tokenizer, reasoning_tokens)
    content = _decode(tokenizer, content_tokens)
    parts: list[dict[str, Any]] = []
    if reasoning:
        parts.append({"type": "thinking", "thinking": reasoning})
    if content or not parts:
        parts.append({"type": "text", "text": content})
    message: Message = {"role": "assistant", "content": parts}
    if tools:
        message["tool_calls"] = tools
    if malformed or stop_count != 1:
        termination = ParseTermination.MALFORMED
    return _ParsedResponse(
        message=message,
        termination=termination,
        reasoning=reasoning,
        content=content,
    )


class KimiK3Renderer(Renderer):
    """Whole-conversation K3 renderer backed by the tokenizer's Python encoder."""

    supports_streaming = True
    supports_per_message_rendering = False
    _preserves_explicit_empty_system_with_tools = True

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_processor: ImageProcessor | None = None,
        *,
        options: KimiK3RenderOptions | None = None,
    ) -> None:
        super().__init__(tokenizer)
        self.image_processor = image_processor
        self.default_options = options or KimiK3RenderOptions()

        tokenizer_module = __import__(type(tokenizer).__module__, fromlist=["*"])
        build_segments = getattr(tokenizer_module, "build_chat_segments", None)
        if not callable(build_segments):
            raise TypeError(
                "Kimi K3 tokenizer must expose encoding_k3.build_chat_segments"
            )
        self._build_chat_segments = build_segments
        encoding_module = __import__(build_segments.__module__, fromlist=["*"])
        normalize_results = getattr(
            encoding_module,
            "normalize_xtml_tool_result_messages",
            None,
        )
        if not callable(normalize_results):
            raise TypeError(
                "Kimi K3 tokenizer must expose normalize_xtml_tool_result_messages"
            )
        self._normalize_tool_results = normalize_results

        media_pad_ids = list(
            tokenizer._encode_text_piece(
                MEDIA_PAD_TOKEN,
                allow_special_tokens=True,
            )
        )
        if len(media_pad_ids) != 1:
            raise ValueError(
                "Kimi K3 tokenizer must encode <|media_pad|> as one special token"
            )
        self.image_placeholder_token_id = int(media_pad_ids[0])

        if image_processor is not None and hasattr(
            image_processor,
            "preserve_image_mode",
        ):
            image_processor.preserve_image_mode = True

    @property
    def has_extension_property(self) -> bool:
        return True

    def get_stop_sequences(self) -> list[str]:
        return [STOP_SEQUENCE]

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        del message, ctx
        raise RendererError("Kimi K3 requires whole-conversation rendering")

    def create_conversation_prefix_with_tools(
        self,
        tools: list[ToolSpec],
        system_prompt: str = "",
    ) -> list[Message]:
        marker: Message = {"role": "tool_declare", "content": ""}
        marker[_TOOL_MARKER_KEY] = copy.deepcopy(tools)  # type: ignore[typeddict-unknown-key]
        prefix = [marker]
        if system_prompt:
            prefix.append(Message(role="system", content=system_prompt))
        return prefix

    def prepare_conversation(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        tools: Iterable[Mapping[str, Any]] | None = None,
        request_kwargs: Mapping[str, Any] | None = None,
    ) -> KimiK3Conversation:
        """Bind API request controls to a whole-conversation render."""
        request_kwargs = request_kwargs or {}
        if "thinking" in request_kwargs:
            thinking_options = KimiK3RenderOptions.from_api(
                thinking=request_kwargs.get("thinking"),
                reasoning_effort=request_kwargs.get("reasoning_effort"),
            )
            thinking = thinking_options.thinking
            thinking_effort = thinking_options.thinking_effort
        elif "reasoning_effort" in request_kwargs and self.default_options.thinking:
            thinking_options = KimiK3RenderOptions.from_api(
                thinking=True,
                reasoning_effort=request_kwargs["reasoning_effort"],
            )
            thinking = thinking_options.thinking
            thinking_effort = thinking_options.thinking_effort
        else:
            thinking = self.default_options.thinking
            thinking_effort = self.default_options.thinking_effort

        tool_choice = (
            _normalize_tool_choice(request_kwargs["tool_choice"])
            if "tool_choice" in request_kwargs
            else self.default_options.tool_choice
        )
        response_format = (
            _deep_sort(request_kwargs["response_format"])
            if "response_format" in request_kwargs
            else copy.deepcopy(self.default_options.response_format)
        )
        return KimiK3Conversation(
            messages,
            tools=tools,
            options=KimiK3RenderOptions(
                thinking=thinking,
                thinking_effort=thinking_effort,
                tool_choice=tool_choice,
                response_format=response_format,
            ),
        )

    def _unpack_request(self, messages: list[Message]) -> _KimiK3Request:
        if isinstance(messages, KimiK3Conversation):
            tools = _normalize_tools(messages.tools)
            options = messages.options
            raw_messages = list(messages)
        else:
            tools = ()
            options = self.default_options
            raw_messages = list(messages)

        native_messages: list[dict[str, Any]] = []
        marker_tools: tuple[dict[str, Any], ...] = ()
        for message in raw_messages:
            if message.get("role") == "tool_declare" and _TOOL_MARKER_KEY in message:
                if marker_tools:
                    raise ValueError("Kimi K3 accepts one tool-declaration prefix")
                marker_tools = _normalize_tools(message[_TOOL_MARKER_KEY])  # type: ignore[typeddict-item]
                continue
            native_messages.append(_native_message(message))

        if tools and marker_tools:
            raise ValueError(
                "Kimi K3 tools were supplied in both the conversation envelope "
                "and the tool prefix"
            )
        return _KimiK3Request(
            messages=tuple(native_messages),
            tools=tools or marker_tools,
            options=options,
        )

    def _normalize_messages_and_media(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[_KimiK3Media]]:
        normalized: list[dict[str, Any]] = []
        media: list[_KimiK3Media] = []
        for raw in messages:
            message = copy.deepcopy(raw)
            content = message.get("content")
            if not isinstance(content, list):
                normalized.append(message)
                continue
            parts: list[dict[str, Any]] = []
            for part in content:
                part_type = part.get("type")
                if part_type == "image":
                    media.append(_normalize_media(part.get("image")))
                    parts.append({"type": "image", "image": IMAGE_PLACEHOLDER})
                elif part_type == "text":
                    parts.append({"type": "text", "text": str(part.get("text", ""))})
                else:
                    raise ValueError(
                        f"Unsupported normalized Kimi K3 content part: {part_type!r}"
                    )
            message["content"] = parts
            normalized.append(message)
        return normalized, media

    def _process_media(
        self,
        media: Sequence[_KimiK3Media],
    ) -> list[_KimiK3ProcessedMedia]:
        if media and self.image_processor is None:
            raise ValueError(
                "Kimi K3 image messages require the release image processor"
            )
        processed: list[_KimiK3ProcessedMedia] = []
        for item in media:
            image_item = {"type": "image", "image": item.open()}
            expected = int(self.image_processor.media_tokens_calculator(image_item))
            prompt = str(
                self.image_processor.make_image_prompt(item.width, item.height)
            )
            batch = self.image_processor.preprocess(
                [image_item],
                return_tensors="pt",
            )
            grid = tuple(int(value) for value in batch["grid_thws"][0].tolist())
            merge = int(self.image_processor.media_proc_cfg["merge_kernel_size"])
            calculated = grid[0] * (grid[1] // merge) * (grid[2] // merge)
            if expected != calculated:
                raise ValueError(
                    f"Kimi K3 image token invariant failed: {expected} != {calculated}"
                )
            processed.append(
                _KimiK3ProcessedMedia(
                    media=item,
                    image_prompt=prompt,
                    expected_tokens=expected,
                )
            )
        return processed

    def _normalized_request(
        self,
        messages: list[Message],
    ) -> tuple[_KimiK3Request, list[_KimiK3ProcessedMedia]]:
        request = self._unpack_request(messages)
        native_messages, media = self._normalize_messages_and_media(request.messages)
        native_messages = list(self._normalize_tool_results(native_messages))
        normalized = _KimiK3Request(
            messages=tuple(native_messages),
            tools=request.tools,
            options=request.options,
        )
        return normalized, self._process_media(media)

    def _native_segments(
        self,
        messages: list[dict[str, Any]],
        request: _KimiK3Request,
        *,
        add_generation_prompt: bool,
        include_request_tail: bool,
        image_prompts: list[str] | None,
    ) -> list[Any]:
        kwargs = request.options.native_kwargs(
            include_request_tail=include_request_tail
        )
        thinking = kwargs.pop("thinking")
        return list(
            self._build_chat_segments(
                messages,
                tools=list(request.tools) or None,
                add_generation_prompt=add_generation_prompt,
                thinking=thinking,
                image_prompts=image_prompts,
                **kwargs,
            )
        )

    def _encode_segments(self, segments: Iterable[Any]) -> list[int]:
        return list(self.tokenizer._encode_chat_segments(list(segments)))

    @staticmethod
    def _replace_image_segments(
        segments: Sequence[Any],
        processed: Sequence[_KimiK3ProcessedMedia],
    ) -> list[Any]:
        materialized: list[Any] = []
        media_index = 0
        for segment in segments:
            if bool(segment.allow_special) and segment.text == IMAGE_PLACEHOLDER:
                if media_index >= len(processed):
                    raise ValueError(
                        "More Kimi K3 image placeholders than image inputs"
                    )
                materialized.append(
                    type(segment)(
                        processed[media_index].image_prompt,
                        allow_special=True,
                    )
                )
                media_index += 1
            else:
                materialized.append(segment)
        if media_index != len(processed):
            raise ValueError("Fewer Kimi K3 image placeholders than image inputs")
        return materialized

    @staticmethod
    def _split_image_prompt(prompt: str) -> tuple[str, str]:
        marker = "<|media_pad|>"
        if prompt.count(marker) != 1 or not prompt.endswith("<|media_end|>"):
            raise ValueError(f"Unexpected Kimi K3 image prompt: {prompt!r}")
        return tuple(prompt.split(marker, 1))  # type: ignore[return-value]

    def _materialize_segments(
        self,
        segments: Sequence[Any],
        processed: Sequence[_KimiK3ProcessedMedia],
        labels: Sequence[str] | None = None,
        *,
        all_tokens: bool = False,
    ) -> tuple[tinker.ModelInput, list[int], list[int], list[float]]:
        chunks: list[tinker.types.ModelInputChunk] = []
        expanded: list[int] = []
        one_pad: list[int] = []
        weights: list[float] = []
        media_index = 0
        for index, segment in enumerate(segments):
            label = labels[index] if labels is not None else "prompt"
            base_weight = 1.0 if label == "all" else 0.0
            if label.startswith("output:"):
                base_weight = float(label.split(":", 1)[1])
            is_image = bool(segment.allow_special) and segment.text == IMAGE_PLACEHOLDER
            if not is_image:
                tokens = list(
                    self.tokenizer._encode_text_piece(
                        segment.text,
                        allow_special_tokens=segment.allow_special,
                    )
                )
                if tokens:
                    chunks.append(tinker.types.EncodedTextChunk(tokens=tokens))
                    expanded.extend(tokens)
                    one_pad.extend(tokens)
                    weights.extend([base_weight] * len(tokens))
                continue

            if media_index >= len(processed):
                raise ValueError("More Kimi K3 image placeholders than image inputs")
            item = processed[media_index]
            media_index += 1
            prefix, suffix = self._split_image_prompt(item.image_prompt)
            prefix_tokens = list(
                self.tokenizer._encode_text_piece(
                    prefix,
                    allow_special_tokens=True,
                )
            )
            suffix_tokens = list(
                self.tokenizer._encode_text_piece(
                    suffix,
                    allow_special_tokens=True,
                )
            )
            chunks.append(tinker.types.EncodedTextChunk(tokens=prefix_tokens))
            chunks.append(
                tinker.types.ImageChunk(
                    data=item.media.data,
                    format=item.media.format,
                    expected_tokens=item.expected_tokens,
                )
            )
            chunks.append(tinker.types.EncodedTextChunk(tokens=suffix_tokens))
            expanded.extend(prefix_tokens)
            expanded.extend([self.image_placeholder_token_id] * item.expected_tokens)
            expanded.extend(suffix_tokens)
            one_pad.extend(prefix_tokens)
            one_pad.append(self.image_placeholder_token_id)
            one_pad.extend(suffix_tokens)
            affix_weight = 1.0 if all_tokens else 0.0
            weights.extend([affix_weight] * len(prefix_tokens))
            weights.extend([0.0] * item.expected_tokens)
            weights.extend([affix_weight] * len(suffix_tokens))

        if media_index != len(processed):
            raise ValueError("Fewer Kimi K3 image placeholders than image inputs")
        return tinker.ModelInput(chunks=chunks), expanded, one_pad, weights

    def _token_in_ids(self, segments: Sequence[Any]) -> list[int]:
        tokens: list[int] = []
        for segment in segments:
            if bool(segment.allow_special) and segment.text == IMAGE_PLACEHOLDER:
                tokens.append(self.image_placeholder_token_id)
            else:
                tokens.extend(
                    self.tokenizer._encode_text_piece(
                        segment.text,
                        allow_special_tokens=segment.allow_special,
                    )
                )
        return tokens

    def render_trace(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = True,
    ) -> KimiK3RenderTrace:
        request, processed = self._normalized_request(messages)
        symbolic = self._native_segments(
            list(request.messages),
            request,
            add_generation_prompt=add_generation_prompt,
            include_request_tail=True,
            image_prompts=None,
        )
        materialized = self._replace_image_segments(symbolic, processed)
        materialized_ids = self._encode_segments(materialized)
        model_input, expanded, one_pad, _ = self._materialize_segments(
            symbolic,
            processed,
        )
        if one_pad != materialized_ids:
            raise ValueError(
                "Kimi K3 image chunks disagree with the release image prompt"
            )
        return KimiK3RenderTrace(
            symbolic_token_ids=tuple(self._encode_segments(symbolic)),
            token_in_ids=tuple(self._token_in_ids(symbolic)),
            materialized_one_pad_ids=tuple(materialized_ids),
            expanded_token_ids=tuple(expanded),
            model_input=model_input,
            processed_media=tuple(processed),
            segment_labels=tuple("prompt" for _ in symbolic),
        )

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: str = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        if role != "assistant":
            raise ValueError("Kimi K3 can generate only assistant messages")
        model_input = self.render_trace(
            messages,
            add_generation_prompt=True,
        ).model_input
        if prefill:
            model_input = model_input.append(
                tinker.types.EncodedTextChunk(
                    tokens=self.tokenizer._encode_text_piece(
                        prefill,
                        allow_special_tokens=False,
                    )
                )
            )
        return model_input

    @staticmethod
    def _message_weight(
        message: dict[str, Any],
        index: int,
        messages: Sequence[dict[str, Any]],
        train_on_what: TrainOnWhat,
    ) -> float:
        role = message.get("role")
        if train_on_what == TrainOnWhat.ALL_TOKENS:
            return 1.0
        if train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE:
            return float(index == len(messages) - 1 and role == "assistant")
        if train_on_what == TrainOnWhat.LAST_ASSISTANT_TURN:
            last_user = max(
                (
                    i
                    for i, candidate in enumerate(messages)
                    if candidate.get("role") == "user"
                ),
                default=-1,
            )
            return float(role == "assistant" and index > last_user)
        if train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
            return float(role == "assistant")
        if train_on_what == TrainOnWhat.ALL_MESSAGES:
            return 1.0
        if train_on_what == TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES:
            return float(role in {"user", "system"})
        if train_on_what == TrainOnWhat.CUSTOMIZED:
            if "trainable" not in message:
                raise ValueError(
                    "CUSTOMIZED Kimi K3 rows require trainable on every message"
                )
            return float(message["trainable"])
        raise RendererError(f"Unknown train_on_what: {train_on_what}")

    @staticmethod
    def _label_message_segments(
        delta: Sequence[Any],
        message: dict[str, Any],
        output_weight: float,
        *,
        all_tokens: bool,
    ) -> list[str]:
        labels = ["all" if all_tokens else "prompt"] * len(delta)
        required_separators = 2 if message.get("role") == "assistant" else 1
        seen = 0
        output_start = len(delta)
        for index, segment in enumerate(delta):
            if bool(segment.allow_special) and segment.text == _SEP:
                seen += 1
                if seen == required_separators:
                    output_start = index + 1
                    break
        output_end = len(delta)
        if (
            delta
            and bool(delta[-1].allow_special)
            and delta[-1].text == "<|end_of_msg|>"
        ):
            output_end -= 1
        for index in range(output_start, output_end):
            labels[index] = "all" if all_tokens else f"output:{output_weight}"
        return labels

    def _supervised_segments_and_labels(
        self,
        request: _KimiK3Request,
        train_on_what: TrainOnWhat,
    ) -> tuple[list[Any], list[str]]:
        messages = list(request.messages)
        all_tokens = train_on_what == TrainOnWhat.ALL_TOKENS
        base = self._native_segments(
            [],
            request,
            add_generation_prompt=False,
            include_request_tail=False,
            image_prompts=None,
        )
        segments = list(base)
        labels = ["all" if all_tokens else "prompt"] * len(base)
        previous = list(base)

        for index, message in enumerate(messages):
            current = self._native_segments(
                messages[: index + 1],
                request,
                add_generation_prompt=False,
                include_request_tail=False,
                image_prompts=None,
            )
            if not _segments_equal(current[: len(previous)], previous):
                raise ValueError(
                    "Kimi K3 progressive rendering stopped being a prefix at "
                    f"message {index}"
                )
            delta = current[len(previous) :]
            weight = self._message_weight(
                message,
                index,
                messages,
                train_on_what,
            )
            segments.extend(delta)
            labels.extend(
                self._label_message_segments(
                    delta,
                    message,
                    weight,
                    all_tokens=all_tokens,
                )
            )
            previous = current

        authoritative = self._native_segments(
            messages,
            request,
            add_generation_prompt=False,
            include_request_tail=True,
            image_prompts=None,
        )
        if not _segments_equal(authoritative[: len(previous)], previous):
            raise ValueError(
                "Kimi K3 request controls do not extend the conversation prefix"
            )
        tail = authoritative[len(previous) :]
        segments.extend(tail)
        labels.extend(["all" if all_tokens else "prompt"] * len(tail))
        if not _segments_equal(segments, authoritative):
            raise ValueError(
                "Kimi K3 supervised trace differs from the release rendering"
            )
        return segments, labels

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        request, processed = self._normalized_request(messages)
        segments, labels = self._supervised_segments_and_labels(
            request,
            train_on_what,
        )
        model_input, expanded, one_pad, weights = self._materialize_segments(
            segments,
            processed,
            labels,
            all_tokens=train_on_what == TrainOnWhat.ALL_TOKENS,
        )
        materialized = self._replace_image_segments(segments, processed)
        materialized_ids = self._encode_segments(materialized)
        if one_pad != materialized_ids:
            raise ValueError(
                "Kimi K3 image chunks disagree with the release image prompt"
            )
        if len(expanded) != len(weights) or model_input.length != len(weights):
            raise ValueError("Kimi K3 supervised token/weight length mismatch")
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
        parsed = _parse_response(self.tokenizer, response)
        return parsed.message, parsed.termination

    def parse_response_streaming(self, response: list[int]) -> Iterator[Any]:
        parsed = _parse_response(self.tokenizer, response)
        yield StreamingMessageHeader(role="assistant")
        content_index = 0
        if parsed.reasoning:
            yield StreamingThinkingDelta(
                thinking=parsed.reasoning,
                content_index=content_index,
            )
            content_index += 1
        if parsed.content:
            yield StreamingTextDelta(
                text=parsed.content,
                content_index=content_index,
            )
        yield parsed.message

    def to_openai_message(self, message: Message) -> dict[str, Any]:
        return _native_message(message)


def _kimi_k3_factory(
    tokenizer: Tokenizer,
    image_processor: ImageProcessor | None = None,
) -> KimiK3Renderer:
    return KimiK3Renderer(tokenizer, image_processor=image_processor)


def _kimi_k3_disable_thinking_factory(
    tokenizer: Tokenizer,
    image_processor: ImageProcessor | None = None,
) -> KimiK3Renderer:
    return KimiK3Renderer(
        tokenizer,
        image_processor=image_processor,
        options=KimiK3RenderOptions(thinking=False, thinking_effort=None),
    )


register_renderer("kimi_k3", _kimi_k3_factory)
register_renderer("kimi_k3_disable_thinking", _kimi_k3_disable_thinking_factory)
