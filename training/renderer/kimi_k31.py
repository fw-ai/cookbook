"""Renderer for the Moonshot Kimi K3.1 XTML chat protocol.

Kimi K3.1 speaks the same XTML protocol as Kimi K3 and ships the same
tokenizer class, vision processor and segment contract, so this renderer is a
thin adaptation of :class:`training.renderer.kimi_k3.KimiK3Renderer`: whole
conversation rendering, image handling and loss-mask assignment are inherited,
and the XTML segments come from the ``encoding_k3.py`` that the checkpoint's
``TikTokenTokenizer`` delegates to. Load the K3.1 checkpoint tokenizer
(``moonshotai/Kimi-K3.1-Pro``, pinned revision) the way ``kimi_k3`` loads K3's;
the fast-tokenizer bundle from ``training.utils.kimi_k31_tokenizer`` serves TITO
only and is rejected here.

What differs from ``kimi_k3``:

* **Thinking tiers.** The pinned K3.1 encoder accepts ``low``, ``high`` and
  ``max`` (upstream dropped ``medium``); a ``medium`` request folds into
  ``high`` as in ``kimi_k3``. A request without a reasoning effort renders no
  ``thinking-effort`` system message, matching Fireworks serving;
  ``kimi_k31_<low|high|max>`` pin a tier and
  ``kimi_k31_disable_thinking`` drops the think channel.
* **Tool arguments.** The K3.1 encoder renders non-string arguments from their
  JSON literal bytes (``1e2`` stays ``1e2``). Dataset normalization hands
  argument strings over unchanged (``preserves_tool_argument_literals``), and
  ``parse_response`` returns arguments that re-render to the sampled tokens.
* **Images.** Token counts and prompts come from ``KimiK31ImageTokenCounter``,
  which reads only ``preprocessor_config.json``; the checkpoint's processor
  module is not loaded. Pixel preprocessing stays in FireTitan.
* **Tool shaping.** Request tools and dynamically declared message-level
  tools drop ``strict``/``container``/``None`` the way Fireworks serving does
  before the encoder sees them.
* **Parsing.** Token-level: literal control text stays text, a trailing proper
  prefix of the stop sequence (what completions keep after stripping the stop
  string) is the stop boundary, and a turn truncated before any channel
  boundary follows the request's thinking mode when the caller passes it.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from transformers.image_processing_utils import BaseImageProcessor

from training.renderer import (
    Message,
    ParseTermination,
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
    ToolCall,
    register_renderer,
)
from training.renderer.image_processing import ImageProcessor
from training.renderer.kimi_k3 import (
    STOP_SEQUENCE,
    KimiK3Conversation,
    KimiK3Renderer,
    KimiK3RenderOptions,
    _count_subsequence,
    _find_subsequence,
)
from training.renderer.kimi_k3 import _native_message as _k3_native_message
from training.renderer.plugins import register_renderer_image_capability
from training.renderer.tokenizer import Tokenizer

THINKING_EFFORTS = ("low", "high", "max")
# API efforts the encoder has no tier for, folded the way ``kimi_k3`` does.
_EFFORT_ALIASES = {"medium": "high", "xhigh": "max"}

logger = logging.getLogger(__name__)

_OPEN = "<|open|>"
_CLOSE = "<|close|>"
_SEP = "<|sep|>"
_END_OF_MSG = "<|end_of_msg|>"
_HEADER_RE = re.compile(r'^(?P<tag>[A-Za-z_]+)(?P<attrs>(?: [A-Za-z_]+="[^"]*")*)$')
_ATTR_RE = re.compile(r' (?P<key>[A-Za-z_]+)="(?P<value>[^"]*)"')


# ---------------------------------------------------------------------------
# Request controls
# ---------------------------------------------------------------------------


def _normalize_reasoning(value: Any) -> tuple[bool, str | None]:
    """Map an API ``reasoning_effort`` to ``(thinking, thinking_effort)``.

    ``None`` keeps thinking on without a ``thinking-effort`` message, which is
    what Fireworks serving renders for a request that omits the field.
    ``medium`` folds into ``high`` and ``xhigh`` into ``max``; every other value
    must be a K3.1 tier.
    """
    if isinstance(value, Mapping):
        value = value.get("effort", value.get("reasoning_effort"))
    if value is False or value == "none":
        return False, None
    if value is None or value is True:
        return True, None
    value = _EFFORT_ALIASES.get(value, value)
    if value not in THINKING_EFFORTS:
        raise ValueError(f"Unsupported Kimi K3.1 reasoning effort: {value!r}")
    return True, value


@dataclass(frozen=True)
class KimiK31RenderOptions(KimiK3RenderOptions):
    """K3 request controls with K3.1's tiers and no default effort message."""

    thinking_effort: str | None = None

    _normalize_reasoning = staticmethod(_normalize_reasoning)

    def __post_init__(self) -> None:
        if self.thinking_effort is not None and self.thinking_effort not in THINKING_EFFORTS:
            raise ValueError(
                "Kimi K3.1 thinking_effort must be None or one of "
                + ", ".join(THINKING_EFFORTS)
                + f"; got {self.thinking_effort!r}"
            )


class KimiK31Conversation(KimiK3Conversation):
    """K3 conversation envelope whose ``options=None`` keeps the renderer's defaults."""

    def __init__(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        tools: Iterable[Mapping[str, Any]] | None = None,
        options: KimiK31RenderOptions | None = None,
    ) -> None:
        super().__init__(messages, tools=tools, options=options)
        self.options = options


def _drop_none(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not None}


def _normalize_tools(tools: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Shape request tools the way Fireworks serving hands them to the encoder.

    Bare function specs are wrapped; ``None`` values, ``container`` and
    ``function.strict`` are dropped (serving excludes them), so a dataset row
    that carries them trains on the prompt inference renders.
    """
    normalized: list[dict[str, Any]] = []
    for raw_tool in tools:
        tool = copy.deepcopy(dict(raw_tool))
        if not (tool.get("type") == "function" and isinstance(tool.get("function"), Mapping)):
            tool = {"type": "function", "function": tool}
        tool = _drop_none(tool)
        tool.pop("container", None)
        function = _drop_none(tool["function"])
        function.pop("strict", None)
        tool["function"] = function
        normalized.append(tool)
    return tuple(normalized)


def _native_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """K3's message projection with dynamically declared tools shaped like request tools."""
    native = _k3_native_message(message)
    if native.get("tools") is not None:
        native["tools"] = [dict(tool) for tool in _normalize_tools(native["tools"])]
    return native


# ---------------------------------------------------------------------------
# Images: token counts from the processor config, no checkpoint Python
# ---------------------------------------------------------------------------


class KimiK31ImageTokenCounter:
    """Count Kimi K3.1 image tokens from the released ``media_proc_cfg``.

    The checkpoint's ``KimiK3VisionProcessor`` decides an image's token count
    purely from its size and ``preprocessor_config.json`` (``navit_resize_image``),
    so rendering needs no checkpoint Python: this class reproduces that
    arithmetic the way Fireworks serving (``KimiK3VLTokenizer``) and FireTitan's
    pixel pipeline do. FireTitan owns pixel preprocessing; the renderer only
    needs the exact pad count declared on each wire ``ImageChunk`` and the
    ``<|media_begin|>`` prompt. ``kimi_k3`` keeps loading the release
    processor; only the K3.1 renderer is routed here.
    """

    _REQUIRED = ("patch_size", "merge_kernel_size", "in_patch_limit", "patch_limit_on_one_side")

    def __init__(self, media_proc_cfg: Mapping[str, Any]) -> None:
        missing = [key for key in self._REQUIRED if key not in media_proc_cfg]
        if missing:
            raise ValueError(f"Kimi K3 media_proc_cfg is missing {', '.join(missing)}")
        self.media_proc_cfg = dict(media_proc_cfg)
        self.patch_size = int(media_proc_cfg["patch_size"])
        self.merge_kernel_size = int(media_proc_cfg["merge_kernel_size"])
        self.in_patch_limit = int(media_proc_cfg["in_patch_limit"])
        self.patch_limit_on_one_side = int(media_proc_cfg["patch_limit_on_one_side"])
        fixed = media_proc_cfg.get("fixed_output_tokens")
        self.fixed_output_tokens = None if fixed is None else int(fixed)
        if min(self.patch_size, self.merge_kernel_size, self.in_patch_limit, self.patch_limit_on_one_side) <= 0:
            raise ValueError("Kimi K3 media_proc_cfg dimensions must be positive")
        # The release processor keeps the caller's image mode; the renderer
        # records it on the materialized media.
        self.preserve_image_mode = True

    @classmethod
    def from_pretrained(cls, model_name: str) -> "KimiK31ImageTokenCounter":
        config, _ = BaseImageProcessor.get_image_processor_dict(model_name)
        media_proc_cfg = config.get("media_proc_cfg")
        if not isinstance(media_proc_cfg, Mapping):
            raise ValueError(f"{model_name}: preprocessor_config.json must contain media_proc_cfg")
        return cls(media_proc_cfg)

    def resize_config(self, width: int, height: int) -> dict[str, int]:
        """``media_utils.navit_resize_image`` from the checkpoint, verbatim arithmetic."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
        patch, merge = self.patch_size, self.merge_kernel_size
        limit = self.patch_limit_on_one_side
        s1 = math.sqrt(self.in_patch_limit / (max(1.0, width // patch) * max(1.0, height // patch)))
        s2 = limit * patch / width
        s3 = limit * patch / height
        scale = min(1.0, s1, s2, s3)
        new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
        new_w = min(new_w, limit * patch)
        new_h = min(new_h, limit * patch)
        factor = merge * patch
        pad_height = (factor - new_h % factor) % factor
        pad_width = (factor - new_w % factor) % factor
        if self.fixed_output_tokens is not None:
            num_tokens = self.fixed_output_tokens
        else:
            token_height = (new_h + pad_height) // factor
            token_width = (new_w + pad_width) // factor
            if token_height * merge > limit or token_width * merge > limit:
                raise ValueError(f"Kimi K3 image {width}x{height} exceeds patch_limit_on_one_side={limit}")
            num_tokens = token_height * token_width
        return {
            "num_tokens": num_tokens,
            "new_width": new_w,
            "new_height": new_h,
            "pad_width": pad_width,
            "pad_height": pad_height,
        }

    def media_tokens_calculator(self, media: Mapping[str, Any]) -> int:
        if media.get("type") != "image":
            raise ValueError(f"Unsupported Kimi K3 media type: {media.get('type')!r}")
        width, height = media["image"].size
        return int(self.resize_config(width, height)["num_tokens"])

    @staticmethod
    def make_image_prompt(width: int, height: int) -> str:
        return f"<|media_begin|>image {width}x{height}<|media_content|><|media_pad|><|media_end|>"

    def grid_thw(self, width: int, height: int) -> tuple[int, int, int]:
        """The ``grid_thws`` row the release processor's ``preprocess`` reports."""
        resize = self.resize_config(width, height)
        return (
            1,
            (resize["new_height"] + resize["pad_height"]) // self.patch_size,
            (resize["new_width"] + resize["pad_width"]) // self.patch_size,
        )


def load_image_token_counter(tokenizer_model: str) -> KimiK31ImageTokenCounter | None:
    """The image processor every K3.1 construction path uses (builders and pickle restore).

    Text rendering needs no processor at all: an artifact without a readable
    ``preprocessor_config.json`` starts text-only, and image rows then fail at
    render time with the renderer's own error instead of every job failing at
    startup.
    """
    try:
        return KimiK31ImageTokenCounter.from_pretrained(tokenizer_model)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning(
            "Kimi K3.1 image token counter unavailable for %s: %s. "
            "Rendering text-only; image rows will fail at render time.",
            tokenizer_model,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Response parsing (token level, so literal control text stays text)
# ---------------------------------------------------------------------------


def _unescape_attr(value: str) -> str:
    """Exact inverse of ``encoding_k3._escape_attr_value`` (``&`` first, then ``"``)."""
    return value.replace("&quot;", '"').replace("&amp;", "&")


class _MalformedToolBlock(Exception):
    """The tools channel does not follow the encoder's XTML grammar."""


@dataclass(frozen=True)
class _ParsedResponse:
    message: Message
    termination: ParseTermination
    reasoning: str
    content: str


class _ResponseParser:
    """Parse sampled K3.1 turns on token ids using the checkpoint tokenizer."""

    def __init__(self, tokenizer: Any, *, thinking: bool) -> None:
        self._tokenizer = tokenizer
        self._thinking = thinking
        self.open_id = self._single_control(_OPEN)
        self.close_id = self._single_control(_CLOSE)
        self.sep_id = self._single_control(_SEP)
        self.end_of_msg_id = self._single_control(_END_OF_MSG)
        self.control_ids = frozenset({self.open_id, self.close_id, self.sep_id, self.end_of_msg_id})
        self.stop_ids = self._encode(STOP_SEQUENCE)
        self._think_to_response = self._encode(f"{_CLOSE}think{_SEP}{_OPEN}response{_SEP}")
        self._close_response = self._encode(f"{_CLOSE}response{_SEP}")
        self._open_tools = self._encode(f"{_OPEN}tools{_SEP}")
        self._close_tools = self._encode(f"{_CLOSE}tools{_SEP}")

    def _encode(self, text: str) -> list[int]:
        return [int(token) for token in self._tokenizer._encode_text_piece(text, allow_special_tokens=True)]

    def _decode(self, tokens: Sequence[int]) -> str:
        return str(self._tokenizer.decode([int(token) for token in tokens]))

    def _single_control(self, text: str) -> int:
        tokens = self._encode(text)
        if len(tokens) != 1:
            raise ValueError(f"Kimi K3.1 tokenizer must encode {text!r} as one special token")
        return tokens[0]

    # -- text channels ------------------------------------------------------

    def _text(self, tokens: Sequence[int]) -> tuple[str, bool]:
        """Decode a text channel; a control token inside it cannot round-trip."""
        clean = not any(token in self.control_ids for token in tokens)
        return self._decode(tokens), clean

    # -- tools channel ------------------------------------------------------

    def _read_open(self, tokens: Sequence[int], index: int) -> tuple[str, dict[str, str], int]:
        if index >= len(tokens) or tokens[index] != self.open_id:
            raise _MalformedToolBlock("content outside a tag")
        end = index + 1
        while end < len(tokens) and tokens[end] != self.sep_id:
            if tokens[end] in self.control_ids:
                raise _MalformedToolBlock("tag header is not closed")
            end += 1
        if end >= len(tokens):
            raise _MalformedToolBlock("block ends mid-tag")
        header = self._decode(tokens[index + 1 : end])
        match = _HEADER_RE.match(header)
        if match is None:
            raise _MalformedToolBlock(f"malformed tag header {header!r}")
        attrs = {item.group("key"): _unescape_attr(item.group("value")) for item in _ATTR_RE.finditer(match["attrs"])}
        return match["tag"], attrs, end + 1

    def _read_close(self, tokens: Sequence[int], index: int, tag: str) -> int:
        if index >= len(tokens) or tokens[index] != self.close_id:
            raise _MalformedToolBlock(f"{tag} tag is not closed")
        end = index + 1
        while end < len(tokens) and tokens[end] != self.sep_id:
            end += 1
        if end >= len(tokens):
            raise _MalformedToolBlock("block ends mid-tag")
        if self._decode(tokens[index + 1 : end]) != tag:
            raise _MalformedToolBlock(f"{tag} closed by another tag")
        return end + 1

    def _read_text(self, tokens: Sequence[int], index: int) -> tuple[str, int]:
        end = index
        while end < len(tokens) and tokens[end] not in self.control_ids:
            end += 1
        return self._decode(tokens[index:end]), end

    @staticmethod
    def _argument_literal(text: str, type_name: str | None) -> str:
        """JSON bytes that ``encoding_k3`` parses back to exactly ``(type, text)``."""
        if type_name in (None, "string"):
            return json.dumps(text, ensure_ascii=False)
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            raise _MalformedToolBlock(f"{type_name} argument is not a JSON literal") from exc
        return text

    def _parse_calls(self, tokens: Sequence[int]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        index = 0
        while index < len(tokens):
            tag, attrs, index = self._read_open(tokens, index)
            if tag != "call":
                raise _MalformedToolBlock(f"tools block holds a {tag!r} tag")
            name = attrs.get("tool")
            if not name:
                raise _MalformedToolBlock("tool call is missing a tool name")
            fields: list[str] = []
            json_block: str | None = None
            while True:
                if index < len(tokens) and tokens[index] == self.close_id:
                    index = self._read_close(tokens, index, "call")
                    break
                inner, inner_attrs, index = self._read_open(tokens, index)
                body, index = self._read_text(tokens, index)
                index = self._read_close(tokens, index, inner)
                if inner == "argument":
                    key = inner_attrs.get("key")
                    if key is None:
                        raise _MalformedToolBlock("tool argument is missing its key")
                    literal = self._argument_literal(body, inner_attrs.get("type"))
                    fields.append(f"{json.dumps(key, ensure_ascii=False)}:{literal}")
                elif inner == "json":
                    # The encoder writes a raw ``json`` block only for arguments it
                    # could not parse as an object. Accept a well-formed object so
                    # the call stays usable; anything else is malformed.
                    try:
                        parsed = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise _MalformedToolBlock("raw JSON arguments are invalid") from exc
                    if not isinstance(parsed, dict):
                        raise _MalformedToolBlock("raw JSON arguments are not an object")
                    json_block = body
                else:
                    raise _MalformedToolBlock(f"tool call holds an unexpected {inner!r} tag")
            arguments = json_block if json_block is not None else "{" + ",".join(fields) + "}"
            calls.append(ToolCall(id=f"{name}:{len(calls)}", function=ToolCall.FunctionBody(name=name, arguments=arguments)))
        return calls

    # -- whole response -----------------------------------------------------

    def _trailing_stop_prefix(self, tokens: Sequence[int]) -> int:
        """Length of the longest proper stop-sequence prefix that ends ``tokens``."""
        for width in range(min(len(self.stop_ids) - 1, len(tokens)), 0, -1):
            if list(tokens[-width:]) == self.stop_ids[:width]:
                return width
        return 0

    def parse(self, response: Iterable[int], *, thinking: bool | None = None) -> _ParsedResponse:
        """Parse one sampled turn.

        ``thinking`` is the mode the turn was rendered with, when the caller knows
        it (a ``prepare_conversation`` request that toggled it). It only decides
        the ambiguous case of a turn truncated before any channel boundary; a
        complete turn is recognized from its own structure.
        """
        thinking_mode = self._thinking if thinking is None else bool(thinking)
        tokens = [int(token) for token in response]
        stop_count = _count_subsequence(tokens, self.stop_ids)
        stop_at = _find_subsequence(tokens, self.stop_ids)
        if stop_at >= 0:
            body = tokens[:stop_at]
            malformed = stop_count != 1
        else:
            # Fireworks completions strip a matched stop string but keep its
            # leading token ids, so a sampled turn can end at the ``<|close|>`` of
            # ``<|close|>message<|sep|>``. That trailing proper prefix is the stop
            # boundary, never content; with no prefix at all the turn never stopped.
            prefix = self._trailing_stop_prefix(tokens)
            body = tokens[: len(tokens) - prefix]
            malformed = prefix == 0

        channel_at = _find_subsequence(body, self._think_to_response)
        if channel_at >= 0:
            reasoning_tokens = body[:channel_at]
            rest = body[channel_at + len(self._think_to_response) :]
        elif _find_subsequence(body, self._close_response) >= 0:
            # A closed response channel with no think boundary is a turn rendered
            # without thinking, whatever this renderer's default mode is.
            reasoning_tokens = []
            rest = body
        elif thinking_mode:
            # The prompt opened ``think``; without its closing boundary nothing
            # after it is agent-visible content.
            reasoning_tokens = body
            rest = []
            malformed = True
        else:
            reasoning_tokens = []
            rest = body

        response_end = _find_subsequence(rest, self._close_response)
        if response_end < 0:
            # The response channel never closed: a truncated or malformed turn.
            content_tokens = rest
            tail: list[int] = []
            malformed = True
        else:
            content_tokens = rest[:response_end]
            tail = rest[response_end + len(self._close_response) :]

        tools: list[ToolCall] = []
        if tail:
            tools_end = _find_subsequence(tail, self._close_tools, len(self._open_tools))
            if tail[: len(self._open_tools)] != self._open_tools or tools_end < 0:
                malformed = True
            else:
                try:
                    tools = self._parse_calls(tail[len(self._open_tools) : tools_end])
                except _MalformedToolBlock:
                    tools = []
                    malformed = True
                if tail[tools_end + len(self._close_tools) :]:
                    # Content after the tools block is outside the turn grammar; the
                    # calls are not trusted either.
                    tools = []
                    malformed = True

        reasoning, reasoning_clean = self._text(reasoning_tokens)
        content, content_clean = self._text(content_tokens)
        malformed = malformed or not (reasoning_clean and content_clean)

        parts: list[dict[str, Any]] = []
        if reasoning:
            parts.append({"type": "thinking", "thinking": reasoning})
        if content or not parts:
            parts.append({"type": "text", "text": content})
        message: Message = {"role": "assistant", "content": parts}
        if tools:
            message["tool_calls"] = tools
        termination = ParseTermination.MALFORMED if malformed else ParseTermination.STOP_SEQUENCE
        return _ParsedResponse(message=message, termination=termination, reasoning=reasoning, content=content)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class KimiK31Renderer(KimiK3Renderer):
    """Kimi K3 renderer adapted to the K3.1 encoder revision."""

    _options_cls = KimiK31RenderOptions
    _conversation_cls = KimiK31Conversation
    _normalize_tools = staticmethod(_normalize_tools)
    _native_message = staticmethod(_native_message)
    # The encoder renders non-string tool arguments from their JSON literal bytes,
    # so dataset normalization must hand argument strings over unchanged.
    preserves_tool_argument_literals = True

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_processor: ImageProcessor | None = None,
        *,
        options: KimiK31RenderOptions | None = None,
    ) -> None:
        if not callable(getattr(tokenizer, "_encode_text_piece", None)) or not hasattr(
            type(tokenizer), "apply_chat_template"
        ):
            raise TypeError(
                "Kimi K3.1 needs the checkpoint's TikTokenTokenizer (load the K3.1 checkpoint "
                "tokenizer with remote code, as for kimi_k3); the fast-tokenizer bundle from "
                "training.utils.kimi_k31_tokenizer only serves TITO"
            )
        super().__init__(tokenizer, image_processor=image_processor, options=options or KimiK31RenderOptions())
        encoder = __import__(self._build_chat_segments.__module__, fromlist=["*"])
        # Both encoders now share the same effort tiers; only K3.1's renders
        # tool arguments from their JSON literal bytes.
        if not callable(getattr(encoder, "_parse_arguments_object", None)):
            raise TypeError(
                "This tokenizer ships the Kimi K3 encoder (no literal tool-argument rendering); "
                "use the kimi_k3 renderer for it, or load the Kimi K3.1 checkpoint tokenizer"
            )
        self._parser = _ResponseParser(tokenizer, thinking=self.default_options.thinking)

    @staticmethod
    def thinking_mode_of(conversation: Any) -> bool | None:
        """The thinking mode a ``prepare_conversation`` envelope was rendered with, if any."""
        options = getattr(conversation, "options", None)
        return None if options is None else bool(options.thinking)

    def parse_response(
        self,
        response: list[int],
        *,
        thinking: bool | None = None,
    ) -> tuple[Message, ParseTermination]:
        """Parse a sampled turn.

        Pass ``thinking`` (for example ``renderer.thinking_mode_of(conversation)``)
        when the request toggled thinking away from this renderer's default: a
        complete turn is recognized from its structure either way, but a turn
        truncated before any channel boundary is reasoning in thinking mode and
        content otherwise.
        """
        parsed = self._parser.parse(response, thinking=thinking)
        return parsed.message, parsed.termination

    def parse_response_streaming(self, response: list[int], *, thinking: bool | None = None) -> Iterator[Any]:
        parsed = self._parser.parse(response, thinking=thinking)
        yield StreamingMessageHeader(role="assistant")
        content_index = 0
        if parsed.reasoning:
            yield StreamingThinkingDelta(thinking=parsed.reasoning, content_index=content_index)
            content_index += 1
        if parsed.content:
            yield StreamingTextDelta(text=parsed.content, content_index=content_index)
        yield parsed.message


def _factory(options: KimiK31RenderOptions):
    def build(tokenizer: Tokenizer, image_processor: ImageProcessor | None = None) -> KimiK31Renderer:
        return KimiK31Renderer(tokenizer, image_processor=image_processor, options=options)

    return build


register_renderer("kimi_k31", _factory(KimiK31RenderOptions()))
register_renderer(
    "kimi_k31_disable_thinking",
    _factory(KimiK31RenderOptions(thinking=False, thinking_effort=None)),
)
for _effort in THINKING_EFFORTS:
    register_renderer(f"kimi_k31_{_effort}", _factory(KimiK31RenderOptions(thinking=True, thinking_effort=_effort)))

# One image-processor route for every way a K3.1 renderer is constructed: the
# supervised builders consult this loader first, and so does pickle restore.
for _name in ("kimi_k31", "kimi_k31_disable_thinking", *(f"kimi_k31_{effort}" for effort in THINKING_EFFORTS)):
    register_renderer_image_capability(
        _name,
        load_image_token_counter,
        supports_images=True,
        supports_tool_images=True,
    )
