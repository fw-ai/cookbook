"""Pinned Kimi K3.1 full-history TITO model-format primitive.

Contract (verified against the checkpoint tokenizer's own ``apply_chat_template``,
which renders with ``encoding_k3.py``):

* Prompts are XTML. ``build_chat_segments`` from the vendored, hash-pinned
  ``encoding_k3.py`` (``_vendored/kimi_k31``) produces segments; structural markers (``<|open|>``,
  ``<|close|>``, ``<|sep|>``, ``<|end_of_msg|>``) encode as special tokens and
  every user/tool/attribute text segment encodes as ordinary text, so a literal
  ``<|open|>`` inside game text never becomes a control token.
* The generation prompt ends ``<|open|>message role="assistant"<|sep|><|open|>think<|sep|>``.
  A completion continues the reasoning, then emits
  ``<|close|>think<|sep|><|open|>response<|sep|>…<|close|>response<|sep|>``,
  an optional ``<|open|>tools<|sep|>…<|close|>tools<|sep|>`` block of calls
  with typed ``argument`` blocks, and stops at ``<|close|>message<|sep|>``. A raw
  ``json`` argument block (the encoder's form for non-object arguments) is a
  parse error.
  History appends ``<|end_of_msg|>`` after that stop, so a stored checkpoint is
  always a prefix of the next full-history render when the parser reproduces the
  sampled bytes.
* Historical reasoning is always rendered (the encoder preserves thinking).
* ``thinking_effort`` is fixed per certification. The default ``kimi_k31``
  renders no effort message, matching Fireworks serving when a request omits
  ``reasoning_effort``; ``kimi_k31_<effort>`` pins one tier.

Exact-token decisions the generic helpers would get wrong for XTML:

* Messages come from the request's wire value, not the canonical copy, because
  canonicalization re-serializes tool arguments with sorted keys while XTML
  renders arguments in their original order.
* Parsed tool arguments are rebuilt as a JSON object string whose non-string
  values keep the sampled literal bytes (``1e2`` stays ``1e2``, ``{"a": 1}`` keeps
  its spacing). ``encoding_k3.normalize_tool_arguments`` re-renders exactly those
  bytes, so the round trip is token-identical.

Certified for full-history rendering only, text-only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from fireworks.training.sdk import (
    TITOChatRequest,
    TITOError,
    TITOIncrementalPrompt,
    TITOParsedAssistant,
)

from training.renderer.tito._errors import renderer_parse_error
from training.renderer.tito._vendored.kimi_k31 import (
    encoding_k3 as _vendored_encoder,
)
from training.renderer.tito.shared import (
    TITORendererCertification,
    _ensure_tool_call_ids,
    _plain,
    validate_no_dynamic_template_fields,
)

KIMI_K31_RENDERER_NAME = "kimi_k31"
THINKING_EFFORTS = ("low", "high", "max")

KIMI_K31_REPOSITORY = "moonshotai/Kimi-K3.1-Pro"
KIMI_K31_REVISION = "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e"
# Upstream path and digest of the encoder vendored under ``_vendored/kimi_k31``;
# the digest is asserted against the vendored file by unit tests.
KIMI_K31_ENCODER_FILE = "encoding_k3.py"
KIMI_K31_ENCODER_SHA256 = "367f0a7ab8800f78f7e09dc31e58544ee7fc46eea992ea05b32d53506a921cc5"
# The checkpoint ships tiktoken files, not ``tokenizer.json``. This is the
# fingerprint of the fast tokenizer written by
#   python -m training.utils.kimi_k31_tokenizer <out-dir>
KIMI_K31_TOKENIZER_FINGERPRINT = "d9fba60895365d367afbee1f75616fbdfc98fb23e5ec6ba24a83a78bac745f26"

_CERTIFICATION_PREFIX = f"kimi-k3.1-pro@{KIMI_K31_REVISION[:8]}-full-history"
_CERTIFICATION_REVISION = "v1"

_OPEN = "<|open|>"
_CLOSE = "<|close|>"
_SEP = "<|sep|>"
_END_OF_MSG = "<|end_of_msg|>"
_STOP = f"{_CLOSE}message{_SEP}"

# ``TikTokenTokenizer._encode_text_piece`` chunking, reproduced so long inputs
# split at the same boundaries as the checkpoint tokenizer.
_TIKTOKEN_MAX_ENCODE_CHARS = 400_000
_MAX_NO_WHITESPACES_CHARS = 25_000

_HEADER_RE = re.compile(r'^(?P<tag>[A-Za-z_]+)(?P<attrs>(?: [A-Za-z_]+="[^"]*")*)$')
_ATTR_RE = re.compile(r' (?P<key>[A-Za-z_]+)="(?P<value>[^"]*)"')
_MEDIA_PART_TYPES = frozenset(
    {
        "image",
        "image_url",
        "input_image",
        "video",
        "video_url",
        "input_video",
        "audio",
        "audio_url",
        "input_audio",
    }
)


def _split_whitespaces_or_nonwhitespaces(text: str, limit: int) -> Iterator[str]:
    """Same slicing as ``TikTokenTokenizer._split_whitespaces_or_nonwhitespaces``."""
    current_len = 0
    current_is_space = text[0].isspace() if text else False
    start = 0
    for index, char in enumerate(text):
        is_space = char.isspace()
        if current_is_space ^ is_space:
            current_len = 1
            current_is_space = is_space
        else:
            current_len += 1
            if current_len > limit:
                yield text[start:index]
                start = index
                current_len = 1
    yield text[start:]


def _xtml_type(value: Any) -> str:
    """Same mapping as ``encoding_k3._xtml_type``."""
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    return "array"


def _unescape_attr(value: str) -> str:
    """Exact inverse of ``encoding_k3._escape_attr_value`` (``&`` first, then ``"``)."""
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _json_block_reason(body: str) -> str:
    """Parse-error code for a raw ``json`` argument block, matching DeepSeek V4's codes."""
    try:
        value = json.loads(body)
    except ValueError:
        return "tool_arguments_invalid_json"
    return "tool_unparsed_content" if isinstance(value, dict) else "tool_arguments_not_object"


def _drop_none(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not None}


def _normalize_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Match how Fireworks serving passes request tools to the encoder.

    Each tool is dumped without ``None`` values and without ``function.strict``
    / ``container``; the encoder then deep-sorts keys, so only the presence of
    keys (not their order) reaches the prompt.
    """
    normalized: list[dict[str, Any]] = []
    for raw_tool in tools:
        tool = _drop_none(_plain(raw_tool))
        tool.pop("container", None)
        function = tool.get("function")
        if isinstance(function, dict):
            function = _drop_none(function)
            function.pop("strict", None)
            tool["function"] = function
        normalized.append(tool)
    return normalized


class KimiK31TITORenderer:
    """Pinned Kimi K3.1 full-history TITO model-format primitive."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        certification: TITORendererCertification,
        thinking_effort: str | None = None,
    ) -> None:
        if thinking_effort is not None and thinking_effort not in THINKING_EFFORTS:
            raise ValueError(
                "Kimi K3.1 thinking_effort must be None or one of "
                + ", ".join(THINKING_EFFORTS)
                + f"; got {thinking_effort!r}"
            )
        self.tokenizer = tokenizer
        self.renderer_id = KIMI_K31_RENDERER_NAME
        self.certification_id = certification.certification_id
        self.tokenizer_fingerprint = certification.tokenizer_fingerprint
        self.thinking_effort = thinking_effort
        self._encoder = _vendored_encoder
        self._special_backend, self._text_backend = self._backends(tokenizer)
        self._open_id = self._single_control(_OPEN)
        self._close_id = self._single_control(_CLOSE)
        self._sep_id = self._single_control(_SEP)
        self._end_of_msg_id = self._single_control(_END_OF_MSG)
        self._control_ids = frozenset({self._open_id, self._close_id, self._sep_id, self._end_of_msg_id})
        self._stop_ids = self._encode_piece(_STOP, allow_special=True)
        self._think_to_response = self._encode_piece(
            f"{_CLOSE}think{_SEP}{_OPEN}response{_SEP}",
            allow_special=True,
        )
        self._close_response = self._encode_piece(f"{_CLOSE}response{_SEP}", allow_special=True)
        self._open_tools = self._encode_piece(f"{_OPEN}tools{_SEP}", allow_special=True)
        self._close_tools = self._encode_piece(f"{_CLOSE}tools{_SEP}", allow_special=True)

    # ------------------------------------------------------------------
    # Tokenization: the checkpoint tokenizer's segment contract
    # ------------------------------------------------------------------

    @staticmethod
    def _backends(tokenizer: Any) -> tuple[Any, Any]:
        """Two private backend copies: one matches specials, one encodes them as text.

        Separate instances keep ``encode_special_tokens`` from being toggled on a
        tokenizer that other threads may share.
        """
        from tokenizers import Tokenizer

        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None or not hasattr(backend, "to_str"):
            raise ValueError("Kimi K3.1 TITO requires the certified fast tokenizer bundle")
        serialized = backend.to_str()
        special = Tokenizer.from_str(serialized)
        special.encode_special_tokens = False
        text = Tokenizer.from_str(serialized)
        text.encode_special_tokens = True
        return special, text

    def _encode_piece(self, text: str, *, allow_special: bool) -> tuple[int, ...]:
        """``TikTokenTokenizer._encode_text_piece`` on the fast backend."""
        backend = self._special_backend if allow_special else self._text_backend
        tokens: list[int] = []
        for offset in range(0, len(text), _TIKTOKEN_MAX_ENCODE_CHARS):
            for piece in _split_whitespaces_or_nonwhitespaces(
                text[offset : offset + _TIKTOKEN_MAX_ENCODE_CHARS],
                _MAX_NO_WHITESPACES_CHARS,
            ):
                if piece:
                    tokens.extend(backend.encode(piece, add_special_tokens=False).ids)
        return tuple(int(token) for token in tokens)

    def _single_control(self, text: str) -> int:
        tokens = self._encode_piece(text, allow_special=True)
        if len(tokens) != 1:
            raise ValueError(f"Kimi K3.1 expected {text!r} to encode as one token")
        return tokens[0]

    def _decode(self, tokens: Sequence[int]) -> str:
        return str(self._special_backend.decode([int(token) for token in tokens], skip_special_tokens=False))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _validate_request(self, request: TITOChatRequest) -> None:
        validate_no_dynamic_template_fields(request)

    def _template_message(self, raw_message: Mapping[str, Any]) -> dict[str, Any]:
        message = _plain(raw_message)
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise TITOError("tito_invalid_request", 400, f"unsupported Kimi K3.1 role {role!r}")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            if not isinstance(content, list):
                raise TITOError("tito_invalid_request", 400, "Kimi K3.1 content must be a string or a part list")
            for part in content:
                part_type = str(part.get("type", "")) if isinstance(part, dict) else ""
                if part_type in _MEDIA_PART_TYPES:
                    raise TITOError(
                        "tito_invalid_request",
                        400,
                        "Kimi K3.1 TITO rollouts are text-only; media needs an "
                        "out-of-band payload that an exact-token rollout does not carry",
                    )
                if not isinstance(part, dict) or part_type != "text" or not isinstance(part.get("text"), str):
                    raise TITOError("tito_invalid_request", 400, "Kimi K3.1 TITO supports text content parts only")
        if role == "system" and message.get("tools"):
            message["tools"] = _normalize_tools(message["tools"])
        return message

    def _template_inputs(
        self,
        request: TITOChatRequest,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        # The wire value keeps the harness's argument strings byte-for-byte;
        # the canonical copy re-sorts their keys, which XTML would render.
        wire = request.wire_value()
        if wire is not None:
            messages = wire.get("messages") or ()
            tools = wire.get("tools") or ()
        else:
            messages = request.messages
            tools = request.tools
        return [self._template_message(message) for message in messages], _normalize_tools(tools)

    def render_segments(self, request: TITOChatRequest) -> list[Any]:
        """The checkpoint encoder's XTML segments for ``request`` (for audits)."""
        self._validate_request(request)
        messages, tools = self._template_inputs(request)
        try:
            return self._encoder.build_chat_segments(
                messages,
                tools=tools or None,
                add_generation_prompt=True,
                thinking=True,
                thinking_effort=self.thinking_effort,
            )
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            raise TITOError("tito_invalid_request", 400, f"Kimi K3.1 XTML render failed: {exc}") from exc

    def render_conversation_tokens(self, request: TITOChatRequest) -> Sequence[int]:
        tokens: list[int] = []
        for segment in self.render_segments(request):
            tokens.extend(self._encode_piece(segment.text, allow_special=segment.allow_special))
        return tuple(tokens)

    def prepare_incremental_prompt(
        self,
        request: TITOChatRequest,
        stored_messages: Sequence[Mapping[str, Any]],
        appended_messages: Sequence[Mapping[str, Any]],
        exact_checkpoint_ids: Sequence[int],
    ) -> TITOIncrementalPrompt | None:
        """Kimi K3.1 is certified for full-history rendering only."""
        del request, stored_messages, appended_messages, exact_checkpoint_ids
        return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _find(haystack: Sequence[int], needle: Sequence[int], start: int = 0) -> int:
        width = len(needle)
        first = needle[0]
        for index in range(start, len(haystack) - width + 1):
            if haystack[index] == first and tuple(haystack[index : index + width]) == tuple(needle):
                return index
        return -1

    def _trailing_stop_prefix(self, tokens: Sequence[int]) -> int:
        """Length of the longest proper stop-sequence prefix that ends ``tokens``."""
        for width in range(min(len(self._stop_ids) - 1, len(tokens)), 0, -1):
            if tuple(tokens[-width:]) == self._stop_ids[:width]:
                return width
        return 0

    def _text(self, tokens: Sequence[int], *, where: str) -> str:
        """Decode a text channel; a control token inside it cannot round-trip."""
        if any(token in self._control_ids for token in tokens):
            raise renderer_parse_error(
                f"Kimi K3.1 {where} contains an XTML control token",
                reason_code="control_token_in_text",
            )
        return self._decode(tokens)

    def _read_open(self, tokens: Sequence[int], index: int) -> tuple[str, dict[str, str], int]:
        if index >= len(tokens):
            raise renderer_parse_error("Kimi K3.1 tool block ends mid-call", reason_code="tool_unparsed_boundary")
        if tokens[index] != self._open_id:
            raise renderer_parse_error(
                "Kimi K3.1 tool block has content outside a tag",
                reason_code="tool_unparsed_content",
            )
        end = index + 1
        while end < len(tokens) and tokens[end] != self._sep_id:
            if tokens[end] in self._control_ids:
                raise renderer_parse_error(
                    "Kimi K3.1 tool tag header is not closed",
                    reason_code="tool_unparsed_content",
                )
            end += 1
        if end >= len(tokens):
            raise renderer_parse_error("Kimi K3.1 tool block ends mid-tag", reason_code="tool_unparsed_boundary")
        header = self._decode(tokens[index + 1 : end])
        match = _HEADER_RE.match(header)
        if match is None:
            raise renderer_parse_error(
                f"malformed Kimi K3.1 tool tag header: {header!r}",
                reason_code="tool_unparsed_content",
            )
        attrs = {item.group("key"): _unescape_attr(item.group("value")) for item in _ATTR_RE.finditer(match["attrs"])}
        return match["tag"], attrs, end + 1

    def _read_close(self, tokens: Sequence[int], index: int, tag: str) -> int:
        if index >= len(tokens):
            raise renderer_parse_error("Kimi K3.1 tool block ends mid-call", reason_code="tool_unparsed_boundary")
        if tokens[index] != self._close_id:
            raise renderer_parse_error(
                f"Kimi K3.1 {tag} tag is not closed",
                reason_code="tool_unparsed_content",
            )
        end = index + 1
        while end < len(tokens) and tokens[end] != self._sep_id:
            end += 1
        if end >= len(tokens):
            raise renderer_parse_error("Kimi K3.1 tool block ends mid-tag", reason_code="tool_unparsed_boundary")
        closed = self._decode(tokens[index + 1 : end])
        if closed != tag:
            raise renderer_parse_error(
                f"Kimi K3.1 closes {closed!r} while {tag!r} is open",
                reason_code="tool_unparsed_content",
            )
        return end + 1

    def _read_text(self, tokens: Sequence[int], index: int) -> tuple[str, int]:
        end = index
        while end < len(tokens) and tokens[end] not in self._control_ids:
            end += 1
        return self._decode(tokens[index:end]), end

    @staticmethod
    def _argument_literal(text: str, type_name: str | None) -> str:
        """JSON bytes that ``encoding_k3`` parses back to exactly (type, text)."""
        if type_name in (None, "string"):
            return json.dumps(text, ensure_ascii=False)
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            raise renderer_parse_error(
                f"Kimi K3.1 {type_name} argument is not a JSON literal: {text[:80]!r}",
                reason_code="tool_invalid_argument",
            ) from exc
        return text

    def _parse_calls(self, tokens: Sequence[int], request: TITOChatRequest) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        index = 0
        while index < len(tokens):
            tag, attrs, index = self._read_open(tokens, index)
            if tag != "call":
                raise renderer_parse_error(
                    f"Kimi K3.1 tools block holds a {tag!r} tag",
                    reason_code="tool_unparsed_content",
                )
            name = attrs.get("tool")
            if not name:
                raise renderer_parse_error(
                    "Kimi K3.1 tool call is missing a tool name",
                    reason_code="tool_missing_name",
                )
            fields: list[str] = []
            while True:
                if index < len(tokens) and tokens[index] == self._close_id:
                    index = self._read_close(tokens, index, "call")
                    break
                inner, inner_attrs, index = self._read_open(tokens, index)
                body, index = self._read_text(tokens, index)
                index = self._read_close(tokens, index, inner)
                if inner == "argument":
                    key = inner_attrs.get("key")
                    if key is None:
                        raise renderer_parse_error(
                            "Kimi K3.1 tool argument is missing its key",
                            reason_code="tool_unparsed_content",
                        )
                    literal = self._argument_literal(body, inner_attrs.get("type"))
                    fields.append(f"{json.dumps(key, ensure_ascii=False)}:{literal}")
                elif inner == "json":
                    # The encoder writes a raw ``json`` block only for arguments
                    # that are not a JSON object, which no harness can execute as
                    # a function call, and it re-renders an object as ``argument``
                    # blocks, so no ``json`` block survives the round trip.
                    raise renderer_parse_error(
                        "Kimi K3.1 tool call carries raw JSON arguments",
                        reason_code=_json_block_reason(body),
                    )
                else:
                    raise renderer_parse_error(
                        f"Kimi K3.1 tool call holds an unexpected {inner!r} tag",
                        reason_code="tool_unparsed_content",
                    )
            arguments = "{" + ",".join(fields) + "}"
            calls.append({"type": "function", "function": {"name": name, "arguments": arguments}})
        return calls

    def _allowed_tool_names(self, request: TITOChatRequest) -> set[str]:
        names = {str((tool.get("function") or {}).get("name")) for tool in request.tools}
        for message in request.messages:
            if message.get("role") == "system":
                names.update(str((tool.get("function") or {}).get("name")) for tool in message.get("tools") or ())
        names.discard("None")
        return names

    def _parse_message(
        self,
        request: TITOChatRequest,
        completion_ids: Sequence[int],
        finish_reason: str,
    ) -> dict[str, Any]:
        tokens = [int(token) for token in completion_ids]
        stop_at = self._find(tokens, self._stop_ids)
        if stop_at >= 0:
            body = tokens[:stop_at]
        else:
            # Fireworks completions strip a matched stop string from the text but
            # keep the stop's leading token ids (``completion_token_ids`` end at
            # the ``<|close|>`` of ``<|close|>message<|sep|>``). A trailing
            # proper prefix of the stop is that boundary, never content; the
            # stored checkpoint still prefixes the next render, which continues
            # with the rest of the stop and ``<|end_of_msg|>``.
            body = tokens[: len(tokens) - self._trailing_stop_prefix(tokens)]
        # A clean stop is the (possibly partial) stop sequence or a non-length
        # finish with a complete structure.
        truncated = stop_at < 0 and finish_reason == "length"

        split = self._find(body, self._think_to_response)
        if split < 0:
            if truncated:
                # The prompt opened ``think``; a length stop before it closes is
                # incomplete reasoning, never agent-visible content. A stop inside
                # the closing tag leaves control tokens that text cannot reproduce.
                return {"role": "assistant", "content": "", "reasoning_content": self._text(body, where="reasoning")}
            raise renderer_parse_error(
                "Kimi K3.1 reasoning channel has no closing boundary",
                reason_code="reasoning_missing_close",
            )
        reasoning = self._text(body[:split], where="reasoning")
        rest = body[split + len(self._think_to_response) :]

        response_end = self._find(rest, self._close_response)
        if response_end < 0:
            if truncated:
                content = self._text(rest, where="response")
                return {"role": "assistant", "content": content, "reasoning_content": reasoning}
            raise renderer_parse_error(
                "Kimi K3.1 response channel has no closing boundary",
                reason_code="response_missing_close",
            )
        content = self._text(rest[:response_end], where="response")
        tail = rest[response_end + len(self._close_response) :]

        calls: list[dict[str, Any]] = []
        if tail:
            if tuple(tail[: len(self._open_tools)]) != self._open_tools:
                raise renderer_parse_error(
                    "Kimi K3.1 completion has content after the response channel",
                    reason_code="tool_unparsed_content",
                )
            tools_end = self._find(tail, self._close_tools, len(self._open_tools))
            if tools_end < 0:
                raise renderer_parse_error(
                    "unparsed Kimi K3.1 tool-call boundary",
                    reason_code="tool_unparsed_boundary",
                )
            calls = self._parse_calls(tail[len(self._open_tools) : tools_end], request)
            if tail[tools_end + len(self._close_tools) :]:
                raise renderer_parse_error(
                    "Kimi K3.1 completion has content after the tools block",
                    reason_code="tool_unparsed_content",
                )
            allowed = self._allowed_tool_names(request)
            if any(call["function"]["name"] not in allowed for call in calls):
                raise renderer_parse_error(
                    "Kimi K3.1 tool call names are absent from the request",
                    reason_code="tool_unknown_name",
                )

        # A closed think block always yields ``reasoning_content``, even when the
        # model wrote no reasoning: harnesses echo the field back as ``""``, and
        # the engine compares messages before tokens, so omitting it would split
        # every turn after an empty think into a ``history_rewrite`` segment.
        # Empty, missing and null reasoning render the same tokens.
        message: dict[str, Any] = {"role": "assistant", "content": content, "reasoning_content": reasoning}
        if calls:
            message["tool_calls"] = calls
        return message

    def parse_assistant(
        self,
        request: TITOChatRequest,
        completion_ids: Sequence[int],
        completion_text: str,
        finish_reason: str,
    ) -> TITOParsedAssistant:
        del completion_text
        message = _ensure_tool_call_ids(
            self._parse_message(request, completion_ids, finish_reason),
            completion_ids,
        )
        output_kind = (
            "tool_calls"
            if message.get("tool_calls")
            else "reasoning" if message.get("reasoning_content") is not None else "text"
        )
        return TITOParsedAssistant(message=message, output_kind=output_kind)

    def render_contract_id(self, request: TITOChatRequest) -> str:
        contract = {
            "renderer_id": self.renderer_id,
            "certification_id": self.certification_id,
            "model": request.model,
            "tools": [dict(tool) for tool in request.tools],
            "renderer_class": type(self).__qualname__,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "thinking_effort": self.thinking_effort,
        }
        return hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()

    def stop_sequences(self, request: TITOChatRequest) -> Sequence[str]:
        del request
        return (_STOP,)


def _build_renderer(thinking_effort: str | None):
    def factory(tokenizer: Any, certification: TITORendererCertification) -> KimiK31TITORenderer:
        return KimiK31TITORenderer(tokenizer, certification=certification, thinking_effort=thinking_effort)

    return factory


def kimi_k31_certifications() -> tuple[TITORendererCertification, ...]:
    """One certification per pinned thinking-effort prompt contract."""
    variants: list[tuple[str, str | None, str]] = [(KIMI_K31_RENDERER_NAME, None, "default")]
    variants.extend((f"{KIMI_K31_RENDERER_NAME}_{effort}", effort, effort) for effort in THINKING_EFFORTS)
    return tuple(
        TITORendererCertification(
            certification_id=f"{_CERTIFICATION_PREFIX}-{label}-{_CERTIFICATION_REVISION}",
            renderer_names=frozenset({name}),
            tokenizer_fingerprint=KIMI_K31_TOKENIZER_FINGERPRINT,
            renderer_factory=_build_renderer(effort),
            # The encoder ships as hash-pinned vendored package source
            # (tito/_vendored/kimi_k31); no checkpoint artifact is fetched.
            packaged_contract=True,
        )
        for name, effort, label in variants
    )


__all__ = [
    "KIMI_K31_ENCODER_FILE",
    "KIMI_K31_ENCODER_SHA256",
    "KIMI_K31_RENDERER_NAME",
    "KIMI_K31_REPOSITORY",
    "KIMI_K31_REVISION",
    "KIMI_K31_TOKENIZER_FINGERPRINT",
    "KimiK31TITORenderer",
    "THINKING_EFFORTS",
    "kimi_k31_certifications",
]
