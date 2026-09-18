"""Pinned Muse Glimmer 30B preserved-reasoning TITO model-format primitive.

Contract (verified against the pinned tokenizer's authoritative ATEM chat
template at ``meta-models/Muse-Glimmer-30B`` revision ``a4e59da5``): the
generation prompt ends at ``<|start|>assistant``; a completion opens a
channel with `` to=<recipient><|message|>``, where ``self`` is the reasoning
channel, ``user`` is the visible channel, and a tool name introduces an
``<atem:function_calls>`` block. Channels continue the same sampled response
across ``<|eom|><|start|>assistant`` junctions and terminate at ``<|eot|>``.

Full-history rendering delegates to the tokenizer's authoritative chat
template with its pinned defaults (``reasoning_strength`` high,
``knowledge_cutoff`` 2026-01-04); the certification fingerprint freezes that
template text. Muse Glimmer is certified for full-history rendering only: the
upstream template flips a terminal ATEM call from ``<|eot|>`` to ``<|eom|>``
when another assistant message is appended, so stored histories are not
prefixes and incremental prompt construction is not claimed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from fireworks.training.sdk import (
    TITOChatRequest,
    TITOIncrementalPrompt,
    TITOParsedAssistant,
)

from training.renderer.tito.shared import (
    _ensure_tool_call_ids,
    _normalize_template_messages,
    _normalize_template_tools,
    validate_no_dynamic_template_fields,
    TITORendererCertification,
)

MUSE_GLIMMER_RENDERER_NAME = "muse_glimmer"

_EOT = "<|eot|>"
_EOM = "<|eom|>"
_START = "<|start|>"
_MESSAGE = "<|message|>"
_SEGMENT_SEPARATOR = _EOM + _START + "assistant"
_ATEM_CALL_RE = re.compile(
    r'<atem:function_calls>\n<atem:invoke name="(?P<name>[^"]*)">\n'
    r"(?P<body>.*?)</atem:invoke>\n</atem:function_calls>",
    re.DOTALL,
)
_ATEM_ARG_RE = re.compile(
    r'<atem:parameter name="(?P<name>[^"]*)">(?P<value>.*?)' r"</atem:parameter>\n",
    re.DOTALL,
)


class MuseGlimmerTITORenderer:
    """Pinned Muse Glimmer 30B preserved-reasoning TITO model-format primitive."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        certification: TITORendererCertification,
    ) -> None:
        self.tokenizer = tokenizer
        self.renderer_id = MUSE_GLIMMER_RENDERER_NAME
        self.certification_id = certification.certification_id
        self.tokenizer_fingerprint = certification.tokenizer_fingerprint
        self._eot_token = self._single_token(_EOT)
        self._stop = (self._eot_token,)

    def _single_token(self, text: str) -> int:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Muse Glimmer expected {text!r} to encode as one token")
        return int(tokens[0])

    def _validate_request(self, request: TITOChatRequest) -> None:
        validate_no_dynamic_template_fields(request)

    def _template_inputs(
        self,
        request: TITOChatRequest,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        wire = request.wire_value()
        if wire is not None:
            messages = wire.get("messages") or ()
            tools = wire.get("tools") or ()
        else:
            messages = request.messages
            tools = request.tools
        return _normalize_template_messages(messages), _normalize_template_tools(tools)

    def _render_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any:
        # Pinned defaults only: reasoning_strength high, knowledge_cutoff
        # 2026-01-04, no per-request template options. The certified tokenizer
        # fingerprint freezes the template text, so these defaults cannot drift.
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    def _render(self, request: TITOChatRequest) -> tuple[int, ...]:
        self._validate_request(request)
        messages, tools = self._template_inputs(request)
        rendered = self._render_template(
            messages,
            tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(rendered, Mapping):
            rendered = rendered["input_ids"]
        return tuple(int(token) for token in rendered)

    def render_conversation_tokens(self, request: TITOChatRequest) -> Sequence[int]:
        return self._render(request)

    def prepare_incremental_prompt(
        self,
        request: TITOChatRequest,
        stored_messages: Sequence[Mapping[str, Any]],
        appended_messages: Sequence[Mapping[str, Any]],
        exact_checkpoint_ids: Sequence[int],
    ) -> TITOIncrementalPrompt | None:
        """Muse Glimmer is certified for full-history rendering only.

        The upstream template mutates a terminal ATEM call from ``<|eot|>`` to
        ``<|eom|>`` when a consecutive assistant message is appended, so an
        exact stored checkpoint is never a prefix of the extended render.
        """
        del request, stored_messages, appended_messages, exact_checkpoint_ids
        return None

    def _parse_message(
        self,
        request: TITOChatRequest,
        completion_ids: Sequence[int],
        finish_reason: str,
    ) -> tuple[dict[str, Any], bool]:
        del finish_reason
        text = str(self.tokenizer.decode([int(token) for token in completion_ids]))
        stop_at = text.find(_EOT)
        clean = stop_at >= 0
        if clean:
            text = text[:stop_at]
        # Sampling begins right after the prompt-owned ``<|start|>assistant``.
        # Be liberal when a caller feeds a full assistant segment back in.
        text = text.removeprefix(_START + "assistant")

        reasoning_parts: list[str] = []
        visible_parts: list[str] = []
        calls: list[dict[str, Any]] = []
        for segment in text.split(_SEGMENT_SEPARATOR):
            if _MESSAGE not in segment:
                if segment.strip():
                    raise ValueError("Muse Glimmer segment has no message boundary")
                continue
            recipient, payload = segment.split(_MESSAGE, 1)
            if not recipient.startswith(" to="):
                raise ValueError("Muse Glimmer segment is missing its recipient")
            recipient = recipient[len(" to=") :]
            if recipient == "self":
                reasoning_parts.append(payload)
                continue
            matches = list(_ATEM_CALL_RE.finditer(payload))
            if matches:
                # Narrated text on a tool channel is dropped, mirroring the
                # upstream template's re-render of tool-call turns (registered
                # divergence for this model family). The ATEM blocks themselves
                # must parse exactly; the renderer must not repair malformed
                # markup into a different tool call.
                for match in matches:
                    name = match.group("name")
                    if not name:
                        raise ValueError("Muse Glimmer tool call is missing a name")
                    arguments = {
                        item.group("name"): self._parse_argument(item.group("value"))
                        for item in _ATEM_ARG_RE.finditer(match.group("body"))
                    }
                    residue = _ATEM_ARG_RE.sub("", match.group("body")).strip()
                    if residue:
                        raise ValueError(
                            f"unparsed Muse Glimmer tool-call content: {residue!r}"
                        )
                    calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    )
                continue
            if recipient == "user":
                visible_parts.append(payload)
                continue
            raise ValueError(f"unknown Muse Glimmer recipient {recipient!r}")

        allowed_tool_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in request.tools
            if (tool.get("function") or {}).get("name")
        }
        if calls and (
            not allowed_tool_names
            or any(
                str(call["function"]["name"]) not in allowed_tool_names
                for call in calls
            )
        ):
            raise ValueError("Muse Glimmer tool call names are absent from the request")

        content = "".join(visible_parts)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if calls:
            message["tool_calls"] = calls
        return message, clean

    @staticmethod
    def _parse_argument(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def parse_assistant(
        self,
        request: TITOChatRequest,
        completion_ids: Sequence[int],
        completion_text: str,
        finish_reason: str,
    ) -> TITOParsedAssistant:
        del completion_text
        message, clean = self._parse_message(request, completion_ids, finish_reason)
        if not clean and finish_reason != "length":
            message, clean = self._parse_message(
                request,
                [*completion_ids, self._eot_token],
                finish_reason,
            )
        if not clean and finish_reason != "length":
            raise ValueError("unclean Muse Glimmer renderer parse")
        message = _ensure_tool_call_ids(message, completion_ids)
        tool_calls = message.get("tool_calls") or []
        if finish_reason == "length" and tool_calls:
            raise ValueError("truncated structured tool call is not protocol-safe")
        output_kind = (
            "tool_calls"
            if tool_calls
            else "reasoning"
            if message.get("reasoning_content") is not None
            else "text"
        )
        return TITOParsedAssistant(message=message, output_kind=output_kind)

    def fallback_assistant_text(
        self,
        request: TITOChatRequest,
        completion_ids: Sequence[int],
        finish_reason: str,
        parser_error: BaseException,
    ) -> str | None:
        del request, finish_reason, parser_error
        del completion_ids
        return None

    def render_contract_id(self, request: TITOChatRequest) -> str:
        contract = {
            "renderer_id": self.renderer_id,
            "certification_id": self.certification_id,
            "model": request.model,
            "tools": [dict(tool) for tool in request.tools],
            "renderer_class": type(self).__qualname__,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
        }
        return hashlib.sha256(
            json.dumps(
                contract,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()

    def stop_sequences(self, request: TITOChatRequest) -> Sequence[str]:
        del request
        return tuple(str(self.tokenizer.decode([token])) for token in self._stop)
