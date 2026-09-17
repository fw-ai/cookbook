"""Pinned Qwen3.8-27B preserved-thinking TITO model-format primitive."""

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

QWEN38_RENDERER_NAME = "qwen3_8"
_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(.*?)>\s*(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_QWEN_TOOL_ARG_RE = re.compile(
    r"<parameter=(.*?)>\n?(.*?)\n?</parameter>",
    re.DOTALL,
)


class Qwen38TITORenderer:
    """Pinned Qwen3.8-27B preserved-thinking TITO model-format primitive.

    Contract (verified against the live qwen3p8 serving style and the pinned
    tokenizer's chat template): the generation prompt opens ``<think>\\n``;
    completions continue reasoning, close with ``</think>``, then emit content
    plus optional ``<tool_call><function=name><parameter=key>value...`` blocks,
    terminating at ``<|im_end|>``. Full-history rendering delegates to the
    tokenizer's authoritative chat template with its pinned defaults
    (``enable_thinking``/``preserve_thinking`` true, ``reasoning_effort``
    xhigh); the certification fingerprint freezes that template text.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        certification: TITORendererCertification,
    ) -> None:
        self.tokenizer = tokenizer
        self.renderer_id = QWEN38_RENDERER_NAME
        self.certification_id = certification.certification_id
        self.tokenizer_fingerprint = certification.tokenizer_fingerprint
        self._im_end_token = self._single_token("<|im_end|>")
        self._endoftext_token = self._single_token("<|endoftext|>")
        self._stop = (self._im_end_token, self._endoftext_token)

    def _single_token(self, text: str) -> int:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Qwen3.8 expected {text!r} to encode as one token")
        return int(tokens[0])

    def _encode(self, text: str) -> list[int]:
        return [
            int(token)
            for token in self.tokenizer.encode(text, add_special_tokens=False)
        ]

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
        # Pinned defaults only: enable_thinking/preserve_thinking true,
        # reasoning_effort xhigh. The certified tokenizer fingerprint freezes
        # the template text, so these defaults cannot drift.
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
        """Qwen3.8 is certified for full-history rendering only."""
        del request, stored_messages, appended_messages, exact_checkpoint_ids
        return None

    def _parse_message(
        self,
        request: TITOChatRequest,
        completion_ids: Sequence[int],
        finish_reason: str,
    ) -> tuple[dict[str, Any], bool]:
        response = [int(token) for token in completion_ids]
        think_prefix = self._encode("<think>")
        # The certified generation suffix opened ``<think>\n`` before sampling.
        # Restore that prompt-owned boundary for every completion, including
        # malformed stops that arrive before ``</think>``.
        if response[: len(think_prefix)] != think_prefix:
            response = [*think_prefix, *response]
        end = len(response)
        for stop in self._stop:
            try:
                end = min(end, response.index(stop))
            except ValueError:
                pass
        clean = end < len(response)
        content = str(self.tokenizer.decode(response[:end])).lstrip("\n")
        reasoning: str | None = None
        if "</think>" in content:
            if not content.startswith("<think>"):
                raise ValueError("Qwen3.8 reasoning close has no opening boundary")
            reasoning, content = content[len("<think>") :].split("</think>", 1)
            content = content.lstrip("\n")
        elif finish_reason == "length":
            # The certified generation prompt ends inside ``<think>``. A length
            # stop before ``</think>`` is incomplete reasoning, never
            # agent-visible content; keep it structured and do not interpret
            # tool markup inside it.
            reasoning = (
                content[len("<think>") :] if content.startswith("<think>") else content
            )
            content = ""
        elif content.startswith("<think>"):
            # A clean stop before the prompt-opened reasoning boundary closes
            # is malformed structured output, not visible assistant text.
            raise ValueError("Qwen3.8 reasoning open has no closing boundary")

        # Tool markup is protocol only after the reasoning boundary. Malformed
        # model output is classified by the engine; the renderer must not
        # manufacture a different tool call in an attempt to repair it.
        calls: list[dict[str, Any]] = []
        cleaned: list[str] = []
        position = 0
        for match in _QWEN_TOOL_CALL_RE.finditer(content):
            cleaned.append(content[position : match.start()])
            name = match.group(1).strip()
            body = match.group(2)
            if not name:
                raise ValueError("Qwen3.8 tool call is missing a function name")
            arguments = {
                item.group(1).strip(): self._parse_argument(item.group(2))
                for item in _QWEN_TOOL_ARG_RE.finditer(body)
            }
            residue = _QWEN_TOOL_ARG_RE.sub("", body).strip()
            if residue:
                raise ValueError(f"unparsed Qwen3.8 tool-call content: {residue!r}")
            calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
            position = match.end()
        cleaned.append(content[position:])
        content = "".join(cleaned)
        if "<tool_call>" in content or "</tool_call>" in content:
            raise ValueError("unparsed Qwen3.8 tool-call boundary")

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
            raise ValueError("Qwen3.8 tool call names are absent from the request")

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
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
                [*completion_ids, self._im_end_token],
                finish_reason,
            )
        if not clean and finish_reason != "length":
            raise ValueError("unclean Qwen3.8 renderer parse")
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
