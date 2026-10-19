"""Full-history DeepSeek V4 rendering with each checkpoint's pinned native encoder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from typing import Any

from fireworks.training.sdk import TITOParseError, TITOChatRequest, TITOError, TITOParsedAssistant

from training.renderer.tito.shared import (
    TITORendererCertification,
    _ensure_tool_call_ids,
    _normalize_template_tools,
    _plain,
    renderer_artifacts,
    validate_no_dynamic_template_fields,
)

DEEPSEEK_V4_RENDERER_NAME = "deepseek_v4_flash_0731"
DEEPSEEK_V4_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_V4_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DEEPSEEK_V4_PRO_RENDERER_NAME = "deepseek_v4_pro_0813"
DEEPSEEK_V4_PRO_REPOSITORY = "deepseek-ai/DeepSeek-V4-Pro-0813"
DEEPSEEK_V4_PRO_REVISION = "72e1d3230f6c080a530b0a1d46f8eb4602340597"
# Both certified releases share these tokenizer and native encoder hashes.
DEEPSEEK_V4_TOKENIZER_FINGERPRINT = "64a749582a69535715f77e8b90470719065cde2688a27376c944eeea408b0525"
DEEPSEEK_V4_ENCODER_FILE = "encoding/encoding_dsv4.py"
DEEPSEEK_V4_ENCODER_SHA256 = "abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6"


class DeepSeekV4TITORenderer:
    """Fixed thinking/low-effort, preserved-history text/tool contract."""

    encoder_file = DEEPSEEK_V4_ENCODER_FILE
    reasoning_effort = "low"

    def __init__(self, tokenizer: Any, *, certification: TITORendererCertification) -> None:
        path = renderer_artifacts(tokenizer, certification)[self.encoder_file]
        spec = importlib.util.spec_from_file_location("tito_pinned_dsv4_encoder", path)
        if spec is None or spec.loader is None:
            raise ValueError("could not load certified DeepSeek V4 encoder")
        encoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(encoder)
        self._encoder = encoder
        self.tokenizer = tokenizer
        self.renderer_id = next(iter(certification.renderer_names))
        self.certification_id = certification.certification_id
        self.tokenizer_fingerprint = certification.tokenizer_fingerprint
        self._eos = tuple(tokenizer.encode(encoder.eos_token, add_special_tokens=False))
        if len(self._eos) != 1:
            raise ValueError("DeepSeek V4 EOS must encode as one token")

    def _template_messages(self, request: TITOChatRequest) -> list[dict[str, Any]]:
        validate_no_dynamic_template_fields(request)
        if "reasoning_history" in request.sampling_fields:
            raise TITOError("tito_invalid_request", 400, "DeepSeek V4 reasoning history is fixed by certification")
        wire = request.wire_value()
        raw_messages = wire.get("messages", ()) if wire is not None else request.messages
        raw_tools = (wire.get("tools") or ()) if wire is not None else request.tools
        messages: list[dict[str, Any]] = []
        for message_index, raw in enumerate(raw_messages):
            if raw.get("role") not in {"system", "user", "assistant", "tool"}:
                raise TITOError("tito_invalid_request", 400, "unsupported DeepSeek V4 role")
            extra_fields = set(raw) - {"role", "content", "reasoning_content", "tool_call_id", "tool_calls", "name"}
            # Admission removes only this empty client response marker from
            # canonical history; render raw arguments without rejecting it.
            if (
                raw.get("role") == "assistant"
                and raw.get("provider_specific_fields") == {"refusal": None}
                and "provider_specific_fields" not in request.messages[message_index]
            ):
                extra_fields.discard("provider_specific_fields")
            if extra_fields:
                raise TITOError("tito_invalid_request", 400, "unsupported DeepSeek V4 message fields")
            content = raw.get("content")
            if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
                if any(
                    not isinstance(part, Mapping)
                    or part.get("type") != "text"
                    or not isinstance(part.get("text"), str)
                    for part in content
                ):
                    raise TITOError("tito_invalid_request", 400, "DeepSeek V4 TITO supports text only")
                content = "".join(part["text"] for part in content)
            if content is not None and not isinstance(content, str):
                raise TITOError("tito_invalid_request", 400, "invalid DeepSeek V4 content")
            message: dict[str, Any] = {"role": raw["role"], "content": content or ""}
            for field in ("reasoning_content", "tool_call_id", "tool_calls"):
                if raw.get(field):
                    message[field] = _plain(raw[field])
            for call in message.get("tool_calls") or ():
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if call.get("type") != "function" or not function.get("name"):
                    raise TITOError("tito_invalid_request", 400, "invalid DeepSeek V4 historical function call")
                if isinstance(arguments, Mapping):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                    function["arguments"] = arguments
                try:
                    if not isinstance(json.loads(arguments), dict):
                        raise ValueError("not an object")
                except (TypeError, ValueError) as exc:
                    raise TITOError(
                        "tito_invalid_request", 400, "DeepSeek V4 historical arguments must be JSON objects"
                    ) from exc
            messages.append(message)
        if messages[-1]["role"] not in {"user", "tool"}:
            raise TITOError(
                "tito_invalid_request", 400, "DeepSeek V4 requires a user message or tool result before sampling"
            )
        tools = _normalize_template_tools(raw_tools)
        for tool in tools:
            if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
                raise TITOError("tito_invalid_request", 400, "DeepSeek V4 supports function tools only")
            tool.pop("container", None)
            tool["function"] = {key: value for key, value in tool["function"].items() if value is not None}
        # Serving lifts top-level tools onto the first system message.
        if tools:
            if not messages or messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})
            messages[0]["tools"] = tools
        return messages

    def render_conversation_tokens(self, request: TITOChatRequest) -> Sequence[int]:
        messages = self._template_messages(request)
        text = self._encoder.encode_messages(
            messages,
            thinking_mode="thinking",
            reasoning_effort=self.reasoning_effort,
            drop_thinking=False,
            add_default_bos_token=True,
        )
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def parse_assistant(
        self,
        request: TITOChatRequest,
        completion_ids: Sequence[int],
        completion_text: str,
        finish_reason: str,
    ) -> TITOParsedAssistant:
        del completion_text
        self._template_messages(request)
        ids = tuple(int(token) for token in completion_ids)
        # The generation prefix already owns <think>. The official parser
        # expects completion-only text and an EOS that serving may omit.
        if ids and ids[-1] == self._eos[0]:
            ids = ids[:-1]
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        reasoning, boundary, content = text.partition(self._encoder.thinking_end_token)
        if finish_reason == "length" and (not boundary or self._encoder.dsml_token not in content):
            message = {
                "role": "assistant",
                "content": content if boundary else "",
                "reasoning_content": reasoning,
            }
        else:
            try:
                message = self._encoder.parse_message_from_completion_text(
                    text + self._encoder.eos_token, thinking_mode="thinking"
                )
            except (AssertionError, ValueError) as exc:
                raise TITOParseError("malformed DeepSeek V4 assistant output") from exc
        calls = message.get("tool_calls") or []
        allowed = {str((tool.get("function") or {}).get("name")) for tool in request.tools}
        if any(call.get("namespace") or str(call["function"]["name"]) not in allowed for call in calls):
            raise TITOParseError("DeepSeek V4 tool call names are absent from the request")
        for call in calls:
            try:
                arguments = json.loads(call["function"]["arguments"])
            except json.JSONDecodeError as exc:
                raise TITOParseError("DeepSeek V4 tool arguments are invalid JSON") from exc
            if not isinstance(arguments, dict):
                raise TITOParseError("DeepSeek V4 tool arguments must be a JSON object")
        if not calls:
            message.pop("tool_calls", None)
        message = _ensure_tool_call_ids(message, completion_ids)
        return TITOParsedAssistant(
            message=message,
            output_kind="tool_calls" if calls else "reasoning",
        )

    def render_contract_id(self, request: TITOChatRequest) -> str:
        contract = {
            "renderer": self.renderer_id,
            "certification": self.certification_id,
            "tokenizer": self.tokenizer_fingerprint,
            "model": request.model,
            "tools": _plain(request.tools),
            "thinking_mode": "thinking",
            "reasoning_effort": self.reasoning_effort,
            "drop_thinking": False,
        }
        return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()

    def stop_sequences(self, request: TITOChatRequest) -> Sequence[str]:
        return (self._encoder.eos_token,)
