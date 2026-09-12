"""Model rendering and parsing for TITO.

Prompt construction delegates to the pinned tokenizer's authoritative chat
template.  This module owns only protocol normalization plus the GLM output
parser; importing it must not load Tinker or Torch. Full-history rendering is
the production default. Incremental rendering is experimental and requires a
model/template-specific suffix-and-junction implementation.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fireworks.training.sdk import (
    TITOChatRequest,
    TITOError,
    TITOIncrementalPrompt,
    TITOParsedAssistant,
    TITORenderer,
    normalize_openai_tool_arguments,
)


_GLM52_RENDERER = "glm_moe_dsa_preserve_thinking"
_KIMI_K3_RENDERER = "kimi_k3_preserve_thinking"
_INCREMENTAL_ANCHOR_SYSTEM = {"role": "system", "content": "TITO anchor"}
_GLM_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_GLM_TOOL_ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)
_DYNAMIC_TEMPLATE_FIELDS = frozenset(
    {
        "chat_template_kwargs",
        "clear_thinking",
        "drop_thinking",
        "enable_thinking",
        "preserve_thinking",
        "reasoning_effort",
        "response_format",
        "thinking",
    }
)


@dataclass(frozen=True)
class TITORendererCertification:
    """Reviewed full-history capability for one renderer/tokenizer contract.

    This base certification does not certify the experimental incremental
    method; renderer authors own that additional model-specific contract.
    """

    certification_id: str
    renderer_names: frozenset[str]
    tokenizer_fingerprint: str
    renderer_factory: Callable[[Any, "TITORendererCertification"], TITORenderer]


def load_sidecar_tokenizer(path: str | Path) -> Any:
    """Load the pinned tokenizer and its bundled authoritative chat template."""
    from transformers import AutoTokenizer

    tokenizer_path = Path(path)
    tokenizer_config = json.loads(
        (tokenizer_path / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    auto_map = tokenizer_config.get("auto_map") or {}
    custom_tokenizer = auto_map.get("AutoTokenizer")
    if custom_tokenizer not in (None, ["tokenization_kimi.TikTokenTokenizer", None]):
        raise ValueError("TITO sidecar does not allow this custom tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=custom_tokenizer is not None,
    )
    if custom_tokenizer is None and not getattr(tokenizer, "chat_template", None):
        raise ValueError("TITO sidecar tokenizer has no bundled chat template")
    return tokenizer


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        contract = {
            "kind": "fast",
            "backend": json.loads(backend.to_str()),
            "chat_template": getattr(tokenizer, "chat_template", None),
            "special_tokens_map": tokenizer.special_tokens_map,
        }
    elif type(tokenizer).__name__ == "TikTokenTokenizer":
        tokenizer_module = __import__(type(tokenizer).__module__, fromlist=["__file__"])
        tokenizer_source = Path(tokenizer_module.__file__).resolve()
        encoding_source = tokenizer_source.with_name("encoding_k3.py")
        vocab_source = Path(tokenizer.vocab_file).resolve()
        if not encoding_source.is_file() or not vocab_source.is_file():
            raise ValueError("Kimi K3 tokenizer bundle is incomplete")
        contract = {
            "kind": "kimi_k3_tiktoken",
            "class": type(tokenizer).__name__,
            "tokenization_sha256": hashlib.sha256(
                tokenizer_source.read_bytes()
            ).hexdigest(),
            "encoding_sha256": hashlib.sha256(
                encoding_source.read_bytes()
            ).hexdigest(),
            "vocab_sha256": hashlib.sha256(vocab_source.read_bytes()).hexdigest(),
            "special_tokens_map": tokenizer.special_tokens_map,
        }
    else:
        raise ValueError(
            "production TITO certification requires a serializable fast tokenizer "
            "or the pinned Kimi K3 TikTokenTokenizer"
        )
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def get_tito_renderer_certification(
    renderer_name: str,
    tokenizer: Any,
) -> TITORendererCertification:
    """Resolve and verify the source-controlled production artifact."""
    certification = _TITO_CERTIFICATION_BY_RENDERER.get(renderer_name)
    if certification is None:
        raise ValueError(
            f"renderer {renderer_name!r} has no production TITO certification"
        )
    actual = _tokenizer_fingerprint(tokenizer)
    if actual != certification.tokenizer_fingerprint:
        raise ValueError(
            "tokenizer does not match TITO certification "
            f"{certification.certification_id!r}"
        )
    return certification


def _ensure_tool_call_ids(
    message: Mapping[str, Any],
    completion_ids: Sequence[int],
) -> dict[str, Any]:
    calls = message.get("tool_calls") or []
    if not calls:
        return dict(message)
    normalized_calls: list[dict[str, Any]] = []
    for index, raw_call in enumerate(calls):
        call = dict(raw_call)
        function = dict(call.get("function") or {})
        call["function"] = function
        if not call.get("id"):
            identity_function = {
                **function,
                "arguments": normalize_openai_tool_arguments(
                    function.get("arguments", "")
                ),
            }
            identity = {
                "completion_ids": [int(token) for token in completion_ids],
                "index": index,
                "function": identity_function,
            }
            digest = hashlib.sha256(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            call["id"] = f"call_{digest[:24]}"
        normalized_calls.append(call)
    return {**message, "tool_calls": normalized_calls}


class GLM52TITORenderer:
    """Pinned GLM-5.2 preserved-thinking TITO model-format primitive."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        certification: TITORendererCertification,
    ) -> None:
        self.tokenizer = tokenizer
        self.renderer_id = _GLM52_RENDERER
        self.certification_id = certification.certification_id
        self.tokenizer_fingerprint = certification.tokenizer_fingerprint
        self._user_token = self._single_token("<|user|>")
        self._observation_token = self._single_token("<|observation|>")
        self._stop = (self._user_token, self._observation_token)

    def _single_token(self, text: str) -> int:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"GLM-5.2 expected {text!r} to encode as one token")
        return int(tokens[0])

    def _encode(self, text: str) -> list[int]:
        return [
            int(token)
            for token in self.tokenizer.encode(text, add_special_tokens=False)
        ]

    def _validate_request(self, request: TITOChatRequest) -> None:
        fields = sorted(_DYNAMIC_TEMPLATE_FIELDS.intersection(request.sampling_fields))
        if fields:
            raise TITOError(
                "tito_invalid_request",
                400,
                "TITO renderer/template options are fixed by the certified "
                "renderer contract; unsupported per-request fields: "
                + ", ".join(fields),
            )

    @staticmethod
    def _plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): GLM52TITORenderer._plain(item) for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [GLM52TITORenderer._plain(item) for item in value]
        return value

    @classmethod
    def _template_tools(
        cls,
        tools: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for raw_tool in tools:
            tool = cls._plain(raw_tool)
            function = tool.get("function")
            if isinstance(function, dict):
                # Match ChatCompletionTool.model_dump() field order at Fireworks
                # chat admission. The GLM template renders this function object
                # directly, so its envelope order is prompt-visible too.
                parameters = function.get("parameters") or {}
                tool["function"] = {
                    "name": function.get("name"),
                    "description": function.get("description"),
                    # ChatCompletionFunction treats parameters as an opaque
                    # mapping, so request admission preserves its key order.
                    "parameters": cls._plain(parameters),
                }
                normalized.append(tool)
                continue
            normalized.append(tool)
        return normalized

    def _template_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized = [self._plain(message) for message in messages]
        for message in normalized:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or ():
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise TITOError(
                            "tito_invalid_request",
                            400,
                            "historical assistant tool arguments are not valid JSON",
                        ) from exc
        return normalized

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
        return self._template_messages(messages), self._template_tools(tools)

    def _render_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any:
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            clear_thinking=False,
            reasoning_effort="max",
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
        """Experimentally join a Miles-style suffix to the sampled checkpoint.

        This copies the linear prompt-construction mechanism from Miles Session
        v2—synthetic assistant anchor, suffix render, model-owned junction—but
        deliberately does not adopt Miles's trajectory tree.
        """

        self._validate_request(request)
        if not stored_messages or not appended_messages:
            return None
        wire = request.wire_value()
        wire_messages = (
            list(wire.get("messages") or ())
            if wire is not None
            else [dict(message) for message in request.messages]
        )
        wire_tools = (
            list(wire.get("tools") or ())
            if wire is not None
            else [dict(tool) for tool in request.tools]
        )
        stored_count = len(stored_messages)
        if len(wire_messages) != stored_count + len(appended_messages):
            return None
        stored_assistant = wire_messages[stored_count - 1]
        if stored_assistant.get("role") != "assistant":
            return None

        dummy_assistant: dict[str, Any] = {
            "role": "assistant",
            "content": "",
            "reasoning_content": " ",
        }
        if stored_assistant.get("tool_calls"):
            dummy_assistant["tool_calls"] = stored_assistant["tool_calls"]
        anchor = self._template_messages([_INCREMENTAL_ANCHOR_SYSTEM, dummy_assistant])
        suffix_messages = self._template_messages(wire_messages[stored_count:])
        tools = self._template_tools(wire_tools)
        rendered_anchor = self._render_template(
            anchor,
            tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        rendered_with_suffix = self._render_template(
            [*anchor, *suffix_messages],
            tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered_anchor, str) or not isinstance(
            rendered_with_suffix, str
        ):
            raise TypeError("tokenizer returned a non-text chat-template render")
        if not rendered_with_suffix.startswith(rendered_anchor):
            return None

        suffix_ids = self._encode(rendered_with_suffix[len(rendered_anchor) :])
        checkpoint = tuple(int(token) for token in exact_checkpoint_ids)
        junction_kind = "append"
        checkpoint_trim_tokens = 0
        boundary_tokens = {self._user_token, self._observation_token}
        if checkpoint and checkpoint[-1] in boundary_tokens:
            removed_boundary = checkpoint[-1]
            checkpoint = checkpoint[:-1]
            checkpoint_trim_tokens = 1
            junction_kind = (
                "deduplicate_role_boundary"
                if suffix_ids and suffix_ids[0] == removed_boundary
                else "replace_role_boundary"
            )
        if not suffix_ids:
            return None
        return TITOIncrementalPrompt(
            prompt_ids=(*checkpoint, *suffix_ids),
            contract_id=f"{self.certification_id}:incremental-v1",
            junction_kind=junction_kind,
            checkpoint_trim_tokens=checkpoint_trim_tokens,
        )

    def _parse_message(
        self,
        request: TITOChatRequest,
        completion_ids: Sequence[int],
        finish_reason: str,
    ) -> tuple[dict[str, Any], bool]:
        response = [int(token) for token in completion_ids]
        think_prefix = self._encode("<think>")
        # The certified generation suffix already opened ``<think>`` before
        # sampling. Restore that prompt-owned boundary for every completion,
        # including malformed stops that arrive before ``</think>``.
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
                raise ValueError("GLM-5.2 reasoning close has no opening boundary")
            reasoning, content = content[len("<think>") :].split("</think>", 1)
            content = content.lstrip("\n")
        elif finish_reason == "length":
            # The certified generation prompt ends inside ``<think>``. If a
            # length stop arrives before ``</think>``, every sampled token is
            # incomplete reasoning—not agent-visible assistant content. Keep
            # it structured and never interpret tool markup inside it.
            reasoning = (
                content[len("<think>") :] if content.startswith("<think>") else content
            )
            content = ""
        elif content.startswith("<think>"):
            # A clean stop before the prompt-opened reasoning boundary closes
            # is malformed structured output, not visible assistant text.
            # Reject it before tool parsing can reinterpret reasoning.
            raise ValueError("GLM-5.2 reasoning open has no closing boundary")

        # Tool markup is protocol only after the reasoning boundary. Malformed
        # model output is classified by the engine; the renderer must not
        # manufacture a different tool call in an attempt to repair it.
        calls: list[dict[str, Any]] = []
        cleaned: list[str] = []
        position = 0
        for match in _GLM_TOOL_CALL_RE.finditer(content):
            cleaned.append(content[position : match.start()])
            body = match.group(1)
            first_arg = body.find("<arg_key>")
            name = body[:first_arg].strip() if first_arg >= 0 else body.strip()
            if not name:
                raise ValueError("GLM-5.2 tool call is missing a function name")
            arguments = {
                item.group(1): self._parse_argument(item.group(2))
                for item in _GLM_TOOL_ARG_RE.finditer(body)
            }
            residue = _GLM_TOOL_ARG_RE.sub("", body).strip().removeprefix(name).strip()
            if residue:
                raise ValueError(f"unparsed GLM-5.2 tool-call content: {residue!r}")
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
            raise ValueError("unparsed GLM-5.2 tool-call boundary")

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
            raise ValueError("GLM-5.2 tool call names are absent from the request")

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
                [*completion_ids, self._user_token],
                finish_reason,
            )
        if not clean and finish_reason != "length":
            raise ValueError("unclean GLM-5.2 renderer parse")
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


_KIMI_OPEN = "<|open|>"
_KIMI_CLOSE = "<|close|>"
_KIMI_SEP = "<|sep|>"
_KIMI_STOP = f"{_KIMI_CLOSE}message{_KIMI_SEP}"
_KIMI_CALL_PATTERN = re.compile(
    rf"{re.escape(_KIMI_OPEN)}call(?P<header>.*?){re.escape(_KIMI_SEP)}"
    rf"(?P<body>.*?){re.escape(_KIMI_CLOSE)}call{re.escape(_KIMI_SEP)}",
    re.DOTALL,
)
_KIMI_ARG_PATTERN = re.compile(
    rf"{re.escape(_KIMI_OPEN)}argument(?P<header>.*?){re.escape(_KIMI_SEP)}"
    rf"(?P<body>.*?){re.escape(_KIMI_CLOSE)}argument{re.escape(_KIMI_SEP)}",
    re.DOTALL,
)
_KIMI_JSON_PATTERN = re.compile(
    rf"{re.escape(_KIMI_OPEN)}json(?P<header>.*?){re.escape(_KIMI_SEP)}"
    rf"(?P<body>.*?){re.escape(_KIMI_CLOSE)}json{re.escape(_KIMI_SEP)}",
    re.DOTALL,
)


def _find_token_subsequence(
    haystack: Sequence[int], needle: Sequence[int], start: int = 0
) -> int:
    if not needle:
        return start
    for index in range(start, len(haystack) - len(needle) + 1):
        if list(haystack[index : index + len(needle)]) == list(needle):
            return index
    return -1


def _kimi_control_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer._encode_text_piece(text, allow_special_tokens=True))


def _kimi_attribute(tag: str, name: str) -> str | None:
    match = re.search(rf'\b{re.escape(name)}="([^"]*)"', tag)
    return html.unescape(match.group(1)) if match else None


def _kimi_typed_argument(value: str, type_name: str | None) -> Any:
    if type_name in {"object", "array", "number", "boolean", "null"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _parse_kimi_tool_calls(tokenizer: Any, tokens: Sequence[int]) -> list[dict[str, Any]]:
    text = str(tokenizer.decode(list(tokens)))
    matches = list(_KIMI_CALL_PATTERN.finditer(text))
    if not matches:
        raise ValueError("Kimi K3 tools block contains no complete tool call")
    calls: list[dict[str, Any]] = []
    for match in matches:
        name = _kimi_attribute(match.group("header"), "tool")
        if not name:
            raise ValueError("Kimi K3 tool call is missing a function name")
        arguments: dict[str, Any] = {}
        json_match = _KIMI_JSON_PATTERN.search(match.group("body"))
        if json_match:
            parsed = json.loads(json_match.group("body"))
            if not isinstance(parsed, dict):
                raise ValueError("Kimi K3 JSON tool arguments must be an object")
            arguments = parsed
        else:
            for argument in _KIMI_ARG_PATTERN.finditer(match.group("body")):
                key = _kimi_attribute(argument.group("header"), "key")
                if key is None:
                    raise ValueError("Kimi K3 tool argument is missing a key")
                arguments[key] = _kimi_typed_argument(
                    argument.group("body"),
                    _kimi_attribute(argument.group("header"), "type"),
                )
        calls.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
        )
    return calls


class KimiK3TITORenderer:
    """Pinned Kimi K3 XTML renderer for the lightweight TITO sidecar."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        certification: TITORendererCertification,
    ) -> None:
        self.tokenizer = tokenizer
        self.renderer_id = _KIMI_K3_RENDERER
        self.certification_id = certification.certification_id
        self.tokenizer_fingerprint = certification.tokenizer_fingerprint

    def _render(self, request: TITOChatRequest) -> tuple[int, ...]:
        wire = request.wire_value()
        messages = GLM52TITORenderer._plain(
            list((wire or {}).get("messages") or request.messages)
        )
        tools = GLM52TITORenderer._plain(
            list((wire or {}).get("tools") or request.tools)
        )
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=True,
            add_generation_prompt=True,
            thinking=True,
            thinking_effort="max",
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
        del request, stored_messages, appended_messages, exact_checkpoint_ids
        return None

    def parse_assistant(
        self,
        request: TITOChatRequest,
        completion_ids: Sequence[int],
        completion_text: str,
        finish_reason: str,
    ) -> TITOParsedAssistant:
        del completion_text
        tokens = [int(token) for token in completion_ids]
        stop = _kimi_control_ids(self.tokenizer, _KIMI_STOP)
        stop_at = _find_token_subsequence(tokens, stop)
        if stop_at < 0:
            if finish_reason == "length":
                return TITOParsedAssistant(
                    message={
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": str(self.tokenizer.decode(tokens)),
                    },
                    output_kind="reasoning",
                )
            tokens.extend(stop)
            stop_at = len(tokens) - len(stop)
        elif _find_token_subsequence(tokens, stop, stop_at + len(stop)) >= 0:
            raise ValueError("Kimi K3 completion contains multiple stop sequences")
        body = tokens[:stop_at]

        close_think_open_response = _kimi_control_ids(
            self.tokenizer,
            f"{_KIMI_CLOSE}think{_KIMI_SEP}{_KIMI_OPEN}response{_KIMI_SEP}",
        )
        close_response = _kimi_control_ids(
            self.tokenizer, f"{_KIMI_CLOSE}response{_KIMI_SEP}"
        )
        open_tools = _kimi_control_ids(
            self.tokenizer, f"{_KIMI_OPEN}tools{_KIMI_SEP}"
        )
        close_tools = _kimi_control_ids(
            self.tokenizer, f"{_KIMI_CLOSE}tools{_KIMI_SEP}"
        )

        channel_at = _find_token_subsequence(body, close_think_open_response)
        reasoning_ids = body[:channel_at] if channel_at >= 0 else []
        response_start = (
            channel_at + len(close_think_open_response) if channel_at >= 0 else 0
        )
        response_end = _find_token_subsequence(body, close_response, response_start)
        if response_end < 0:
            raise ValueError("Kimi K3 completion has no closed response channel")
        content_ids = body[response_start:response_end]
        remainder_start = response_end + len(close_response)

        calls: list[dict[str, Any]] = []
        tools_at = _find_token_subsequence(body, open_tools, remainder_start)
        if tools_at >= 0:
            tools_end = _find_token_subsequence(
                body, close_tools, tools_at + len(open_tools)
            )
            if tools_end < 0:
                raise ValueError("Kimi K3 completion has an unclosed tools channel")
            calls = _parse_kimi_tool_calls(
                self.tokenizer, body[tools_at + len(open_tools) : tools_end]
            )

        reasoning = str(self.tokenizer.decode(reasoning_ids))
        content = str(self.tokenizer.decode(content_ids))
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
        message = _ensure_tool_call_ids(message, completion_ids)
        allowed_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in request.tools
            if (tool.get("function") or {}).get("name")
        }
        if calls and any(
            str(call["function"]["name"]) not in allowed_names for call in calls
        ):
            raise ValueError("Kimi K3 tool call names are absent from the request")
        return TITOParsedAssistant(
            message=message,
            output_kind="tool_calls" if calls else "reasoning" if reasoning else "text",
        )

    def fallback_assistant_text(
        self,
        request: TITOChatRequest,
        completion_ids: Sequence[int],
        finish_reason: str,
        parser_error: BaseException,
    ) -> str | None:
        del request, completion_ids, finish_reason, parser_error
        return None

    def render_contract_id(self, request: TITOChatRequest) -> str:
        contract = {
            "renderer_id": self.renderer_id,
            "certification_id": self.certification_id,
            "model": request.model,
            "tools": [dict(tool) for tool in request.tools],
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
        return (_KIMI_STOP,)


def _build_glm52_tito_renderer(
    tokenizer: Any,
    certification: TITORendererCertification,
) -> TITORenderer:
    return GLM52TITORenderer(tokenizer, certification=certification)


def _build_kimi_k3_tito_renderer(
    tokenizer: Any,
    certification: TITORendererCertification,
) -> TITORenderer:
    return KimiK3TITORenderer(tokenizer, certification=certification)


_TITO_RENDERER_CERTIFICATIONS = (
    TITORendererCertification(
        certification_id="glm-5.2-preserved@b4734de4-v7",
        renderer_names=frozenset({_GLM52_RENDERER}),
        tokenizer_fingerprint=(
            "5591741bd28d5acb92d4b7d735e0084d4d76d9ce50e2afe99aec6b01e1ef3ef0"
        ),
        renderer_factory=_build_glm52_tito_renderer,
    ),
    TITORendererCertification(
        certification_id="kimi-k3-preserved@9f62e4e9-v1",
        renderer_names=frozenset({_KIMI_K3_RENDERER}),
        tokenizer_fingerprint=(
            "3d98398caa8f9429f16be14e95efc65767ac09f8a7944e306ae0e0d7f72aef58"
        ),
        renderer_factory=_build_kimi_k3_tito_renderer,
    ),
)
_TITO_CERTIFICATION_BY_RENDERER = {
    renderer_name: certification
    for certification in _TITO_RENDERER_CERTIFICATIONS
    for renderer_name in certification.renderer_names
}


def build_sidecar_tito_renderer(
    tokenizer: Any,
    renderer_name: str,
) -> TITORenderer:
    """Build a renderer admitted to the lightweight agent-sidecar runtime."""
    certification = get_tito_renderer_certification(renderer_name, tokenizer)
    return certification.renderer_factory(tokenizer, certification)


__all__ = [
    "GLM52TITORenderer",
    "KimiK3TITORenderer",
    "TITORendererCertification",
    "build_sidecar_tito_renderer",
    "get_tito_renderer_certification",
    "load_sidecar_tokenizer",
]
