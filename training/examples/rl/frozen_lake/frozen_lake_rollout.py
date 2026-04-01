"""FrozenLake multi-turn tool-call rollout processor.

Uses the generic :class:`FireworksV1CompletionsClient` from eval-protocol
with a FrozenLake-specific tool-call parser.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fireworks import AsyncFireworks
from openai.types import CompletionUsage

from eval_protocol.integrations.fireworks_v1_completions_client import (
    FireworksV1CompletionsClient,
    ParsedToolCall,
    _normalize_token_id_sequence,
    strip_chat_special_tokens,
    to_openai_tool_calls,
)
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

from training.examples.rl.frozen_lake.frozen_lake_env import (
    build_frozen_lake_tool_env,
    build_frozen_lake_user_prompt,
)
from training.examples.rl.frozen_lake.frozen_lake_schema import (
    FROZEN_LAKE_ACTIONS,
    FROZEN_LAKE_TOOLS,
    TOOL_NAME_LAKE_MOVE,
    normalize_parsed_tool_call,
    parse_first_frozen_lake_tool_call,
    parse_first_frozen_lake_tool_call_with_content,
    parse_tool_call_with_fallback,
)

logger = logging.getLogger(__name__)
_TOKENIZER_CACHE: Dict[str, Any] = {}

DEFAULT_SYSTEM_PROMPT_INSTRUCTIONS = (
    "You are an RL policy for FrozenLake.\n"
    "Pick the action that moves toward G while avoiding H.\n"
    "Always respond with exactly one tool call and no text."
)

DEFAULT_VISUAL_PROMPT_TEMPLATE = (
    "You are playing FrozenLake. The image shows the current grid. "
    "The current textual observation is below.\n"
    "{observation}\n\n"
    "Tiles are labeled S, F, H, and G. The agent is marked with a red dot. "
    "Use exactly one lake_move tool call with action LEFT, DOWN, RIGHT, or UP."
)


def _get_hf_tokenizer(tokenizer_name_or_path: str):
    tok = _TOKENIZER_CACHE.get(tokenizer_name_or_path)
    if tok is not None:
        return tok
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    _TOKENIZER_CACHE[tokenizer_name_or_path] = tok
    return tok


def _is_kimi_tokenizer_name(tokenizer_name_or_path: str) -> bool:
    lowered = str(tokenizer_name_or_path or "").lower()
    return "kimi-k2.5" in lowered or "kimi-k2p5" in lowered


def _build_multimodal_image_content(*, image_url: str, text: Optional[str] = None) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": str(image_url)}},
    ]
    if text is not None:
        parts.append({"type": "text", "text": str(text)})
    return parts


def _normalize_multimodal_content(content: Any) -> Any:
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    normalized_parts: List[Dict[str, Any]] = []
    for part in content:
        if isinstance(part, dict):
            part_type = str(part.get("type") or "").strip().lower()
            if part_type == "image":
                normalized_parts.append({"type": "image"})
                continue
            if part_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    normalized_parts.append({"type": "image_url", "image_url": dict(image_url)})
                elif image_url is not None:
                    normalized_parts.append({"type": "image_url", "image_url": {"url": str(image_url)}})
                else:
                    normalized_parts.append({"type": "image"})
                continue
            if part_type == "text":
                normalized_parts.append({"type": "text", "text": str(part.get("text") or "")})
                continue
            if "text" in part:
                normalized_parts.append({"type": "text", "text": str(part.get("text") or "")})
                continue
        normalized_parts.append({"type": "text", "text": str(part)})
    return normalized_parts


def _sanitize_messages_for_multimodal_template(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        sanitized_msg: Dict[str, Any] = {
            "role": str(msg.get("role", "user")),
            "content": _normalize_multimodal_content(msg.get("content")),
        }
        if msg.get("tool_calls") is not None:
            sanitized_msg["tool_calls"] = msg.get("tool_calls")
        if msg.get("tool_call_id") is not None:
            sanitized_msg["tool_call_id"] = msg.get("tool_call_id")
        if msg.get("name") is not None:
            sanitized_msg["name"] = msg.get("name")
        sanitized.append(sanitized_msg)
    return sanitized


def _build_multimodal_fallback_prompt_text(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> str:
    chunks: List[str] = []
    if tools:
        chunks.append("TOOLS:")
        for tool in tools:
            function = tool.get("function", {})
            chunks.append(
                json.dumps(
                    {
                        "name": function.get("name"),
                        "description": function.get("description"),
                        "parameters": function.get("parameters"),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        chunks.append("")

    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        content = msg.get("content")
        if isinstance(content, list):
            rendered_parts: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    rendered_parts.append(str(part))
                    continue
                part_type = str(part.get("type") or "").strip().lower()
                if part_type in {"image", "image_url"}:
                    rendered_parts.append("<image>")
                else:
                    rendered_parts.append(str(part.get("text") or ""))
            rendered = "".join(rendered_parts)
        else:
            rendered = str(content or "")
        chunks.append(f"{role}: {rendered}")
        if msg.get("tool_calls"):
            chunks.append(f"{role}_TOOL_CALLS: {json.dumps(msg['tool_calls'], ensure_ascii=False)}")
    chunks.append("ASSISTANT:")
    return "\n".join(chunks)


def _build_visual_user_prompt(
    *,
    prompt_template: Optional[str],
    observation: str,
    default_prompt: str,
) -> str:
    template = str(prompt_template or default_prompt)
    if "{observation}" in template:
        return template.replace("{observation}", observation)
    observation_suffix = f"\n\nCurrent textual observation:\n{observation}" if observation else ""
    return f"{template.rstrip()}{observation_suffix}"


def _strip_trailing_text_suffix(text: str, suffix: str) -> str:
    if suffix and text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def _strip_trailing_token_suffix(token_ids: List[int], suffix_ids: List[int]) -> Tuple[List[int], int]:
    normalized_token_ids = [int(x) for x in list(token_ids)]
    normalized_suffix_ids = [int(x) for x in list(suffix_ids)]
    if (
        normalized_suffix_ids
        and len(normalized_token_ids) >= len(normalized_suffix_ids)
        and normalized_token_ids[-len(normalized_suffix_ids):] == normalized_suffix_ids
    ):
        return normalized_token_ids[:-len(normalized_suffix_ids)], len(normalized_suffix_ids)
    return normalized_token_ids, 0


def _build_kimi_toolcall_generation_prefill_token_ids(tokenizer_name_or_path: str) -> List[int]:
    if not _is_kimi_tokenizer_name(tokenizer_name_or_path):
        return []
    tokenizer = _get_hf_tokenizer(tokenizer_name_or_path)
    return _normalize_token_id_sequence(
        tokenizer.encode("<|tool_calls_section_begin|>", add_special_tokens=False)
    )


def _build_kimi_toolcall_generation_prefill_text(tokenizer_name_or_path: str) -> str:
    if not _is_kimi_tokenizer_name(tokenizer_name_or_path):
        return ""
    return "<|tool_calls_section_begin|>"


class FireworksV1ImageCompletionsClient:
    """Multimodal `/v1/completions` client that mirrors local tokenized chat mode."""

    def __init__(
        self,
        *,
        model_id: str,
        tokenizer_name_or_path: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
        request_params: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        enable_thinking: Optional[bool] = None,
        tool_call_parser=None,
        default_tools: Optional[List[Dict[str, Any]]] = None,
    ):
        self.model_id = model_id
        self.tokenizer_name_or_path = tokenizer_name_or_path or model_id
        self._api_key = api_key
        self._base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_params = dict(request_params or {})
        self.logprobs = logprobs
        self.enable_thinking = enable_thinking
        self.tool_call_parser = tool_call_parser
        self.default_tools = default_tools or []
        self._tokenizer = None
        self._client = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

    def _get_client(self) -> AsyncFireworks:
        if self._client is None:
            self._client = AsyncFireworks(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _get_hf_tokenizer(self.tokenizer_name_or_path)
        return self._tokenizer

    def _thinking_kwargs(self) -> Dict[str, Any]:
        if self.enable_thinking is not None:
            return {"thinking": self.enable_thinking}
        return {}

    def _strip_generation_think_from_token_ids(self, token_ids: List[int]) -> List[int]:
        if self.enable_thinking is not False or not _is_kimi_tokenizer_name(self.tokenizer_name_or_path):
            return token_ids
        tokenizer = self._get_tokenizer()
        think_suffix_ids = _normalize_token_id_sequence(
            tokenizer.encode("<think>", add_special_tokens=False)
        )
        if think_suffix_ids and token_ids[-len(think_suffix_ids):] == think_suffix_ids:
            return token_ids[:-len(think_suffix_ids)]
        return token_ids

    def _strip_generation_think_from_text(self, text: str) -> str:
        if self.enable_thinking is not False or not _is_kimi_tokenizer_name(self.tokenizer_name_or_path):
            return text
        stripped = _strip_trailing_text_suffix(text, "<think>")
        stripped = _strip_trailing_text_suffix(stripped, "\n")
        return stripped

    def _strip_generation_assistant_suffix_from_text(self, text: str) -> str:
        return _strip_trailing_text_suffix(text, self.assistant_turn_suffix_text())

    def _apply_chat_template(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any:
        tokenizer = self._get_tokenizer()
        sanitized_messages = _sanitize_messages_for_multimodal_template(messages=messages)
        thinking_kw = self._thinking_kwargs()
        try:
            return tokenizer.apply_chat_template(
                sanitized_messages,
                tools=tools,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **thinking_kw,
            )
        except Exception as exc:
            if tools:
                logger.debug("Multimodal chat template with tools failed, retrying without tools: %s", exc)
                return tokenizer.apply_chat_template(
                    sanitized_messages,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                    **thinking_kw,
                )
            raise

    def build_prompt_token_ids(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> List[int]:
        tokenizer = self._get_tokenizer()
        try:
            token_ids = self._apply_chat_template(
                messages=messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
            )
            return self._strip_generation_think_from_token_ids(
                _normalize_token_id_sequence(token_ids)
            )
        except Exception as exc:
            logger.debug("Multimodal tokenized prompt build failed, using text fallback: %s", exc)
            fallback_prompt = self.build_prompt_text(messages=messages, tools=tools)
            return _normalize_token_id_sequence(tokenizer.encode(fallback_prompt, add_special_tokens=False))

    def build_prompt_text(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> str:
        try:
            prompt_text = str(
                self._apply_chat_template(
                    messages=messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            return self._strip_generation_think_from_text(prompt_text)
        except Exception as exc:
            logger.warning(
                "Multimodal chat template build failed for %s: %s",
                self.tokenizer_name_or_path,
                exc,
            )
            sanitized_messages = _sanitize_messages_for_multimodal_template(messages=messages)
            return self._strip_generation_think_from_text(
                _build_multimodal_fallback_prompt_text(sanitized_messages, tools=tools)
            )

    def build_tool_response_suffix_token_ids(
        self,
        *,
        tool_message: Dict[str, Any],
    ) -> List[int]:
        tokenizer = self._get_tokenizer()
        suffix_messages = [tool_message]
        try:
            token_ids = self._apply_chat_template(
                messages=suffix_messages,
                tools=None,
                tokenize=True,
                add_generation_prompt=True,
            )
            return self._strip_generation_think_from_token_ids(
                _normalize_token_id_sequence(token_ids)
            )
        except Exception as exc:
            logger.debug("Multimodal tool suffix build failed, using text fallback: %s", exc)
            fallback_prompt = self.build_prompt_text(messages=suffix_messages, tools=None)
            return _normalize_token_id_sequence(tokenizer.encode(fallback_prompt, add_special_tokens=False))

    def build_tool_response_suffix_text(
        self,
        *,
        tool_message: Dict[str, Any],
    ) -> str:
        suffix_messages = [tool_message]
        return self.build_prompt_text(messages=suffix_messages, tools=None)

    def encode_assistant_turn_suffix(self) -> List[int]:
        tokenizer = self._get_tokenizer()
        suffix_text = "<|im_end|>" if _is_kimi_tokenizer_name(self.tokenizer_name_or_path) else "<|im_end|>\n"
        return _normalize_token_id_sequence(
            tokenizer.encode(suffix_text, add_special_tokens=False)
        )

    def assistant_turn_suffix_text(self) -> str:
        return "<|im_end|>" if _is_kimi_tokenizer_name(self.tokenizer_name_or_path) else "<|im_end|>\n"

    def decode_token_ids(self, *, token_ids: List[int]) -> str:
        if not token_ids:
            return ""
        tokenizer = self._get_tokenizer()
        try:
            return str(
                tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        except TypeError:
            return str(tokenizer.decode(token_ids))

    async def create_completion_from_prompt_ids(
        self,
        *,
        prompt_token_ids: List[int],
        prompt_text: Optional[str] = None,
        images: List[str],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        active_tools = tools if tools is not None else (self.default_tools or None)
        normalized_prompt_token_ids = [int(x) for x in list(prompt_token_ids)]
        if prompt_text is None:
            prompt_text = self.decode_token_ids(token_ids=normalized_prompt_token_ids)
        request_payload = {
            **self.request_params,
            "model": self.model_id,
            "prompt": prompt_text,
            "images": images,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "return_token_ids": True,
            "raw_output": True,
        }
        if self.logprobs:
            request_payload["logprobs"] = True

        max_retries = 40
        base_delay = 10.0
        for attempt in range(max_retries + 1):
            try:
                response = await self._get_client().completions.create(**request_payload)
                break
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
                err_str = str(exc)
                is_transient = (
                    status in (425, 429, 502, 503, 504)
                    or "model_not_ready" in err_str
                    or "hot loading" in err_str
                    or "Model not found" in err_str
                    or "DEPLOYMENT_SCALING_UP" in err_str
                )
                if not is_transient or attempt >= max_retries:
                    raise
                delay = min(base_delay * (2 ** attempt), 60.0)
                logger.info(
                    "Retryable multimodal error (attempt %d/%d, status=%s), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    status,
                    delay,
                    err_str[:200],
                )
                await asyncio.sleep(delay)

        response_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        choices = response_dict.get("choices") or []
        if not choices:
            raise ValueError("Fireworks /v1/completions response did not include choices")

        choice = choices[0]
        finish_reason = str(choice.get("finish_reason") or "unknown")
        raw_output = choice.get("raw_output") if isinstance(choice.get("raw_output"), dict) else {}
        raw_completion_token_ids = _normalize_token_id_sequence(
            choice.get("token_ids") or raw_output.get("completion_token_ids") or []
        )
        choice_prompt_token_ids = _normalize_token_id_sequence(
            choice.get("prompt_token_ids") or raw_output.get("prompt_token_ids") or normalized_prompt_token_ids
        )
        completion_token_ids = self._strip_generation_think_from_token_ids(raw_completion_token_ids)
        completion_token_ids, _ = _strip_trailing_token_suffix(
            completion_token_ids,
            self.encode_assistant_turn_suffix(),
        )
        completion_text = self.decode_token_ids(token_ids=completion_token_ids)
        if not completion_text:
            completion_text = self._strip_generation_assistant_suffix_from_text(
                self._strip_generation_think_from_text(str(choice.get("text") or ""))
            )

        completion_logprobs: List[float] = []
        choice_logprobs = choice.get("logprobs")
        if isinstance(choice_logprobs, dict):
            token_logprobs = choice_logprobs.get("token_logprobs") or []
            completion_logprobs = [float(lp) if lp is not None else 0.0 for lp in token_logprobs]
            completion_logprobs = completion_logprobs[: len(completion_token_ids)]

        if self.tool_call_parser is not None:
            parsed_output = self.tool_call_parser(completion_text, completion_token_ids, active_tools)
            parsed_tool_call: Optional[ParsedToolCall] = parsed_output.get("parsed_tool_call")
            assistant_content = str(parsed_output.get("assistant_content", "") or "")
            parser_name = str(parsed_output.get("parser", "external"))
            message_payload: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content,
            }
            if parsed_tool_call is not None:
                message_payload["tool_calls"] = to_openai_tool_calls(parsed_tool_call)
        else:
            assistant_content = strip_chat_special_tokens(completion_text)
            parser_name = "none"
            message_payload = {"role": "assistant", "content": assistant_content}

        usage_obj = response_dict.get("usage") or {}
        result: Dict[str, Any] = {
            "choices": [
                {
                    "message": message_payload,
                    "finish_reason": finish_reason,
                    "raw_output": {**dict(raw_output or {}), "tool_call_parser": parser_name},
                }
            ],
            "usage": {
                "prompt_tokens": int(usage_obj.get("prompt_tokens", len(choice_prompt_token_ids))),
                "completion_tokens": int(usage_obj.get("completion_tokens", len(completion_token_ids))),
                "total_tokens": int(
                    usage_obj.get("total_tokens", len(choice_prompt_token_ids) + len(completion_token_ids))
                ),
            },
            "prompt_text": prompt_text,
            "prompt_ids": list(choice_prompt_token_ids),
            "completion_ids": list(completion_token_ids),
            "completion_text": completion_text,
            "finish_reason": finish_reason,
            "raw_output": {**dict(raw_output or {}), "tool_call_parser": parser_name},
        }
        if completion_logprobs:
            result["completion_logprobs"] = completion_logprobs
        return result

    async def create_completion(
        self,
        *,
        messages: List[Dict[str, Any]],
        images: List[str],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        active_tools = tools if tools is not None else (self.default_tools or None)
        prompt_token_ids = self.build_prompt_token_ids(messages=messages, tools=active_tools)
        return await self.create_completion_from_prompt_ids(
            prompt_token_ids=prompt_token_ids,
            images=images,
            tools=active_tools,
        )


def _to_message_payload(messages: Iterable[Message]) -> List[Dict[str, Any]]:
    return [m.model_dump(exclude_none=True) for m in messages]


def _resolve_tool_call_action(tool_calls: Any) -> Tuple[str, str]:
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError("Assistant response must include at least one tool call")

    first = tool_calls[0]
    if not isinstance(first, dict):
        raise ValueError("Tool call must be an object")

    function = first.get("function")
    if not isinstance(function, dict):
        raise ValueError("Tool call.function must be an object")

    name = str(function.get("name") or "").strip()
    if name != TOOL_NAME_LAKE_MOVE:
        raise ValueError(f"Unexpected tool '{name}', expected '{TOOL_NAME_LAKE_MOVE}'")

    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Tool call.arguments must be an object")

    action = str(arguments.get("action") or "").strip().upper()
    if action not in FROZEN_LAKE_ACTIONS:
        raise ValueError(f"Invalid action '{action}', expected one of {FROZEN_LAKE_ACTIONS}")

    tool_call_id = str(first.get("id") or "")
    return action, tool_call_id


def _merge_request_params(
    *,
    completion_params: Dict[str, Any],
    processor_kwargs: Dict[str, Any],
    default_request_params: Dict[str, Any],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(default_request_params)
    if isinstance(processor_kwargs.get("request_params"), dict):
        merged.update(processor_kwargs["request_params"])

    ignored_keys = {
        "model", "temperature", "max_tokens", "base_url", "api_key",
        "tokenizer_name_or_path", "tokenizer_path",
        "system_prompt", "user_prompt_template", "visual_prompt_template",
        "observation_mode",
        "allow_plaintext_action_fallback",
    }
    for key, value in completion_params.items():
        if key in ignored_keys:
            continue
        if key == "extra_body" and isinstance(value, dict):
            merged.update(value)
            continue
        merged[key] = value
    return merged


def _looks_like_tool_parse_error_message(message: str) -> bool:
    lowered = str(message or "").lower()
    return (
        "tool_call" in lowered
        or "tool calls" in lowered
        or "valid json" in lowered
        or "unsupported tool" in lowered
        or "invalid action" in lowered
    )


# ---------------------------------------------------------------------------
# Frozen-lake-specific tool-call parser (passed to generic client)
# ---------------------------------------------------------------------------

def build_frozen_lake_tool_call_parser(
    *,
    allow_plaintext_action_fallback: bool = False,
    tokenizer_getter=None,
    model_id: str = "",
) -> Any:
    """Return a ``ToolCallParserFn`` for the FrozenLake domain."""

    def _try_vllm_parser(
        completion_text: str,
        completion_token_ids: List[int],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        parser_name = ""
        if not parser_name:
            if "qwen3" in model_id.lower():
                parser_name = "qwen3xml"
        if not parser_name:
            return None
        try:
            from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
        except Exception:
            return None
        try:
            tokenizer = tokenizer_getter() if tokenizer_getter else None
            if tokenizer is None:
                return None
            parser_cls = ToolParserManager.get_tool_parser(parser_name)
            parser = parser_cls(tokenizer)
            request = SimpleNamespace(tools=tools or FROZEN_LAKE_TOOLS)
            sig = inspect.signature(parser.extract_tool_calls)
            if "token_ids" in sig.parameters:
                extracted = parser.extract_tool_calls(
                    model_output=completion_text, request=request,
                    token_ids=completion_token_ids,
                )
            else:
                extracted = parser.extract_tool_calls(
                    model_output=completion_text, request=request,
                )
            extracted_tool_calls = list(getattr(extracted, "tool_calls", []) or [])
            if not extracted_tool_calls:
                return None
            first = extracted_tool_calls[0]
            function = getattr(first, "function", None)
            parsed = normalize_parsed_tool_call(
                tool_call_id=getattr(first, "id", None),
                name=getattr(function, "name", None),
                arguments=getattr(function, "arguments", None),
            )
            parsed_content = str(getattr(extracted, "content", "") or "")
            return {
                "parsed_tool_call": parsed,
                "assistant_content": parsed_content,
                "parser": f"vllm:{parser_name}",
            }
        except Exception as exc:
            logger.debug("vLLM tool parser '%s' failed: %s", parser_name, exc)
            return None

    def parser_fn(
        completion_text: str,
        completion_token_ids: List[int],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        vllm_result = _try_vllm_parser(completion_text, completion_token_ids, tools)
        if vllm_result is not None:
            content = strip_chat_special_tokens(str(vllm_result.get("assistant_content", "") or ""))
            vllm_result["assistant_content"] = content
            if content:
                vllm_result["non_tool_content"] = content
            return vllm_result

        try:
            parsed_tool_call, non_tool_content = parse_first_frozen_lake_tool_call_with_content(completion_text)
            cleaned = strip_chat_special_tokens(non_tool_content)
            return {
                "parsed_tool_call": parsed_tool_call,
                "assistant_content": cleaned,
                "non_tool_content": cleaned,
                "parser": "json_schema",
            }
        except Exception:
            parsed = parse_tool_call_with_fallback(
                completion_text, allow_plaintext=allow_plaintext_action_fallback,
            )
            return {
                "parsed_tool_call": parsed,
                "assistant_content": strip_chat_special_tokens(completion_text),
                "parser": "fallback",
            }

    return parser_fn


# ---------------------------------------------------------------------------
# Rollout Processor
# ---------------------------------------------------------------------------

class FrozenLakeToolRolloutProcessor(RolloutProcessor):
    """Rollout processor for strict JSON tool-calling on FrozenLake."""

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        request_params: Optional[Dict[str, Any]] = None,
        allow_plaintext_action_fallback: bool = False,
        logprobs: bool = True,
        enable_thinking: Optional[bool] = False,
        observation_mode: str = "text",
        max_parse_retries: int = 0,
    ):
        self.model_id = model_id
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.request_params = dict(request_params or {})
        self.allow_plaintext_action_fallback = allow_plaintext_action_fallback
        self.logprobs = logprobs
        self.enable_thinking = enable_thinking
        self.observation_mode = observation_mode
        self.max_parse_retries = max_parse_retries

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        completion_params = dict(config.completion_params or {})
        processor_kwargs = dict(config.kwargs or {})
        semaphore = config.semaphore
        max_steps = int(config.steps or 30)

        model_id = str(completion_params.get("model") or processor_kwargs.get("model_id") or self.model_id or "")
        if not model_id:
            raise ValueError("model id is required for FrozenLakeToolRolloutProcessor")

        tokenizer_name_or_path = (
            completion_params.get("tokenizer_name_or_path")
            or completion_params.get("tokenizer_path")
            or processor_kwargs.get("tokenizer_name_or_path")
            or processor_kwargs.get("tokenizer_path")
            or self.tokenizer_name_or_path
            or model_id
        )
        if tokenizer_name_or_path == model_id and _is_kimi_tokenizer_name(model_id):
            tokenizer_name_or_path = "moonshotai/Kimi-K2.5"
        api_key = completion_params.get("api_key") or processor_kwargs.get("api_key") or self.api_key
        base_url = completion_params.get("base_url") or processor_kwargs.get("base_url") or self.base_url
        temperature = float(completion_params.get("temperature", processor_kwargs.get("temperature", self.temperature)))
        max_tokens = int(completion_params.get("max_tokens", processor_kwargs.get("max_tokens", self.max_tokens)))
        request_params = _merge_request_params(
            completion_params=completion_params,
            processor_kwargs=processor_kwargs,
            default_request_params=self.request_params,
        )
        allow_plaintext_action_fallback = bool(
            completion_params.get(
                "allow_plaintext_action_fallback",
                processor_kwargs.get(
                    "allow_plaintext_action_fallback",
                    self.allow_plaintext_action_fallback,
                ),
            )
        )
        observation_mode = str(
            completion_params.get("observation_mode")
            or processor_kwargs.get("observation_mode")
            or self.observation_mode
        ).strip().lower()
        max_parse_retries = int(
            completion_params.get("max_parse_retries")
            or processor_kwargs.get("max_parse_retries")
            or self.max_parse_retries
        )
        if observation_mode == "image":
            request_params.setdefault("thinking", {"type": "disabled"})
        fallback_system_prompt = DEFAULT_SYSTEM_PROMPT_INSTRUCTIONS
        default_system_prompt = (
            completion_params.get("system_prompt")
            or processor_kwargs.get("system_prompt")
            or self.system_prompt
            or fallback_system_prompt
        )
        default_user_prompt_template = (
            processor_kwargs.get("user_prompt_template")
            or self.user_prompt_template
            or completion_params.get("user_prompt_template")
            or None
        )
        default_visual_prompt = (
            processor_kwargs.get("visual_prompt_template")
            or completion_params.get("visual_prompt_template")
            or DEFAULT_VISUAL_PROMPT_TEMPLATE
        )
        _shared_tokenizer = None
        _shared_tokenizer_lock = asyncio.Lock()

        async def _load_shared_tokenizer():
            nonlocal _shared_tokenizer
            if _shared_tokenizer is not None:
                return _shared_tokenizer
            async with _shared_tokenizer_lock:
                if _shared_tokenizer is None:
                    # Load once per rollout invocation so per-row clients share
                    # the same tokenizer instead of re-fetching from HF.
                    _shared_tokenizer = _get_hf_tokenizer(str(tokenizer_name_or_path))
            return _shared_tokenizer

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            start_time = time.perf_counter()
            shared_tokenizer = await _load_shared_tokenizer()

            tool_call_parser = build_frozen_lake_tool_call_parser(
                allow_plaintext_action_fallback=allow_plaintext_action_fallback,
                tokenizer_getter=lambda: shared_tokenizer,
                model_id=model_id,
            )
            text_client = None
            image_client = None
            if observation_mode == "image":
                image_client = FireworksV1ImageCompletionsClient(
                    model_id=model_id,
                    tokenizer_name_or_path=str(tokenizer_name_or_path),
                    api_key=str(api_key) if api_key is not None else None,
                    base_url=str(base_url) if base_url is not None else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_params=request_params,
                    logprobs=self.logprobs,
                    enable_thinking=self.enable_thinking,
                    tool_call_parser=tool_call_parser,
                    default_tools=FROZEN_LAKE_TOOLS,
                )
                image_client._tokenizer = shared_tokenizer
            else:
                text_client = FireworksV1CompletionsClient(
                    model_id=model_id,
                    tokenizer_name_or_path=str(tokenizer_name_or_path),
                    api_key=str(api_key) if api_key is not None else None,
                    base_url=str(base_url) if base_url is not None else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_params=request_params,
                    logprobs=self.logprobs,
                    enable_thinking=self.enable_thinking,
                    tool_call_parser=tool_call_parser,
                    default_tools=FROZEN_LAKE_TOOLS,
                )
                text_client._tokenizer = shared_tokenizer
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            all_prompt_ids: List[int] = []
            all_completion_ids: List[int] = []
            step_rewards: List[float] = []
            tool_call_traces: List[Dict[str, Any]] = []
            model_request_traces: List[Dict[str, Any]] = []
            token_turn_traces: List[Dict[str, Any]] = []
            latest_finish_reason: Optional[str] = None
            latest_raw_output: Optional[Dict[str, Any]] = None
            rollout_error: Optional[str] = None
            assistant_toolcall_prefill_ids: List[int] = []
            assistant_toolcall_prefill_text = ""
            if _is_kimi_tokenizer_name(str(tokenizer_name_or_path)):
                assistant_toolcall_prefill_ids = _build_kimi_toolcall_generation_prefill_token_ids(
                    str(tokenizer_name_or_path)
                )
                assistant_toolcall_prefill_text = _build_kimi_toolcall_generation_prefill_text(
                    str(tokenizer_name_or_path)
                )

            try:
                dataset_info = row.input_metadata.dataset_info or {}
                if not isinstance(dataset_info, dict):
                    dataset_info = {}
                env_context = dataset_info.get("environment_context")
                if not isinstance(env_context, dict):
                    env_context = {}
                row_user_prompt_template = dataset_info.get("user_prompt_template")
                if not isinstance(row_user_prompt_template, str) or not row_user_prompt_template:
                    row_user_prompt_template = default_user_prompt_template
                row_visual_prompt_template = dataset_info.get("visual_prompt_template")
                if not isinstance(row_visual_prompt_template, str) or not row_visual_prompt_template:
                    row_visual_prompt_template = default_visual_prompt

                env = build_frozen_lake_tool_env(environment_context=env_context, max_steps=max_steps)
                state = env.reset()
                observation = str(state.get("observation") or "")
                done = bool(state.get("terminated") or state.get("truncated"))

                messages = list(row.messages)
                if default_system_prompt and not any(m.role == "system" for m in messages):
                    messages.insert(0, Message(role="system", content=default_system_prompt))
                row.tools = row.tools or list(FROZEN_LAKE_TOOLS)
                current_prompt_ids: List[int] = []
                current_images: List[str] = []
                current_prompt_text = ""
                if observation_mode != "image":
                    initial_user_prompt = build_frozen_lake_user_prompt(
                        user_prompt_template=str(row_user_prompt_template) if row_user_prompt_template else None,
                        observation=observation,
                    )
                    messages.append(Message(role="user", content=initial_user_prompt))
                    current_prompt_ids = text_client.build_prompt_token_ids(
                        messages=_to_message_payload(messages=messages),
                        tools=row.tools,
                    )
                else:
                    initial_visual_prompt = _build_visual_user_prompt(
                        prompt_template=str(row_visual_prompt_template) if row_visual_prompt_template else None,
                        observation=observation,
                        default_prompt=str(default_visual_prompt),
                    )
                    initial_image_url = env.render_image_data_url()
                    current_images = [initial_image_url]
                    messages.append(
                        Message(
                            role="user",
                            content=_build_multimodal_image_content(
                                image_url=initial_image_url,
                                text=initial_visual_prompt,
                            ),
                        )
                    )
                    message_payloads = _to_message_payload(messages=messages)
                    current_prompt_ids = image_client.build_prompt_token_ids(
                        messages=message_payloads,
                        tools=row.tools,
                    )
                    current_prompt_text = image_client.build_prompt_text(
                        messages=message_payloads,
                        tools=row.tools,
                    )

                for step_index in range(max_steps):
                    if done:
                        break
                    if getattr(config, "logger", None):
                        try:
                            row_for_log = row.model_copy(deep=True)
                            row_for_log.messages = list(messages)
                            config.logger.log(row_for_log)
                        except Exception:
                            pass

                    base_prompt_ids = list(current_prompt_ids)
                    base_prompt_text = str(current_prompt_text)

                    model_request_traces.append(
                        {
                            "step_index": step_index + 1,
                            "prompt_ids": list(base_prompt_ids),
                            "prompt_token_count": len(base_prompt_ids),
                            "tools": list(row.tools or []),
                            "observation_mode": observation_mode,
                        }
                    )
                    request_prompt_ids = list(base_prompt_ids)
                    request_prompt_text = base_prompt_text
                    if assistant_toolcall_prefill_ids:
                        request_prompt_ids.extend(assistant_toolcall_prefill_ids)
                        if observation_mode == "image":
                            request_prompt_text = f"{request_prompt_text}{assistant_toolcall_prefill_text}"
                        model_request_traces[-1]["assistant_prefill_len"] = len(assistant_toolcall_prefill_ids)
                        model_request_traces[-1]["assistant_prefill_text"] = assistant_toolcall_prefill_text
                    model_request_traces[-1]["logical_prompt_ids"] = list(base_prompt_ids)
                    model_request_traces[-1]["logical_prompt_token_count"] = len(base_prompt_ids)
                    model_request_traces[-1]["prompt_ids"] = list(request_prompt_ids)
                    model_request_traces[-1]["prompt_token_count"] = len(request_prompt_ids)
                    parse_retry_count = 0
                    while True:
                        try:
                            if observation_mode == "image":
                                completion = await image_client.create_completion_from_prompt_ids(
                                    prompt_token_ids=request_prompt_ids,
                                    prompt_text=request_prompt_text,
                                    images=current_images,
                                    tools=row.tools,
                                )
                                model_request_traces[-1]["image_count"] = len(current_images)
                                model_request_traces[-1]["prompt_text"] = request_prompt_text
                            else:
                                completion = await text_client.create_completion_from_prompt_ids(
                                    prompt_token_ids=request_prompt_ids,
                                    tools=row.tools,
                                )
                            break
                        except Exception as exc:
                            if parse_retry_count >= max_parse_retries or not _looks_like_tool_parse_error_message(str(exc)):
                                rollout_error = str(exc)
                                tool_call_traces.append({"step_index": step_index + 1, "error": rollout_error})
                                completion = None
                                break
                            parse_retry_count += 1
                            model_request_traces[-1]["parse_retry_count"] = parse_retry_count
                            model_request_traces[-1].setdefault("parse_retry_errors", []).append(str(exc))
                            await asyncio.sleep(0.25 * parse_retry_count)
                    if completion is None:
                        break

                    usage_payload = completion.get("usage") or {}
                    usage["prompt_tokens"] += int(usage_payload.get("prompt_tokens", 0))
                    usage["completion_tokens"] += int(usage_payload.get("completion_tokens", 0))
                    usage["total_tokens"] += int(usage_payload.get("total_tokens", 0))

                    provider_prompt_ids = [int(x) for x in list(completion.get("prompt_ids") or [])]
                    prompt_ids = [int(x) for x in list(base_prompt_ids)]
                    raw_completion_ids = [int(x) for x in list(completion.get("completion_ids") or [])]
                    if observation_mode == "image":
                        assistant_suffix_ids = image_client.encode_assistant_turn_suffix()
                    else:
                        assistant_suffix_ids = text_client.encode_special_suffix()
                    completion_ids = (
                        [int(x) for x in list(assistant_toolcall_prefill_ids)]
                        + list(raw_completion_ids)
                        + [int(x) for x in list(assistant_suffix_ids)]
                    )
                    completion_text = str(completion.get("completion_text") or "")
                    all_prompt_ids.extend(prompt_ids)
                    all_completion_ids.extend(completion_ids)
                    if provider_prompt_ids and provider_prompt_ids != request_prompt_ids:
                        model_request_traces[-1]["provider_prompt_ids"] = provider_prompt_ids

                    choice = (completion.get("choices") or [{}])[0]
                    message_payload = choice.get("message") or {}
                    latest_finish_reason = str(completion.get("finish_reason") or choice.get("finish_reason") or "")
                    raw_output = completion.get("raw_output")
                    if isinstance(raw_output, dict):
                        latest_raw_output = raw_output

                    assistant_message_payload = {
                        "role": "assistant",
                        "content": completion_text or str(message_payload.get("content", "") or ""),
                        "tool_calls": message_payload.get("tool_calls"),
                    }
                    assistant_message = Message.model_validate(assistant_message_payload)
                    messages.append(assistant_message)

                    try:
                        action, tool_call_id = _resolve_tool_call_action(message_payload.get("tool_calls"))
                        state = env.step(action)
                        reward = float(state.get("reward", 0.0))
                        done = bool(state.get("terminated") or state.get("truncated"))
                        observation = str(state.get("observation") or "")
                    except Exception as exc:
                        rollout_error = str(exc)
                        tool_call_traces.append({"step_index": step_index + 1, "error": rollout_error})
                        break
                    step_rewards.append(reward)
                    turn_trace: Dict[str, Any] = {
                        "step_index": step_index + 1,
                        "prompt_ids": list(prompt_ids),
                        "completion_ids": list(completion_ids),
                        "step_reward": reward,
                        "tool_call_parser": (
                            raw_output.get("tool_call_parser") if isinstance(raw_output, dict) else None
                        ),
                    }
                    if observation_mode == "image":
                        turn_trace["prompt_text"] = base_prompt_text
                        turn_trace["observation_mode"] = "image"
                    completion_logprobs = completion.get("completion_logprobs")
                    if completion_logprobs:
                        turn_trace["completion_logprobs"] = [float(lp) for lp in completion_logprobs]
                    token_turn_traces.append(turn_trace)

                    tool_result = {
                        "action": action,
                        "reward": reward,
                        "terminated": bool(state.get("terminated", False)),
                        "truncated": bool(state.get("truncated", False)),
                        "position": state.get("position"),
                        "row": state.get("row"),
                        "col": state.get("col"),
                        "tile": state.get("tile"),
                        "step_index": state.get("step_index", step_index + 1),
                    }
                    tool_image_url = env.render_image_data_url() if observation_mode == "image" else None
                    tool_message_payload = {
                        "role": "tool",
                        "name": TOOL_NAME_LAKE_MOVE,
                        "tool_call_id": tool_call_id or None,
                        "content": (
                            _build_multimodal_image_content(image_url=tool_image_url)
                            if observation_mode == "image" and tool_image_url
                            else json.dumps(tool_result, separators=(",", ":"))
                        ),
                    }
                    messages.append(Message.model_validate(tool_message_payload))

                    if observation_mode == "image":
                        model_request_traces[-1]["assistant_turn_len"] = len(completion_ids)
                        if not done:
                            assistant_turn_ids = list(completion_ids)
                            assistant_turn_text = (
                                str(assistant_toolcall_prefill_text or "")
                                + str(completion_text or "")
                                + image_client.assistant_turn_suffix_text()
                            )
                            tool_suffix_ids = image_client.build_tool_response_suffix_token_ids(
                                tool_message=tool_message_payload,
                            )
                            tool_suffix_text = image_client.build_tool_response_suffix_text(
                                tool_message=tool_message_payload,
                            )
                            tool_suffix_text = image_client.decode_token_ids(
                                token_ids=tool_suffix_ids
                            )
                            current_prompt_ids = (
                                list(prompt_ids) + list(assistant_turn_ids) + list(tool_suffix_ids)
                            )
                            current_prompt_text = (
                                str(base_prompt_text)
                                + assistant_turn_text
                                + tool_suffix_text
                            )
                            if tool_image_url is not None:
                                current_images = list(current_images) + [tool_image_url]
                            model_request_traces[-1]["tool_suffix_len"] = len(tool_suffix_ids)
                        else:
                            current_prompt_ids = []
                            current_prompt_text = ""
                    else:
                        assistant_turn_ids = list(completion_ids)
                        tool_suffix_ids = text_client.build_tool_response_suffix_token_ids(
                            tool_message=tool_message_payload
                        )
                        current_prompt_ids = list(prompt_ids) + list(assistant_turn_ids) + list(tool_suffix_ids)
                        model_request_traces[-1]["assistant_turn_len"] = len(completion_ids)
                        model_request_traces[-1]["tool_suffix_len"] = len(tool_suffix_ids)
                    tool_call_traces.append(
                        {
                            "step_index": step_index + 1,
                            "tool_call_id": tool_call_id,
                            "tool_name": TOOL_NAME_LAKE_MOVE,
                            "action": action,
                            "reward": reward,
                            "terminated": bool(state.get("terminated", False)),
                            "truncated": bool(state.get("truncated", False)),
                            "position": state.get("position"),
                        }
                    )

                row.messages = messages
                row.execution_metadata.usage = CompletionUsage(
                    prompt_tokens=int(usage["prompt_tokens"]),
                    completion_tokens=int(usage["completion_tokens"]),
                    total_tokens=int(usage["total_tokens"]),
                )
                row.execution_metadata.rollout_duration_seconds = time.perf_counter() - start_time
                row.execution_metadata.finish_reason = latest_finish_reason
                row.execution_metadata.tool_call_count = len(
                    [trace for trace in tool_call_traces if trace.get("tool_name") == TOOL_NAME_LAKE_MOVE]
                )
                row.execution_metadata.raw_output = latest_raw_output

                extra = row.execution_metadata.extra if isinstance(row.execution_metadata.extra, dict) else {}
                extra["prompt_ids"] = list(all_prompt_ids)
                extra["completion_ids"] = list(all_completion_ids)
                extra["step_rewards"] = list(step_rewards)
                extra["tool_call_traces"] = list(tool_call_traces)
                extra["tools_input"] = list(row.tools or [])
                extra["model_request_traces"] = list(model_request_traces)
                extra["token_turn_traces"] = list(token_turn_traces)
                extra["observation_mode"] = observation_mode
                extra["tool_call_generation_mode"] = (
                    "plaintext_fallback" if allow_plaintext_action_fallback else "prompt_only"
                )
                if rollout_error:
                    extra["rollout_error"] = rollout_error
                row.execution_metadata.extra = extra

                if getattr(config, "logger", None):
                    try:
                        config.logger.log(row)
                    except Exception:
                        pass
                return row
            finally:
                if text_client is not None:
                    await text_client.close()
                if image_client is not None:
                    await image_client.close()

        async def _sem_wrapper(target_row: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                return await process_row(target_row)

        return [asyncio.create_task(_sem_wrapper(row)) for row in rows]
