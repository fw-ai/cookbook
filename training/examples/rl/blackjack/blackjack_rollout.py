"""Blackjack multi-turn tool-call rollout processor.

Uses the generic :class:`FireworksV1CompletionsClient` from eval-protocol
with a Blackjack-specific tool-call parser.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

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

from training.examples.rl.blackjack.blackjack_env import (
    build_blackjack_tool_env,
    build_blackjack_user_prompt,
)
from training.examples.rl.blackjack.blackjack_schema import (
    BLACKJACK_ACTIONS,
    BLACKJACK_TOOLS,
    TOOL_NAME_BLACKJACK_ACTION,
    normalize_parsed_tool_call,
    parse_first_blackjack_tool_call_with_content,
    parse_tool_call_with_fallback,
)

logger = logging.getLogger(__name__)
_TOKENIZER_CACHE: Dict[str, Any] = {}

DEFAULT_SYSTEM_PROMPT = (
    "You are playing Blackjack against a dealer.\n"
    "Rules:\n"
    "- Number cards are worth their face value. Face cards (J, Q, K) are worth 10. Aces are worth 11 unless that would bust you, in which case they are worth 1.\n"
    "- You lose if your hand exceeds 21 (bust).\n"
    "- After you stick, the dealer draws until reaching 17 or higher.\n"
    "- You win if the dealer busts, or if your sum is higher than the dealer's without busting.\n"
    "Always respond with exactly one tool call and no other text."
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
        return normalized_token_ids[: -len(normalized_suffix_ids)], len(normalized_suffix_ids)
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


def _to_message_payload(messages) -> List[Dict[str, Any]]:
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
    if name != TOOL_NAME_BLACKJACK_ACTION:
        raise ValueError(f"Unexpected tool '{name}', expected '{TOOL_NAME_BLACKJACK_ACTION}'")

    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Tool call.arguments must be an object")

    action = str(arguments.get("action") or "").strip().lower()
    if action not in BLACKJACK_ACTIONS:
        raise ValueError(f"Invalid action '{action}', expected one of {BLACKJACK_ACTIONS}")

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
        "system_prompt", "user_prompt_template",
        "allow_plaintext_action_fallback",
        "natural", "sab",
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
# Blackjack-specific tool-call parser
# ---------------------------------------------------------------------------

def build_blackjack_tool_call_parser(
    *,
    allow_plaintext_action_fallback: bool = False,
    tokenizer_getter=None,
    model_id: str = "",
) -> Any:
    """Return a ``ToolCallParserFn`` for the Blackjack domain."""

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
            request = SimpleNamespace(tools=tools or BLACKJACK_TOOLS)
            sig = inspect.signature(parser.extract_tool_calls)
            if "token_ids" in sig.parameters:
                extracted = parser.extract_tool_calls(
                    model_output=completion_text,
                    request=request,
                    token_ids=completion_token_ids,
                )
            else:
                extracted = parser.extract_tool_calls(
                    model_output=completion_text,
                    request=request,
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
            parsed_tool_call, non_tool_content = parse_first_blackjack_tool_call_with_content(completion_text)
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

class BlackjackToolRolloutProcessor(RolloutProcessor):
    """Rollout processor for strict JSON tool-calling on Blackjack."""

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 128,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        request_params: Optional[Dict[str, Any]] = None,
        allow_plaintext_action_fallback: bool = False,
        logprobs: bool = True,
        enable_thinking: Optional[bool] = False,
        max_parse_retries: int = 0,
        natural: bool = False,
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
        self.max_parse_retries = max_parse_retries
        self.natural = natural
        self.sab = False  # set via completion_params/processor_kwargs

    def __call__(
        self, rows: List[EvaluationRow], config: RolloutProcessorConfig
    ) -> List[asyncio.Task[EvaluationRow]]:
        completion_params = dict(config.completion_params or {})
        processor_kwargs = dict(config.kwargs or {})
        semaphore = config.semaphore
        max_steps = int(config.steps or 20)

        model_id = str(
            completion_params.get("model")
            or processor_kwargs.get("model_id")
            or self.model_id
            or ""
        )
        if not model_id:
            raise ValueError("model id is required for BlackjackToolRolloutProcessor")

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
        temperature = float(
            completion_params.get("temperature", processor_kwargs.get("temperature", self.temperature))
        )
        max_tokens = int(
            completion_params.get("max_tokens", processor_kwargs.get("max_tokens", self.max_tokens))
        )
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
        natural = bool(
            completion_params.get("natural", processor_kwargs.get("natural", self.natural))
        )
        sab = bool(
            completion_params.get("sab", processor_kwargs.get("sab", self.sab))
        )
        max_parse_retries = int(
            completion_params.get("max_parse_retries")
            or processor_kwargs.get("max_parse_retries")
            or self.max_parse_retries
        )
        default_system_prompt = (
            completion_params.get("system_prompt")
            or processor_kwargs.get("system_prompt")
            or self.system_prompt
            or DEFAULT_SYSTEM_PROMPT
        )
        default_user_prompt_template = (
            processor_kwargs.get("user_prompt_template")
            or self.user_prompt_template
            or completion_params.get("user_prompt_template")
            or None
        )

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            start_time = time.perf_counter()

            tool_call_parser = build_blackjack_tool_call_parser(
                allow_plaintext_action_fallback=allow_plaintext_action_fallback,
                tokenizer_getter=lambda: _get_hf_tokenizer(str(tokenizer_name_or_path)),
                model_id=model_id,
            )
            client = FireworksV1CompletionsClient(
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
                default_tools=BLACKJACK_TOOLS,
            )

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

                env = build_blackjack_tool_env(
                    environment_context=env_context,
                    natural=natural,
                    sab=sab,
                    max_steps=max_steps,
                )
                state = env.reset()
                observation = str(state.get("observation") or "")
                done = bool(state.get("terminated") or state.get("truncated"))

                messages = list(row.messages)
                if default_system_prompt and not any(m.role == "system" for m in messages):
                    messages.insert(0, Message(role="system", content=default_system_prompt))
                row.tools = row.tools or list(BLACKJACK_TOOLS)

                initial_user_prompt = build_blackjack_user_prompt(
                    user_prompt_template=str(row_user_prompt_template) if row_user_prompt_template else None,
                    observation=observation,
                )
                messages.append(Message(role="user", content=initial_user_prompt))
                current_prompt_ids = client.build_prompt_token_ids(
                    messages=_to_message_payload(messages),
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

                    model_request_traces.append(
                        {
                            "step_index": step_index + 1,
                            "prompt_ids": list(base_prompt_ids),
                            "prompt_token_count": len(base_prompt_ids),
                            "tools": list(row.tools or []),
                        }
                    )

                    request_prompt_ids = list(base_prompt_ids)
                    if assistant_toolcall_prefill_ids:
                        request_prompt_ids.extend(assistant_toolcall_prefill_ids)
                        model_request_traces[-1]["assistant_prefill_len"] = len(assistant_toolcall_prefill_ids)
                        model_request_traces[-1]["assistant_prefill_text"] = assistant_toolcall_prefill_text
                    model_request_traces[-1]["logical_prompt_ids"] = list(base_prompt_ids)
                    model_request_traces[-1]["logical_prompt_token_count"] = len(base_prompt_ids)
                    model_request_traces[-1]["prompt_ids"] = list(request_prompt_ids)
                    model_request_traces[-1]["prompt_token_count"] = len(request_prompt_ids)

                    parse_retry_count = 0
                    completion = None
                    while True:
                        try:
                            completion = await client.create_completion_from_prompt_ids(
                                prompt_token_ids=request_prompt_ids,
                                tools=row.tools,
                            )
                            break
                        except Exception as exc:
                            if (
                                parse_retry_count >= max_parse_retries
                                or not _looks_like_tool_parse_error_message(str(exc))
                            ):
                                rollout_error = str(exc)
                                tool_call_traces.append({"step_index": step_index + 1, "error": rollout_error})
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

                    prompt_ids = [int(x) for x in list(base_prompt_ids)]
                    raw_completion_ids = [int(x) for x in list(completion.get("completion_ids") or [])]
                    assistant_suffix_ids = client.encode_special_suffix()
                    completion_ids = (
                        [int(x) for x in list(assistant_toolcall_prefill_ids)]
                        + list(raw_completion_ids)
                        + [int(x) for x in list(assistant_suffix_ids)]
                    )
                    completion_text = str(completion.get("completion_text") or "")
                    all_prompt_ids.extend(prompt_ids)
                    all_completion_ids.extend(completion_ids)

                    provider_prompt_ids = [int(x) for x in list(completion.get("prompt_ids") or [])]
                    if provider_prompt_ids and provider_prompt_ids != request_prompt_ids:
                        model_request_traces[-1]["provider_prompt_ids"] = provider_prompt_ids

                    choice = (completion.get("choices") or [{}])[0]
                    message_payload = choice.get("message") or {}
                    latest_finish_reason = str(
                        completion.get("finish_reason") or choice.get("finish_reason") or ""
                    )
                    raw_output = completion.get("raw_output")
                    if isinstance(raw_output, dict):
                        latest_raw_output = raw_output

                    assistant_message_payload = {
                        "role": "assistant",
                        "content": completion_text or str(message_payload.get("content", "") or ""),
                        "tool_calls": message_payload.get("tool_calls"),
                    }
                    messages.append(Message.model_validate(assistant_message_payload))

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
                    completion_logprobs = completion.get("completion_logprobs")
                    if completion_logprobs:
                        turn_trace["completion_logprobs"] = [float(lp) for lp in completion_logprobs]
                    token_turn_traces.append(turn_trace)

                    tool_result = {
                        "action": action,
                        "reward": reward,
                        "terminated": bool(state.get("terminated", False)),
                        "truncated": bool(state.get("truncated", False)),
                        "player_sum": state.get("player_sum"),
                        "dealer_card": state.get("dealer_card"),
                        "usable_ace": state.get("usable_ace"),
                        "player_cards": state.get("player_cards"),
                        "dealer_cards": state.get("dealer_cards"),
                        "step_index": state.get("step_index", step_index + 1),
                    }
                    tool_message_payload = {
                        "role": "tool",
                        "name": TOOL_NAME_BLACKJACK_ACTION,
                        "tool_call_id": tool_call_id or None,
                        "content": json.dumps(tool_result, separators=(",", ":")),
                    }
                    messages.append(Message.model_validate(tool_message_payload))

                    assistant_turn_ids = list(completion_ids)
                    tool_suffix_ids = client.build_tool_response_suffix_token_ids(
                        tool_message=tool_message_payload
                    )
                    current_prompt_ids = list(prompt_ids) + list(assistant_turn_ids) + list(tool_suffix_ids)
                    model_request_traces[-1]["assistant_turn_len"] = len(completion_ids)
                    model_request_traces[-1]["tool_suffix_len"] = len(tool_suffix_ids)

                    tool_call_traces.append(
                        {
                            "step_index": step_index + 1,
                            "tool_call_id": tool_call_id,
                            "tool_name": TOOL_NAME_BLACKJACK_ACTION,
                            "action": action,
                            "reward": reward,
                            "terminated": bool(state.get("terminated", False)),
                            "truncated": bool(state.get("truncated", False)),
                            "player_sum": state.get("player_sum"),
                            "dealer_card": state.get("dealer_card"),
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
                    [t for t in tool_call_traces if t.get("tool_name") == TOOL_NAME_BLACKJACK_ACTION]
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
                await client.close()

        async def _sem_wrapper(target_row: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                return await process_row(target_row)

        return [asyncio.create_task(_sem_wrapper(row)) for row in rows]
