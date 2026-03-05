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

from openai.types import CompletionUsage

from eval_protocol.integrations.fireworks_v1_completions_client import (
    FireworksV1CompletionsClient,
    ParsedToolCall,
    strip_chat_special_tokens,
    to_openai_tool_calls,
)
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

from training.examples.frozen_lake.frozen_lake_env import (
    build_frozen_lake_tool_env,
    build_frozen_lake_user_prompt,
)
from training.examples.frozen_lake.frozen_lake_schema import (
    FROZEN_LAKE_ACTIONS,
    FROZEN_LAKE_TOOLS,
    TOOL_NAME_LAKE_MOVE,
    normalize_parsed_tool_call,
    parse_first_frozen_lake_tool_call,
    parse_first_frozen_lake_tool_call_with_content,
    parse_tool_call_with_fallback,
)

logger = logging.getLogger(__name__)


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
        "system_prompt", "user_prompt_template",
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
        default_system_prompt = (
            completion_params.get("system_prompt")
            or processor_kwargs.get("system_prompt")
            or self.system_prompt
            or (
                "/no_think\n"
                "You are an RL policy for FrozenLake.\n"
                "Pick the action that moves toward G while avoiding H.\n"
                "Always respond with exactly one tool call, no text."
            )
        )
        default_user_prompt_template = (
            processor_kwargs.get("user_prompt_template")
            or self.user_prompt_template
            or completion_params.get("user_prompt_template")
        )

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            start_time = time.perf_counter()

            tool_call_parser = build_frozen_lake_tool_call_parser(
                allow_plaintext_action_fallback=allow_plaintext_action_fallback,
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
                default_tools=FROZEN_LAKE_TOOLS,
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

                env = build_frozen_lake_tool_env(environment_context=env_context, max_steps=max_steps)
                state = env.reset()
                observation = str(state.get("observation") or "")
                done = bool(state.get("terminated") or state.get("truncated"))

                messages = list(row.messages)
                if default_system_prompt and not any(m.role == "system" for m in messages):
                    messages.insert(0, Message(role="system", content=default_system_prompt))
                row.tools = row.tools or list(FROZEN_LAKE_TOOLS)
                initial_user_prompt = build_frozen_lake_user_prompt(
                    user_prompt_template=str(row_user_prompt_template) if row_user_prompt_template else None,
                    observation=observation,
                )
                messages.append(Message(role="user", content=initial_user_prompt))
                current_prompt_ids = client.build_prompt_token_ids(
                    messages=_to_message_payload(messages=messages),
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

                    model_request_traces.append(
                        {
                            "step_index": step_index + 1,
                            "prompt_ids": list(current_prompt_ids),
                            "prompt_token_count": len(current_prompt_ids),
                            "tools": list(row.tools or []),
                        }
                    )
                    try:
                        completion = await client.create_completion_from_prompt_ids(
                            prompt_token_ids=current_prompt_ids,
                            tools=row.tools,
                        )
                    except Exception as exc:
                        rollout_error = str(exc)
                        tool_call_traces.append({"step_index": step_index + 1, "error": rollout_error})
                        break

                    usage_payload = completion.get("usage") or {}
                    usage["prompt_tokens"] += int(usage_payload.get("prompt_tokens", 0))
                    usage["completion_tokens"] += int(usage_payload.get("completion_tokens", 0))
                    usage["total_tokens"] += int(usage_payload.get("total_tokens", 0))

                    prompt_ids = [int(x) for x in list(completion.get("prompt_ids") or current_prompt_ids)]
                    completion_ids = [int(x) for x in list(completion.get("completion_ids") or [])]
                    all_prompt_ids.extend(prompt_ids)
                    all_completion_ids.extend(completion_ids)

                    choice = (completion.get("choices") or [{}])[0]
                    message_payload = choice.get("message") or {}
                    latest_finish_reason = str(completion.get("finish_reason") or choice.get("finish_reason") or "")
                    raw_output = completion.get("raw_output")
                    if isinstance(raw_output, dict):
                        latest_raw_output = raw_output

                    assistant_message_payload = {
                        "role": "assistant",
                        "content": message_payload.get("content", ""),
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
                    completion_logprobs = completion.get("completion_logprobs")
                    if completion_logprobs:
                        turn_trace["completion_logprobs"] = [float(lp) for lp in completion_logprobs]
                    token_turn_traces.append(turn_trace)

                    tool_result = {
                        "observation": observation,
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
                    tool_message_payload = {
                        "role": "tool",
                        "name": TOOL_NAME_LAKE_MOVE,
                        "tool_call_id": tool_call_id or None,
                        "content": json.dumps(tool_result, separators=(",", ":")),
                    }
                    messages.append(Message.model_validate(tool_message_payload))

                    im_end_ids = client.encode_special_suffix()
                    assistant_turn_ids = list(completion_ids) + im_end_ids
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
