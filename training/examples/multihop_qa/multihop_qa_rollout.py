"""Multi-hop QA rollout processor for IGPO training.

Uses :class:`FireworksV1CompletionsClient` from eval-protocol for tokenised
completions with logprobs, combined with a local :class:`SearchQAEnv` for
TF-IDF paragraph retrieval.  Text-only (no image mode).

The processor records ``token_turn_traces`` and ``model_request_traces`` in
the same format as ``FrozenLakeToolRolloutProcessor``, so
:func:`compute_model_output_spans` works without modification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

from training.examples.multihop_qa.search_env import (
    MULTIHOP_QA_TOOLS,
    TOOL_NAME_SEARCH,
    TOOL_NAME_SUBMIT,
    SearchQAEnv,
    build_search_qa_env,
)

logger = logging.getLogger(__name__)
_TOKENIZER_CACHE: Dict[str, Any] = {}


def _get_hf_tokenizer(tokenizer_name_or_path: str):
    tok = _TOKENIZER_CACHE.get(tokenizer_name_or_path)
    if tok is not None:
        return tok
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    _TOKENIZER_CACHE[tokenizer_name_or_path] = tok
    return tok


def _to_message_payload(messages: Iterable[Message]) -> List[Dict[str, Any]]:
    return [m.model_dump(exclude_none=True) for m in messages]


def _resolve_tool_call(tool_calls: Any) -> Tuple[str, Dict[str, Any], str]:
    """Extract (tool_name, arguments, tool_call_id) from a parsed tool_calls list."""
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError("Assistant response must include at least one tool call")

    first = tool_calls[0]
    if not isinstance(first, dict):
        raise ValueError("Tool call must be an object")

    function = first.get("function")
    if not isinstance(function, dict):
        raise ValueError("Tool call.function must be an object")

    name = str(function.get("name", "")).strip().lower()
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        arguments = {}

    tool_call_id = str(first.get("id") or f"toolcall_{uuid.uuid4().hex[:12]}")
    return name, arguments, tool_call_id


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_TAIL_RE = re.compile(r"<think>.*", re.DOTALL)


def _strip_think_block(text: str) -> str:
    """Remove ``<think>...</think>`` blocks and truncated ``<think>...`` tails.

    Models like Qwen3 emit reasoning traces wrapped in ``<think>`` tags.
    These consume token budget and pollute the prompt context for subsequent
    turns.  Stripping them keeps the conversation compact and avoids
    ``prompt is too long`` errors.  Adjust or disable this for models
    that do not produce thinking tokens.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_TAIL_RE.sub("", text)
    return text.strip()


def _build_tool_call_parser():
    """Build a tool call parser for search/submit_answer tools."""
    valid_names = {TOOL_NAME_SEARCH, TOOL_NAME_SUBMIT}

    def parser(output_text: str, token_ids=None, active_tools=None, **kwargs):
        stripped = _strip_think_block(output_text).strip()

        tc_start = stripped.find("<tool_call>")
        if tc_start >= 0:
            tc_end = stripped.find("</tool_call>", tc_start)
            inner = stripped[tc_start + len("<tool_call>"):tc_end] if tc_end > 0 else stripped[tc_start + len("<tool_call>"):]
            inner = inner.strip()
            brace = inner.find("{")
            if brace >= 0:
                stripped = inner[brace:]

        start = stripped.find("{")
        if start < 0:
            raise ValueError("No JSON object found in model output")
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(stripped[start:])
        if not isinstance(obj, dict):
            raise ValueError("Expected JSON object")

        tool_calls = obj.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            first = tool_calls[0]
            name = str(first.get("name", "")).strip().lower()
            arguments = first.get("arguments", {})
        else:
            name = str(obj.get("name", "")).strip().lower()
            arguments = obj.get("arguments", {})

        if not name or name not in valid_names:
            raise ValueError(f"Unknown tool '{name}'. Expected one of {valid_names}")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        non_json_prefix = stripped[:start].strip() if start > 0 else ""

        return {
            "parsed_tool_call": ParsedToolCall(
                tool_call_id=f"toolcall_{uuid.uuid4().hex[:12]}",
                name=name,
                arguments=arguments if isinstance(arguments, dict) else {},
            ),
            "assistant_content": non_json_prefix,
            "parser": "multihop_qa",
        }

    return parser


# ---------------------------------------------------------------------------
# Rollout Processor
# ---------------------------------------------------------------------------


class MultiHopQARolloutProcessor(RolloutProcessor):
    """Rollout processor for multi-hop QA with search and answer submission tools."""

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
        request_params: Optional[Dict[str, Any]] = None,
        logprobs: bool = True,
        enable_thinking: Optional[bool] = False,  # Qwen3 thinking mode; not all API versions accept this
        search_top_k: int = 2,
        turn_callback: Optional[Any] = None,
    ):
        self.model_id = model_id
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.request_params = dict(request_params or {})
        self.logprobs = logprobs
        self.enable_thinking = enable_thinking
        self.search_top_k = search_top_k
        self.turn_callback = turn_callback

    def __call__(
        self, rows: List[EvaluationRow], config: RolloutProcessorConfig
    ) -> List[asyncio.Task[EvaluationRow]]:
        completion_params = dict(config.completion_params or {})
        processor_kwargs = dict(config.kwargs or {})
        semaphore = config.semaphore
        max_steps = int(config.steps or 8)

        model_id = str(
            completion_params.get("model")
            or processor_kwargs.get("model_id")
            or self.model_id
            or ""
        )
        if not model_id:
            raise ValueError("model id is required for MultiHopQARolloutProcessor")

        tokenizer_name_or_path = (
            completion_params.get("tokenizer_name_or_path")
            or processor_kwargs.get("tokenizer_name_or_path")
            or self.tokenizer_name_or_path
            or model_id
        )
        api_key = (
            completion_params.get("api_key")
            or processor_kwargs.get("api_key")
            or self.api_key
        )
        base_url = (
            completion_params.get("base_url")
            or processor_kwargs.get("base_url")
            or self.base_url
        )
        temperature = float(
            completion_params.get(
                "temperature",
                processor_kwargs.get("temperature", self.temperature),
            )
        )
        max_tokens = int(
            completion_params.get(
                "max_tokens",
                processor_kwargs.get("max_tokens", self.max_tokens),
            )
        )
        search_top_k = int(
            processor_kwargs.get("search_top_k", self.search_top_k)
        )
        turn_callback = processor_kwargs.get("turn_callback") or self.turn_callback

        default_system_prompt = (
            completion_params.get("system_prompt")
            or processor_kwargs.get("system_prompt")
            or self.system_prompt
            or (
                "You are a research assistant. Answer the question by searching "
                "for relevant information. Always respond with exactly one tool "
                "call (search or submit_answer) and no additional text."
            )
        )

        _shared_tokenizer = None
        _shared_tokenizer_lock = asyncio.Lock()

        async def _load_shared_tokenizer():
            nonlocal _shared_tokenizer
            if _shared_tokenizer is not None:
                return _shared_tokenizer
            async with _shared_tokenizer_lock:
                if _shared_tokenizer is None:
                    _shared_tokenizer = _get_hf_tokenizer(str(tokenizer_name_or_path))
            return _shared_tokenizer

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            start_time = time.perf_counter()
            shared_tokenizer = await _load_shared_tokenizer()
            tool_call_parser = _build_tool_call_parser()

            text_client = FireworksV1CompletionsClient(
                model_id=model_id,
                tokenizer_name_or_path=str(tokenizer_name_or_path),
                api_key=str(api_key) if api_key is not None else None,
                base_url=str(base_url) if base_url is not None else None,
                temperature=temperature,
                max_tokens=max_tokens,
                request_params=dict(self.request_params),
                logprobs=self.logprobs,
                enable_thinking=self.enable_thinking,
                tool_call_parser=tool_call_parser,
                default_tools=MULTIHOP_QA_TOOLS,
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

            try:
                dataset_info = row.input_metadata.dataset_info or {}
                if not isinstance(dataset_info, dict):
                    dataset_info = {}

                context = dataset_info.get("context") or {}
                ground_truth = str(dataset_info.get("ground_truth", ""))
                question = str(dataset_info.get("question", ""))

                env = build_search_qa_env(
                    context=context,
                    ground_truth=ground_truth,
                    max_steps=max_steps,
                    search_top_k=search_top_k,
                )
                env.reset()

                messages = list(row.messages)
                if default_system_prompt and not any(m.role == "system" for m in messages):
                    messages.insert(0, Message(role="system", content=default_system_prompt))

                row.tools = row.tools or list(MULTIHOP_QA_TOOLS)

                current_prompt_ids: List[int] = text_client.build_prompt_token_ids(
                    messages=_to_message_payload(messages),
                    tools=row.tools,
                )

                _MAX_PARSE_RETRIES = 3
                done = False
                for step_index in range(max_steps):
                    if done:
                        break

                    prompt_ids = list(current_prompt_ids)

                    model_request_traces.append({
                        "step_index": step_index + 1,
                        "prompt_ids": list(prompt_ids),
                        "prompt_token_count": len(prompt_ids),
                        "tools": list(row.tools or []),
                    })

                    completion = None
                    parse_ok = False
                    last_parse_error = ""
                    for _retry in range(_MAX_PARSE_RETRIES):
                        try:
                            completion = await text_client.create_completion_from_prompt_ids(
                                prompt_token_ids=list(prompt_ids),
                                tools=row.tools,
                            )
                        except Exception as exc:
                            last_parse_error = str(exc)
                            continue

                        choice = (completion.get("choices") or [{}])[0]
                        message_payload = choice.get("message") or {}
                        try:
                            tool_name, arguments, tool_call_id = _resolve_tool_call(
                                message_payload.get("tool_calls")
                            )
                            parse_ok = True
                            break
                        except Exception as exc:
                            last_parse_error = str(exc)
                            completion = None

                    if not parse_ok:
                        rollout_error = last_parse_error
                        tool_call_traces.append({
                            "step_index": step_index + 1,
                            "error": rollout_error,
                        })
                        break

                    usage_payload = completion.get("usage") or {}
                    usage["prompt_tokens"] += int(usage_payload.get("prompt_tokens", 0))
                    usage["completion_tokens"] += int(usage_payload.get("completion_tokens", 0))
                    usage["total_tokens"] += int(usage_payload.get("total_tokens", 0))

                    raw_completion_ids = [
                        int(x) for x in list(completion.get("completion_ids") or [])
                    ]
                    assistant_suffix_ids = text_client.encode_special_suffix()
                    completion_ids = list(raw_completion_ids) + [
                        int(x) for x in list(assistant_suffix_ids)
                    ]
                    completion_text = shared_tokenizer.decode(
                        raw_completion_ids, skip_special_tokens=False,
                    ) if raw_completion_ids else str(completion.get("completion_text") or "")
                    clean_completion_text = _strip_think_block(completion_text)

                    all_prompt_ids.extend(prompt_ids)
                    all_completion_ids.extend(completion_ids)

                    latest_finish_reason = str(
                        completion.get("finish_reason")
                        or choice.get("finish_reason")
                        or ""
                    )
                    raw_output = completion.get("raw_output")
                    if isinstance(raw_output, dict):
                        latest_raw_output = raw_output

                    assistant_message_payload = {
                        "role": "assistant",
                        "content": clean_completion_text or str(
                            _strip_think_block(message_payload.get("content", "") or "")
                        ),
                        "tool_calls": message_payload.get("tool_calls"),
                    }
                    messages.append(Message.model_validate(assistant_message_payload))

                    try:
                        state = env.step(tool_name, arguments)
                        reward = float(state.get("reward", 0.0))
                        done = bool(
                            state.get("terminated") or state.get("truncated")
                        )
                    except Exception as exc:
                        rollout_error = str(exc)
                        tool_call_traces.append({
                            "step_index": step_index + 1,
                            "error": rollout_error,
                        })
                        break

                    step_rewards.append(reward)

                    turn_trace: Dict[str, Any] = {
                        "step_index": step_index + 1,
                        "prompt_ids": list(prompt_ids),
                        "completion_ids": list(completion_ids),
                        "step_reward": reward,
                    }
                    completion_logprobs = completion.get("completion_logprobs")
                    if completion_logprobs:
                        turn_trace["completion_logprobs"] = [
                            float(lp) for lp in completion_logprobs
                        ]
                    token_turn_traces.append(turn_trace)

                    tool_result_content = json.dumps(
                        state, separators=(",", ":"), ensure_ascii=False
                    )
                    tool_message_payload = {
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": tool_call_id or None,
                        "content": tool_result_content,
                    }
                    messages.append(Message.model_validate(tool_message_payload))

                    tool_call_traces.append({
                        "step_index": step_index + 1,
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "reward": reward,
                        "terminated": bool(state.get("terminated", False)),
                        "truncated": bool(state.get("truncated", False)),
                    })

                    tool_suffix_ids = text_client.build_tool_response_suffix_token_ids(
                        tool_message=tool_message_payload
                    )
                    clean_assistant_ids = shared_tokenizer.encode(
                        clean_completion_text, add_special_tokens=False
                    )
                    clean_turn_ids = clean_assistant_ids + [
                        int(x) for x in list(assistant_suffix_ids)
                    ]
                    current_prompt_ids = (
                        list(prompt_ids)
                        + list(clean_turn_ids)
                        + list(tool_suffix_ids)
                    )
                    model_request_traces[-1]["assistant_turn_len"] = len(
                        completion_ids
                    )
                    model_request_traces[-1]["tool_suffix_len"] = len(
                        tool_suffix_ids
                    )

                    if turn_callback is not None:
                        cb_prefix = list(current_prompt_ids)
                        await turn_callback(
                            row_id=row.input_metadata.row_id,
                            prefix_tokens=cb_prefix,
                            step_index=step_index,
                            done=done,
                        )

                row.messages = messages
                row.execution_metadata.usage = CompletionUsage(
                    prompt_tokens=int(usage["prompt_tokens"]),
                    completion_tokens=int(usage["completion_tokens"]),
                    total_tokens=int(usage["total_tokens"]),
                )
                row.execution_metadata.rollout_duration_seconds = (
                    time.perf_counter() - start_time
                )
                row.execution_metadata.finish_reason = latest_finish_reason
                row.execution_metadata.tool_call_count = len(
                    [t for t in tool_call_traces if t.get("tool_name")]
                )
                row.execution_metadata.raw_output = latest_raw_output

                extra = (
                    row.execution_metadata.extra
                    if isinstance(row.execution_metadata.extra, dict)
                    else {}
                )
                extra["prompt_ids"] = list(all_prompt_ids)
                extra["completion_ids"] = list(all_completion_ids)
                extra["step_rewards"] = list(step_rewards)
                extra["tool_call_traces"] = list(tool_call_traces)
                extra["tools_input"] = list(row.tools or [])
                extra["model_request_traces"] = list(model_request_traces)
                extra["token_turn_traces"] = list(token_turn_traces)
                extra["ground_truth"] = ground_truth
                if rollout_error:
                    extra["rollout_error"] = rollout_error
                row.execution_metadata.extra = extra

                return row
            finally:
                if text_client is not None:
                    await text_client.close()

        async def _sem_wrapper(target_row: EvaluationRow) -> EvaluationRow:
            async with semaphore:
                return await process_row(target_row)

        return [asyncio.create_task(_sem_wrapper(row)) for row in rows]
