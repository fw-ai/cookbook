"""Multi-hop QA rollout processor (eval-protocol).

The model answers a question by calling ``search(query)`` until it calls
``submit_answer(answer)``. The processor executes each call against the row's
:class:`~training.examples.rl.multihop_qa.search_env.SearchQAEnv` and keeps the
conversation as token ids: the next prompt is the previous prompt, the
completion verbatim, the turn's close and the tool reply. The episode's
``token_turn_traces`` therefore pack into one training sample
(:mod:`training.utils.rl.rollout.turn_traces`) whose turn spans IGPO scores.

Tool calls are parsed by the cookbook renderer, which reads the model's native
format (Qwen3 JSON, Qwen3.5/3.6 XML functions, Kimi sections, ...).
``completion_params`` carry the endpoint: ``model``, ``base_url``,
``api_key``, ``temperature`` and ``max_tokens``; every other key is sent with
the ``/v1/completions`` request.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from eval_protocol.integrations.fireworks_v1_completions_client import (
    FireworksV1CompletionsClient,
)
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

from training.examples.rl.multihop_qa.search_env import (
    MULTIHOP_QA_TOOLS,
    TOOL_NAME_SEARCH,
    TOOL_NAME_SUBMIT,
    build_search_qa_env,
)
from training.utils.rl.rollout.turn_traces import append_turn_close, tool_reply_ids

DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question by searching for "
    "relevant information. Always respond with exactly one tool call "
    "(search or submit_answer) and no additional text."
)
"""Used only for rows without a system message (``prepare_data.py`` adds one)."""

MAX_ATTEMPTS_PER_TURN = 3
"""Resample a turn this many times when its output holds no valid tool call."""


def parse_search_tool_call(
    renderer: Any, completion_ids: list[int]
) -> tuple[str, dict[str, Any]]:
    """The completion's first ``search``/``submit_answer`` call."""
    message, _ = renderer.parse_response(list(completion_ids))
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        raise ValueError("Completion has no parseable tool call.")
    function = tool_calls[0].function
    if function.name not in (TOOL_NAME_SEARCH, TOOL_NAME_SUBMIT):
        raise ValueError(
            f"Unknown tool {function.name!r}; expected {TOOL_NAME_SEARCH!r} "
            f"or {TOOL_NAME_SUBMIT!r}."
        )
    arguments = json.loads(function.arguments or "{}")
    if not isinstance(arguments, dict):
        raise ValueError("Tool call arguments must be a JSON object.")
    return function.name, arguments


class MultiHopQARolloutProcessor(RolloutProcessor):
    """Plays multi-hop QA search episodes; ``config.steps`` caps the tool calls."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        renderer: Any,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        search_top_k: int = 2,
    ):
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.system_prompt = system_prompt
        self.search_top_k = search_top_k

    def __call__(
        self, rows: list[EvaluationRow], config: RolloutProcessorConfig
    ) -> list[asyncio.Task[EvaluationRow]]:
        async def bounded(row: EvaluationRow) -> EvaluationRow:
            async with config.semaphore:
                return await self.play(row, config)

        return [asyncio.create_task(bounded(row)) for row in rows]

    async def play(self, row: EvaluationRow, config: RolloutProcessorConfig) -> EvaluationRow:
        """Play one episode and record its token traces on ``row``."""
        params = dict(config.completion_params or {})
        client = FireworksV1CompletionsClient(
            model_id=params.pop("model"),
            api_key=params.pop("api_key", None),
            base_url=params.pop("base_url", None),
            temperature=float(params.pop("temperature", 1.0)),
            max_tokens=int(params.pop("max_tokens", 512)),
            request_params=params,
            logprobs=True,
            enable_thinking=False,
            default_tools=MULTIHOP_QA_TOOLS,
        )
        client._tokenizer = self.tokenizer
        info = row.input_metadata.dataset_info or {}
        env = build_search_qa_env(
            context=info.get("context") or {},
            ground_truth=str(info.get("ground_truth", "")),
            max_steps=int(config.steps),
            search_top_k=self.search_top_k,
        )
        env.reset()
        messages = list(row.messages)
        if not any(message.role == "system" for message in messages):
            messages.insert(0, Message(role="system", content=self.system_prompt))

        traces: list[dict[str, Any]] = []
        step_rewards: list[float] = []
        finish_reason = None
        error = None
        try:
            prompt_ids = client.build_prompt_token_ids(
                messages=[message.model_dump(exclude_none=True) for message in messages],
                tools=MULTIHOP_QA_TOOLS,
            )
            close_ids = client.encode_special_suffix()
            for _ in range(int(config.steps)):
                completion, text, (name, arguments) = await self._sample_tool_call(
                    client, prompt_ids
                )
                state = env.step(name, arguments)
                traces.append(
                    {
                        "prompt_ids": prompt_ids,
                        "completion_ids": completion["completion_ids"],
                        "completion_logprobs": completion["completion_logprobs"],
                    }
                )
                step_rewards.append(float(state["reward"]))
                finish_reason = completion["finish_reason"]
                call_id = f"call_{uuid.uuid4().hex[:12]}"
                tool_message = {
                    "role": "tool",
                    "name": name,
                    "tool_call_id": call_id,
                    "content": json.dumps(state, ensure_ascii=False, separators=(",", ":")),
                }
                messages.append(
                    Message.model_validate(
                        {
                            "role": "assistant",
                            "content": text,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(arguments)},
                                }
                            ],
                        }
                    )
                )
                messages.append(Message.model_validate(tool_message))
                if state["terminated"] or state["truncated"]:
                    break
                prompt_ids = (
                    prompt_ids
                    + append_turn_close(completion["completion_ids"], close_ids)
                    + tool_reply_ids(self.tokenizer, tool_message, enable_thinking=False)
                )
        except Exception as exc:  # noqa: BLE001 -- a failed turn ends the episode
            error = str(exc)
        finally:
            await client.close()

        row.messages = messages
        row.execution_metadata.finish_reason = finish_reason
        row.execution_metadata.extra = {
            "token_turn_traces": traces,
            "step_rewards": step_rewards,
            **({"rollout_error": error} if error else {}),
        }
        return row

    async def _sample_tool_call(
        self, client: FireworksV1CompletionsClient, prompt_ids: list[int]
    ) -> tuple[dict[str, Any], str, tuple[str, dict[str, Any]]]:
        attempt = 1
        while True:
            completion = await client.create_completion_from_prompt_ids(
                prompt_token_ids=prompt_ids
            )
            text = self.tokenizer.decode(
                completion["completion_ids"], skip_special_tokens=False
            )
            try:
                return completion, text, parse_search_tool_call(
                    self.renderer, completion["completion_ids"]
                )
            except ValueError:
                if attempt == MAX_ATTEMPTS_PER_TURN:
                    raise
                attempt += 1
