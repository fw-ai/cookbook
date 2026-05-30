"""Convert Eval Protocol rollout rows into async-RL RolloutSample objects."""

from __future__ import annotations

import logging
from typing import Any

from eval_protocol.models import EvaluationRow

from training.utils.rl.rollout import (
    InferenceCall,
    MessageTrajectoryAssembler,
    PrefixMismatch,
    RolloutSample,
    TITOTokenizer,
    TrajectoryAssembler,
)

logger = logging.getLogger(__name__)


def evaluation_row_to_rollout_sample(
    row: EvaluationRow,
    *,
    tokenizer: Any | None = None,
    tokenizer_id: str | None = None,
) -> RolloutSample | None:
    """Convert one evaluated remote rollout into a token-native sample.

    Preferred input is ``execution_metadata.extra["token_turn_traces"]``:
    a list of per-turn ``prompt_ids``, ``completion_ids`` and
    ``completion_logprobs`` entries.  This preserves the exact token/logprob
    alignment needed by GRPO.
    """
    extra = row.execution_metadata.extra or {}
    traces = extra.get("token_turn_traces") or []
    if traces:
        return _from_token_turn_traces(row, traces, tokenizer_id=tokenizer_id)

    if tokenizer is not None:
        return _from_messages(row, tokenizer=tokenizer)

    logger.warning("remote rollout row missing token_turn_traces and no tokenizer fallback was provided")
    return None


def _reward(row: EvaluationRow) -> float:
    if row.evaluation_result is not None:
        return float(row.evaluation_result.score or 0.0)
    extra = row.execution_metadata.extra or {}
    rewards = extra.get("step_rewards") or []
    if rewards:
        return float(rewards[-1])
    return 0.0


def _finish_reason(row: EvaluationRow, traces: list[Any] | None = None) -> str:
    if row.execution_metadata.finish_reason:
        return str(row.execution_metadata.finish_reason)
    if traces:
        last = traces[-1]
        if isinstance(last, dict) and last.get("finish_reason"):
            return str(last["finish_reason"])
    return "stop"


def _coerce_ints(values: Any) -> list[int]:
    return [int(value) for value in list(values or []) if value is not None]


def _coerce_floats(values: Any) -> list[float]:
    return [float(value) if value is not None else 0.0 for value in list(values or [])]


def _completion_from_payload(payload: dict[str, Any]) -> tuple[list[int], list[float]]:
    completion_ids = _coerce_ints(payload.get("completion_ids") or payload.get("completion_token_ids"))
    completion_logprobs = _coerce_floats(payload.get("completion_logprobs"))
    return completion_ids, completion_logprobs


def _assistant_tokens_from_logprobs(message: Any) -> tuple[list[int], list[float]]:
    content_entries: list[Any] = []
    logprobs = getattr(message, "logprobs", None)
    if isinstance(logprobs, dict):
        content_entries = list(logprobs.get("content") or [])

    token_ids: list[int] = []
    token_logprobs: list[float] = []
    for entry in content_entries:
        if not isinstance(entry, dict):
            continue
        token_id = entry.get("token_id")
        if token_id is None:
            return [], []
        token_ids.append(int(token_id))
        token_logprobs.append(float(entry.get("logprob") or 0.0))
    return token_ids, token_logprobs


def _from_token_turn_traces(
    row: EvaluationRow,
    traces: list[Any],
    *,
    tokenizer_id: str | None,
) -> RolloutSample | None:
    assembler = TrajectoryAssembler(tokenizer_id=tokenizer_id)
    extra = row.execution_metadata.extra or {}
    payload_by_turn: dict[int, dict[str, Any]] = {}
    for payload in list(extra.get("assistant_turn_payloads") or []):
        if not isinstance(payload, dict):
            continue
        turn_index = payload.get("assistant_turn_index")
        if turn_index is None:
            continue
        payload_by_turn[int(turn_index)] = payload
    assistant_messages = row.get_assistant_messages()

    try:
        for idx, trace in enumerate(traces):
            if not isinstance(trace, dict):
                logger.warning("token_turn_traces[%d] is not a dict; dropping sample", idx)
                return None
            prompt_ids = _coerce_ints(trace.get("prompt_ids"))
            if not prompt_ids:
                logger.warning("token trace %d missing prompt_ids; dropping sample", idx)
                return None
            assistant_turn_index = int(trace.get("assistant_turn_index", trace.get("turn", idx)))
            completion_ids, completion_logprobs = _completion_from_payload(trace)
            if (not completion_ids or not completion_logprobs) and assistant_turn_index in payload_by_turn:
                completion_ids, completion_logprobs = _completion_from_payload(payload_by_turn[assistant_turn_index])
            if (not completion_ids or not completion_logprobs) and assistant_turn_index < len(assistant_messages):
                completion_ids, completion_logprobs = _assistant_tokens_from_logprobs(
                    assistant_messages[assistant_turn_index]
                )
            if not completion_ids:
                logger.warning(
                    "token trace %d missing completion token data for assistant turn %d; dropping sample",
                    idx,
                    assistant_turn_index,
                )
                return None
            if len(completion_ids) != len(completion_logprobs):
                logger.warning(
                    "token trace %d completion/logprob length mismatch (%d/%d); dropping sample",
                    idx,
                    len(completion_ids),
                    len(completion_logprobs),
                )
                return None
            assembler.add_call(
                InferenceCall(
                    input_tokens=prompt_ids,
                    output_tokens=completion_ids,
                    output_logprobs=completion_logprobs,
                    finish_reason=str(trace.get("finish_reason") or "stop"),
                )
            )
    except (PrefixMismatch, ValueError) as exc:
        logger.warning("failed to assemble token_turn_traces; dropping sample: %s", exc)
        return None

    tokens, logprobs, loss_mask = assembler.to_flat()
    return _build_sample(
        row,
        tokens=tokens,
        logprobs=logprobs,
        loss_mask=loss_mask,
        finish_reason=_finish_reason(row, traces),
    )


def _message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "dump_mdoel_for_chat_completion_request"):
        return message.dump_mdoel_for_chat_completion_request()
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return {k: v for k, v in message.items() if v is not None}
    return {
        "role": getattr(message, "role", "user"),
        "content": getattr(message, "content", ""),
    }


def _assistant_tokens_and_logprobs(message: Any, tokenizer: Any) -> tuple[list[int], list[float]]:
    content_entries: list[Any] = []
    logprobs = getattr(message, "logprobs", None)
    if isinstance(logprobs, dict):
        content_entries = list(logprobs.get("content") or [])

    token_ids: list[int] = []
    token_logprobs: list[float] = []
    for entry in content_entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("token_id") is not None:
            token_ids.append(int(entry["token_id"]))
        token_logprobs.append(float(entry.get("logprob") or 0.0))

    if token_ids and len(token_ids) == len(token_logprobs):
        return token_ids, token_logprobs

    text = str(getattr(message, "content", "") or "")
    encoded = list(tokenizer.encode(text, add_special_tokens=False))
    if len(encoded) == len(token_logprobs):
        return [int(token_id) for token_id in encoded], token_logprobs

    return [], []


def _from_messages(row: EvaluationRow, *, tokenizer: Any) -> RolloutSample | None:
    messages = [_message_to_dict(message) for message in row.messages]
    if not messages:
        return None

    assembler = MessageTrajectoryAssembler(TITOTokenizer(tokenizer))
    try:
        for idx, message in enumerate(row.messages):
            if getattr(message, "role", None) != "assistant":
                continue
            request_messages = messages[:idx]
            assistant_message = messages[idx]
            completion_ids, completion_logprobs = _assistant_tokens_and_logprobs(message, tokenizer)
            if not completion_ids or len(completion_ids) != len(completion_logprobs):
                logger.warning("assistant message %d missing token/logprob data; dropping sample", idx)
                return None
            prompt_ids = assembler.prepare_next_input(request_messages)
            assembler.add_assistant_response(
                request_messages=request_messages,
                assistant_message=assistant_message,
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion_ids,
                completion_logprobs=completion_logprobs,
                finish_reason=_finish_reason(row),
            )
    except Exception as exc:
        logger.warning("failed to assemble messages fallback; dropping sample: %s", exc)
        return None

    tokens, logprobs, loss_mask = assembler.trajectory.to_flat()
    return _build_sample(
        row,
        tokens=tokens,
        logprobs=logprobs,
        loss_mask=loss_mask,
        finish_reason=_finish_reason(row),
    )


def _build_sample(
    row: EvaluationRow,
    *,
    tokens: list[int],
    logprobs: list[float],
    loss_mask: list[int],
    finish_reason: str,
) -> RolloutSample | None:
    if len(tokens) < 2:
        return None
    if len(tokens) != len(logprobs) or len(tokens) != len(loss_mask):
        logger.warning(
            "token/logprob/loss_mask length mismatch (%d/%d/%d); dropping sample",
            len(tokens),
            len(logprobs),
            len(loss_mask),
        )
        return None
    if not any(loss_mask):
        logger.warning("loss_mask has no generated tokens; dropping sample")
        return None

    extra = row.execution_metadata.extra or {}
    routing_matrices = extra.get("routing_matrices")
    if routing_matrices is not None and len(routing_matrices) != len(tokens):
        logger.warning(
            "routing_matrices length %d does not match token length %d; ignoring router replay",
            len(routing_matrices),
            len(tokens),
        )
        routing_matrices = None

    return RolloutSample(
        tokens=tokens,
        logprobs=logprobs,
        loss_mask=loss_mask,
        reward=_reward(row),
        routing_matrices=routing_matrices,
        finish_reason=finish_reason,
        text=_final_assistant_text(row),
    )


def _final_assistant_text(row: EvaluationRow) -> str:
    for message in reversed(row.messages):
        if getattr(message, "role", None) == "assistant":
            return str(getattr(message, "content", "") or "")
    return ""
