"""Convert Eval Protocol chat rows into async RL RolloutSample objects."""

from __future__ import annotations

import logging
from typing import Any

from eval_protocol.models import EvaluationRow, Message

from training.utils import normalize_messages
from training.utils.rl.rollout import (
    InferenceCall,
    PrefixMismatch,
    RolloutSample,
    TrajectoryAssembler,
    model_input_to_token_ids,
)

logger = logging.getLogger(__name__)


def evaluation_row_to_rollout_sample(
    row: EvaluationRow,
    *,
    renderer: Any,
    tokenizer_id: str | None = None,
) -> RolloutSample | None:
    """Pack a completed Eval Protocol chat rollout for async RL training.

    This example intentionally converts the hydrated chat transcript after the
    fact with the cookbook renderer. Production RL integrations should prefer
    prompt/completion token ids emitted by the generation path itself.
    """
    messages = [_message_to_dict(message) for message in row.messages]
    assistant_indices = [
        idx for idx, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        logger.warning("completed row has no assistant messages; dropping sample")
        return None

    payload_by_turn = _assistant_payloads_by_turn(row)
    assembler = TrajectoryAssembler(tokenizer_id=tokenizer_id)
    try:
        for turn_index, message_index in enumerate(assistant_indices):
            prompt_ids = _render_prompt_ids(renderer, messages[:message_index])
            completion_ids, completion_logprobs = _completion_for_turn(payload_by_turn, turn_index)
            if not prompt_ids or not completion_ids:
                logger.warning("assistant turn %d missing prompt/completion ids; dropping sample", turn_index)
                return None
            if len(completion_ids) != len(completion_logprobs):
                logger.warning(
                    "assistant turn %d completion/logprob length mismatch (%d/%d); dropping sample",
                    turn_index,
                    len(completion_ids),
                    len(completion_logprobs),
                )
                return None

            assembler.add_call(
                InferenceCall(
                    input_tokens=prompt_ids,
                    output_tokens=completion_ids,
                    output_logprobs=completion_logprobs,
                    finish_reason=_finish_reason(row),
                )
            )
    except (PrefixMismatch, ValueError) as exc:
        logger.warning("failed to assemble renderer-backed chat rollout; dropping sample: %s", exc)
        return None

    tokens, logprobs, loss_mask = assembler.to_flat()
    return _build_sample(row, tokens=tokens, logprobs=logprobs, loss_mask=loss_mask)


def _message_to_dict(message: Message | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, Message):
        return message.dump_mdoel_for_chat_completion_request()
    return {k: v for k, v in dict(message).items() if v is not None}


def _render_prompt_ids(renderer: Any, messages: list[dict[str, Any]]) -> list[int]:
    model_input = renderer.build_generation_prompt(normalize_messages(messages))
    return model_input_to_token_ids(model_input)


def _assistant_payloads_by_turn(row: EvaluationRow) -> dict[int, dict[str, Any]]:
    payloads: dict[int, dict[str, Any]] = {}
    extra = row.execution_metadata.extra or {}
    for idx, payload in enumerate(extra.get("assistant_turn_payloads") or []):
        if not isinstance(payload, dict):
            continue
        try:
            turn_index = int(payload.get("assistant_turn_index", idx))
        except (TypeError, ValueError):
            logger.warning("assistant_turn_payloads[%d] has invalid assistant_turn_index; ignoring", idx)
            continue
        payloads[turn_index] = payload
    return payloads


def _completion_for_turn(
    payload_by_turn: dict[int, dict[str, Any]],
    turn_index: int,
) -> tuple[list[int], list[float]]:
    payload = payload_by_turn.get(turn_index) or {}
    return _completion_from_payload(payload)


def _completion_from_payload(payload: dict[str, Any]) -> tuple[list[int], list[float]]:
    ids = _coerce_ints(payload.get("completion_ids") or payload.get("completion_token_ids"))
    logprobs = _coerce_floats(payload.get("completion_logprobs"))
    return ids, logprobs


def _coerce_ints(values: Any) -> list[int]:
    return [int(value) for value in list(values or []) if value is not None]


def _coerce_floats(values: Any) -> list[float]:
    return [float(value) if value is not None else 0.0 for value in list(values or [])]


def _reward(row: EvaluationRow) -> float:
    if row.evaluation_result is not None:
        return float(row.evaluation_result.score or 0.0)
    return 0.0


def _finish_reason(row: EvaluationRow) -> str:
    if row.execution_metadata.finish_reason:
        return str(row.execution_metadata.finish_reason)
    return "stop"


def _final_assistant_text(row: EvaluationRow) -> str:
    for message in reversed(row.messages):
        if getattr(message, "role", None) == "assistant":
            return str(getattr(message, "content", "") or "")
    return ""


def _build_sample(
    row: EvaluationRow,
    *,
    tokens: list[int],
    logprobs: list[float],
    loss_mask: list[int],
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

    return RolloutSample(
        tokens=tokens,
        logprobs=logprobs,
        loss_mask=loss_mask,
        reward=_reward(row),
        finish_reason=_finish_reason(row),
        text=_final_assistant_text(row),
    )
