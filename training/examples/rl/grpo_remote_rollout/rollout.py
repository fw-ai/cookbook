"""RemoteRolloutProcessor-backed rollout factory for async GRPO."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from eval_protocol.models import EvaluationRow, Message, Status
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

from training.examples.rl.grpo_remote_rollout.convert import evaluation_row_to_rollout_sample
from training.examples.rl.grpo_remote_rollout.reward import grade_row
from training.recipes.async_rl_loop import RolloutFn, RolloutSetup
from training.utils.rl.rollout import RolloutSample

logger = logging.getLogger(__name__)


def _build_evaluation_row(
    sample_prompt: dict[str, Any],
    *,
    completion_params: dict[str, Any],
    run_id: str,
    invocation_id: str,
    experiment_id: str,
) -> EvaluationRow:
    raw_messages = sample_prompt.get("messages") or []
    if not raw_messages:
        raise ValueError("sample row must include OpenAI-style `messages`")

    row = EvaluationRow(messages=[Message.model_validate(m) for m in raw_messages])
    row_id = str(sample_prompt.get("id") or uuid.uuid4().hex[:8])
    row.input_metadata.row_id = f"{row_id}-{uuid.uuid4().hex[:8]}"
    row.input_metadata.completion_params = dict(completion_params)
    row.input_metadata.dataset_info = {
        key: value
        for key, value in sample_prompt.items()
        if key not in {"messages"}
    }
    row.execution_metadata.rollout_id = uuid.uuid4().hex
    row.execution_metadata.run_id = run_id
    row.execution_metadata.invocation_id = invocation_id
    row.execution_metadata.experiment_id = experiment_id
    return row


async def _run_single_row(
    processor: RemoteRolloutProcessor,
    row: EvaluationRow,
    config: RolloutProcessorConfig,
) -> EvaluationRow:
    tasks = processor([row], config)
    if len(tasks) != 1:
        raise RuntimeError(f"expected one rollout task, got {len(tasks)}")
    result = await tasks[0]
    if not isinstance(result, EvaluationRow):
        raise TypeError(f"RemoteRolloutProcessor returned {type(result).__name__}")
    return result


def make_rollout_fn(setup: RolloutSetup) -> RolloutFn:
    """Build an async rollout_fn using Eval Protocol RemoteRolloutProcessor."""
    extras = dict(setup.extras or {})
    remote_base_url = (
        extras.get("remote_rollout_base_url")
        or os.environ.get("REMOTE_ROLLOUT_BASE_URL")
    )
    if not remote_base_url:
        raise ValueError(
            "Set REMOTE_ROLLOUT_BASE_URL or pass rollout_extras['remote_rollout_base_url']."
        )

    tracing_base_url = (
        extras.get("tracing_base_url")
        or os.environ.get("EP_MODEL_BASE_URL")
        or "https://tracing.fireworks.ai"
    )
    timeout_seconds = float(extras.get("remote_timeout_seconds", 1800.0))
    poll_interval = float(extras.get("remote_poll_interval", 1.0))
    max_turns = int(extras.get("max_turns", 2))

    processor = RemoteRolloutProcessor(
        remote_base_url=str(remote_base_url),
        model_base_url=str(tracing_base_url),
        # Required so trace hydration includes completion payloads with per-token logprobs.
        include_payloads=True,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
    )

    run_id = uuid.uuid4().hex
    invocation_id = uuid.uuid4().hex
    experiment_id = uuid.uuid4().hex

    async def rollout_fn(sample_prompt: dict[str, Any]) -> RolloutSample | None:
        completion_params = {
            **dict(setup.sample_kwargs),
            "model": setup.model,
            "base_url": setup.inference_base_url,
            "tokenizer_model": setup.tokenizer_id,
            "max_turns": max_turns,
            "logprobs": True,
        }
        row = _build_evaluation_row(
            sample_prompt,
            completion_params=completion_params,
            run_id=run_id,
            invocation_id=invocation_id,
            experiment_id=experiment_id,
        )
        config = RolloutProcessorConfig(
            completion_params=completion_params,
            mcp_config_path="",
            semaphore=asyncio.Semaphore(1),
            steps=max_turns,
        )

        try:
            completed = await _run_single_row(processor, row, config)
        except Exception:
            logger.exception("remote rollout failed; dropping sample row_id=%s", row.input_metadata.row_id)
            return None

        status = completed.rollout_status
        if status and (status.is_error() or status.code in (Status.Code.NOT_FOUND, Status.Code.DEADLINE_EXCEEDED)):
            logger.warning(
                "remote rollout status error row_id=%s status=%s",
                row.input_metadata.row_id,
                status,
            )
            return None

        graded = grade_row(completed, sample_prompt)
        return evaluation_row_to_rollout_sample(
            graded,
            tokenizer=setup.tokenizer,
            tokenizer_id=setup.tokenizer_id,
        )

    return rollout_fn
