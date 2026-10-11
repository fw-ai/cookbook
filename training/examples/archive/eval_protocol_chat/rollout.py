"""Eval Protocol chat rollout factory for async RL.

Deprecated: archived example, covered by ``rl/frozen_lake/`` (Eval Protocol rollouts). See ``examples/archive/README.md``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from eval_protocol.models import EvaluationRow, Message, Status
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

from training.examples.archive.eval_protocol_chat.convert import evaluation_row_to_rollout_sample
from training.examples.archive.eval_protocol_chat.reward import grade_row
from training.utils import build_renderer
from training.utils.rl.rollout import RolloutRun

if TYPE_CHECKING:
    from training.recipes.async_rl_loop import RolloutFn, RolloutSetup

logger = logging.getLogger(__name__)


@dataclass
class ChatRolloutConfig:
    """Remote Eval Protocol server settings; bind with ``functools.partial(make_rollout_fn, task=...)``."""

    remote_rollout_base_url: str
    tracing_base_url: str = "https://tracing.fireworks.ai"
    renderer_name: str = ""
    remote_timeout_seconds: float = 1800.0
    remote_poll_interval: float = 1.0


def _chat_api_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/inference"):
        return f"{base}/v1"
    if base.endswith("/v1"):
        return base
    if ".direct.fireworks.ai" in base:
        return f"{base}/v1"
    return f"{base}/inference/v1"


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

    row = EvaluationRow(messages=[Message.model_validate(message) for message in raw_messages])
    row_id = str(sample_prompt.get("id") or uuid.uuid4().hex[:8])
    row.input_metadata.row_id = f"{row_id}-{uuid.uuid4().hex[:8]}"
    row.input_metadata.completion_params = dict(completion_params)
    row.input_metadata.dataset_info = {
        key: value
        for key, value in sample_prompt.items()
        if key != "messages"
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


def make_rollout_fn(setup: "RolloutSetup", *, task: ChatRolloutConfig) -> "RolloutFn":
    processor = RemoteRolloutProcessor(
        remote_base_url=task.remote_rollout_base_url,
        model_base_url=task.tracing_base_url,
        include_payloads=True,
        timeout_seconds=task.remote_timeout_seconds,
        poll_interval=task.remote_poll_interval,
    )
    renderer = build_renderer(setup.tokenizer, setup.tokenizer_id, task.renderer_name)

    run_id = uuid.uuid4().hex
    invocation_id = uuid.uuid4().hex
    experiment_id = uuid.uuid4().hex

    async def rollout_fn(sample_prompt: dict[str, Any]) -> RolloutRun | None:
        completion_params = {
            **dict(setup.sample_kwargs),
            "model": setup.model,
            "base_url": _chat_api_base_url(setup.inference_base_url),
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
            steps=1,
        )

        try:
            completed = await _run_single_row(processor, row, config)
        except Exception:
            logger.exception("Eval Protocol chat rollout failed; dropping row_id=%s", row.input_metadata.row_id)
            return None

        status = completed.rollout_status
        if status and (status.is_error() or status.code in (Status.Code.NOT_FOUND, Status.Code.DEADLINE_EXCEEDED)):
            logger.warning(
                "Eval Protocol chat rollout status error row_id=%s status=%s",
                row.input_metadata.row_id,
                status,
            )
            return None

        try:
            graded = grade_row(completed, sample_prompt)
            sample = evaluation_row_to_rollout_sample(
                graded,
                renderer=renderer,
                tokenizer_id=setup.tokenizer_id,
            )
        except Exception:
            logger.exception("failed to convert Eval Protocol chat row; dropping row_id=%s", row.input_metadata.row_id)
            return None
        if sample is None:
            return None
        return RolloutRun(segments=[sample])

    return rollout_fn
