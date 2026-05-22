"""Helpers for recipe runner state transitions."""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

from training.utils.runner import RunnerIO, RunStatus

logger = logging.getLogger(__name__)

ResourceCallback = Callable[..., None]
"""Callback invoked once remote training resources have been provisioned."""


def notify_resources_ready(
    infra: Any,
    on_resources_ready: ResourceCallback | None,
) -> None:
    """Notify orchestration that training resources are ready.

    Callback failures are logged but never propagated because artifact
    writers should not be able to abort training.
    """
    if on_resources_ready is None:
        return
    try:
        on_resources_ready(
            deployment_id=getattr(infra, "deployment_id", None),
            policy_job_id=getattr(infra, "policy_job_id", None),
            reference_job_id=getattr(infra, "reference_job_id", None),
        )
    except Exception as exc:  # noqa: BLE001 - artifact writer must not abort training
        logger.warning(
            "on_resources_ready callback raised %r; ignoring",
            exc,
            exc_info=True,
        )


def estimate_async_total_steps(
    *,
    step_offset: int,
    total_items: int,
    prior_rows_consumed: int,
    prompt_groups_per_step: int,
    ppo_n_minibatches: int,
) -> int:
    """Estimate async RL optimizer steps for status reporting."""
    remaining = max(0, total_items - prior_rows_consumed)
    rollout_batches = math.ceil(remaining / max(1, prompt_groups_per_step))
    return step_offset + rollout_batches * max(1, ppo_n_minibatches)


def write_running_step(
    runner: RunnerIO,
    *,
    step: int,
    total_steps: int,
    metrics: dict[str, Any],
    tokens: int = 0,
    message: str = "training",
) -> None:
    """Append metrics and refresh running status + metadata."""
    runner.append_metrics(step, metrics, tokens=tokens)
    runner.write_status(
        RunStatus.RUNNING,
        step=step,
        total_steps=total_steps,
        message=message,
    )
    runner.write_metadata()


def start_running(
    runner: RunnerIO,
    *,
    step: int = 0,
    total_steps: int = 0,
    message: str = "training",
) -> None:
    """Start accelerator accounting and mark the run as active."""
    runner.start_training()
    runner.write_status(
        RunStatus.RUNNING,
        step=step,
        total_steps=total_steps,
        message=message,
    )


def write_completed(
    runner: RunnerIO,
    *,
    step: int,
    total_steps: int | None = None,
    message: str = "done",
) -> None:
    """Write successful terminal state and final metadata."""
    runner.write_status(
        RunStatus.COMPLETED,
        step=step,
        total_steps=step if total_steps is None else total_steps,
        message=message,
    )
    runner.write_metadata()
