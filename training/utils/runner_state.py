"""Helpers for recipe runner state transitions."""

from __future__ import annotations

from typing import Any

from training.utils.runner import RunnerIO, RunStatus

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
