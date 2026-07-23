"""Tests for the independent async RL producer metric stream."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import pytest

from training.utils.rl.async_rl.telemetry import (
    ProducerMetricsReporter,
    ProducerStepMetricsReducer,
    producer_metric_values,
)


def _snapshot(**overrides) -> dict[str, int | float | bool | None]:
    snapshot: dict[str, int | float | bool | None] = {
        "published_version": 0,
        "in_flight_samples": 0,
        "admission_capacity": 0,
        "staleness_capacity": 0,
        "concurrency_capacity": None,
        "rows_submitted": 0,
        "rows_accepted": 0,
        "rows_rejected": 0,
        "completion_refill_attempts": 0,
        "completion_refill_rows_submitted": 0,
        "recoverable_errors": 0,
        "source_exhausted": False,
    }
    snapshot.update(overrides)
    return snapshot


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not reached")
        await asyncio.sleep(0)


def test_metric_values_omit_disabled_concurrency_gate() -> None:
    metrics = producer_metric_values(
        _snapshot(
            published_version=3,
            in_flight_samples=8,
            admission_capacity=16,
            staleness_capacity=16,
            rows_submitted=5,
            rows_accepted=3,
            rows_rejected=1,
            completion_refill_attempts=7,
            completion_refill_rows_submitted=2,
            recoverable_errors=1,
        )
    )

    assert metrics["producer/in_flight_samples"] == 8
    assert metrics["producer/rows_completed_total"] == 4
    assert metrics["producer/completion_refill_attempts_total"] == 7
    assert "producer/refill_attempts_total" not in metrics
    assert "producer/concurrency_capacity_samples" not in metrics


def test_step_reducer_time_weights_gauges_and_omits_disabled_capacity() -> None:
    now = [0.0]
    reducer = ProducerStepMetricsReducer(clock=lambda: now[0])
    reducer.observe(
        _snapshot(
            in_flight_samples=2,
            admission_capacity=6,
            staleness_capacity=8,
        )
    )
    now[0] = 2.0
    reducer.observe(
        _snapshot(
            in_flight_samples=4,
            admission_capacity=4,
            staleness_capacity=6,
        )
    )
    now[0] = 5.0
    metrics = reducer.finish_step(
        _snapshot(
            in_flight_samples=8,
            admission_capacity=0,
            staleness_capacity=2,
        )
    )

    assert metrics["async/in_flight_samples_mean"] == 16 / 5
    assert metrics["async/admission_capacity_samples_mean"] == 24 / 5
    assert metrics["async/staleness_capacity_samples_mean"] == 34 / 5
    assert "async/concurrency_capacity_samples_mean" not in metrics

    now[0] = 7.0
    next_step = reducer.finish_step(
        _snapshot(
            in_flight_samples=1,
            admission_capacity=5,
            staleness_capacity=5,
        )
    )
    assert next_step["async/in_flight_samples_mean"] == 8.0


def test_reports_on_an_independent_event_axis() -> None:
    async def scenario() -> None:
        snapshot = _snapshot(
            admission_capacity=2,
            staleness_capacity=2,
            concurrency_capacity=2,
        )
        records: list[dict] = []
        reporter = ProducerMetricsReporter(
            snapshot_fn=lambda: snapshot,
            metrics_fn=records.append,
            interval_s=0.01,
        )
        reporter.start()
        await _wait_until(lambda: len(records) == 1)
        snapshot["completion_refill_attempts"] = 1
        await _wait_until(lambda: len(records) == 2)
        await reporter.aclose()

        assert [record["producer/event"] for record in records] == [0, 1, 2]
        assert records[1]["producer/completion_refill_attempts_total"] == 1
        assert records[1]["producer/published_version"] == 0

    asyncio.run(scenario())


def test_metrics_failure_is_not_silent() -> None:
    async def scenario() -> None:
        def fail_metrics(_metrics: dict) -> None:
            raise OSError("metrics ledger unavailable")

        reporter = ProducerMetricsReporter(
            snapshot_fn=lambda: _snapshot(source_exhausted=True),
            metrics_fn=fail_metrics,
            interval_s=0.01,
        )
        reporter.start()
        await _wait_until(lambda: reporter.failure is not None)
        with pytest.raises(OSError, match="ledger unavailable"):
            await reporter.aclose()

    asyncio.run(scenario())
