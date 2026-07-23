"""Acceptance tests for the batch-native asynchronous RL coordinator."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from training.utils.rl.async_rl.errors import (
    CircuitBreakerConfig,
    CircuitBreakerTripped,
    ErrorClassification,
    ErrorDisposition,
    RecoverableCircuitBreaker,
    classify_rollout_error,
)
from training.utils.rl.async_rl import (
    AsyncRLCoordinator,
    OptimizerBatch,
    RolloutRow,
)
from training.utils.rl.async_rl.batch import balanced_chunk_targets
from training.utils.rl.async_rl.telemetry import AsyncRLTelemetry
from training.utils.rl.rollout import RolloutRun, RolloutSample


def _run(awaitable):
    return asyncio.run(awaitable)


def _rollout_run(reward: float = 0.0) -> RolloutRun:
    return RolloutRun(
        segments=[
            RolloutSample(
                tokens=[1, 2],
                logprobs=[0.0, -0.1],
                loss_mask=[0, 1],
                reward=reward,
            )
        ]
    )


def _passthrough_advantages(rewards: list[float]) -> list[float]:
    return list(rewards)


class SamplingRequestError(RuntimeError):
    """Shape of the SDK terminal sampling error across supported SDK versions."""

    __module__ = "fireworks.training.sdk.sampling_observability"

    def __init__(
        self,
        *,
        attempts: int,
        final_status: int | None = None,
        final_error_kind: str | None = None,
    ) -> None:
        super().__init__("sampling failed")
        self.attempts = attempts
        self.final_status = final_status
        self.final_error_kind = final_error_kind


class DeploymentSamplerTimeoutError(SamplingRequestError):
    __module__ = "fireworks.training.sdk.sampling_observability"


def _row(
    row_id: int,
    *,
    reward: float | None = None,
    run_factory: Callable[[int], Awaitable[RolloutRun | None]] | None = None,
    on_resolved: Callable[[str], None] | None = None,
) -> RolloutRow:
    if run_factory is None:

        async def run_factory(_sub_index: int) -> RolloutRun:
            return _rollout_run(float(row_id) if reward is None else reward)

    return RolloutRow(
        row_id=row_id,
        run_factory=run_factory,
        on_resolved=on_resolved,
    )


def _coordinator(
    rows: Iterable[RolloutRow],
    **overrides,
) -> AsyncRLCoordinator:
    options = {
        "rows": rows,
        "completions_per_prompt": 1,
        "prompt_groups_per_step": 2,
        "training_chunks_per_step": 2,
        "max_head_off_policy_versions": 0,
        "advantage_fn": _passthrough_advantages,
    }
    options.update(overrides)
    return AsyncRLCoordinator(**options)


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not reached")
        await asyncio.sleep(0)


async def _consume(batch: OptimizerBatch) -> list[int]:
    sizes = []
    async for chunk in batch.chunks():
        sizes.append(chunk.actual_groups)
    return sizes


def _telemetry() -> AsyncRLTelemetry:
    return AsyncRLTelemetry(
        producer_metrics_fn=lambda _metrics: None,
        step_metrics_fn=lambda _metrics, _step: None,
    )


@asynccontextmanager
async def _observed(
    coordinator: AsyncRLCoordinator,
    telemetry: AsyncRLTelemetry,
):
    async with coordinator:
        telemetry.start(coordinator.snapshot)
        try:
            yield
        finally:
            await telemetry.aclose()


def _finish_step(
    telemetry: AsyncRLTelemetry,
    batch: OptimizerBatch,
    *,
    trained_against_version: int,
) -> dict:
    return telemetry.finish_step(
        batch=batch,
        trained_against_version=trained_against_version,
        prompt_groups=batch.prompt_groups,
        fwd_bwd_results=[],
        optim_result=SimpleNamespace(metrics={}),
        timing_metrics={},
        weight_sync_time=0.0,
        learning_rate=1e-5,
    )


class TestValidation:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"completions_per_prompt": 0}, "completions_per_prompt"),
            ({"prompt_groups_per_step": 0}, "prompt_groups_per_step"),
            ({"training_chunks_per_step": 0}, "training_chunks_per_step"),
            ({"max_head_off_policy_versions": -1}, "max_head_off_policy_versions"),
            ({"global_step": -1}, "global_step"),
            ({"max_concurrent_rollouts": 0}, "max_concurrent_rollouts"),
            (
                {
                    "completions_per_prompt": 2,
                    "max_concurrent_rollouts": 1,
                },
                "whole row",
            ),
            ({"completions_per_prompt": 2, "min_group_size": 3}, "min_group_size"),
        ],
    )
    def test_coordinator_rejects_invalid_values(self, overrides, match):
        with pytest.raises((TypeError, ValueError), match=match):
            _coordinator([], **overrides)

    def test_balanced_chunk_targets_are_fixed_and_nonempty(self):
        assert balanced_chunk_targets(8, 3) == (3, 3, 2)
        assert balanced_chunk_targets(2, 8) == (1, 1)

    def test_row_id_must_be_hashable(self):
        with pytest.raises(TypeError, match="hashable"):
            RolloutRow(row_id=[], run_factory=lambda _index: None)  # type: ignore[arg-type]


def test_completion_refills_during_physical_training() -> None:
    """A retired rollout immediately refills capacity while trainer is blocked."""

    async def scenario() -> None:
        releases = {index: asyncio.Event() for index in range(1, 5)}
        submitted = {index: asyncio.Event() for index in range(5)}

        def make_row(index: int) -> RolloutRow:
            async def factory(_sub_index: int) -> RolloutRun:
                submitted[index].set()
                if index:
                    await releases[index].wait()
                return _rollout_run(float(index))

            return _row(index, run_factory=factory)

        telemetry = _telemetry()
        coordinator = _coordinator(
            (make_row(index) for index in range(5)),
            max_head_off_policy_versions=2,
            max_concurrent_rollouts=3,
        )
        train_started = threading.Event()
        train_release = threading.Event()

        def blocked_train() -> None:
            train_started.set()
            assert train_release.wait(timeout=2.0)

        async with _observed(coordinator, telemetry):
            batch = await asyncio.wait_for(coordinator.next_batch(), timeout=1.0)
            assert batch is not None
            chunks = batch.chunks()
            first = await anext(chunks)
            assert first.index == 0

            await asyncio.wait_for(submitted[3].wait(), timeout=1.0)
            trainer = asyncio.create_task(
                coordinator.run_blocking(
                    "train_chunk",
                    blocked_train,
                    optimizer_batch=batch,
                )
            )
            await _wait_until(train_started.is_set)
            before = coordinator.snapshot()
            releases[1].set()

            await asyncio.wait_for(submitted[4].wait(), timeout=1.0)
            snapshot = coordinator.snapshot()
            assert (
                snapshot["completion_refill_attempts"]
                > before["completion_refill_attempts"]
            )
            assert (
                snapshot["completion_refill_rows_submitted"]
                > before["completion_refill_rows_submitted"]
            )

            train_release.set()
            await trainer
            await chunks.aclose()

    _run(scenario())


def test_completion_refill_attempt_is_visible_when_staleness_gate_is_full() -> None:
    """A completion retry is counted even when it cannot submit another row."""

    async def scenario() -> None:
        releases = {index: asyncio.Event() for index in (2, 3)}
        fifth_submitted = asyncio.Event()

        def make_row(index: int) -> RolloutRow:
            async def factory(_sub_index: int) -> RolloutRun:
                if index == 4:
                    fifth_submitted.set()
                if index in releases:
                    await releases[index].wait()
                return _rollout_run(float(index))

            return _row(index, run_factory=factory)

        telemetry = _telemetry()
        coordinator = _coordinator(
            (make_row(index) for index in range(5)),
            max_head_off_policy_versions=1,
        )
        train_started = threading.Event()
        train_release = threading.Event()

        def blocked_train() -> None:
            train_started.set()
            assert train_release.wait(timeout=2.0)

        async with _observed(coordinator, telemetry):
            batch = await asyncio.wait_for(coordinator.next_batch(), timeout=1.0)
            assert batch is not None
            chunks = batch.chunks()
            await anext(chunks)
            trainer = asyncio.create_task(
                coordinator.run_blocking(
                    "train_chunk",
                    blocked_train,
                    optimizer_batch=batch,
                )
            )
            await _wait_until(train_started.is_set)
            before = coordinator.snapshot()
            releases[2].set()
            await _wait_until(
                lambda: (
                    int(
                        coordinator.snapshot()["completion_refill_attempts"]
                        or 0
                    )
                    > int(before["completion_refill_attempts"] or 0)
                )
            )

            snapshot = coordinator.snapshot()
            assert (
                snapshot["completion_refill_rows_submitted"]
                == before["completion_refill_rows_submitted"]
            )
            assert snapshot["rows_submitted"] == before["rows_submitted"]
            assert snapshot["staleness_capacity"] == 0
            assert not fifth_submitted.is_set()

            train_release.set()
            await trainer
            await chunks.aclose()

    _run(scenario())


def test_staleness_gate_blocks_refill_until_publish() -> None:
    """Strict on-policy admission never submits batch 2 before version 1."""

    async def scenario() -> None:
        third_submitted = asyncio.Event()

        def make_row(index: int) -> RolloutRow:
            async def factory(_sub_index: int) -> RolloutRun:
                if index == 2:
                    third_submitted.set()
                return _rollout_run(float(index))

            return _row(index, run_factory=factory)

        telemetry = _telemetry()
        coordinator = _coordinator(
            (make_row(index) for index in range(3)),
            max_concurrent_rollouts=3,
        )
        async with _observed(coordinator, telemetry):
            batch = await asyncio.wait_for(coordinator.next_batch(), timeout=1.0)
            assert batch is not None
            assert await _consume(batch) == [1, 1]

            await asyncio.sleep(0)
            before = telemetry.snapshot()
            assert before["rows_submitted"] == 2
            assert before["staleness_capacity"] == 0
            assert not third_submitted.is_set()

            published = coordinator.publish(batch)
            await asyncio.wait_for(third_submitted.wait(), timeout=1.0)
            assert published.trained_against_version == 0
            assert coordinator.published_version == 1

    _run(scenario())


def test_later_chunk_queues_while_first_chunk_trains() -> None:
    async def scenario() -> None:
        second_release = asyncio.Event()

        async def second_factory(_sub_index: int) -> RolloutRun:
            await second_release.wait()
            return _rollout_run(1.0)

        telemetry = _telemetry()
        coordinator = _coordinator(
            [_row(0), _row(1, run_factory=second_factory)],
        )
        train_started = threading.Event()
        train_release = threading.Event()

        def blocked_train() -> None:
            train_started.set()
            assert train_release.wait(timeout=2.0)

        async with _observed(coordinator, telemetry):
            batch = await asyncio.wait_for(coordinator.next_batch(), timeout=1.0)
            assert batch is not None
            chunks = batch.chunks()
            await anext(chunks)
            trainer = asyncio.create_task(
                coordinator.run_blocking(
                    "train_chunk",
                    blocked_train,
                    optimizer_batch=batch,
                )
            )
            await _wait_until(train_started.is_set)
            second_release.set()
            await _wait_until(lambda: batch.ready_chunks == 1)

            train_release.set()
            await trainer
            assert (await anext(chunks)).index == 1
            with pytest.raises(StopAsyncIteration):
                await anext(chunks)
            published = coordinator.publish(batch)
            metrics = _finish_step(
                telemetry,
                batch,
                trained_against_version=published.trained_against_version,
            )
            assert metrics["async/realized_training_chunks"] == 2

    _run(scenario())


def test_metrics_failure_does_not_poison_active_batch() -> None:
    async def scenario() -> None:
        release_second = asyncio.Event()

        async def second_factory(_sub_index: int) -> RolloutRun:
            await release_second.wait()
            return _rollout_run(1.0)

        records: list[dict] = []

        def fail_metrics(_metrics: dict) -> None:
            raise OSError("metrics unavailable")

        telemetry = AsyncRLTelemetry(
            producer_metrics_fn=fail_metrics,
            step_metrics_fn=lambda _metrics, _step: records.append(_metrics),
        )
        coordinator = _coordinator(
            [_row(0), _row(1, run_factory=second_factory)],
        )
        async with coordinator:
            telemetry.start(coordinator.snapshot)
            batch = await coordinator.next_batch()
            assert batch is not None
            chunks = batch.chunks()
            await anext(chunks)

            await _wait_until(lambda: telemetry.failure is not None)
            release_second.set()
            assert (await anext(chunks)).index == 1

            with pytest.raises(OSError, match="metrics unavailable"):
                await telemetry.aclose()

    _run(scenario())


def test_one_optimizer_and_hotload_per_full_or_partial_batch() -> None:
    async def scenario() -> None:
        coordinator = _coordinator((_row(index) for index in range(5)))
        calls = {"train": 0, "optimizer": 0, "hotload": 0}
        batch_shapes = []

        def count(name: str) -> None:
            calls[name] += 1

        async with coordinator:
            while (batch := await coordinator.next_batch()) is not None:
                sizes = []
                async for chunk in batch.chunks():
                    sizes.append(chunk.actual_groups)
                    await coordinator.run_blocking(
                        "train_chunk",
                        count,
                        "train",
                        optimizer_batch=batch,
                    )
                await coordinator.run_blocking(
                    "optimizer",
                    count,
                    "optimizer",
                    optimizer_batch=batch,
                )
                await coordinator.run_blocking(
                    "weight_sync",
                    count,
                    "hotload",
                    optimizer_batch=batch,
                )
                batch_shapes.append((sizes, int(batch.partial)))
                coordinator.publish(batch)

            assert calls == {"train": 5, "optimizer": 3, "hotload": 3}
            assert batch_shapes == [([1, 1], 0), ([1, 1], 0), ([1], 1)]
            assert coordinator.global_step == 3

    _run(scenario())


def test_train_wall_time_ends_with_optimizer() -> None:
    async def scenario() -> None:
        telemetry = _telemetry()
        coordinator = _coordinator(
            [_row(0)],
            prompt_groups_per_step=1,
            training_chunks_per_step=1,
        )
        async with _observed(coordinator, telemetry):
            batch = await coordinator.next_batch()
            assert batch is not None
            await _consume(batch)
            await coordinator.run_blocking(
                "train_chunk",
                lambda: None,
                optimizer_batch=batch,
            )
            await coordinator.run_blocking(
                "optimizer",
                lambda: None,
                optimizer_batch=batch,
            )

            await coordinator.run_blocking(
                "weight_sync",
                time.sleep,
                0.01,
                optimizer_batch=batch,
            )
            published = coordinator.publish(batch)
            metrics = _finish_step(
                telemetry,
                batch,
                trained_against_version=published.trained_against_version,
            )
            assert 0 < metrics["perf/train_step_wall_time"] < 0.01

    _run(scenario())


def test_raw_rewards_include_filtered_groups_but_not_failed_rollouts() -> None:
    async def scenario() -> None:
        async def dropped(_sub_index: int) -> None:
            return None

        rows = [
            _row(0),
            _row(1),
            _row(2),
            _row(3, run_factory=dropped),
            _row(4),
            _row(5),
        ]
        telemetry = _telemetry()
        coordinator = _coordinator(
            rows,
            dynamic_filter_fn=lambda group: group.rewards != [1.0],
        )
        async with _observed(coordinator, telemetry):
            first = await coordinator.next_batch()
            assert first is not None
            await _consume(first)
            first_published = coordinator.publish(first)
            first_metrics = _finish_step(
                telemetry,
                first,
                trained_against_version=first_published.trained_against_version,
            )

            second = await coordinator.next_batch()
            assert second is not None
            await _consume(second)
            second_published = coordinator.publish(second)
            second_metrics = _finish_step(
                telemetry,
                second,
                trained_against_version=second_published.trained_against_version,
            )

            assert first_metrics["rollout/raw_reward"] == 1.0
            assert first_metrics["rollout/raw_samples"] == 3
            assert first_metrics["rollout/filtered_samples"] == 2
            assert second_metrics["rollout/raw_reward"] == 4.5
            assert second_metrics["rollout/raw_samples"] == 2

    _run(scenario())


def test_recoverable_rollout_error_drops_and_continues() -> None:
    async def scenario() -> None:
        async def timeout(_sub_index: int) -> RolloutRun:
            raise TimeoutError("transient deployment timeout")

        telemetry = _telemetry()
        coordinator = _coordinator(
            [_row(0, run_factory=timeout), _row(1)],
            prompt_groups_per_step=1,
            training_chunks_per_step=1,
        )
        async with _observed(coordinator, telemetry):
            batch = await asyncio.wait_for(coordinator.next_batch(), timeout=1.0)
            assert batch is not None
            assert await _consume(batch) == [1]
            coordinator.publish(batch)
            assert await coordinator.next_batch() is None
            final = telemetry.final_stats()
            assert final["recoverable_errors"] == 1
            assert final["sample_fails"] == 1

    _run(scenario())


@pytest.mark.parametrize(
    "error",
    [
        SamplingRequestError(
            attempts=3,
            final_status=503,
            final_error_kind="http_status",
        ),
        DeploymentSamplerTimeoutError(
            attempts=3,
            final_status=504,
            final_error_kind="timeout",
        ),
        SamplingRequestError(
            attempts=3,
            final_error_kind="connection",
        ),
    ],
)
def test_sdk_terminal_sampling_errors_are_recoverable(error: BaseException) -> None:
    classification = classify_rollout_error(error)
    assert classification.disposition is ErrorDisposition.RECOVERABLE


def test_sdk_terminal_nonretryable_status_is_fatal() -> None:
    error = SamplingRequestError(
        attempts=1,
        final_status=400,
        final_error_kind="http_status",
    )
    classification = classify_rollout_error(error)
    assert classification.disposition is ErrorDisposition.FATAL


def test_unknown_rollout_error_is_fatal() -> None:
    async def scenario() -> None:
        async def broken(_sub_index: int) -> RolloutRun:
            raise RuntimeError("rollout contract bug")

        coordinator = _coordinator(
            [_row(0, run_factory=broken)],
            prompt_groups_per_step=1,
            training_chunks_per_step=1,
        )
        async with coordinator:
            with pytest.raises(RuntimeError, match="contract bug"):
                await asyncio.wait_for(coordinator.next_batch(), timeout=1.0)

    _run(scenario())


def test_invalid_rollout_return_is_fatal() -> None:
    async def scenario() -> None:
        async def invalid(_sub_index: int):
            return "not-a-rollout-run"

        coordinator = _coordinator(
            [_row(0, run_factory=invalid)],  # type: ignore[arg-type]
            prompt_groups_per_step=1,
            training_chunks_per_step=1,
            error_classifier=lambda _error: ErrorClassification(
                ErrorDisposition.RECOVERABLE,
                "custom_recoverable_label",
            ),
        )
        async with coordinator:
            with pytest.raises(TypeError, match="must return RolloutRun or None"):
                await asyncio.wait_for(coordinator.next_batch(), timeout=1.0)

    _run(scenario())


def test_unexpected_rollout_cancellation_is_fatal() -> None:
    async def scenario() -> None:
        async def cancelled(_sub_index: int) -> RolloutRun:
            raise asyncio.CancelledError

        coordinator = _coordinator(
            [_row(0, run_factory=cancelled)],
            prompt_groups_per_step=1,
            training_chunks_per_step=1,
        )
        async with coordinator:
            with pytest.raises(RuntimeError, match="unexpectedly cancelled"):
                await asyncio.wait_for(coordinator.next_batch(), timeout=1.0)

    _run(scenario())


def test_circuit_breaker_bounds_transient_failures() -> None:
    breaker = RecoverableCircuitBreaker(
        CircuitBreakerConfig(
            max_consecutive_failures=2,
            rolling_window_size=4,
            rolling_min_observations=4,
            max_failure_rate=1.0,
        )
    )
    error = TimeoutError("deployment unavailable")
    classification = classify_rollout_error(error)
    assert classification.disposition is ErrorDisposition.RECOVERABLE
    breaker.record_failure(error, classification)
    with pytest.raises(CircuitBreakerTripped, match="consecutive_failures"):
        breaker.record_failure(error, classification)


def test_accepted_cursor_is_not_durable_before_publish() -> None:
    async def scenario() -> None:
        resolved: list[tuple[int, str]] = []
        rows = [
            _row(
                index,
                on_resolved=lambda reason, index=index: resolved.append(
                    (index, reason)
                ),
            )
            for index in range(2)
        ]
        coordinator = _coordinator(rows)
        async with coordinator:
            batch = await asyncio.wait_for(coordinator.next_batch(), timeout=1.0)
            assert batch is not None
            await _consume(batch)
            assert resolved == []
            published = coordinator.publish(batch)
            assert published.resolved_rows == 2
            assert resolved == [(0, "accepted"), (1, "accepted")]

    _run(scenario())
