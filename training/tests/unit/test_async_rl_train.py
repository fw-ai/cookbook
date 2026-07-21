"""Acceptance tests for the batch-native asynchronous RL coordinator."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Iterable

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
    run_async_rl_lifecycle,
)
from training.utils.rl.async_rl.batch import balanced_chunk_targets
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

        async with coordinator:
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
            releases[1].set()

            await asyncio.wait_for(submitted[4].wait(), timeout=1.0)
            snapshot = coordinator.snapshot()
            assert snapshot["completion_refills_during_train"] >= 1

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

        coordinator = _coordinator(
            (make_row(index) for index in range(3)),
            max_concurrent_rollouts=3,
        )
        async with coordinator:
            batch = await asyncio.wait_for(coordinator.next_batch(), timeout=1.0)
            assert batch is not None
            assert await _consume(batch) == [1, 1]

            await asyncio.sleep(0)
            before = coordinator.snapshot()
            assert before["rows_submitted"] == 2
            assert before["staleness_capacity"] == 0
            assert not third_submitted.is_set()

            metrics = coordinator.publish(batch)
            await asyncio.wait_for(third_submitted.wait(), timeout=1.0)
            assert metrics["ctx/current_version"] == 0
            assert coordinator.published_version == 1

    _run(scenario())


def test_later_chunk_queues_while_first_chunk_trains() -> None:
    async def scenario() -> None:
        second_release = asyncio.Event()

        async def second_factory(_sub_index: int) -> RolloutRun:
            await second_release.wait()
            return _rollout_run(1.0)

        coordinator = _coordinator([_row(0), _row(1, run_factory=second_factory)])
        train_started = threading.Event()
        train_release = threading.Event()

        def blocked_train() -> None:
            train_started.set()
            assert train_release.wait(timeout=2.0)

        async with coordinator:
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
            metrics = coordinator.publish(batch)
            assert metrics["async/max_ready_training_chunks_during_train"] == 1

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
        coordinator = _coordinator(
            [_row(0)],
            prompt_groups_per_step=1,
            training_chunks_per_step=1,
        )
        async with coordinator:
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
            assert batch._train_started_at is not None
            assert batch._train_finished_at is not None
            expected = batch._train_finished_at - batch._train_started_at

            await coordinator.run_blocking(
                "weight_sync",
                time.sleep,
                0.01,
                optimizer_batch=batch,
            )
            metrics = coordinator.publish(batch)
            assert metrics["perf/train_wall_time"] == expected

    _run(scenario())


def test_filter_and_failure_metrics_are_batch_scoped() -> None:
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
        coordinator = _coordinator(
            rows,
            dynamic_filter_fn=lambda group: group.rewards != [1.0],
        )
        async with coordinator:
            first = await coordinator.next_batch()
            assert first is not None
            await _consume(first)
            first_metrics = coordinator.publish(first)

            second = await coordinator.next_batch()
            assert second is not None
            await _consume(second)
            second_metrics = coordinator.publish(second)

            assert first_metrics["valid_prompt_groups"] == 2
            assert first_metrics["filter_drops"] == 1
            assert first_metrics["sample_fails"] == 0
            assert first_metrics["total_sampled"] == 3
            assert second_metrics["valid_prompt_groups"] == 2
            assert second_metrics["filter_drops"] == 0
            assert second_metrics["sample_fails"] == 1
            assert second_metrics["total_sampled"] == 3

    _run(scenario())


def test_recoverable_rollout_error_drops_and_continues() -> None:
    async def scenario() -> None:
        async def timeout(_sub_index: int) -> RolloutRun:
            raise TimeoutError("transient deployment timeout")

        coordinator = _coordinator(
            [_row(0, run_factory=timeout), _row(1)],
            prompt_groups_per_step=1,
            training_chunks_per_step=1,
        )
        async with coordinator:
            batch = await asyncio.wait_for(coordinator.next_batch(), timeout=1.0)
            assert batch is not None
            assert await _consume(batch) == [1]
            coordinator.publish(batch)
            assert await coordinator.next_batch() is None
            final = coordinator.final_stats()
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
            metrics = coordinator.publish(batch)
            assert metrics["resolved_rows"] == 2
            assert resolved == [(0, "accepted"), (1, "accepted")]

    _run(scenario())


def test_lifecycle_forwards_post_step_callback() -> None:
    def callback(_metrics) -> None:
        pass

    async def training(received):
        return received

    assert (
        _run(run_async_rl_lifecycle(training, post_step_metrics_fn=callback))
        is callback
    )
