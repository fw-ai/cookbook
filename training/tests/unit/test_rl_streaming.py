"""Ready-group requests, mathematical diagnostics and failure barriers."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID

import pytest
from fireworks.training import sdk
from training.utils.rl.algorithm import grpo as objective
from training.utils.rl.async_rl.batch import OptimizerBatch
from training.utils.rl.losses import PromptGroup, combine_prompt_groups
from training.utils.rl.streaming import (
    SubmittedGroup,
    add_batch_diagnostics,
    drain_groups,
    merge_policy_batches,
    retain_submission,
    streaming_metrics,
    submission_window,
    track_group,
)


def group(index=0, samples=1):
    return PromptGroup(
        data=[
            sdk.Datum(
                model_input=sdk.ModelInput.from_ints([1, 2]),
                loss_fn_inputs={
                    "target_tokens": sdk.TensorData(data=[2, 3], dtype="int64", shape=[2]),
                    "weights": sdk.TensorData(data=[0.0, 1.0], dtype="float32", shape=[2]),
                },
            )
            for _ in range(samples)
        ],
        advantages=[1.0] * samples,
        prompt_len=2,
        rewards=[float(index)] * samples,
        inf_logprobs=[[0.0, -0.2]] * samples,
    )


def output(count, logprob=-0.2):
    return SimpleNamespace(
        metrics={"loss:sum": float(count)},
        loss_fn_outputs=[
            {"logprobs": sdk.TensorData(data=[0.0, logprob], dtype="float32", shape=[2])} for _ in range(count)
        ],
    )


async def retain(submissions, batch, result_async, **kwargs):
    async def worker():
        return track_group(
            SimpleNamespace(result_async=result_async),
            batch,
            ready_at=0,
            ready_utc=0,
            submit_started_at=0,
            **kwargs,
        )

    return await retain_submission(submissions, worker())


@pytest.mark.parametrize("failed", [False, True])
def test_debug_context_does_not_replace_the_native_result(caplog, failed):
    caplog.set_level(logging.DEBUG, logger="training.utils.rl.streaming")

    async def scenario():
        async def result_async(timeout=None):
            if failed:
                raise RuntimeError("native failure")
            return output(1)

        submissions = []
        await retain(
            submissions,
            combine_prompt_groups([group()]),
            result_async,
            context={"rows": [UUID("12345678-1234-5678-1234-567812345678")]},
        )
        if failed:
            with pytest.raises(RuntimeError, match="native failure"):
                await drain_groups(submissions)
        else:
            assert (await drain_groups(submissions))[0].metrics == {"loss:sum": 1.0}

    asyncio.run(scenario())
    assert "12345678-1234-5678-1234-567812345678" in caplog.text


def test_prepared_inputs_and_weighted_diagnostics_survive_request_merging():
    batches = [
        replace(
            combine_prompt_groups([group(index, samples)]),
            token_advantages=[[0.0, index + 0.5]] * samples,
            anchor_logprobs=[[0.0, -index - 0.3]] * samples,
        )
        for index, samples in [(0, 2), (1, 1)]
    ]
    merged = merge_policy_batches(batches)
    assert merged.anchor_logprobs == [[0.0, -0.3], [0.0, -0.3], [0.0, -1.3]]
    assert merged.token_advantages == [[0.0, 0.5], [0.0, 0.5], [0.0, 1.5]]
    assert merged.groups == [g for batch in batches for g in batch.groups]
    datums = objective.build_datums(merged, objective.GRPOConfig())
    for datum, expected in zip(datums, merged.token_advantages, strict=True):
        assert datum.loss_fn_inputs["advantages"].data == pytest.approx(expected)
    results = [output(2), output(1, -0.4)]
    results[1].metrics["active_tokens"] = 99
    add_batch_diagnostics([SimpleNamespace(batch=b) for b in batches], results, objective.GRPOConfig(), objective)
    assert results[0].metrics["active_tokens"] == 3
    assert "active_tokens" not in results[1].metrics
    assert results[0].metrics["ppo_ratio_mean"] == pytest.approx((2 * math.exp(0.1) + math.exp(0.9)) / 3)


def test_coalesced_request_metrics_count_every_group_ready_time():
    submission = SubmittedGroup(
        batch=combine_prompt_groups([group(i) for i in range(5)]),
        future=None,
        ready_at=1,
        submitted_at=10,
        ready_utc=0,
        submitted_utc=0,
        completed_at=12,
        group_ready_times=(1, 2, 3, 4, 5),
    )
    metrics = streaming_metrics([submission])
    assert metrics["perf/group_ready_to_submit_time_sum"] == 35
    assert metrics["train/forward_backward_requests"] == 1
    assert metrics["train/groups_per_request_mean"] == metrics["train/groups_per_request_max"] == 5


def test_async_ready_queue_propagates_failure_before_sending_queued_groups():
    async def scenario():
        batch = OptimizerBatch(batch_id=1, target_groups=2)
        batch._put_group(batch._append_group(sequence=0, group=group(), submit_version=0, source_token=0))
        batch._fail(RuntimeError("rollout failed"))
        with pytest.raises(RuntimeError, match="rollout failed"):
            await anext(batch.ready_group_batches())
        assert not batch._consumed

    asyncio.run(scenario())


def test_async_groups_submit_without_waiting_for_earlier_training_results():
    async def scenario():
        first, second, release_training = asyncio.Event(), asyncio.Event(), asyncio.Event()
        batch = OptimizerBatch(batch_id=1, target_groups=2)
        batch._put_group(batch._append_group(sequence=0, group=group(0), submit_version=0, source_token=0))
        submissions = []

        async def consume():
            async for ready in batch.ready_group_batches():
                data = combine_prompt_groups([g.group for g in ready])

                async def result_async(timeout=None, data=data):
                    assert timeout == 17
                    await release_training.wait()
                    return output(len(data.data))

                await retain(submissions, data, result_async, timeout=17)
                (first if len(submissions) == 1 else second).set()

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(first.wait(), 1)
        assert not consumer.done() and not submissions[0].waiter.done()
        batch._put_group(batch._append_group(sequence=1, group=group(1), submit_version=0, source_token=1))
        batch._seal()
        await asyncio.wait_for(second.wait(), 1)
        assert not submissions[0].waiter.done()
        await consumer
        assert [s.batch.groups[0].rewards for s in submissions] == [[0.0], [1.0]]
        release_training.set()
        assert len(await drain_groups(submissions)) == 2

    asyncio.run(asyncio.wait_for(scenario(), 1))


@pytest.mark.parametrize("failure", [False, True], ids=["cancelled-drain", "failed-request"])
def test_drain_joins_every_request_before_propagating_cancellation_or_failure(failure):
    async def scenario():
        release = asyncio.Event()
        submissions = []
        for index in range(2):

            async def result_async(timeout=None, index=index):
                if failure and index == 0:
                    raise RuntimeError("training failed")
                await release.wait()
                return output(1)

            await retain(submissions, combine_prompt_groups([group(index)]), result_async)
        drain = asyncio.create_task(drain_groups(submissions))
        await asyncio.sleep(0)
        if not failure:
            drain.cancel()
        await asyncio.sleep(0)
        assert not drain.done()
        assert all(not s.waiter.cancelled() for s in submissions)
        release.set()
        with pytest.raises(RuntimeError if failure else asyncio.CancelledError):
            await drain
        assert all(s.waiter.done() for s in submissions)

    asyncio.run(scenario())


def test_failed_request_aborts_stalled_async_group_consumer():
    async def scenario():
        owner = asyncio.current_task()
        batch = OptimizerBatch(batch_id=1, target_groups=2)
        batch._put_group(batch._append_group(sequence=0, group=group(), submit_version=0, source_token=0))

        async def result_async(timeout=None):
            raise RuntimeError("remote training failure")

        with pytest.raises(RuntimeError, match="remote training failure"):
            async with submission_window() as submissions:
                async for ready in batch.ready_group_batches():
                    await retain(
                        submissions, combine_prompt_groups([g.group for g in ready]), result_async, abort_task=owner
                    )
        assert not batch._consumed
        assert len(submissions) == 1 and submissions[0].waiter.done()

    asyncio.run(asyncio.wait_for(scenario(), 1))


def test_cancel_during_submission_retains_and_drains_native_future():
    async def scenario():
        started, allow_submit, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        observed = []

        async def result_async(timeout=None):
            await finish.wait()
            observed.append("completed")
            return output(1)

        async def worker():
            started.set()
            await allow_submit.wait()
            return track_group(
                SimpleNamespace(result_async=result_async),
                combine_prompt_groups([group()]),
                ready_at=0,
                ready_utc=0,
                submit_started_at=0,
            )

        async def run():
            async with submission_window() as submissions:
                await retain_submission(submissions, worker())

        task = asyncio.create_task(run())
        await started.wait()
        task.cancel()
        allow_submit.set()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert observed == ["completed"]

    asyncio.run(scenario())


def test_training_failure_wins_over_owner_cancellation():
    async def scenario():
        async def result_async(timeout=None):
            raise RuntimeError("native backward failed")

        with pytest.raises(RuntimeError, match="native backward failed"):
            async with submission_window() as submissions:
                await retain(submissions, combine_prompt_groups([group()]), result_async)
                raise asyncio.CancelledError

    asyncio.run(scenario())
