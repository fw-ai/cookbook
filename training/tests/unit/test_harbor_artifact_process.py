"""Real artifact-process values, failure recovery and input-file lifetime."""

import asyncio
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path

import pytest

from fireworks.training.sdk import (
    TITOChatRequest,
    TITOMetricSummary,
    TITOParsedAssistant,
    TITOResponseAttempt,
    TITOSegmentResult,
    TITOTrajectoryArtifact,
    TITOTurn,
)
from training.examples.rl.harbor.tito._artifact_io import ArtifactProcessPool
from training.examples.rl.harbor.tito.rollout import materialize_harbor_trajectory
from training.examples.rl.harbor.tito.trial import (
    HarborTrialOutcome,
    _finish_harbor_trial,
)
from training.utils.rl.async_rl.errors import RecoverableRolloutError


@pytest.fixture
def pool():
    worker = ArtifactProcessPool()
    yield worker
    asyncio.run(worker.aclose())


def _wait_for_release(started: Path, release: Path, finished: Path) -> int:
    started.write_text(str(os.getpid()))
    deadline = time.monotonic() + 15
    while not release.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("test did not release worker")
        time.sleep(0.005)
    finished.write_text("finished")
    return os.getpid()


def test_cancellation_drains_inputs_and_close_reaps_worker(pool, tmp_path):
    async def check():
        started, release, finished = (
            tmp_path / name for name in ("start", "release", "end")
        )
        task = asyncio.create_task(
            pool.run(_wait_for_release, started, release, finished)
        )
        try:
            async with asyncio.timeout(20):
                while not started.exists():
                    await asyncio.sleep(0.01)
            pid = int(started.read_text())
            assert pid != os.getpid()
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done() and not finished.exists()
            release.touch()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.exists()
            assert await pool.run(os.getpid) == pid
            await pool.aclose()
            assert pid not in {child.pid for child in multiprocessing.active_children()}
            with pytest.raises(RuntimeError, match="closed"):
                await pool.run(os.getpid)
        finally:
            release.touch()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())


def test_worker_crash_does_not_poison_next_trial(pool):
    original_pid = asyncio.run(pool.run(os.getpid))
    with pytest.raises(BrokenProcessPool):
        asyncio.run(pool.run(os._exit, 23))
    assert asyncio.run(pool.run(os.getpid)) != original_pid


def _write_artifact(root: Path, *, routing: bool) -> HarborTrialOutcome:
    turn = TITOTurn(
        turn_id="one",
        request=TITOChatRequest(messages=({"role": "user", "content": "q"},)),
        assistant=TITOParsedAssistant(
            message={"role": "assistant", "content": "answer"}
        ),
        exact_prompt_ids=(1, 2),
        exact_completion_ids=(3, 4),
        inference_logprobs=(-0.1, -0.3),
        sampling_logprobs=(-0.2, -0.4),
        routing_matrices=("route-3", "route-4") if routing else None,
        response_id="response-one",
        finish_reason="stop",
        prompt_disposition="new_segment",
        prefix_match_tokens=None,
        realign_from_token=None,
        realigned_masked_tokens=0,
        requested_output_tokens=8,
        effective_output_tokens=8,
        context_remaining_tokens=100,
        server_metrics=None,
        sampler_wall_seconds=0.1,
        logical_request_id="logical-one",
        upstream_response_id="upstream-one",
        upstream_attempts=1,
    )
    artifact = TITOTrajectoryArtifact(
        trajectory_id="trajectory",
        serving_affinity_key_hash="hash",
        metadata={"nested": {"value": [1, "text"]}},
        status="completed",
        terminal_reason=None,
        segments=(
            TITOSegmentResult("segment", "initial", "contract", (turn,), "done"),
        ),
        calls=(),
        response_attempts=(TITOResponseAttempt("attempt", "one", "completed", 2.0),),
        metrics=TITOMetricSummary(counters={}, distributions={}),
        started_at=1.0,
        finished_at=2.0,
    )
    encoded = artifact.pack()
    directory = root / "artifacts/tito/compact"
    directory.mkdir(parents=True)
    (directory / "trajectory.tito").write_bytes(encoded)
    (directory / "trajectory.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                trajectory_id=artifact.trajectory_id,
                status=artifact.status,
                terminal_reason=artifact.terminal_reason,
                bytes=len(encoded),
                sha256=hashlib.sha256(encoded).hexdigest(),
            )
        )
    )
    (directory / "COMPLETE").touch()
    return HarborTrialOutcome("task", "trial", root, None, {}, None, None)


def _completion_options(**overrides):
    return (
        dict(
            raw_rewards={"reward": 0.0, "partial": 0.75},
            reward_key="partial",
            terminal_failure_reward=None,
            has_exception=False,
            retry_names=frozenset(),
            retryable_e2b_timeout=False,
            retryable_sidecar_readiness=False,
        )
        | overrides
    )


@pytest.mark.parametrize("routing,debug", [(False, False), (True, True)])
def test_process_returns_exact_rollout_values(pool, tmp_path, routing, debug):
    outcome = _write_artifact(tmp_path, routing=routing)
    result = asyncio.run(
        pool.run(
            _finish_harbor_trial,
            outcome,
            materializer=partial(
                materialize_harbor_trajectory,
                max_context_tokens=4096,
                debug_enabled=debug,
            ),
            **_completion_options(),
        )
    )
    assert (result.task_name, result.trial_name) == ("task", "trial")
    assert result.rewards == {"reward": 0.0, "partial": 0.75}
    assert result.trajectory_artifact is None
    (segment,) = result.rollout.segments
    assert segment.tokens == [1, 2, 3, 4]
    assert segment.logprobs == [0.0, 0.0, -0.2, -0.4]
    assert segment.raw_logprobs == [0.0, 0.0, -0.1, -0.3]
    assert segment.loss_mask == [0, 0, 1, 1]
    assert segment.reward == result.reward == 0.75
    assert segment.routing_matrices == (["", "route-3", "route-4"] if routing else None)


def test_artifact_failures_leave_worker_usable(pool, tmp_path):
    outcome = _write_artifact(tmp_path, routing=False)
    convert = partial(
        pool.run,
        _finish_harbor_trial,
        outcome,
        materializer=partial(
            materialize_harbor_trajectory, max_context_tokens=4096, debug_enabled=False
        ),
    )
    for overrides, message in [
        ({"raw_rewards": {"partial": float("nan")}}, "non-finite reward"),
        ({"raw_rewards": {}}, "did not produce a usable reward"),
        ({"retryable_e2b_timeout": True}, "command stream did not open"),
    ]:
        with pytest.raises(RecoverableRolloutError, match=message):
            asyncio.run(convert(**_completion_options(**overrides)))
    manifest_path = tmp_path / "artifacts/tito/compact/trajectory.json"
    manifest = manifest_path.read_text()
    invalid = json.loads(manifest)
    invalid["trajectory_id"] = "another-trial"
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(RecoverableRolloutError, match="identity mismatch"):
        asyncio.run(convert(**_completion_options()))
    manifest_path.write_text(manifest)
    assert (
        asyncio.run(convert(**_completion_options())).rollout.segments[0].reward == 0.75
    )


def test_pi_process_reconciles_lifecycle_and_timeout_metadata(pool, tmp_path):
    from training.examples.rl.harbor.pi.rollout import _materialize_pi_trajectory

    outcome = _write_artifact(tmp_path, routing=True)
    convert = partial(
        pool.run,
        _finish_harbor_trial,
        outcome,
        materializer=partial(
            _materialize_pi_trajectory, max_context_tokens=4096, debug_enabled=False
        ),
        **_completion_options(),
    )
    lifecycle = tmp_path / "agent" / "pi.txt"
    lifecycle.parent.mkdir()
    with pytest.raises(RecoverableRolloutError, match="no lifecycle stream"):
        asyncio.run(convert())
    lifecycle.write_text(
        json.dumps(dict(type="compaction_end", reason="overflow", willRetry=True))
    )
    with pytest.raises(RecoverableRolloutError, match="could not be reconciled"):
        asyncio.run(convert())
    lifecycle.write_text(
        json.dumps(
            dict(
                type="tool_execution_end",
                toolName="bash",
                toolCallId="call-one",
                result="timed out after 900 seconds",
            )
        )
    )
    result = asyncio.run(convert())
    assert result.rollout.metadata["pi_abandoned_turn_count"] == 0
    assert result.rollout.metadata["harness_tool_timeout_count"] == 1
    assert result.rollout.segments[0].routing_matrices == ["", "route-3", "route-4"]


def test_drain_bound_discards_a_wedged_worker(pool, tmp_path, monkeypatch):
    """A worker that outlives the drain bound must not own the pool's process."""

    from training.examples.rl.harbor.tito import _artifact_io

    monkeypatch.setattr(_artifact_io, "_DRAIN_TIMEOUT_SECONDS", 1.0)
    started, release, finished = (
        tmp_path / name for name in ("start", "release", "end")
    )

    async def check():
        task = asyncio.create_task(
            pool.run(_wait_for_release, started, release, finished)
        )
        async with asyncio.timeout(20):
            while not started.exists():
                await asyncio.sleep(0.01)
        wedged_pid = int(started.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The wedged worker is still running: the pool must have dropped it.
        assert not finished.exists()
        assert pool._executor is None  # noqa: SLF001
        async with asyncio.timeout(60):
            assert await pool.run(os.getpid) != wedged_pid

    try:
        asyncio.run(check())
    finally:
        release.touch()
