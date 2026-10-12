from __future__ import annotations

import asyncio

import pytest

from training.utils.rl.heldout import (
    evaluation_rollout_fn,
    make_heldout_evaluation,
    rollout_heldout_rows,
    run_evaluation,
)
from training.utils.rl.rollout import RolloutRun, RolloutSample


def _run(reward: float) -> RolloutRun:
    return RolloutRun(
        segments=[
            RolloutSample(tokens=[1, 2], logprobs=[0.0, -0.1], loss_mask=[0, 1], reward=reward)
        ]
    )


def test_rows_are_grouped_and_failures_become_none() -> None:
    calls: list[tuple[str, int, int, bool]] = []
    in_flight = 0
    peak = 0

    async def rollout(row, *, row_index: int, sample_index: int, evaluation: bool):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        calls.append((row["id"], row_index, sample_index, evaluation))
        if row["id"] == "b" and sample_index == 1:
            raise RuntimeError("judge failed")
        if row["id"] == "b":
            return None
        return _run(float(sample_index))

    runs = asyncio.run(
        rollout_heldout_rows(
            evaluation_rollout_fn(rollout),
            [{"id": "a"}, {"id": "b"}],
            samples_per_row=2,
            max_concurrency=1,
        )
    )

    assert [[run.segments[0].reward if run else None for run in row] for row in runs] == [
        [0.0, 1.0],
        [None, None],
    ]
    assert sorted(calls) == [
        ("a", 0, 0, True),
        ("a", 0, 1, True),
        ("b", 1, 0, True),
        ("b", 1, 1, True),
    ]
    assert peak == 1


def test_evaluation_passes_fixed_rows_to_metrics_fn() -> None:
    rows = [{"id": "a"}]
    seen = {}

    def metrics_fn(eval_rows, runs):
        seen["rows"] = eval_rows
        return {"eval/reward": runs[0][0].segments[0].reward}

    async def rollout(_row, **_context):
        return _run(0.75)

    evaluate = make_heldout_evaluation(rows, samples_per_row=1, metrics_fn=metrics_fn)
    rows[0]["id"] = "mutated"

    assert asyncio.run(evaluate(3, rollout)) == {"eval/reward": 0.75}
    assert seen["rows"] == ({"id": "a"},)


@pytest.mark.parametrize(
    ("rows", "samples", "concurrency"),
    [([], 1, None), ([{"id": "a"}], 0, None), ([{"id": "a"}], 1, 0)],
)
def test_invalid_heldout_settings_fail_fast(rows, samples, concurrency) -> None:
    async def rollout(_row, **_context):
        return None

    with pytest.raises(ValueError):
        evaluate = make_heldout_evaluation(
            rows,
            samples_per_row=samples,
            metrics_fn=lambda _rows, _runs: {},
            max_concurrency=concurrency,
        )
        asyncio.run(evaluate(0, rollout))


def test_run_evaluation_reports_wall_time_and_failure() -> None:
    seen = []

    async def rollout(row, *, evaluation: bool):
        seen.append(evaluation)
        return _run(1.0)

    async def evaluate(step, rollout_fn):
        await rollout_fn({"id": "a"})
        return {"eval/reward": float(step)}

    async def broken(step, rollout_fn):
        raise RuntimeError("judge down")

    metrics = asyncio.run(run_evaluation(2, evaluate, rollout))
    failed = asyncio.run(run_evaluation(3, broken, rollout))

    assert seen == [True]
    assert (metrics["eval/reward"], metrics["eval/failed"]) == (2.0, 0.0)
    assert metrics["eval/wall_time"] >= 0.0
    assert failed["eval/failed"] == 1.0 and "eval/reward" not in failed
