import asyncio
from types import SimpleNamespace

from eval_protocol.models import EvaluationRow, InputMetadata
from eval_protocol.pytest import evaluation_test

from training.utils.rl.rollout import (
    RolloutSample,
    load_eval_protocol_input_rows,
    make_eval_protocol_rollout_fn_factory,
)


def _run(coro):
    return asyncio.run(coro)


def _sample_from_row(row: EvaluationRow) -> RolloutSample:
    reward = float((row.execution_metadata.extra or {}).get("reward", 0.0))
    return RolloutSample(
        tokens=[1, 2, 3],
        logprobs=[0.0, -0.1, -0.2],
        loss_mask=[0, 1, 1],
        reward=reward,
    )


def test_load_eval_protocol_input_rows_from_decorated_evaluator():
    rows = [
        EvaluationRow(input_metadata=InputMetadata(row_id="keep")),
        EvaluationRow(input_metadata=InputMetadata(row_id="drop")),
    ]

    @evaluation_test(input_rows=[rows], filtered_row_ids=["keep"])
    def evaluator(row: EvaluationRow) -> EvaluationRow:
        return row

    loaded = load_eval_protocol_input_rows(evaluator)

    assert [row.input_metadata.row_id for row in loaded] == ["keep"]
    assert loaded[0] is not rows[0]


def test_eval_protocol_adapter_invokes_processor_one_row_and_scores():
    calls: list[dict] = []

    class FakeProcessor:
        def setup(self):
            calls.append({"setup": True})

        def __call__(self, rows, config):
            calls.append({
                "row_ids": [row.input_metadata.row_id for row in rows],
                "completion_params": config.completion_params,
                "steps": config.steps,
                "semaphore_value": config.semaphore._value,
                "kwargs": config.kwargs,
            })

            async def finish():
                row = rows[0]
                row.execution_metadata.extra = {"reward": 0.25}
                return row

            return [finish()]

    input_row = EvaluationRow(input_metadata=InputMetadata(row_id="row-1"))

    @evaluation_test(
        input_rows=[[input_row]],
        completion_params=[{"model": "decorator-model", "temperature": 0.7}],
        rollout_processor=FakeProcessor(),
        rollout_processor_kwargs={"custom": "value"},
        steps=7,
    )
    def evaluator(row: EvaluationRow) -> EvaluationRow:
        row.execution_metadata.extra["reward"] = 1.0
        return row

    rollout_fn = make_eval_protocol_rollout_fn_factory(
        evaluator,
        sample_converter=_sample_from_row,
    )(SimpleNamespace(model="runtime-model", sample_kwargs={"top_p": 0.9}))

    sample = _run(rollout_fn(input_row))

    assert sample.reward == 1.0
    assert calls == [
        {"setup": True},
        {
            "row_ids": ["row-1"],
            "completion_params": {
                "model": "runtime-model",
                "temperature": 0.7,
                "top_p": 0.9,
            },
            "steps": 7,
            "semaphore_value": 1,
            "kwargs": {"custom": "value"},
        },
    ]


def test_eval_protocol_adapter_accepts_custom_row_factory():
    class FakeProcessor:
        def setup(self):
            pass

        def __call__(self, rows, _config):
            async def finish():
                row = rows[0]
                row.execution_metadata.extra = {"reward": 0.5}
                return row

            return [finish()]

    @evaluation_test(
        input_rows=[[EvaluationRow(input_metadata=InputMetadata(row_id="placeholder"))]],
        rollout_processor=FakeProcessor(),
    )
    def evaluator(row: EvaluationRow) -> EvaluationRow:
        return row

    def row_factory(row: dict) -> EvaluationRow:
        return EvaluationRow(input_metadata=InputMetadata(row_id=row["id"]))

    rollout_fn = make_eval_protocol_rollout_fn_factory(
        evaluator,
        row_factory=row_factory,
        sample_converter=_sample_from_row,
    )(None)

    sample = _run(rollout_fn({"id": "custom-row"}))

    assert sample.reward == 0.5
