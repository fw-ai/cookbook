"""Unit tests for crash-safe PromptGroup spill."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import tinker

from training.utils.rl.async_rl.batch import balanced_chunk_targets
from training.utils.rl.async_rl.errors import ErrorClassification, ErrorDisposition
from training.utils.rl.async_rl.group_spill import (
    clear_batch,
    load_batch,
    write_group,
)
from training.utils.rl.async_rl.producer import RolloutProducer, RolloutRow
from training.utils.rl.losses import PromptGroup


def _datum(tokens: list[int]) -> tinker.Datum:
    n = len(tokens)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=tokens, dtype="int64", shape=[n]
            ),
            "weights": tinker.TensorData(
                data=[1.0] * n, dtype="float32", shape=[n]
            ),
        },
    )


def _group(reward: float = 0.5) -> PromptGroup:
    data = [_datum([10, 11, 12]), _datum([20, 21, 22])]
    return PromptGroup(
        data=data,
        advantages=[0.1, -0.1],
        ref_logprobs=None,
        prompt_len=1,
        rewards=[reward, reward],
        inf_logprobs=[[-1.0, -1.1], [-2.0, -2.1]],
        raw_inf_logprobs=[[-1.0, -1.1], [-2.0, -2.1]],
        completion_lens=[2, 2],
        truncated=[False, False],
    )


def test_write_load_roundtrip(tmp_path: Path) -> None:
    g = _group(0.75)
    write_group(
        tmp_path,
        batch_id=6,
        sequence=0,
        group=g,
        source_token="row-a",
        submit_version=5,
        target_groups=2,
        chunk_targets=(1, 1),
    )
    write_group(
        tmp_path,
        batch_id=6,
        sequence=1,
        group=_group(0.25),
        source_token="row-b",
        submit_version=5,
        target_groups=2,
        chunk_targets=(1, 1),
    )
    loaded = load_batch(tmp_path, 6)
    assert loaded is not None
    assert loaded.batch_id == 6
    assert loaded.complete
    assert loaded.realized_groups == 2
    assert loaded.groups[0].source_token == "row-a"
    assert loaded.groups[0].group.rewards == [0.75, 0.75]
    assert loaded.groups[0].group.data[0].loss_fn_inputs["target_tokens"].data == [
        10,
        11,
        12,
    ]
    assert loaded.groups[1].source_token == "row-b"


def test_clear_batch(tmp_path: Path) -> None:
    write_group(
        tmp_path,
        batch_id=3,
        sequence=0,
        group=_group(),
        source_token="r0",
        submit_version=2,
        target_groups=1,
        chunk_targets=(1,),
    )
    assert load_batch(tmp_path, 3) is not None
    clear_batch(tmp_path, 3)
    assert load_batch(tmp_path, 3) is None


def _make_producer(rows: list[RolloutRow], output: asyncio.Queue, **kwargs) -> RolloutProducer:
    defaults = dict(
        completions_per_prompt=2,
        prompt_groups_per_step=2,
        training_chunks_per_step=2,
        max_head_off_policy_versions=1,
        max_concurrent_rollouts=2,
        advantage_fn=lambda rewards: [0.0] * len(rewards),
        with_reference=False,
        router_replay_completion_only=False,
        min_group_size=1,
        max_incomplete_group_retries=0,
        dynamic_filter_fn=None,
        initial_version=0,
        resolved_rows_offset=0,
        resolved_rows_fn=None,
        error_classifier=lambda _exc: ErrorClassification(
            disposition=ErrorDisposition.FATAL, reason="fatal"
        ),
        circuit_breaker=None,
    )
    defaults.update(kwargs)
    return RolloutProducer(rows=rows, output=output, **defaults)


@pytest.mark.asyncio
async def test_producer_restore_partial_skips_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASYNC_RL_TRAIN_SPILL_DIR", str(tmp_path))
    chunk_targets = balanced_chunk_targets(2, 2)
    write_group(
        tmp_path,
        batch_id=1,
        sequence=0,
        group=_group(1.0),
        source_token="row-0",
        submit_version=0,
        target_groups=2,
        chunk_targets=chunk_targets,
    )

    rolled: list[str] = []

    async def boom(_version: int):
        rolled.append("row-0")
        raise AssertionError("should not roll spilled row row-0")

    async def live(_version: int):
        rolled.append("row-1")
        return None

    rows = [
        RolloutRow(row_id="row-0", run_factory=boom),
        RolloutRow(row_id="row-1", run_factory=live),
    ]
    output: asyncio.Queue = asyncio.Queue()
    producer = _make_producer(rows, output)
    n = producer.restore_from_spill()
    assert n == 1
    assert "row-0" not in rolled
    assert producer._fill_batch is not None
    assert producer._fill_batch.realized_groups == 1
    assert producer._fill_batch.batch_id == 1


@pytest.mark.asyncio
async def test_producer_restore_complete_exposes_sealed_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fire-drill: full spill reload → sealed batch ready without re-roll."""

    monkeypatch.setenv("ASYNC_RL_TRAIN_SPILL_DIR", str(tmp_path))
    chunk_targets = balanced_chunk_targets(2, 2)
    for seq, token, reward in [(0, "row-0", 1.0), (1, "row-1", 0.0)]:
        write_group(
            tmp_path,
            batch_id=1,
            sequence=seq,
            group=_group(reward),
            source_token=token,
            submit_version=0,
            target_groups=2,
            chunk_targets=chunk_targets,
        )

    rolled: list[str] = []

    def boom(row_id: str):
        async def _run(_version: int):
            rolled.append(row_id)
            raise AssertionError(f"should not roll {row_id}")

        return _run

    rows = [
        RolloutRow(row_id="row-0", run_factory=boom("row-0")),
        RolloutRow(row_id="row-1", run_factory=boom("row-1")),
        RolloutRow(row_id="row-2", run_factory=boom("row-2")),
    ]
    output: asyncio.Queue = asyncio.Queue()
    producer = _make_producer(rows, output)
    n = producer.restore_from_spill()
    assert n == 2
    assert rolled == []
    assert producer._fill_batch is None  # sealed
    batch = output.get_nowait()
    assert batch.batch_id == 1
    assert batch.sealed
    assert batch.realized_groups == 2
    assert batch._accepted_sequences == [0, 1]
    # Remaining dataset row was not consumed.
    assert next(producer._rows).row_id == "row-2"
