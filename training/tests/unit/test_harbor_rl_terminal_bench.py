from __future__ import annotations

import sys

import pytest

from training.examples.rl.harbor_rl_terminal_bench.train_serverless import (
    parse_args,
    select_rows,
    split_rows,
    training_rows,
)


def _rows(*names: str) -> list[dict[str, object]]:
    return [
        {"task_name": name, "harbor_task_config": {"path": f"/tasks/{name}"}}
        for name in names
    ]


def test_split_rows_is_disjoint_and_deterministic() -> None:
    train, evaluation = split_rows(_rows("c", "a", "d", "b"), ["d", "b"])

    assert [row["task_name"] for row in train] == ["a", "c"]
    assert [row["task_name"] for row in evaluation] == ["d", "b"]


def test_select_rows_rejects_missing_and_duplicate_tasks() -> None:
    rows = _rows("a", "b")

    with pytest.raises(ValueError, match="contains duplicates"):
        select_rows(rows, ["a", "a"])
    with pytest.raises(ValueError, match="missing tasks"):
        select_rows(rows, ["missing"])


def test_training_rows_shuffles_once_then_cycles() -> None:
    rows = _rows("a", "b", "c")

    selected = training_rows(rows, max_rows=8, seed=7)
    names = [row["task_name"] for row in selected]

    assert names[:3] == names[3:6]
    assert names[:2] == names[6:]
    assert set(names[:3]) == {"a", "b", "c"}


def test_router_replay_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_serverless",
            "--harbor-dataset",
            "/tasks",
            "--harbor-trials-dir",
            "/trials",
            "--no-router-replay",
        ],
    )

    assert parse_args().router_replay is False
