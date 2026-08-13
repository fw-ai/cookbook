from __future__ import annotations

import asyncio
import builtins
import importlib.util
import json

import pytest

from training.examples.rl.harbor_rl_opencode.dabstep import (
    AdaptiveTaskSelector,
    DABstepManifest,
    rows_for_tasks,
    usable_group_probability,
)
from training.examples.rl.harbor_rl_opencode import prepare_dabstep_tasks
from training.examples.rl.harbor.evaluate import (
    evaluate_rows,
    make_fixed_evaluation,
)
from training.utils.rl.rollout import RolloutRun, RolloutSample


def _manifest_document() -> dict:
    train = [f"dabstep-{index}" for index in range(67)]
    holdout = [f"dabstep-{index}" for index in range(67, 75)]
    selected = [*train, *holdout]
    return {
        "schema_version": 1,
        "tasks": {
            "train": train,
            "holdout": holdout,
            "reference_step0": train[:6],
        },
        "profile": {
            name: {"attempts": 5, "solved": index % 6}
            for index, name in enumerate(selected)
        },
        "content_sha256": {name: "a" * 64 for name in selected},
        "source": {"kind": "test"},
    }


def test_dabstep_manifest_validates_fixed_membership(tmp_path):
    document = _manifest_document()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    manifest = DABstepManifest.load(path)

    assert len(manifest.train_tasks) == 67
    assert len(manifest.holdout_tasks) == 8
    assert not set(manifest.train_tasks) & set(manifest.holdout_tasks)
    assert rows_for_tasks(
        [{"task_name": name} for name in reversed(manifest.train_tasks)],
        manifest.train_tasks[:2],
    ) == [{"task_name": "dabstep-0"}, {"task_name": "dabstep-1"}]


def test_dabstep_manifest_rejects_split_overlap(tmp_path):
    document = _manifest_document()
    document["tasks"]["holdout"][0] = document["tasks"]["train"][0]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        DABstepManifest.load(path)


def test_dabstep_manifest_rejects_content_drift(tmp_path, monkeypatch):
    document = _manifest_document()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    manifest = DABstepManifest.load(path)
    for name in (*manifest.train_tasks, *manifest.holdout_tasks):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(
        "training.examples.rl.harbor_rl_opencode.dabstep._directory_sha256",
        lambda path: "b" * 64,
    )

    with pytest.raises(ValueError, match="content drift"):
        manifest.verify_task_root(tmp_path)


def test_dabstep_manifest_rejects_sign_insensitive_scorer(tmp_path, monkeypatch):
    document = _manifest_document()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    manifest = DABstepManifest.load(path)
    first_task = tmp_path / manifest.train_tasks[0]
    (first_task / "tests").mkdir(parents=True)
    (first_task / "tests" / "scorer.py").write_text(
        'match = re.search(r"(\\d*\\.\\d+|\\d+\\.?\\d*)%?", value)\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "training.examples.rl.harbor_rl_opencode.dabstep._directory_sha256",
        lambda path: "a" * 64,
    )

    with pytest.raises(ValueError, match="sign-insensitive numeric scorer"):
        manifest.verify_task_root(tmp_path)


def test_dabstep_hash_dependency_is_loaded_only_when_needed(tmp_path, monkeypatch):
    from training.examples.rl.harbor_rl_opencode import dabstep

    original_import = builtins.__import__

    def import_without_dirhash(name, *args, **kwargs):
        if name == "dirhash":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_dirhash)

    with pytest.raises(RuntimeError, match="example-only 'dirhash' dependency"):
        dabstep._directory_sha256(tmp_path)


@pytest.mark.parametrize("quote", ["'", '"'])
def test_prepare_dabstep_tasks_preserves_numeric_signs(tmp_path, quote):
    source = tmp_path / "dabstep-1"
    (source / "environment").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "task.toml").write_text("", encoding="utf-8")
    (source / "environment" / "Dockerfile").write_text(
        "FROM debian:bookworm\n",
        encoding="utf-8",
    )
    numeric_pattern = rf"r{quote}(\d*\.\d+|\d+\.?\d*)%?{quote}"
    (source / "tests" / "scorer.py").write_text(
        "import re\n\n"
        "def extract_numeric(value):\n"
        f"    match = re.search({numeric_pattern}, value)\n"
        "    return float(match.group(1)) if match else None\n",
        encoding="utf-8",
    )

    [prepared] = prepare_dabstep_tasks.prepare(
        source,
        tmp_path / "prepared",
        "1.18.10",
    )

    scorer = prepared / "tests" / "scorer.py"
    spec = importlib.util.spec_from_file_location("prepared_dabstep_scorer", scorer)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.extract_numeric("-2.18") == -2.18
    assert module.extract_numeric("+2.18") == 2.18


def test_prepare_dabstep_tasks_rejects_unknown_scorer(tmp_path):
    source = tmp_path / "dabstep-1"
    (source / "environment").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "task.toml").write_text("", encoding="utf-8")
    (source / "environment" / "Dockerfile").write_text(
        "FROM debian:bookworm\n",
        encoding="utf-8",
    )
    (source / "tests" / "scorer.py").write_text(
        "raise RuntimeError('unknown scorer')\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="known unsigned numeric scorer pattern"):
        prepare_dabstep_tasks.prepare(
            source,
            tmp_path / "prepared",
            "1.18.10",
        )


def test_adaptive_selector_is_seeded_and_distinct_per_batch():
    rows = [{"task_name": f"dabstep-{index}"} for index in range(12)]
    profile = {
        row["task_name"]: {"attempts": 5, "solved": index % 6}
        for index, row in enumerate(rows)
    }

    async def select(seed: int):
        selector = AdaptiveTaskSelector(
            task_rows=rows,
            profile=profile,
            group_size=8,
            groups_per_batch=8,
            seed=seed,
        )
        selected = [
            (await selector.row_for_group(index))["task_name"] for index in range(8)
        ]
        return selected

    first = asyncio.run(select(20260728))
    second = asyncio.run(select(20260728))

    assert first == second
    assert len(first) == len(set(first)) == 8
    assert usable_group_probability(0.5, 8) == pytest.approx(254 / 256)


def test_adaptive_selector_replaces_retried_sample_observations():
    rows = [{"task_name": f"dabstep-{index}"} for index in range(8)]
    selector = AdaptiveTaskSelector(
        task_rows=rows,
        profile={row["task_name"]: {} for row in rows},
        group_size=2,
        groups_per_batch=2,
        seed=3,
    )

    async def record():
        await selector.row_for_group(0)
        await selector.record(0, 0, 0.0)
        await selector.record(0, 0, 1.0)
        await selector.record(0, 1, 1.0)

    asyncio.run(record())

    assert selector._observations[0] == {0: 1.0, 1: 1.0}


def test_adaptive_selector_counts_dropped_trajectory_as_unsolved_attempt():
    rows = [{"task_name": f"dabstep-{index}"} for index in range(8)]
    selector = AdaptiveTaskSelector(
        task_rows=rows,
        profile={row["task_name"]: {} for row in rows},
        group_size=2,
        groups_per_batch=2,
        seed=3,
    )

    async def record():
        task_name = (await selector.row_for_group(0))["task_name"]
        await selector.record(0, 0, 1.0)
        await selector.record(0, 1, None)
        return task_name

    task_name = asyncio.run(record())

    assert selector._stats[task_name].attempts == 2
    assert selector._stats[task_name].solved == 1
    assert selector._observations[0] == {0: 1.0, 1: None}


def test_adaptive_selector_state_round_trip_preserves_next_batch():
    rows = [{"task_name": f"dabstep-{index}"} for index in range(8)]
    profile = {row["task_name"]: {} for row in rows}
    original = AdaptiveTaskSelector(
        task_rows=rows,
        profile=profile,
        group_size=2,
        groups_per_batch=2,
        seed=7,
    )

    async def advance(selector):
        for group_index in range(2):
            await selector.row_for_group(group_index)
            await selector.record(group_index, 0, 0.0)
            await selector.record(group_index, 1, 1.0)

    asyncio.run(advance(original))
    serialized = json.loads(json.dumps(original.state_dict()))
    restored = AdaptiveTaskSelector(
        task_rows=rows,
        profile=profile,
        group_size=2,
        groups_per_batch=2,
        seed=999,
    )
    restored.load_state_dict(serialized)

    async def next_batch(selector):
        return [
            (await selector.row_for_group(group_index))["task_name"]
            for group_index in range(2, 4)
        ]

    assert asyncio.run(next_batch(original)) == asyncio.run(next_batch(restored))
    assert original.state_dict()["stats"] == restored.state_dict()["stats"]


def test_evaluate_rows_aggregates_logical_run_lengths_and_failures():
    active = 0
    peak = 0

    async def rollout_fn(row, *, sample_index, **kwargs):
        nonlocal active, peak
        del kwargs
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        if row["task_name"] == "bad":
            raise TimeoutError("test timeout")
        return RolloutRun(
            segments=[
                RolloutSample(
                    tokens=[1, 2, 3],
                    logprobs=[0.0, -0.1, -0.2],
                    loss_mask=[0, 1, 1],
                    reward=float(sample_index),
                )
            ],
            run_id=f"{row['task_name']}-{sample_index}",
            metadata={
                "task_name": row["task_name"],
                "history_wipes": sample_index,
                "append_token_mismatches": sample_index,
            },
        )

    metrics = asyncio.run(
        evaluate_rows(
            rollout_fn,
            [{"task_name": "good"}, {"task_name": "bad"}],
            completions_per_prompt=2,
            metric_prefix="eval",
            step=0,
            max_concurrency=2,
        )
    )

    assert peak == 2
    assert metrics["eval/attempted_trajectories"] == 4
    assert metrics["eval/completed_trajectories"] == 2
    assert metrics["eval/failed_trajectories"] == 2
    assert metrics["eval/no_trajectory"] == 0
    assert metrics["eval/failure/TimeoutError"] == 2
    assert metrics["eval/trainable_tokens_mean"] == 2
    assert metrics["eval/trainable_tokens_min"] == 2
    assert metrics["eval/trainable_tokens_max"] == 2
    assert metrics["eval/history_wipes"] == 1
    assert metrics["eval/append_token_mismatches"] == 1


def test_fixed_evaluation_keeps_one_sample_count_per_call():
    calls: list[tuple[str, int, int]] = []

    async def rollout_fn(row, *, epoch, sample_index, **kwargs):
        del kwargs
        calls.append((row["task_name"], epoch, sample_index))
        return RolloutRun(
            segments=[
                RolloutSample(
                    tokens=[1, 2],
                    logprobs=[0.0, -0.1],
                    loss_mask=[0, 1],
                    reward=1.0,
                )
            ],
            run_id=f"{row['task_name']}-{epoch}-{sample_index}",
            metadata={"task_name": row["task_name"]},
        )

    evaluate = make_fixed_evaluation(
        [{"task_name": "a"}, {"task_name": "b"}],
        completions_per_prompt=3,
    )

    first = asyncio.run(evaluate(0, rollout_fn))
    second = asyncio.run(evaluate(5, rollout_fn))

    assert first["eval/attempted_trajectories"] == 6
    assert second["eval/completed_trajectories"] == 6
    assert {step for _, step, _ in calls} == {0, 5}
    assert len(calls) == 12
