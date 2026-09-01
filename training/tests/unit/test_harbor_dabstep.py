from __future__ import annotations

import asyncio
import builtins
import importlib.util
import json
from types import MethodType

import pytest

from training.examples.rl.harbor.recipes.dabstep.manifest import (
    AdaptiveTaskSelector,
    DABstepManifest,
    rows_for_tasks,
    usable_group_probability,
)
from training.examples.rl.harbor.recipes.dabstep import (
    prepare_tasks as prepare_dabstep_tasks,
)
from training.examples.rl.harbor.recipes.dabstep.dataset import (
    DEFAULT_WAVE_SIZE,
    FrozenDABstepDataset,
    ProgressiveDABstepTasks,
    freeze_default_split,
    make_progressive_rollout_factory,
    shuffle_dataset_for_run,
)
from training.examples.rl.harbor.recipes import train_pi as pi_train
from training.examples.rl.harbor.tito.evaluate import (
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
        "training.examples.rl.harbor.recipes.dabstep.manifest._directory_sha256",
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
        "training.examples.rl.harbor.recipes.dabstep.manifest._directory_sha256",
        lambda path: "a" * 64,
    )

    with pytest.raises(ValueError, match="sign-insensitive numeric scorer"):
        manifest.verify_task_root(tmp_path)


def test_dabstep_hash_dependency_is_loaded_only_when_needed(tmp_path, monkeypatch):
    from training.examples.rl.harbor.recipes.dabstep import manifest as dabstep

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


def test_dabstep_dataset_freezes_complete_snapshot_order(tmp_path, monkeypatch):
    names = ("task-2", "task-1", "task-4", "task-3")
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for name in names:
        task = tasks / name
        (task / "tests").mkdir(parents=True)
        (task / "task.toml").write_text("", encoding="utf-8")
        (task / "tests" / "scorer.py").write_text("score = 1\n", encoding="utf-8")
    snapshot = {
        "dataset": {
            "source_task_count": 4,
            "parent_status": "match",
            "stored_ref": "adyen/dabstep@test",
            "recomputed_ref": "adyen/dabstep@test",
        },
        "selection": {"count": 4},
        "tasks": [{"name": name} for name in names],
    }
    (tasks / ".fw-ai-package-snapshot.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    monkeypatch.setattr(
        "training.examples.rl.harbor.recipes.dabstep.dataset."
        "EXPECTED_DEFAULT_SPLIT_TASKS",
        4,
    )
    monkeypatch.setattr(
        "training.examples.rl.harbor.recipes.dabstep.dataset._directory_sha256",
        lambda path: f"hash-{path.name}",
    )

    dataset = freeze_default_split(tasks, manifest_path=tmp_path / "manifest.json")

    assert dataset.task_names == names
    assert dataset.evaluation_tasks == names
    assert [row["task_name"] for row in dataset.rollout_rows()] == list(names)
    assert json.loads(dataset.manifest_path.read_text())["task_names"] == list(names)


def test_dabstep_dataset_persists_seeded_run_order(tmp_path):
    names = tuple(f"task-{index}" for index in range(8))
    dataset = FrozenDABstepDataset(
        task_root=tmp_path,
        task_names=names,
        evaluation_tasks=names[:4],
        content_sha256={name: name for name in names},
        scorer_sha256={name: name for name in names},
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="dataset-digest",
    )
    order_path = tmp_path / "run-order.json"

    shuffled = shuffle_dataset_for_run(dataset, seed=19, order_path=order_path)
    repeated = shuffle_dataset_for_run(dataset, seed=19, order_path=order_path)

    assert shuffled.task_names == repeated.task_names
    assert shuffled.task_names != names
    assert set(shuffled.task_names) == set(names)
    assert shuffled.evaluation_tasks == names[:4]
    assert [row["task_name"] for row in shuffled.evaluation_rows()] == list(names[:4])
    document = json.loads(order_path.read_text())
    assert document["shuffle_seed"] == 19
    assert document["task_names"] == list(shuffled.task_names)
    assert document["task_count"] == len(names)
    assert DEFAULT_WAVE_SIZE == 64


def test_dabstep_pi_recipe_uses_sdk_managed_resources_and_offpolicy_two(tmp_path):
    args = pi_train.parse_args(
        [
            "--base-model",
            "accounts/example/models/policy",
            "--tokenizer-model",
            "example/tokenizer",
            "--renderer-name",
            "example-renderer",
            "--harbor-dataset",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--shuffle-seed",
            "17",
        ]
    )

    config = pi_train._build_config(args, run_dir=tmp_path, row_count=450)

    assert config.completions_per_prompt == 8
    assert config.prompt_groups_per_step == 8
    assert config.pipeline_chunks_per_step == 2
    assert config.max_head_offpolicy_versions == 2
    assert config.server_side_grpo is True
    assert config.trainer.job_id is None
    assert config.trainer.training_shape_id is None
    assert config.deployment.deployment_id is None
    assert config.deployment.deployment_shape is None
    assert config.deployment.hot_load_trainer_job is None
    assert config.cleanup_on_exit is True
    assert not hasattr(args, "training_shape_id")
    assert not hasattr(args, "deployment_shape")


def test_dabstep_launch_manifest_records_automatic_resource_selection(tmp_path):
    args = pi_train.parse_args(
        [
            "--base-model",
            "accounts/example/models/policy",
            "--tokenizer-model",
            "example/tokenizer",
            "--renderer-name",
            "example-renderer",
            "--harbor-dataset",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--shuffle-seed",
            "17",
        ]
    )
    output = tmp_path / "launch.json"

    pi_train._write_launch_manifest(
        output,
        args=args,
        task_names=("task-a", "task-b"),
        evaluation_tasks=("task-a",),
        dataset_manifest_sha256="a" * 64,
    )

    document = json.loads(output.read_text())
    assert document["harness"] == "pi"
    assert document["environment"] == "e2b"
    assert document["resources"] == {
        "cleanup_on_exit": True,
        "replica_count": 4,
        "selection": "automatic",
    }
    assert document["training"]["max_head_offpolicy_versions"] == 2


def test_progressive_dabstep_prepares_only_sequential_training_waves(tmp_path):
    names = tuple(f"task-{index}" for index in range(6))
    dataset = FrozenDABstepDataset(
        task_root=tmp_path,
        task_names=names,
        evaluation_tasks=names[:4],
        content_sha256={name: name for name in names},
        scorer_sha256={name: name for name in names},
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="digest",
    )
    waves = ProgressiveDABstepTasks(
        dataset,
        run_root=tmp_path / "run",
        wave_size=2,
        context_limit=4096,
        output_limit=1024,
    )
    prepared: list[int] = []

    async def prepare(self, wave):
        prepared.append(wave)
        start = wave * self.wave_size
        stop = min(start + self.wave_size, len(names))
        self._rows.update(
            {
                index: {"task_name": names[index], "prepared": True}
                for index in range(start, stop)
            }
        )
        self._ready_waves.add(wave)

    waves._prepare_wave = MethodType(prepare, waves)

    async def exercise():
        first = await waves.row_for(dataset.rollout_rows()[0], evaluation=False)
        with pytest.raises(RuntimeError, match="crossed"):
            await waves.row_for(dataset.rollout_rows()[4], evaluation=False)
        evaluated = await waves.row_for(dataset.rollout_rows()[4], evaluation=True)
        second = await waves.row_for(dataset.rollout_rows()[2], evaluation=False)
        return first, evaluated, second

    first, evaluated, second = asyncio.run(exercise())
    assert first == {"task_name": "task-0", "prepared": True}
    assert evaluated == {"task_name": "task-4", "prepared": True}
    assert second == {"task_name": "task-2", "prepared": True}
    assert prepared == [0, 2, 1]


def test_progressive_rollout_factory_keeps_harness_out_of_dataset_logic(tmp_path):
    names = ("task-a", "task-b", "task-c", "task-d")
    dataset = FrozenDABstepDataset(
        task_root=tmp_path,
        task_names=names,
        evaluation_tasks=names,
        content_sha256={name: name for name in names},
        scorer_sha256={name: name for name in names},
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="digest",
    )
    waves = ProgressiveDABstepTasks(
        dataset,
        run_root=tmp_path / "run",
        context_limit=4096,
        output_limit=1024,
    )

    async def row_for(placeholder, *, evaluation):
        return {"task_name": placeholder["task_name"], "prepared": evaluation}

    waves.row_for = row_for
    calls = []

    def base_factory(setup):
        del setup

        async def base(row, *, evaluation=False, **context):
            calls.append((row, evaluation, context))
            return "result"

        async def close():
            calls.append("closed")

        base.aclose = close
        return base

    rollout = make_progressive_rollout_factory(waves, base_factory)(object())

    async def exercise():
        result = await rollout(
            dataset.rollout_rows()[0], evaluation=True, sample_index=3
        )
        await rollout.aclose()
        return result

    assert asyncio.run(exercise()) == "result"
    assert calls == [
        (
            {"task_name": "task-a", "prepared": True},
            True,
            {"sample_index": 3},
        ),
        "closed",
    ]


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
                "tito_metrics": {
                    "tito/lineage/boundary_reason_history_rewrite": sample_index,
                    "tito/lineage/prefix_mismatch": sample_index,
                    "tito/lineage/new_segment": 1,
                    "tito/lineage/prefix_check": 0,
                    "tito/trajectory/policy_turns_count": 1,
                    "tito/trajectory/policy_turns_sum": 1,
                    "tito/trajectory/policy_turns_mean": 1,
                    "tito/trajectory/policy_turns_min": 1,
                    "tito/trajectory/policy_turns_max": 1,
                    "tito/turn/completion_tokens_count": 1,
                    "tito/turn/completion_tokens_sum": 2 + sample_index,
                    "tito/turn/completion_tokens_mean": 2 + sample_index,
                    "tito/turn/completion_tokens_min": 2 + sample_index,
                    "tito/turn/completion_tokens_max": 2 + sample_index,
                },
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
    assert metrics["tito/trajectory/count"] == 2
    assert metrics["tito/turn/count"] == 2
    assert metrics["tito/turn/output_tokens_mean"] == 2.5
    assert metrics["tito/turn/output_tokens_min"] == 2
    assert metrics["tito/turn/output_tokens_max"] == 3
    assert not any(name.startswith("tito/debug/") for name in metrics)


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
