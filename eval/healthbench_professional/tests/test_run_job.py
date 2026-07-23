from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.models.job.config import JobConfig

from healthbench_professional import adapter, run_job


def _prepared_dataset(root: Path, indices: list[int]) -> Path:
    for index in indices:
        dataset_id = f"synthetic-{index}"
        task = root / f"healthbench_professional_{dataset_id}"
        (task / "tests").mkdir(parents=True)
        (task / "task.toml").write_text('schema_version = "1.3"\n', encoding="utf-8")
        (task / "tests/source.json").write_text(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "dataset_repo": adapter.DATASET_ID,
                    "dataset_index": index,
                    "dataset_revision": adapter.DATASET_REVISION,
                }
            ),
            encoding="utf-8",
        )
    return root


def test_resolved_job_applies_model_limit_and_preserves_fixed_contract(
    tmp_path: Path,
) -> None:
    dataset = _prepared_dataset(tmp_path / "dataset", [2, 0, 1])
    model = "accounts/example/deployments/example"
    config = run_job.resolve_job_config(
        model=model,
        dataset_dir=dataset,
        jobs_dir=tmp_path / "jobs",
        job_name="healthbench-smoke",
        concurrency=3,
        limit=2,
        quiet=True,
    )

    validated = JobConfig.model_validate(config)
    assert validated.job_name == "healthbench-smoke"
    assert validated.jobs_dir == (tmp_path / "jobs").resolve()
    assert validated.n_attempts == 8
    assert validated.n_concurrent_trials == 3
    assert validated.quiet is True
    assert validated.agents[0].model_name == model
    assert validated.agents[0].kwargs == {
        "require_token_trajectory": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16384,
        "request_timeout_seconds": 1800.0,
    }
    assert validated.datasets[0].path == dataset.resolve()
    assert validated.datasets[0].task_names == [
        "healthbench_professional_synthetic-0",
        "healthbench_professional_synthetic-1",
    ]
    assert validated.datasets[0].n_tasks == 2


def test_full_resolved_job_requires_all_525_prepared_tasks(tmp_path: Path) -> None:
    short = _prepared_dataset(tmp_path / "short", list(range(2)))
    with pytest.raises(ValueError, match="exactly 525"):
        run_job.resolve_job_config(dataset_dir=short)

    full = _prepared_dataset(tmp_path / "full", list(range(adapter.EXPECTED_ROWS)))
    config = run_job.resolve_job_config(dataset_dir=full)
    dataset = config["datasets"][0]
    assert dataset["task_names"] == [
        f"healthbench_professional_synthetic-{index}"
        for index in range(adapter.EXPECTED_ROWS)
    ]
    assert dataset["n_tasks"] == adapter.EXPECTED_ROWS


def test_resolved_job_rejects_stray_harbor_tasks(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path / "dataset", [0])
    stray = prepared / "unrelated"
    (stray / "tests").mkdir(parents=True)
    (stray / "task.toml").write_text('schema_version = "1.3"\n', encoding="utf-8")
    (stray / "tests/source.json").write_text(
        json.dumps(
            {
                "dataset_id": "synthetic-stray",
                "dataset_repo": adapter.DATASET_ID,
                "dataset_index": 1,
                "dataset_revision": adapter.DATASET_REVISION,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected Harbor task"):
        run_job.resolve_job_config(dataset_dir=prepared, limit=1)


def test_resolved_job_rejects_non_fireworks_model_and_bad_source(
    tmp_path: Path,
) -> None:
    dataset = _prepared_dataset(tmp_path / "dataset", [0])
    with pytest.raises(ValueError, match="full Fireworks"):
        run_job.resolve_job_config(
            model="moonshot/kimi-k2.6", dataset_dir=dataset, limit=1
        )

    source = dataset / "healthbench_professional_synthetic-0/tests/source.json"
    source.write_text(
        json.dumps(
            {
                "dataset_id": "synthetic-0",
                "dataset_repo": adapter.DATASET_ID,
                "dataset_index": 0,
                "dataset_revision": "wrong",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected dataset revision"):
        run_job.resolve_job_config(dataset_dir=dataset, limit=1)


def test_run_harbor_uses_same_python_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "job.json"
    config.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command, check):
        assert check is False
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(run_job.subprocess, "run", fake_run)
    assert run_job.run_harbor(config) == 7
    assert calls == [
        [
            run_job.sys.executable,
            "-m",
            "harbor.cli.main",
            "run",
            "--config",
            str(config),
            "--yes",
        ]
    ]
