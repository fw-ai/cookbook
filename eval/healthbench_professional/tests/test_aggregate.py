from __future__ import annotations

import json
from pathlib import Path

import pytest

from healthbench_professional.aggregate import (
    aggregate_rewards,
    main,
    read_rewards,
    summarize_job,
)


def test_aggregate_means_before_clipping() -> None:
    result = aggregate_rewards(
        [
            {
                "overall_score_length_adjusted": 1.2,
                "overall_score": -0.2,
                "specialty_synthetic": 1.2,
            },
            {
                "overall_score_length_adjusted": 0.9,
                "overall_score": 0.1,
            },
        ]
    )
    # The adjusted mean is 1.05 and the raw mean is -0.05; only those means clip.
    assert result["overall_score_length_adjusted"] == 1.0
    assert result["overall_score"] == 0.0
    assert result["specialty_synthetic"] == 1.0
    assert result["overall_score_length_adjusted:n_samples"] == 2
    assert result["specialty_synthetic:n_samples"] == 1
    assert result["n_trials"] == 2


def test_aggregate_does_not_clip_samples_before_mean() -> None:
    result = aggregate_rewards(
        [
            {"overall_score_length_adjusted": 1.4, "overall_score": 1.4},
            {"overall_score_length_adjusted": -0.6, "overall_score": -0.6},
        ]
    )
    assert result["overall_score_length_adjusted"] == pytest.approx(0.4)
    assert result["overall_score"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "rewards, match",
    [
        ([], "empty"),
        ([None], "has no HealthBench reward"),
        ([{"overall_score": 0.5}], "missing required metric"),
        (
            [{"overall_score_length_adjusted": float("nan"), "overall_score": 0.5}],
            "must be finite",
        ),
    ],
)
def test_aggregate_fails_closed_on_invalid_trials(rewards, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        aggregate_rewards(rewards)


def test_uv_script_io_contract(tmp_path: Path) -> None:
    input_path = tmp_path / "rewards.jsonl"
    output_path = tmp_path / "metric.json"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"overall_score_length_adjusted": 0.4, "overall_score": 0.5}
                ),
                json.dumps(
                    {"overall_score_length_adjusted": 0.6, "overall_score": 0.7}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert len(read_rewards(input_path)) == 2
    main(input_path, output_path)
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["overall_score_length_adjusted"] == pytest.approx(0.5)
    assert result["overall_score"] == pytest.approx(0.6)


def _write_synthetic_job(job_dir: Path) -> None:
    (job_dir / "config.json").parent.mkdir(parents=True, exist_ok=True)
    (job_dir / "config.json").write_text(
        json.dumps(
            {
                "n_attempts": 2,
                "agents": [{"model_name": "accounts/synthetic/models/test"}],
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "result.json").write_text(
        json.dumps({"finished_at": "2026-01-01T00:00:00Z", "n_total_trials": 4}),
        encoding="utf-8",
    )
    trial_number = 0
    for dataset_id, scores in {
        "synthetic-a": (0.2, 0.4),
        "synthetic-b": (0.6, 0.8),
    }.items():
        for score in scores:
            trial_dir = job_dir / f"trial-{trial_number}"
            verifier_dir = trial_dir / "verifier"
            verifier_dir.mkdir(parents=True)
            (trial_dir / "config.json").write_text(
                json.dumps({"agent": {"model_name": "accounts/synthetic/models/test"}}),
                encoding="utf-8",
            )
            (verifier_dir / "healthbench_result.json").write_text(
                json.dumps(
                    {
                        "dataset_id": dataset_id,
                        "metrics": {
                            "overall_score_length_adjusted": score,
                            "overall_score": score,
                        },
                    }
                ),
                encoding="utf-8",
            )
            trial_number += 1


def test_summarize_complete_harbor_job_directory(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_synthetic_job(job_dir)
    summary = summarize_job(job_dir)
    assert summary["metrics"]["overall_score_length_adjusted"] == pytest.approx(0.5)
    assert summary["diagnostics"] == {
        "job_dir": str(job_dir),
        "expected_trials": 4,
        "completed_results": 4,
        "missing_results": 0,
        "n_tasks": 2,
        "n_attempts": 2,
        "models": {"accounts/synthetic/models/test": 4},
    }
    assert summary["by_model"]["accounts/synthetic/models/test"]["n_trials"] == 4


def test_summarize_job_fails_on_missing_details(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_synthetic_job(job_dir)
    next(job_dir.rglob("verifier/healthbench_result.json")).unlink()
    with pytest.raises(ValueError, match="expected_trials=4.*missing_results=1"):
        summarize_job(job_dir)
