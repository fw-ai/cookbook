"""Aggregate Harbor HealthBench rewards, clipping only after the mean."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REQUIRED_METRICS = ("overall_score_length_adjusted", "overall_score")


def _numeric_reward(value: object, *, metric: str, trial_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Trial {trial_index} metric {metric!r} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Trial {trial_index} metric {metric!r} must be finite")
    return value


def aggregate_rewards(
    rewards: Sequence[Mapping[str, object] | None],
) -> dict[str, float | int]:
    """Mean each metric over matching trials, then clip its mean to [0, 1]."""

    if not rewards:
        raise ValueError("Cannot aggregate an empty HealthBench reward set")

    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for trial_index, reward in enumerate(rewards):
        if reward is None:
            raise ValueError(f"Trial {trial_index} has no HealthBench reward")
        if not isinstance(reward, Mapping):
            raise ValueError(f"Trial {trial_index} reward must be an object")
        for metric in REQUIRED_METRICS:
            if metric not in reward:
                raise ValueError(
                    f"Trial {trial_index} reward is missing required metric {metric!r}"
                )
        for metric, value in reward.items():
            if not isinstance(metric, str) or not metric:
                raise ValueError(f"Trial {trial_index} has an invalid metric name")
            values_by_metric[metric].append(
                _numeric_reward(value, metric=metric, trial_index=trial_index)
            )

    result: dict[str, float | int] = {}
    ordered_metrics = [
        *REQUIRED_METRICS,
        *sorted(set(values_by_metric) - set(REQUIRED_METRICS)),
    ]
    for metric in ordered_metrics:
        values = values_by_metric[metric]
        mean = sum(values) / len(values)
        result[metric] = min(1.0, max(0.0, mean))
        result[f"{metric}:n_samples"] = len(values)
    result["n_trials"] = len(rewards)
    return result


def read_rewards(path: Path) -> list[dict[str, object] | None]:
    rewards: list[dict[str, object] | None] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            reward = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid reward JSON on line {line_number}") from exc
        if reward is not None and not isinstance(reward, dict):
            raise ValueError(f"Reward line {line_number} must be an object or null")
        rewards.append(reward)
    return rewards


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def summarize_job(job_dir: Path) -> dict[str, Any]:
    """Validate and summarize a completed Harbor job directory."""

    job_dir = Path(job_dir)
    config_path = job_dir / "config.json"
    result_path = job_dir / "result.json"
    if not config_path.is_file() or not result_path.is_file():
        raise ValueError(
            f"Harbor job must contain config.json and result.json: {job_dir}"
        )

    config = _read_json_object(config_path)
    job_result = _read_json_object(result_path)
    if not job_result.get("finished_at"):
        raise ValueError(
            f"Harbor job is incomplete (finished_at is missing): {job_dir}"
        )
    expected_trials = job_result.get("n_total_trials")
    if (
        isinstance(expected_trials, bool)
        or not isinstance(expected_trials, int)
        or expected_trials < 1
    ):
        raise ValueError(
            "Harbor result.json requires a positive integer n_total_trials"
        )

    detail_paths = sorted(job_dir.rglob("verifier/healthbench_result.json"))
    if len(detail_paths) != expected_trials:
        missing = max(0, expected_trials - len(detail_paths))
        extra = max(0, len(detail_paths) - expected_trials)
        raise ValueError(
            "Incomplete HealthBench job: "
            f"expected_trials={expected_trials}, completed_results={len(detail_paths)}, "
            f"missing_results={missing}, extra_results={extra}"
        )

    n_attempts = config.get("n_attempts")
    agents = config.get("agents")
    if (
        isinstance(n_attempts, bool)
        or not isinstance(n_attempts, int)
        or n_attempts < 1
    ):
        raise ValueError("Harbor config.json requires a positive integer n_attempts")
    if not isinstance(agents, list) or not agents:
        raise ValueError("Harbor config.json requires at least one agent")
    trials_per_task = n_attempts * len(agents)
    if expected_trials % trials_per_task:
        raise ValueError(
            f"n_total_trials={expected_trials} is not divisible by attempts*agents={trials_per_task}"
        )
    expected_tasks = expected_trials // trials_per_task

    rewards: list[dict[str, object]] = []
    rewards_by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    counts_by_dataset_id: dict[str, int] = defaultdict(int)
    for details_path in detail_paths:
        details = _read_json_object(details_path)
        metrics = details.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(
                f"HealthBench details missing metrics object: {details_path}"
            )
        dataset_id = details.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"HealthBench details missing dataset_id: {details_path}")

        trial_dir = details_path.parent.parent
        trial_config_path = trial_dir / "config.json"
        trial_config = _read_json_object(trial_config_path)
        agent = trial_config.get("agent")
        model_name = agent.get("model_name") if isinstance(agent, dict) else None
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(
                f"Trial config missing agent.model_name: {trial_config_path}"
            )

        # aggregate_rewards performs the numeric and finite-value validation.
        rewards.append(metrics)
        rewards_by_model[model_name].append(metrics)
        counts_by_dataset_id[dataset_id] += 1

    if len(counts_by_dataset_id) != expected_tasks:
        raise ValueError(
            "HealthBench task completeness mismatch: "
            f"expected_tasks={expected_tasks}, observed_task_ids={len(counts_by_dataset_id)}"
        )
    wrong_attempt_counts = {
        dataset_id: count
        for dataset_id, count in counts_by_dataset_id.items()
        if count != trials_per_task
    }
    if wrong_attempt_counts:
        raise ValueError(
            f"HealthBench task attempt counts must equal {trials_per_task}: {wrong_attempt_counts}"
        )

    expected_model_counts: dict[str, int] = defaultdict(int)
    for agent in agents:
        if not isinstance(agent, dict) or not isinstance(agent.get("model_name"), str):
            raise ValueError("Every configured HealthBench agent requires model_name")
        expected_model_counts[agent["model_name"]] += expected_tasks * n_attempts
    actual_model_counts = {
        model: len(model_rewards) for model, model_rewards in rewards_by_model.items()
    }
    if actual_model_counts != dict(expected_model_counts):
        raise ValueError(
            "HealthBench per-model completeness mismatch: "
            f"expected={dict(expected_model_counts)}, actual={actual_model_counts}"
        )

    return {
        "metrics": aggregate_rewards(rewards),
        "by_model": {
            model: aggregate_rewards(model_rewards)
            for model, model_rewards in sorted(rewards_by_model.items())
        },
        "diagnostics": {
            "job_dir": str(job_dir),
            "expected_trials": expected_trials,
            "completed_results": len(detail_paths),
            "missing_results": 0,
            "n_tasks": expected_tasks,
            "n_attempts": n_attempts,
            "models": actual_model_counts,
        },
    }


def main(input_path: Path, output_path: Path) -> None:
    result = (
        summarize_job(input_path)
        if input_path.is_dir()
        else aggregate_rewards(read_rewards(input_path))
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-path", type=Path, required=True)
    parser.add_argument("-o", "--output-path", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_path, args.output_path)
