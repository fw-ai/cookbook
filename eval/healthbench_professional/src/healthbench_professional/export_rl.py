"""Export validated HealthBench ATIF trajectories as trainer-ready JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .trajectory import (
    ExactTokenTrajectory,
    TrajectoryContractError,
    load_atif_trajectory,
)


RL_SCHEMA_VERSION = "healthbench-professional-rl-v1"
HEALTHBENCH_RESULT_FILENAME = "healthbench_result.json"


def verifier_result_path_for(trajectory_path: Path | str) -> Path:
    """Return ``<trial>/verifier/healthbench_result.json`` for an agent trace."""

    path = Path(trajectory_path)
    if path.parent.name != "agent":
        raise TrajectoryContractError(
            "trajectory must live at <trial>/agent/trajectory.json to locate its reward"
        )
    return path.parent.parent / "verifier" / HEALTHBENCH_RESULT_FILENAME


def load_healthbench_reward(path: Path | str) -> float:
    """Load the canonical length-adjusted HealthBench Professional score."""

    result = load_healthbench_result(path)
    reward = result["metrics"].get("overall_score_length_adjusted")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TrajectoryContractError(
            "metrics.overall_score_length_adjusted must be a finite number"
        )
    reward = float(reward)
    if not math.isfinite(reward):
        raise TrajectoryContractError(
            "metrics.overall_score_length_adjusted must be a finite number"
        )
    return reward


def load_healthbench_result(path: Path | str) -> dict[str, Any]:
    """Load a verifier result while preserving optional provenance fields."""

    result_path = Path(path)
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrajectoryContractError(
            f"could not load HealthBench verifier result {result_path}: {exc}"
        ) from exc
    if not isinstance(result, dict) or not isinstance(result.get("metrics"), dict):
        raise TrajectoryContractError("verifier result must contain a metrics object")
    return result


def make_rl_record(
    exact: ExactTokenTrajectory,
    *,
    reward: float,
    verifier_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert exact token arrays to the cookbook's flat RL representation."""

    prompt = list(exact.prompt_token_ids)
    completion = list(exact.completion_token_ids)
    behavior_logprobs = list(exact.sampling_logprobs)
    record: dict[str, Any] = {
        "schema_version": RL_SCHEMA_VERSION,
        "prompt_token_ids": prompt,
        "completion_token_ids": completion,
        "sampling_logprobs": behavior_logprobs,
        "tokens": prompt + completion,
        "logprobs": [0.0] * len(prompt) + behavior_logprobs,
        "loss_mask": [0] * len(prompt) + [1] * len(completion),
        "reward": float(reward),
        "finish_reason": exact.finish_reason,
        # Keep raw model completion only for observability. Token IDs and
        # behavior logprobs remain authoritative and are never derived from it.
        "text": exact.raw_completion,
        "model": exact.model,
        "settings": dict(exact.settings),
    }
    if exact.provider_model is not None:
        record["provider_model"] = exact.provider_model
    if verifier_result is not None:
        provenance = verifier_result.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        dataset_id = verifier_result.get("dataset_id", provenance.get("dataset_id"))
        slices = verifier_result.get("slices", provenance.get("slices"))
        if dataset_id is not None:
            record["dataset_id"] = dataset_id
        if slices is not None:
            record["slices"] = slices
    return record


def export_rl_record(
    trajectory_path: Path | str,
    *,
    verifier_result_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate one persisted ATIF trajectory and attach its verifier reward."""

    exact = load_atif_trajectory(trajectory_path)
    result_path = (
        Path(verifier_result_path)
        if verifier_result_path is not None
        else verifier_result_path_for(trajectory_path)
    )
    verifier_result = load_healthbench_result(result_path)
    reward = verifier_result["metrics"].get("overall_score_length_adjusted")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TrajectoryContractError(
            "metrics.overall_score_length_adjusted must be a finite number"
        )
    reward = float(reward)
    if not math.isfinite(reward):
        raise TrajectoryContractError(
            "metrics.overall_score_length_adjusted must be a finite number"
        )
    return make_rl_record(
        exact,
        reward=reward,
        verifier_result=verifier_result,
    )


def discover_trajectory_paths(inputs: list[Path | str]) -> list[Path]:
    """Expand explicit trace files and Harbor job directories deterministically."""

    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw_input in inputs:
        input_path = Path(raw_input)
        if input_path.is_file():
            candidates = [input_path]
        elif input_path.is_dir():
            candidates = sorted(
                path
                for path in input_path.rglob("trajectory.json")
                if path.parent.name == "agent"
            )
            if not candidates:
                raise TrajectoryContractError(
                    f"no <trial>/agent/trajectory.json files found under {input_path}"
                )
        else:
            raise TrajectoryContractError(
                f"trajectory input does not exist: {input_path}"
            )

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                raise TrajectoryContractError(
                    f"duplicate trajectory input discovered: {candidate}"
                )
            seen.add(resolved)
            discovered.append(candidate)
    if not discovered:
        raise TrajectoryContractError("no trajectory inputs were provided")
    return discovered


def write_rl_jsonl(records: list[dict[str, Any]], output_path: Path | str) -> Path:
    """Write validated RL records as newline-delimited JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export HealthBench Professional ATIF trajectories for RL"
    )
    parser.add_argument(
        "trajectories",
        nargs="+",
        type=Path,
        help="One or more Harbor job directories or <trial>/agent/trajectory.json files",
    )
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args(argv)
    trajectories = discover_trajectory_paths(args.trajectories)
    records = [export_rl_record(path) for path in trajectories]
    write_rl_jsonl(records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HEALTHBENCH_RESULT_FILENAME",
    "RL_SCHEMA_VERSION",
    "discover_trajectory_paths",
    "export_rl_record",
    "load_healthbench_result",
    "load_healthbench_reward",
    "main",
    "make_rl_record",
    "verifier_result_path_for",
    "write_rl_jsonl",
]
