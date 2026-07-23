from __future__ import annotations

import json

import pytest

from healthbench_professional.export_rl import (
    RL_SCHEMA_VERSION,
    discover_trajectory_paths,
    export_rl_record,
    load_healthbench_reward,
    write_rl_jsonl,
)
from healthbench_professional.trajectory import (
    TrajectoryContractError,
    build_atif_trajectory,
    extract_fireworks_response,
    write_atif_trajectory,
)


SETTINGS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 16_384,
    "raw_output": True,
    "return_token_ids": True,
    "logprobs": True,
    "context_length_exceeded_behavior": "error",
    "n": 1,
    "stream": False,
}
MESSAGES = [{"role": "user", "content": "Synthetic question"}]


def make_trial(tmp_path, name="trial-a"):
    trial = tmp_path / name
    agent_dir = trial / "agent"
    verifier_dir = trial / "verifier"
    agent_dir.mkdir(parents=True)
    verifier_dir.mkdir(parents=True)
    response = {
        "model": "accounts/provider/models/base",
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Visible"},
                "raw_output": {
                    "prompt_token_ids": [1, 2],
                    "completion_token_ids": [3, 4],
                    "completion": "Raw completion",
                    "completion_logprobs": {
                        "content": [
                            {"token_id": 3, "logprob": -8.0, "sampling_logprob": -0.1},
                            {"token_id": 4, "logprob": -9.0, "sampling_logprob": -0.2},
                        ]
                    },
                },
            }
        ],
    }
    exact = extract_fireworks_response(
        response,
        model="accounts/example/deployments/synthetic",
        settings=SETTINGS,
    )
    document = build_atif_trajectory(
        exact,
        instruction=json.dumps({"schema_version": 1, "messages": MESSAGES}),
        messages=MESSAGES,
        agent_name="healthbench-professional",
        agent_version="0.1.0",
    )
    trajectory_path = write_atif_trajectory(agent_dir / "trajectory.json", document)
    result = {
        "dataset_id": "synthetic-1",
        "slices": {"specialty": "synthetic"},
        "metrics": {
            "overall_score": 0.75,
            "overall_score_length_adjusted": 0.625,
        },
    }
    result_path = verifier_dir / "healthbench_result.json"
    result_path.write_text(json.dumps(result))
    return trajectory_path, result_path


def test_exports_flat_rl_record_from_persisted_atif_and_reward(tmp_path) -> None:
    trajectory_path, _ = make_trial(tmp_path)

    record = export_rl_record(trajectory_path)

    assert record == {
        "schema_version": RL_SCHEMA_VERSION,
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3, 4],
        "sampling_logprobs": [-0.1, -0.2],
        "tokens": [1, 2, 3, 4],
        "logprobs": [0.0, 0.0, -0.1, -0.2],
        "loss_mask": [0, 0, 1, 1],
        "reward": 0.625,
        "finish_reason": "stop",
        "text": "Raw completion",
        "model": "accounts/example/deployments/synthetic",
        "settings": SETTINGS,
        "provider_model": "accounts/provider/models/base",
        "dataset_id": "synthetic-1",
        "slices": {"specialty": "synthetic"},
    }


def test_reward_loader_requires_finite_length_adjusted_score(tmp_path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"metrics": {"overall_score_length_adjusted": "bad"}}))
    with pytest.raises(TrajectoryContractError, match="finite"):
        load_healthbench_reward(path)


def test_recursive_discovery_is_sorted_and_rejects_duplicates(tmp_path) -> None:
    second, _ = make_trial(tmp_path, "trial-z")
    first, _ = make_trial(tmp_path, "trial-a")

    assert discover_trajectory_paths([tmp_path]) == [first, second]
    with pytest.raises(TrajectoryContractError, match="duplicate"):
        discover_trajectory_paths([tmp_path, first])

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TrajectoryContractError, match="no .*trajectory"):
        discover_trajectory_paths([empty])


def test_writes_jsonl(tmp_path) -> None:
    trajectory_path, _ = make_trial(tmp_path)
    record = export_rl_record(trajectory_path)
    output = write_rl_jsonl([record], tmp_path / "rollouts.jsonl")

    assert json.loads(output.read_text()) == record
