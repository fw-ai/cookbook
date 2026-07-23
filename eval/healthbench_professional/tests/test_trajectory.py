from __future__ import annotations

import copy
import json
import math

import pytest

from healthbench_professional.trajectory import (
    TrajectoryContractError,
    build_atif_trajectory,
    extract_fireworks_response,
    load_atif_trajectory,
    validate_atif_trajectory,
    validate_sampling_settings,
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
MESSAGES = [
    {"role": "system", "content": "You are a clinician."},
    {"role": "user", "content": "Earlier question"},
    {"role": "assistant", "content": "Earlier answer"},
    {"role": "user", "content": "What should I do next?"},
]


def response_fixture() -> dict:
    raw_logprobs = [
        {"token_id": 30, "logprob": -9.0, "sampling_logprob": -0.1},
        {"token_id": 31, "logprob": -8.0, "sampling_logprob": -0.2},
        {"token_id": 32, "logprob": -7.0, "sampling_logprob": -0.3},
    ]
    return {
        "id": "request-1",
        "model": "accounts/provider/models/base",
        "prompt_token_ids": [10, 11, 12],
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Visible answer",
                    "reasoning_content": "Private reasoning",
                },
                # Parsed chat logprobs cover only visible content. They must not
                # replace or invalidate the raw completion sequence.
                "logprobs": {
                    "content": [
                        {
                            "token_id": 32,
                            "logprob": -7.0,
                            "sampling_logprob": -0.3,
                        }
                    ]
                },
                "raw_output": {
                    "prompt_token_ids": [10, 11, 12],
                    "completion_token_ids": [30, 31, 32],
                    "completion": "<think>Private reasoning</think>Visible answer",
                    "completion_logprobs": {"content": raw_logprobs},
                },
            }
        ],
    }


def exact_fixture():
    return extract_fireworks_response(
        response_fixture(),
        model="accounts/example/deployments/model",
        settings=SETTINGS,
    )


def test_extracts_authoritative_raw_token_sequence_without_retokenizing() -> None:
    exact = exact_fixture()

    assert exact.prompt_token_ids == (10, 11, 12)
    assert exact.completion_token_ids == (30, 31, 32)
    assert exact.sampling_logprobs == (-0.1, -0.2, -0.3)
    assert exact.visible_text == "Visible answer"
    assert exact.raw_completion == "<think>Private reasoning</think>Visible answer"
    assert exact.reasoning_content == "Private reasoning"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.pop("prompt_token_ids"), "prompt_token_ids"),
        (
            lambda value: value["choices"][0]["raw_output"].pop("completion_token_ids"),
            "completion_token_ids",
        ),
        (
            lambda value: value["choices"][0]["raw_output"].pop("completion_logprobs"),
            "completion_logprobs",
        ),
        (
            lambda value: value["choices"][0]["raw_output"]["completion_logprobs"][
                "content"
            ][0].pop("sampling_logprob"),
            "sampling_logprob",
        ),
        (
            lambda value: value["choices"][0]["raw_output"]["completion_logprobs"][
                "content"
            ][0].update(token_id=999),
            "does not match",
        ),
        (
            lambda value: value["choices"][0]["raw_output"]["completion_logprobs"][
                "content"
            ][0].update(sampling_logprob=math.nan),
            "finite",
        ),
        (
            lambda value: value["choices"][0]["raw_output"]["completion_logprobs"][
                "content"
            ].pop(),
            "entries",
        ),
        (
            lambda value: value["choices"][0].pop("finish_reason"),
            "finish_reason",
        ),
    ],
)
def test_response_extraction_fails_closed(mutation, match: str) -> None:
    response = response_fixture()
    mutation(response)

    with pytest.raises(TrajectoryContractError, match=match):
        extract_fireworks_response(
            response,
            model="accounts/example/deployments/model",
            settings=SETTINGS,
        )


def test_raw_logprob_is_never_a_sampling_logprob_fallback() -> None:
    response = response_fixture()
    for item in response["choices"][0]["raw_output"]["completion_logprobs"]["content"]:
        item.pop("sampling_logprob")

    with pytest.raises(TrajectoryContractError, match="sampling_logprob"):
        extract_fireworks_response(
            response,
            model="accounts/example/deployments/model",
            settings=SETTINGS,
        )


def test_matching_full_choice_logprobs_are_cross_checked() -> None:
    response = response_fixture()
    response["choices"][0]["logprobs"] = copy.deepcopy(
        response["choices"][0]["raw_output"]["completion_logprobs"]
    )
    response["choices"][0]["logprobs"]["content"][1]["sampling_logprob"] = -9.9

    with pytest.raises(TrajectoryContractError, match="disagrees"):
        extract_fireworks_response(
            response,
            model="accounts/example/deployments/model",
            settings=SETTINGS,
        )


def test_atif_round_trip_preserves_context_and_exact_arrays(tmp_path) -> None:
    instruction = json.dumps({"schema_version": 1, "messages": MESSAGES})
    document = build_atif_trajectory(
        exact_fixture(),
        instruction=instruction,
        messages=MESSAGES,
        agent_name="healthbench-professional",
        agent_version="0.1.0",
        session_id="session-1",
    )

    assert [step["source"] for step in document["steps"]] == [
        "system",
        "user",
        "agent",
        "user",
        "agent",
    ]
    assert document["steps"][2]["is_copied_context"] is True
    assert "metrics" not in document["steps"][2]
    generated_metrics = document["steps"][-1]["metrics"]
    assert generated_metrics["prompt_token_ids"] == [10, 11, 12]
    assert generated_metrics["completion_token_ids"] == [30, 31, 32]
    assert generated_metrics["logprobs"] == [-0.1, -0.2, -0.3]
    assert generated_metrics["extra"]["settings"] == SETTINGS

    path = write_atif_trajectory(tmp_path / "trajectory.json", document)
    loaded = load_atif_trajectory(path)
    assert loaded == exact_fixture()


def test_persisted_atif_validation_rejects_misaligned_arrays() -> None:
    document = build_atif_trajectory(
        exact_fixture(),
        instruction=json.dumps({"schema_version": 1, "messages": MESSAGES}),
        messages=MESSAGES,
        agent_name="healthbench-professional",
        agent_version="0.1.0",
    )
    document["steps"][-1]["metrics"]["logprobs"].pop()

    with pytest.raises(TrajectoryContractError, match="entries"):
        validate_atif_trajectory(document)


def test_settings_require_strict_non_streaming_token_capture() -> None:
    for forbidden in ("top_k", "echo", "reasoning_effort"):
        settings = dict(SETTINGS)
        settings[forbidden] = None
        with pytest.raises(TrajectoryContractError, match="omit"):
            validate_sampling_settings(settings)

    settings = dict(SETTINGS, context_length_exceeded_behavior="truncate")
    with pytest.raises(TrajectoryContractError, match="error"):
        validate_sampling_settings(settings)
