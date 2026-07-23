from __future__ import annotations

import json

import pytest
from harbor.models.agent.context import AgentContext

from healthbench_professional.agent import (
    ANSWER_PATH,
    HealthBenchProfessionalAgent,
    REQUEST_MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    normalize_fireworks_model,
    parse_instruction_messages,
)
from healthbench_professional.trajectory import (
    TrajectoryContractError,
    load_atif_trajectory,
)


MESSAGES = [
    {"role": "system", "content": "Answer as a medical professional."},
    {"role": "user", "content": "Synthetic question"},
]


def response_fixture() -> dict:
    return {
        "id": "request-123",
        "model": "accounts/provider/models/base",
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Visible synthetic answer",
                    "reasoning_content": "Synthetic reasoning",
                },
                "raw_output": {
                    "prompt_token_ids": [1, 2],
                    "completion_token_ids": [3, 4],
                    "completion": "<think>Synthetic reasoning</think>Visible synthetic answer",
                    "completion_logprobs": {
                        "content": [
                            {"token_id": 3, "logprob": -5.0, "sampling_logprob": -0.1},
                            {"token_id": 4, "logprob": -6.0, "sampling_logprob": -0.2},
                        ]
                    },
                },
            }
        ],
    }


class FakeCompletions:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class FakeClient:
    def __init__(self, response: dict) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(response)


class FakeEnvironment:
    def __init__(self) -> None:
        self.uploads = []

    async def upload_file(self, source_path, target_path):
        self.uploads.append((source_path, target_path))


@pytest.mark.asyncio
async def test_agent_preserves_messages_and_records_exact_rl_trace(tmp_path) -> None:
    client = FakeClient(response_fixture())
    environment = FakeEnvironment()
    context = AgentContext()
    instruction = json.dumps({"schema_version": 1, "messages": MESSAGES})
    agent = HealthBenchProfessionalAgent(
        logs_dir=tmp_path,
        model_name="fireworks_ai/accounts/example/deployments/synthetic",
        client=client,
        require_token_trajectory=True,
    )
    agent.session_id = "session-123"

    await agent.run(instruction, environment, context)

    request = client.chat.completions.kwargs
    assert request["messages"] == MESSAGES
    assert request == {
        "model": "accounts/example/deployments/synthetic",
        "messages": MESSAGES,
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
    assert "top_k" not in request
    assert "echo" not in request
    assert "reasoning_effort" not in request

    response_path = tmp_path / "response.txt"
    assert response_path.read_text() == "Visible synthetic answer"
    assert environment.uploads == [(response_path, ANSWER_PATH)]

    exact = load_atif_trajectory(tmp_path / "trajectory.json")
    assert exact.prompt_token_ids == (1, 2)
    assert exact.completion_token_ids == (3, 4)
    assert exact.sampling_logprobs == (-0.1, -0.2)
    assert context.rollout_details == [exact.rollout_detail()]
    assert context.metadata == {
        "finish_reason": "stop",
        "model": "accounts/example/deployments/synthetic",
        "provider_model": "accounts/provider/models/base",
        "reasoning_present": True,
        "request_id": "request-123",
        "require_token_trajectory": True,
        "settings": exact.settings,
    }


@pytest.mark.asyncio
async def test_agent_fails_before_writing_answer_when_trace_is_invalid(
    tmp_path,
) -> None:
    response = response_fixture()
    response["choices"][0]["raw_output"]["completion_logprobs"]["content"][0].pop(
        "sampling_logprob"
    )
    client = FakeClient(response)
    environment = FakeEnvironment()
    agent = HealthBenchProfessionalAgent(
        logs_dir=tmp_path,
        model_name="accounts/example/deployments/synthetic",
        client=client,
    )

    with pytest.raises(TrajectoryContractError, match="sampling_logprob"):
        await agent.run(
            json.dumps({"schema_version": 1, "messages": MESSAGES}),
            environment,
            AgentContext(),
        )

    assert not (tmp_path / "response.txt").exists()
    assert not (tmp_path / "trajectory.json").exists()
    assert environment.uploads == []


def test_strict_trace_setting_cannot_be_disabled(tmp_path) -> None:
    with pytest.raises(ValueError, match="must remain true"):
        HealthBenchProfessionalAgent(
            logs_dir=tmp_path,
            model_name="accounts/example/deployments/synthetic",
            client=FakeClient(response_fixture()),
            require_token_trajectory=False,
        )

    with pytest.raises(ValueError, match="request_timeout_seconds must remain"):
        HealthBenchProfessionalAgent(
            logs_dir=tmp_path,
            model_name="accounts/example/deployments/synthetic",
            client=FakeClient(response_fixture()),
            request_timeout_seconds=60,
        )


def test_model_normalization_and_instruction_validation() -> None:
    assert (
        normalize_fireworks_model("fireworks_ai/accounts/a/models/b")
        == "accounts/a/models/b"
    )
    assert normalize_fireworks_model("accounts/a/models/b") == "accounts/a/models/b"
    instruction = json.dumps({"schema_version": 1, "messages": MESSAGES})
    assert parse_instruction_messages(instruction) == MESSAGES

    with pytest.raises(TrajectoryContractError, match="schema_version"):
        parse_instruction_messages(
            json.dumps({"schema_version": 2, "messages": MESSAGES})
        )


@pytest.mark.asyncio
async def test_harbor_extra_env_supplies_sdk_credentials_without_persisting_them(
    tmp_path, monkeypatch
) -> None:
    captured = {}
    fake_client = FakeClient(response_fixture())

    def fake_async_fireworks(**kwargs):
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr(
        "healthbench_professional.agent.AsyncFireworks", fake_async_fireworks
    )
    agent = HealthBenchProfessionalAgent(
        logs_dir=tmp_path,
        model_name="accounts/example/deployments/synthetic",
        extra_env={
            "FIREWORKS_API_KEY": "synthetic-secret",
            "FIREWORKS_BASE_URL": "https://synthetic.invalid",
        },
    )
    context = AgentContext()
    await agent.run(
        json.dumps({"schema_version": 1, "messages": MESSAGES}),
        FakeEnvironment(),
        context,
    )

    assert captured == {
        "api_key": "synthetic-secret",
        "base_url": "https://synthetic.invalid",
        "timeout": REQUEST_TIMEOUT_SECONDS,
        "max_retries": REQUEST_MAX_RETRIES,
    }
    persisted = (tmp_path / "trajectory.json").read_text()
    context_json = context.model_dump_json()
    assert "synthetic-secret" not in persisted
    assert "synthetic-secret" not in context_json
