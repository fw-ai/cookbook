import json
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")
from harbor.agents.installed.base import ApiRateLimitError, NonZeroAgentExitCodeError

from training.examples.rl.harbor.opencode.agent import ConfigurableOpenCode


@pytest.fixture
def agent(tmp_path):
    return ConfigurableOpenCode(
        logs_dir=tmp_path, sidecar_bundle_path="unused", sidecar_launch_spec="{}",
        context_limit=262144, output_limit=131072, tool_timeout_seconds=6900,
        version="1.18.8",
    )


def event(kind, **data):
    return json.dumps({"type": kind, "timestamp": 1, "sessionID": "session", **data})


LAUNCHER = "opencode run --format=json; test -s /tmp/fireworks-tito-opencode/agent-status"


@pytest.mark.parametrize("kind", ["text", "reasoning", "tool_use", "step_start", "step_finish"])
def test_task_content_is_not_a_provider_rate_limit(agent, kind):
    output = event(kind, part={"text": "GitHub rate limit; use local copy"})
    result = SimpleNamespace(return_code=-1, stdout=output, stderr="")
    error = agent._classify_exec_error(LAUNCHER, result)
    assert type(error) is NonZeroAgentExitCodeError
    assert result.stdout == output  # Artifacts and original exec result unchanged.


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_actual_session_error_is_preserved(agent, stream):
    output = event("error", error={"name": "APIError", "data": {
        "statusCode": 429, "message": "Too many requests",
    }})
    output += "\n" + event("text", part={"text": "Continue using local files"})
    result = SimpleNamespace(return_code=1, stdout="", stderr="")
    setattr(result, stream, output)
    assert isinstance(agent._classify_exec_error(LAUNCHER, result), ApiRateLimitError)


def test_raw_startup_diagnostic_is_preserved(agent):
    result = SimpleNamespace(return_code=1, stdout="", stderr="Too many requests")
    assert isinstance(agent._classify_exec_error(LAUNCHER, result), ApiRateLimitError)


def test_setup_commands_keep_original_classification(agent):
    result = SimpleNamespace(return_code=1, stdout=event("text", part={"text": "rate limit"}), stderr="")
    assert isinstance(agent._classify_exec_error("install opencode", result), ApiRateLimitError)


@pytest.mark.parametrize("output", [
    '{"type":"text","part":{"text":"rate limit"}}',
    '{"type":"unknown","timestamp":1,"sessionID":"s","part":{"text":"rate limit"}}',
    'malformed JSON with rate limit',
])
def test_unknown_or_partial_format_is_not_silently_discarded(agent, output):
    result = SimpleNamespace(return_code=1, stdout=output, stderr="")
    assert isinstance(agent._classify_exec_error(LAUNCHER, result), ApiRateLimitError)
