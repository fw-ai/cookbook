from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import requests

from training.utils.training_session_client import TrainingSessionClient, TrainingSessionHandle


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeManager:
    def __init__(self, list_payload: dict[str, Any], post_payloads: list[dict[str, Any]]):
        self.account_id = "acct"
        self.list_payload = list_payload
        self.post_payloads = list(post_payloads)
        self.get_calls: list[tuple[str, int]] = []
        self.post_calls: list[tuple[str, dict[str, Any], int]] = []

    def _get(self, path: str, timeout: int) -> _FakeResponse:
        self.get_calls.append((path, timeout))
        return _FakeResponse(self.list_payload)

    def _post(self, path: str, json: dict[str, Any], timeout: int) -> _FakeResponse:
        self.post_calls.append((path, json, timeout))
        payload = self.post_payloads.pop(0)
        return _FakeResponse(payload)


class _FakeHTTPSession:
    def __init__(self, response: _FakeResponse | Exception):
        self.headers: dict[str, str] = {}
        self.response = response
        self.calls: list[tuple[str, dict[str, Any], int]] = []

    def post(self, url: str, json: dict[str, Any], timeout: int) -> _FakeResponse:
        self.calls.append((url, json, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _FakeDatum:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def model_dump(self, mode: str = "json", exclude_none: bool = True) -> dict[str, Any]:
        assert mode == "json"
        assert exclude_none is True
        return self._payload


def test_create_session_reuses_newest_parent_job() -> None:
    manager = _FakeManager(
        list_payload={
            "trainingSessionJobs": [
                {"name": "accounts/acct/trainingSessionJobs/job-new"},
                {"name": "accounts/acct/trainingSessionJobs/job-old"},
            ]
        },
        post_payloads=[
            {"name": "accounts/acct/trainingSessionJobs/job-new/trainingSessions/session-123"},
        ],
    )
    http_session = _FakeHTTPSession(_FakeResponse({}))
    client = TrainingSessionClient(
        api_key="test-key",
        base_url="https://api.unit.test",
        manager=manager,
        request_session=http_session,
    )

    handle = client.create_session("accounts/fireworks/models/qwen3-8b")

    assert handle._job_id == "job-new"
    assert handle._training_session_id == "session-123"
    assert manager.get_calls[0][1] == 30
    assert "trainingSessionJobs" in manager.get_calls[0][0]
    assert "base_model%3D%22accounts%2Ffireworks%2Fmodels%2Fqwen3-8b%22" in manager.get_calls[0][0]
    assert manager.post_calls == [
        ("/v1/accounts/acct/trainingSessionJobs/job-new/trainingSessions", {}, 30)
    ]


def test_create_session_creates_parent_when_missing() -> None:
    manager = _FakeManager(
        list_payload={"trainingSessionJobs": []},
        post_payloads=[
            {"name": "accounts/acct/trainingSessionJobs/job-created"},
            {"name": "accounts/acct/trainingSessionJobs/job-created/trainingSessions/session-456"},
        ],
    )
    client = TrainingSessionClient(
        api_key="test-key",
        base_url="https://api.unit.test",
        manager=manager,
        request_session=_FakeHTTPSession(_FakeResponse({})),
    )

    handle = client.create_session("accounts/fireworks/models/qwen3-8b")

    assert handle._job_id == "job-created"
    assert handle._training_session_id == "session-456"
    assert manager.post_calls == [
        (
            "/v1/accounts/acct/trainingSessionJobs",
            {"baseModel": "accounts/fireworks/models/qwen3-8b"},
            30,
        ),
        (
            "/v1/accounts/acct/trainingSessionJobs/job-created/trainingSessions",
            {},
            30,
        ),
    ]


def test_load_state_resets_private_sequence_counter() -> None:
    manager = _FakeManager(list_payload={"trainingSessionJobs": []}, post_payloads=[{}])
    handle = TrainingSessionHandle(
        manager=manager,
        request_session=_FakeHTTPSession(_FakeResponse({})),
        base_url="https://api.unit.test",
        account_id="acct",
        job_id="job-123",
        training_session_id="session-123",
    )
    handle._next_seq_id = 7

    handle.load_state("/tmp/adapters/ref-a")

    assert handle._next_seq_id == 2
    assert manager.post_calls == [
        (
            "/v1/accounts/acct/trainingSessionJobs/job-123/trainingSessions/session-123:loadState",
            {
                "name": "accounts/acct/trainingSessionJobs/job-123/trainingSessions/session-123",
                "path": "/tmp/adapters/ref-a",
            },
            30,
        )
    ]


def test_forward_uses_session_scoped_path_and_increments_sequence() -> None:
    http_session = _FakeHTTPSession(
        _FakeResponse(
            {
                "loss_fn_output_type": "forward",
                "loss_fn_outputs": [
                    {
                        "logprobs": {
                            "data": [-0.5, -0.6],
                            "dtype": "float32",
                            "shape": [2],
                        }
                    }
                ],
                "metrics": {"loss": 1.23},
            }
        )
    )
    handle = TrainingSessionHandle(
        manager=_FakeManager(list_payload={"trainingSessionJobs": []}, post_payloads=[]),
        request_session=http_session,
        base_url="https://api.unit.test",
        account_id="acct",
        job_id="job-123",
        training_session_id="session-123",
    )

    result = handle.forward([_FakeDatum({"value": 1})], "cross_entropy")

    assert http_session.calls == [
        (
            "https://api.unit.test/training/v1/trainingSessionJobs/acct/job-123/trainingSessions/session-123/forward",
            {
                "forward_input": {
                    "data": [{"value": 1}],
                    "loss_fn": "cross_entropy",
                },
                "seq_id": 1,
            },
            600,
        )
    ]
    assert handle._next_seq_id == 2
    assert result.loss_fn_output_type == "forward"
    assert result.loss_fn_outputs[0]["logprobs"].data == [-0.5, -0.6]


def test_forward_failure_keeps_sequence_and_invalidates_handle() -> None:
    handle = TrainingSessionHandle(
        manager=_FakeManager(list_payload={"trainingSessionJobs": []}, post_payloads=[]),
        request_session=_FakeHTTPSession(requests.Timeout("timed out")),
        base_url="https://api.unit.test",
        account_id="acct",
        job_id="job-123",
        training_session_id="session-123",
    )
    handle._next_seq_id = 3

    with pytest.raises(requests.Timeout):
        handle.forward([], "cross_entropy")

    assert handle._next_seq_id == 3
    with pytest.raises(RuntimeError, match="no longer usable"):
        handle.forward([], "cross_entropy")


def test_forward_http_error_keeps_handle_usable() -> None:
    http_session = _FakeHTTPSession(_FakeResponse({"error": "boom"}, status_code=500))
    handle = TrainingSessionHandle(
        manager=_FakeManager(list_payload={"trainingSessionJobs": []}, post_payloads=[]),
        request_session=http_session,
        base_url="https://api.unit.test",
        account_id="acct",
        job_id="job-123",
        training_session_id="session-123",
    )
    handle._next_seq_id = 4

    with pytest.raises(requests.HTTPError):
        handle.forward([], "cross_entropy")

    assert handle._next_seq_id == 4

    http_session.response = _FakeResponse(
        {
            "loss_fn_output_type": "forward",
            "loss_fn_outputs": [],
        }
    )
    handle.forward([], "cross_entropy")
    assert handle._next_seq_id == 5
