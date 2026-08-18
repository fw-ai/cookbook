"""Unit tests for ``training.utils.serverless``."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from training.utils import serverless as serverless_utils
from training.utils.account import FireworksAccountProvenanceError


class _FakeService:
    instances = []

    def __init__(self, *, base_url, api_key, default_headers):
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers
        self.training_session_id = "ts-1234"
        self.training_session_name = (
            "accounts/test-account-id/trainingSessions/ts-1234"
        )
        self.lora_creation_calls = 0
        self.closed = False
        self.instances.append(self)

    def create_lora_training_client(self, base_model, rank, alpha):
        self.lora_creation_calls += 1
        assert base_model == "accounts/fireworks/models/qwen3-4b"
        assert rank == 8
        assert alpha == 32
        return SimpleNamespace(
            model_id="run-abcdef:train:0",
            run_id="run-abcdef",
            run_name="accounts/test-account-id/trainingRuns/run-abcdef",
        )

    def close(self):
        self.closed = True


class _FakeFireworksClient:
    def __init__(self, *, api_key, base_url, additional_headers):
        self.api_key = api_key
        self.base_url = base_url
        self.additional_headers = additional_headers
        self.account_id = "test-account-id"
        self.closed = False

    def close(self):
        self.closed = True

    def list_training_session_checkpoints(self, name, *, page_size=200):
        return [{"name": f"{name}/checkpoints/step-8", "pageSize": page_size}]


def test_setup_serverless_training_uses_service_training_session_id(
    monkeypatch, tmp_path
):
    created = {}

    monkeypatch.setattr(
        serverless_utils,
        "assert_expected_fireworks_account",
        lambda **_kwargs: "test-account-id",
    )
    monkeypatch.setattr(serverless_utils, "FiretitanServiceClient", _FakeService)
    monkeypatch.setattr(serverless_utils, "FireworksClient", _FakeFireworksClient)

    def fake_from_training_client(training_client, **kwargs):
        created["training_client"] = training_client
        created["client_kwargs"] = kwargs
        client = MagicMock()
        client.resolve_checkpoint_path.return_value = "path://unused"
        return client

    monkeypatch.setattr(
        serverless_utils.ReconnectableClient,
        "from_training_client",
        fake_from_training_client,
    )

    cfg = SimpleNamespace(
        base_model="accounts/fireworks/models/qwen3-4b",
        lora_rank=8,
        max_seq_len=512,
        step_timeout=None,
        log_path=str(tmp_path),
    )
    with ExitStack() as stack:
        _service, _client, ckpt, session_id, max_seq_len = (
            serverless_utils.setup_serverless_training(
                cfg,
                api_key="fw-test-key",
                base_url="https://api.example.test",
                additional_headers={"x-test": "1"},
                stack=stack,
            )
        )

    assert session_id == "ts-1234"
    assert max_seq_len == 512
    assert created["client_kwargs"]["job_id"] == "ts-1234"
    assert ckpt._trainer_id == "ts-1234"
    assert ckpt._current_run_id == "run-abcdef"
    assert ckpt._fw_client.list_checkpoints("ts-1234") == [
        {
            "name": "accounts/test-account-id/trainingSessions/ts-1234/checkpoints/step-8",
            "pageSize": 200,
        }
    ]


def test_account_mismatch_prevents_serverless_session_creation(monkeypatch, tmp_path):
    _FakeService.instances = []
    monkeypatch.setattr(serverless_utils, "FiretitanServiceClient", _FakeService)

    def reject_account(**_kwargs):
        raise FireworksAccountProvenanceError("wrong authenticated account")

    monkeypatch.setattr(
        serverless_utils,
        "assert_expected_fireworks_account",
        reject_account,
    )
    cfg = SimpleNamespace(
        base_model="accounts/fireworks/models/qwen3-4b",
        lora_rank=8,
        max_seq_len=512,
        step_timeout=None,
        log_path=str(tmp_path),
    )

    with ExitStack() as stack, pytest.raises(
        FireworksAccountProvenanceError, match="wrong authenticated account"
    ):
        serverless_utils.setup_serverless_training(
            cfg,
            api_key="fw-test-key",
            base_url="https://api.example.test",
            additional_headers=None,
            stack=stack,
            expected_account_id="research-train",
        )

    assert _FakeService.instances == []


def test_post_create_session_account_mismatch_closes_and_rejects(monkeypatch):
    service = _FakeService(
        base_url="https://api.example.test/training/v1/serverless",
        api_key="fw-test-key",
        default_headers=None,
    )
    service.training_session_name = (
        "accounts/other-account/trainingSessions/ts-1234"
    )
    monkeypatch.setattr(
        serverless_utils,
        "assert_expected_fireworks_account",
        lambda **_kwargs: "research-train",
    )

    with pytest.raises(
        FireworksAccountProvenanceError,
        match="session account differs",
    ):
        serverless_utils.create_lora_training_client_for_account(
            service,
            expected_account_id="research-train",
            base_model="accounts/fireworks/models/qwen3-4b",
            rank=8,
            alpha=32,
        )

    assert service.lora_creation_calls == 1
    assert service.closed is True
