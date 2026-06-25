"""Unit tests for ``training.utils.serverless``."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock

from training.utils import serverless as serverless_utils


class _FakeService:
    def __init__(self, *, base_url, api_key, default_headers):
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers
        self.training_session_id = "ts-1234"

    def create_lora_training_client(self, base_model, rank):
        assert base_model == "accounts/fireworks/models/qwen3-4b"
        assert rank == 8
        return SimpleNamespace(
            model_id="run-abcdef:train:0",
            run_id="run-abcdef",
            run_name="accounts/e2e-fireoptimizer/trainingRuns/run-abcdef",
        )


class _FakeFireworksClient:
    def __init__(self, *, api_key, base_url, additional_headers):
        self.api_key = api_key
        self.base_url = base_url
        self.additional_headers = additional_headers
        self.account_id = "e2e-fireoptimizer"
        self.closed = False

    def close(self):
        self.closed = True

    def list_training_session_checkpoints(self, name, *, page_size=200):
        return [{"name": f"{name}/checkpoints/step-8", "pageSize": page_size}]


def test_setup_serverless_training_uses_service_training_session_id(
    monkeypatch, tmp_path
):
    created = {}

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
                base_url="http://personal-api-gateway",
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
            "name": "accounts/e2e-fireoptimizer/trainingSessions/ts-1234/checkpoints/step-8",
            "pageSize": 200,
        }
    ]
