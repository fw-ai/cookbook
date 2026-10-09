"""Unit tests for ``training.utils.serverless``."""

from contextlib import ExitStack
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from training.utils import serverless as serverless_utils


class _LegacyFakeService:
    def __init__(self, *, base_url, api_key, default_headers):
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers
        self.training_session_id = "ts-1234"
        self.events = []

    def close(self):
        pass

    def create_lora_training_client(self, base_model, rank, alpha):
        self.events.append("create-training-client")
        assert base_model == "accounts/fireworks/models/qwen3-4b"
        assert rank == 8
        assert alpha == 32
        return SimpleNamespace(
            model_id="run-abcdef:train:0",
            run_id="run-abcdef",
            run_name="accounts/test-account-id/trainingRuns/run-abcdef",
        )


class _FakeService(_LegacyFakeService):
    def _enable_serverless_supervised_409_retry(self):
        self.events.append("enable-supervised-retry")


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


@pytest.mark.parametrize(
    "service_cls", [_FakeService, _LegacyFakeService], ids=["retry-hook", "legacy-sdk"]
)
def test_setup_serverless_training_uses_service_training_session_id(
    monkeypatch, tmp_path, service_cls
):
    created = {}

    monkeypatch.setattr(serverless_utils, "FiretitanServiceClient", service_cls)
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
    expected_events = ["enable-supervised-retry"] if service_cls is _FakeService else []
    assert _service.events == expected_events + ["create-training-client"]
    assert created["client_kwargs"]["job_id"] == "ts-1234"
    assert ckpt._trainer_id == "ts-1234"
    assert ckpt._current_run_id == "run-abcdef"
    assert ckpt._fw_client.list_checkpoints("ts-1234") == [
        {
            "name": "accounts/test-account-id/trainingSessions/ts-1234/checkpoints/step-8",
            "pageSize": 200,
        }
    ]


def test_enable_supervised_retry_uses_optional_sdk_hook():
    hook = MagicMock()
    service = SimpleNamespace(_enable_serverless_supervised_409_retry=hook)

    serverless_utils.enable_serverless_supervised_409_retry(service)

    hook.assert_called_once_with()


def test_enable_supervised_retry_accepts_installed_sdk_service():
    # No session reservation or HTTP request: exercise the installed SDK class
    # in both staged-SDK and published-minimum compatibility runs.
    service = object.__new__(serverless_utils.FiretitanServiceClient)
    service.holder = SimpleNamespace()

    serverless_utils.enable_serverless_supervised_409_retry(service)

    has_retry_hook = hasattr(service, "_enable_serverless_supervised_409_retry")
    assert (
        getattr(service.holder, "_fireworks_serverless_supervised_409_retry", False)
        is has_retry_hook
    )


def test_enable_supervised_retry_accepts_sdk_without_hook():
    serverless_utils.enable_serverless_supervised_409_retry(SimpleNamespace())


def test_enable_supervised_retry_does_not_hide_hook_failure():
    service = SimpleNamespace(
        _enable_serverless_supervised_409_retry=MagicMock(
            side_effect=RuntimeError("holder unavailable")
        )
    )

    with pytest.raises(RuntimeError, match="holder unavailable"):
        serverless_utils.enable_serverless_supervised_409_retry(service)


@pytest.mark.parametrize(
    ("module_name", "run_cls"),
    [
        ("serverless_sft.support_triage_sft", "ServerlessTriageSFT"),
        ("serverless_dpo.ultrafeedback_dpo", "ServerlessUltraFeedbackDPO"),
    ],
    ids=["sft", "dpo"],
)
@pytest.mark.parametrize(
    "resume_from", ["", "acct/run-example/checkpoint-0002"], ids=["fresh", "resume"]
)
@pytest.mark.parametrize(
    "with_retry_hook", [True, False], ids=["retry-hook", "legacy-sdk"]
)
def test_standalone_supervised_example_opts_in_before_create_or_resume(
    monkeypatch, tmp_path, module_name, run_cls, resume_from, with_retry_hook
):
    module = import_module(f"training.examples.{module_name}")
    events = []

    class StopAfterTrainingClientRequest(Exception):
        pass

    class FakeService:
        def __init__(self, *, api_key, base_url):
            assert base_url == "https://api.example.test/training/v1/serverless"

        def create_lora_training_client(self, *, base_model, rank):
            events.append("create")
            raise StopAfterTrainingClientRequest

        def create_training_client_from_state_with_optimizer(self, checkpoint):
            assert checkpoint == resume_from
            events.append("resume")
            raise StopAfterTrainingClientRequest

    if with_retry_hook:
        monkeypatch.setattr(
            FakeService,
            "_enable_serverless_supervised_409_retry",
            lambda self: events.append("enable-supervised-retry"),
            raising=False,
        )
    monkeypatch.setattr(module, "FiretitanServiceClient", FakeService)
    monkeypatch.setattr(module, "_load_rows", lambda path: [{"messages": []}] * 2)
    monkeypatch.setattr(module, "get_tokenizer", lambda model: object())
    monkeypatch.setattr(module, "build_renderer", lambda *args: object())
    dataset = tmp_path / "dataset.jsonl"
    dataset.touch()
    cfg = module.Config(
        api_key="fw-test-key",
        api_base_url="https://api.example.test",
        dataset=str(dataset),
        renderer_name="test-renderer",
        resume_from=resume_from,
    )
    if hasattr(cfg, "eval_pairs"):
        cfg.eval_pairs = 0

    with pytest.raises(StopAfterTrainingClientRequest):
        getattr(module, run_cls)(cfg)

    expected_events = ["enable-supervised-retry"] if with_retry_hook else []
    assert events == expected_events + ["resume" if resume_from else "create"]
