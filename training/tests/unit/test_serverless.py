"""Unit tests for ``training.utils.serverless``."""

from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
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
    monkeypatch, tmp_path, caplog, service_cls
):
    created = {}
    caplog.set_level("INFO", logger=serverless_utils.__name__)

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
    # Another process continues this run with the fully qualified reference.
    assert "init_from_checkpoint='test-account-id/run-abcdef/<checkpoint>'" in caplog.text


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
    ("module_name", "recipe_name"),
    [
        ("sft.train_sft_serverless", "sft_loop"),
        ("dpo.train_dpo_serverless", "dpo_loop"),
    ],
    ids=["sft", "dpo"],
)
def test_serverless_supervised_examples_run_recipe_in_serverless_mode(
    monkeypatch, tmp_path, module_name, recipe_name
):
    # The examples are assemblies: CLI flags -> recipe Config(serverless=True)
    # -> recipe main. setup_serverless_training (tested above) owns the session.
    module = import_module(f"training.examples.{module_name}")
    recipe = import_module(f"training.recipes.{recipe_name}")
    received = []
    monkeypatch.setattr(recipe, "main", lambda cfg: received.append(cfg) or {"steps": 0})
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text("{}\n")
    reference = "acct/run-" + "a" * 32 + "/step-3"

    module.main(
        [
            "--dataset",
            str(dataset),
            "--log-path",
            str(tmp_path / "logs"),
            "--init-from-checkpoint",
            reference,
            "--output-model-id",
            "triage-model",
        ]
    )

    (cfg,) = received
    assert isinstance(cfg, recipe.Config)
    assert cfg.serverless is True
    assert cfg.lora_rank > 0
    assert cfg.max_seq_len and cfg.max_seq_len > 0
    assert cfg.base_model == "accounts/fireworks/models/kimi-k3"
    assert cfg.tokenizer_model == "moonshotai/Kimi-K3"
    assert cfg.dataset == str(dataset)
    assert cfg.log_path == str(tmp_path / "logs")
    assert cfg.init_from_checkpoint == reference
    assert cfg.output_model_id == "triage-model"
    assert cfg.wandb.entity is None


def test_serverless_sft_example_defaults_to_bundled_dataset():
    module = import_module("training.examples.sft.train_sft_serverless")

    cfg = module.build_config(module.parse_args([]))

    assert Path(cfg.dataset).is_file()
    assert cfg.output_model_id is None
    assert cfg.init_from_checkpoint is None


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.fireworks.ai",
        "https://api.fireworks.ai/",
        "https://api.fireworks.ai/training/v1",
        "https://api.fireworks.ai/training/v1/serverless",
        "https://api.fireworks.ai/training/v1/serverless/",
    ],
)
def test_serverless_urls_accept_a_root_or_surface_base_url(base_url):
    from training.utils.serverless import gateway_root_url, serverless_base_url

    assert serverless_base_url(base_url) == "https://api.fireworks.ai/training/v1/serverless"
    assert gateway_root_url(base_url) == "https://api.fireworks.ai"
