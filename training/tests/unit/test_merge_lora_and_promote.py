from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "tools" / "merge_lora_and_promote.py"
)
_SPEC = importlib.util.spec_from_file_location("merge_lora_and_promote_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = module
_SPEC.loader.exec_module(module)


def _cfg(**overrides) -> object:
    defaults = dict(
        base_model=None,
        adapter_gcs=None,
        lora_rank=None,
        adapter_model=None,
        training_shape="",
        output_model_id="merged-out",
        region=None,
        snapshot_name="merged-base",
        keep_trainer=False,
        trainer_timeout_s=3600.0,
        op_timeout_s=3000.0,
        checkpoint_poll_timeout_s=900.0,
        promote_poll_timeout_s=1800.0,
    )
    defaults.update(overrides)
    return module.MergeConfig(**defaults)


@pytest.mark.parametrize(
    "signed_url,expected",
    [
        (
            "https://storage.googleapis.com/fw-bucket/adapters/lora-1/adapter_config.json"
            "?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Signature=abc",
            "gs://fw-bucket/adapters/lora-1",
        ),
        (
            "https://fw-bucket.storage.googleapis.com/adapters/lora-1/adapter_config.json?sig=x",
            "gs://fw-bucket/adapters/lora-1",
        ),
        (
            "https://storage.googleapis.com/fw-bucket/adapters/my%20lora/adapter_config.json",
            "gs://fw-bucket/adapters/my lora",
        ),
    ],
)
def test_gcs_dir_from_signed_url(signed_url: str, expected: str) -> None:
    assert module._gcs_dir_from_signed_url(signed_url) == expected


def test_gcs_dir_from_signed_url_rejects_bucket_root() -> None:
    with pytest.raises(ValueError):
        module._gcs_dir_from_signed_url("https://storage.googleapis.com/fw-bucket")


def test_resolve_adapter_source_uses_explicit_flags_without_api_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(*_args, **_kwargs):
        raise AssertionError("explicit flags must not trigger control-plane reads")

    monkeypatch.setattr(module, "_get_json", _fail)

    source = module._resolve_adapter_source(
        "https://api.fireworks.ai",
        "key",
        _cfg(
            base_model="accounts/fireworks/models/qwen3-8b",
            adapter_gcs="gs://bucket/adapters/lora-1",
            lora_rank=8,
        ),
    )

    assert source == module.AdapterSource(
        base_model="accounts/fireworks/models/qwen3-8b",
        adapter_gcs="gs://bucket/adapters/lora-1",
        lora_rank=8,
    )


def _stub_get_json(monkeypatch: pytest.MonkeyPatch, model: dict, files: dict) -> list[str]:
    """Record requested paths and answer model / download-endpoint reads."""
    requested: list[str] = []

    def _get_json(_base_url: str, _api_key: str, path: str) -> dict:
        requested.append(path)
        if path.endswith(":getDownloadEndpoint"):
            return {"filenameToSignedUrls": files}
        return model

    monkeypatch.setattr(module, "_get_json", _get_json)
    return requested


def test_resolve_adapter_source_reads_base_rank_and_directory_from_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _stub_get_json(
        monkeypatch,
        model={
            "kind": "HF_PEFT_ADDON",
            "peftDetails": {"baseModel": "accounts/fireworks/models/kimi-k2p6", "r": 32},
        },
        files={
            "adapter_config.json": "https://storage.googleapis.com/b/adapters/l1/adapter_config.json?s=1",
            "adapter_model.safetensors": "https://storage.googleapis.com/b/adapters/l1/adapter_model.safetensors?s=1",
        },
    )

    source = module._resolve_adapter_source(
        "https://api.fireworks.ai", "key", _cfg(adapter_model="accounts/acct/models/lora-1")
    )

    assert source == module.AdapterSource(
        base_model="accounts/fireworks/models/kimi-k2p6",
        adapter_gcs="gs://b/adapters/l1",
        lora_rank=32,
    )
    assert requested == [
        "accounts/acct/models/lora-1",
        "accounts/acct/models/lora-1:getDownloadEndpoint",
    ]


def test_resolve_adapter_source_explicit_flags_win_over_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _stub_get_json(
        monkeypatch,
        model={"peftDetails": {"baseModel": "accounts/fireworks/models/kimi-k2p6", "r": 32}},
        files={},
    )

    source = module._resolve_adapter_source(
        "https://api.fireworks.ai",
        "key",
        _cfg(
            adapter_model="accounts/acct/models/lora-1",
            adapter_gcs="gs://override/dir",
            lora_rank=16,
        ),
    )

    assert source.adapter_gcs == "gs://override/dir"
    assert source.lora_rank == 16
    assert source.base_model == "accounts/fireworks/models/kimi-k2p6"
    assert requested == ["accounts/acct/models/lora-1"]


def test_resolve_adapter_source_rejects_non_peft_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_get_json(monkeypatch, model={"kind": "HF_BASE_MODEL"}, files={})

    with pytest.raises(ValueError, match="not a LoRA/PEFT model"):
        module._resolve_adapter_source(
            "https://api.fireworks.ai", "key", _cfg(adapter_model="accounts/acct/models/base-1")
        )


def test_resolve_adapter_source_requires_adapter_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_get_json(
        monkeypatch,
        model={"peftDetails": {"baseModel": "accounts/fireworks/models/qwen3-8b", "r": 8}},
        files={"adapter_model.safetensors": "https://storage.googleapis.com/b/d/x.safetensors"},
    )

    with pytest.raises(ValueError, match="no adapter_config.json"):
        module._resolve_adapter_source(
            "https://api.fireworks.ai", "key", _cfg(adapter_model="accounts/acct/models/lora-1")
        )


class _LazyService:
    """Mirrors the SDK: the trainer is provisioned on the first client call."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._provisioned = False
        self.closed = False

    @property
    def managed_trainer_job_id(self) -> str | None:
        return "trainer-1" if self._provisioned else None

    @property
    def trainer_job_id(self) -> str:
        self._events.append("read-job-id")
        if not self._provisioned:
            raise RuntimeError("SDK-managed service did not resolve trainer job id.")
        return "trainer-1"

    def create_lora_training_client(self, base_model: str, rank: int):
        self._events.append(f"create-client:{base_model}:{rank}")
        self._provisioned = True
        return SimpleNamespace(
            load_adapter=lambda path: SimpleNamespace(
                result=lambda timeout: self._events.append(f"load-adapter:{path}")
            ),
            save_weights_for_sampler_ext=lambda name, checkpoint_type: SimpleNamespace(
                path="gs://ckpt", snapshot_name=name
            ),
        )

    def close(self) -> None:
        self.closed = True


def _stub_main_dependencies(monkeypatch: pytest.MonkeyPatch, service) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "key")
    monkeypatch.setattr(module, "build_service_client", lambda **_kwargs: service)
    monkeypatch.setattr(
        module,
        "FireworksClient",
        lambda **_kwargs: SimpleNamespace(
            account_id="acct",
            list_checkpoints=lambda _job_id: [
                {"name": "x/checkpoints/merged-base", "promotable": True, "createTime": "1"}
            ],
        ),
    )
    monkeypatch.setattr(
        module,
        "TrainerJobManager",
        lambda **_kwargs: SimpleNamespace(promote_checkpoint=lambda **_kw: None),
    )
    monkeypatch.setattr(
        module,
        "_poll_model_until_ready",
        lambda *_args, **_kwargs: {
            "name": "accounts/acct/models/merged-out",
            "state": "READY",
            "kind": "HF_BASE_MODEL",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_lora_and_promote.py",
            "--base-model", "accounts/fireworks/models/qwen3-8b",
            "--adapter-gcs", "gs://bucket/adapters/lora-1",
            "--lora-rank", "8",
            "--output-model-id", "merged-out",
        ],
    )


def test_main_creates_training_client_before_reading_trainer_job_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    service = _LazyService(events)
    _stub_main_dependencies(monkeypatch, service)

    module.main()

    assert events.index("create-client:accounts/fireworks/models/qwen3-8b:8") < events.index(
        "read-job-id"
    )
    assert "load-adapter:gs://bucket/adapters/lora-1" in events
    assert service.closed


def test_main_explains_masked_provisioning_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    service = _LazyService(events)

    def _boom(*_args, **_kwargs):
        raise RuntimeError(
            "ERROR: Trainer job training-api-service-261f8eb5 failed\n"
            "  Cause: Internal error occurred"
        )

    service.create_lora_training_client = _boom  # type: ignore[method-assign]
    _stub_main_dependencies(monkeypatch, service)

    with pytest.raises(RuntimeError) as excinfo:
        module.main()

    message = str(excinfo.value)
    assert "accelerator, region" in message
    assert "--region unset" in message
    assert "Internal error occurred" in str(excinfo.value.__cause__)
    assert service.closed


def test_parse_args_requires_explicit_inputs_without_adapter_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["merge_lora_and_promote.py", "--output-model-id", "merged-out", "--lora-rank", "8"],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_parse_args_accepts_adapter_model_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_lora_and_promote.py",
            "--adapter-model", "accounts/acct/models/lora-1",
            "--output-model-id", "merged-out",
        ],
    )

    cfg = module.parse_args()

    assert cfg.adapter_model == "accounts/acct/models/lora-1"
    assert (cfg.base_model, cfg.adapter_gcs, cfg.lora_rank) == (None, None, None)
