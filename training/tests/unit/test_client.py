from __future__ import annotations

from types import SimpleNamespace

import training.utils.client as module


def test_build_base_model_reference_id_has_base_prefix() -> None:
    """Base-model reference IDs must never look like LoRA session IDs."""
    assert (
        module.build_base_model_reference_id("accounts/test/models/qwen3-4b")
        == "base::accounts/test/models/qwen3-4b"
    )


def test_reconnectable_client_uses_manual_model_id_override(monkeypatch) -> None:
    """Manual model IDs should skip create_model and target the base model directly."""
    events: dict[str, object] = {}

    class FakeHolder:
        def get_training_client_id(self) -> int:
            events["training_client_id_calls"] = (
                int(events.get("training_client_id_calls", 0)) + 1
            )
            return 7

    class FakeServiceClient:
        def __init__(self, base_url: str, api_key: str, **kwargs) -> None:
            events["service_init"] = {
                "base_url": base_url,
                "api_key": api_key,
                "kwargs": kwargs,
            }
            self.holder = FakeHolder()

        def create_training_client(self, *args, **kwargs):
            raise AssertionError(
                "create_training_client should not run for model_id overrides"
            )

    class FakeTrainingClient:
        def __init__(self, holder, model_seq_id: int, model_id: str) -> None:
            self.holder = holder
            self.model_seq_id = model_seq_id
            self.model_id = model_id
            events["manual_client"] = {
                "model_seq_id": model_seq_id,
                "model_id": model_id,
            }

    monkeypatch.setattr(module, "FiretitanServiceClient", FakeServiceClient)
    monkeypatch.setattr(module, "FiretitanTrainingClient", FakeTrainingClient)

    endpoint = SimpleNamespace(base_url="https://trainer.unit.test", job_id="job-123")
    model_id = module.build_base_model_reference_id("accounts/test/models/qwen3-4b")

    client = module.ReconnectableClient(
        rlor_mgr=SimpleNamespace(),
        job_id="job-123",
        base_model="accounts/test/models/qwen3-4b",
        lora_rank=8,
        fw_api_key="fw-key",
        endpoint=endpoint,
        model_id_override=model_id,
    )

    assert events["manual_client"] == {
        "model_seq_id": 7,
        "model_id": model_id,
    }
    assert events["training_client_id_calls"] == 1
    assert events["service_init"] == {
        "base_url": "https://trainer.unit.test",
        "api_key": "tml-local",
        "kwargs": {
            "default_headers": {
                "X-API-Key": "fw-key",
                "Authorization": "Bearer fw-key",
            }
        },
    }
    assert client.inner.model_id == model_id
