from __future__ import annotations


import pytest

import training.recipes.orpo_loop as module
from training.utils import supervised as supervised_utils


class _StopAfterProvisioning(RuntimeError):
    pass


def _build_service_kwargs(monkeypatch, cfg: module.Config) -> dict:
    calls = []

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "validate_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "validate_warm_start_config",
        lambda *args, **kwargs: None,
    )

    def fake_build_service_client(**kwargs):
        calls.append(kwargs)
        raise _StopAfterProvisioning

    monkeypatch.setattr(module, "build_service_client", fake_build_service_client)

    with pytest.raises(_StopAfterProvisioning):
        module.main(cfg)

    assert len(calls) == 1
    return calls[0]


def test_config_uses_shared_default_weight_decay():
    cfg = module.Config(log_path="/tmp/orpo_test_logs")

    assert cfg.weight_decay == pytest.approx(module.DEFAULT_ADAM["weight_decay"])


def test_main_rejects_invalid_base_model(monkeypatch):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(log_path="/tmp/orpo_test_logs", base_model="qwen3-4b", dataset="/tmp/pairs.jsonl", tokenizer_model="Qwen/Qwen3-4B")

    with pytest.raises(RuntimeError, match="Invalid base_model"):
        module.main(cfg)


def test_main_rejects_invalid_output_model_id(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(
        log_path=str(tmp_path),
        base_model="accounts/test/models/qwen3-4b",
        dataset="/tmp/pairs.jsonl",
        tokenizer_model="Qwen/Qwen3-4B",
        output_model_id="bad_name",
    )

    with pytest.raises(RuntimeError, match="Invalid output_model_id|output_model_id.*invalid|invalid.*output_model_id"):
        module.main(cfg)


@pytest.mark.parametrize(
    ("mode", "resolved_renderer"),
    [
        ("interleaved", "qwen3_6_interleaved"),
        ("preserved", "qwen3_6_preserved"),
    ],
)
def test_main_resolves_thinking_history_before_provisioning(
    monkeypatch,
    tmp_path,
    mode,
    resolved_renderer,
):
    captured: dict = {}

    def fake_resolve_renderer_snapshot(**kwargs):
        captured.update(kwargs)
        return resolved_renderer

    monkeypatch.setattr(
        module,
        "resolve_renderer_snapshot",
        fake_resolve_renderer_snapshot,
    )
    cfg = module.Config(
        log_path=str(tmp_path),
        base_model="accounts/fireworks/models/qwen3p6-27b",
        dataset="/tmp/preferences.jsonl",
        tokenizer_model="Qwen/Qwen3.6-27B",
        renderer_name=resolved_renderer,
        renderer_name_is_resolved=True,
        thinking_trace_history_mode=mode,
    )

    _build_service_kwargs(monkeypatch, cfg)

    assert captured == {
        "tokenizer_model": "Qwen/Qwen3.6-27B",
        "renderer_name": resolved_renderer,
        "thinking_trace_history_mode": mode,
        "renderer_name_is_resolved": True,
    }


def test_stale_renderer_snapshot_reuses_persisted_renderer(monkeypatch, tmp_path):
    monkeypatch.setattr(
        supervised_utils,
        "resolve_thinking_trace_renderer_plan",
        lambda *_args, **_kwargs: pytest.fail("live registry was consulted"),
    )
    cfg = module.Config(
        log_path=str(tmp_path),
        base_model="accounts/fireworks/models/qwen3p6-27b",
        dataset="/tmp/preferences.jsonl",
        tokenizer_model="Qwen/Qwen3.6-27B",
        renderer_name="qwen3_6_preserved",
        renderer_name_is_resolved=True,
        thinking_trace_history_mode="preserved",
    )

    _build_service_kwargs(monkeypatch, cfg)


def test_eager_rendering_builds_validated_concrete_renderer(monkeypatch, tmp_path):
    class StopAfterRendererBuild(RuntimeError):
        pass

    class FakeRunner:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write_status(self, *args, **kwargs):
            pass

        def set_accelerator_info(self, *args, **kwargs):
            pass

    class FakeService:
        accelerator_type = None
        accelerator_count = None
        training_profile = None
        trainer_job_id = "trainer-job"
        max_context_length = 4096

        def close(self):
            pass

        def create_training_client(self, *args, **kwargs):
            return object()

    class FakeCheckpoints:
        def __init__(self, *args, **kwargs):
            pass

        def resume(self, **kwargs):
            return None

    captured: dict = {}
    tokenizer = object()

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(module, "RunnerIO", lambda _cfg: FakeRunner())
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "validate_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "validate_warm_start_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "resolve_renderer_snapshot",
        lambda **kwargs: "qwen3_6_preserve_thinking",
    )
    monkeypatch.setattr(module, "build_service_client", lambda **kwargs: FakeService())
    monkeypatch.setattr(
        module.ReconnectableClient,
        "from_training_client",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(module, "TrainingCheckpoints", FakeCheckpoints)
    def fake_load_tokenizer(model, revision=None, trust_remote_code=None):
        captured.update(
            tokenizer_source=model,
            tokenizer_revision=revision,
            tokenizer_trust_remote_code=trust_remote_code,
        )
        return tokenizer

    monkeypatch.setattr(module, "load_tokenizer", fake_load_tokenizer)

    def fake_build_renderer(actual_tokenizer, tokenizer_model, renderer_name):
        captured.update(
            tokenizer=actual_tokenizer,
            tokenizer_model=tokenizer_model,
            renderer_name=renderer_name,
        )
        raise StopAfterRendererBuild

    monkeypatch.setattr(
        module,
        "build_renderer_from_resolved_name",
        fake_build_renderer,
    )
    cfg = module.Config(
        log_path=str(tmp_path),
        dataset="/tmp/preferences.jsonl",
        tokenizer_model="Qwen/Qwen3.6-27B",
        tokenizer_revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        tokenizer_trust_remote_code=False,
    )

    with pytest.raises(StopAfterRendererBuild):
        module.main(cfg)

    assert captured == {
        "tokenizer": tokenizer,
        "tokenizer_model": "Qwen/Qwen3.6-27B",
        "renderer_name": "qwen3_6_preserve_thinking",
        "tokenizer_source": "Qwen/Qwen3.6-27B",
        "tokenizer_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "tokenizer_trust_remote_code": False,
    }


def test_shuffled_pair_cache_is_seeded_without_mutating_source():
    pair_cache = [{"id": i} for i in range(5)]

    first_order = module._shuffled_pair_cache(pair_cache, seed=17, epoch=0)
    second_order = module._shuffled_pair_cache(pair_cache, seed=17, epoch=0)

    assert first_order == second_order
    assert pair_cache == [{"id": i} for i in range(5)]


def test_legacy_orpo_lr_schedule_preserves_warmup_floor():
    cfg = module.Config(
        log_path="/tmp/orpo_test_logs",
        lr_schedule="cosine",
        warmup_ratio=0.2,
        min_lr_ratio=0.1,
    )

    assert module._uses_legacy_orpo_lr_schedule(cfg)
    assert module._compute_legacy_orpo_lr(
        0,
        10,
        peak_lr=1.0,
        warmup_ratio=0.2,
        min_lr_ratio=0.1,
        schedule="cosine",
    ) == pytest.approx(0.1)
    assert module._compute_legacy_orpo_lr(
        1,
        10,
        peak_lr=1.0,
        warmup_ratio=0.2,
        min_lr_ratio=0.1,
        schedule="cosine",
    ) == pytest.approx(0.55)


def test_nested_orpo_lr_scheduler_uses_shared_scheduler():
    cfg = module.Config(
        log_path="/tmp/orpo_test_logs",
        lr_scheduler={"type": "cosine", "warmup_ratio": 0.2, "min_lr_ratio": 0.1},
        lr_schedule="cosine",
        warmup_ratio=0.2,
        min_lr_ratio=0.1,
    )

    assert not module._uses_legacy_orpo_lr_schedule(cfg)
    scheduler = module.normalize_lr_scheduler_spec(
        cfg.lr_scheduler,
        legacy_lr_schedule=cfg.lr_schedule,
        legacy_warmup_ratio=cfg.warmup_ratio,
        legacy_min_lr_ratio=cfg.min_lr_ratio,
    )
    assert scheduler.type == "cosine"
    assert scheduler.warmup_ratio == pytest.approx(0.2)
    assert scheduler.min_lr_ratio == pytest.approx(0.1)
