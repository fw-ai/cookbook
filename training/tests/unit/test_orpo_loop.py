from __future__ import annotations


import pytest

import training.recipes.orpo_loop as module


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
