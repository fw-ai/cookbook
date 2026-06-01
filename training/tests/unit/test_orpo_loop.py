from __future__ import annotations


import pytest

import training.recipes.orpo_loop as module


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
