from __future__ import annotations

import pytest

from training.examples.serverless_rl.countdown_rl import (
    Config,
    _validate_config,
    _validate_datum_length,
    _validate_prompt_budget,
)


def test_serverless_defaults_use_supported_model_and_matching_tokenizer():
    cfg = Config()
    assert cfg.base_model == "accounts/fireworks/models/qwen3p6-27b"
    assert cfg.tokenizer_model == "Qwen/Qwen3.6-27B"
    assert cfg.max_seq_len == 65536
    assert cfg.lora_rank > 0


@pytest.mark.parametrize("max_seq_len", [0, -1])
def test_serverless_rejects_invalid_sequence_bound(max_seq_len):
    cfg = Config(max_seq_len=max_seq_len)
    with pytest.raises(ValueError, match="max_seq_len > 0"):
        _validate_config(cfg)


def test_serverless_rejects_non_lora_config():
    cfg = Config(lora_rank=0)
    with pytest.raises(ValueError, match="lora_rank > 0"):
        _validate_config(cfg)


def test_serverless_prompt_budget_accepts_exact_bound():
    _validate_prompt_budget(512, 512, 1024)


def test_serverless_prompt_budget_rejects_overflow():
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        _validate_prompt_budget(513, 512, 1024)


def test_serverless_datum_length_accepts_exact_bound():
    _validate_datum_length(1024, 1024)


def test_serverless_datum_length_rejects_overflow():
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        _validate_datum_length(1025, 1024)
