"""Tests for cookbook and SDK default values."""

from __future__ import annotations


def test_grpo_temperature():
    from training.recipes.rl_loop import Config

    assert Config().temperature == 1.0


def test_grpo_max_completion_tokens():
    from training.recipes.rl_loop import Config

    assert Config().max_completion_tokens == 1024


def test_is_config_defaults():
    from training.utils.rl.importance_sampling import ISConfig

    cfg = ISConfig()
    assert cfg.eps_clip == 0.2
    assert cfg.tis_cap == 5.0


def test_cispo_config_defaults():
    from training.recipes.rl_loop import Config

    cfg = Config()
    assert cfg.cispo.eps_low == 0.2
    assert cfg.cispo.eps_high == 0.28


def test_cispo_is_valid_policy_loss():
    from training.recipes.rl_loop import Config

    cfg = Config(policy_loss="cispo")
    assert cfg.policy_loss == "cispo"


def test_dpo_has_tokenizer_model():
    from training.recipes.dpo_loop import Config

    assert hasattr(Config(), "tokenizer_model")
