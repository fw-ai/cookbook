"""Tests for deprecated parameter backward compatibility.

Verifies that __post_init__ on recipe Configs correctly migrates old
parameter names and emits visible deprecation warnings.
"""

import logging
import pytest

from training.recipes.sft_loop import Config as SFTConfig
from training.recipes.dpo_loop import Config as DPOConfig
from training.recipes.orpo_loop import Config as ORPOConfig
from training.utils.config import DeployConfig


class TestTokenizerModelDeprecation:
    """tokenizer_model -> hf_tokenizer_name migration."""

    def test_sft_config_migrates_tokenizer_model(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = SFTConfig(
                log_path="/tmp/test",
                base_model="accounts/fw/models/qwen3-8b",
                dataset="/tmp/data.jsonl",
                tokenizer_model="Qwen/Qwen3-8B",
            )
        assert cfg.hf_tokenizer_name == "Qwen/Qwen3-8B"
        assert cfg.tokenizer_model is None  # cleared after migration
        assert "DEPRECATED" in caplog.text
        assert "tokenizer_model" in caplog.text

    def test_dpo_config_migrates_tokenizer_model(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = DPOConfig(
                log_path="/tmp/test",
                base_model="accounts/fw/models/qwen3-8b",
                dataset="/tmp/data.jsonl",
                tokenizer_model="Qwen/Qwen3-8B",
            )
        assert cfg.hf_tokenizer_name == "Qwen/Qwen3-8B"
        assert "DEPRECATED" in caplog.text

    def test_orpo_config_migrates_tokenizer_model(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = ORPOConfig(
                log_path="/tmp/test",
                base_model="accounts/fw/models/qwen3-8b",
                dataset="/tmp/data.jsonl",
                tokenizer_model="Qwen/Qwen3-8B",
            )
        assert cfg.hf_tokenizer_name == "Qwen/Qwen3-8B"
        assert "DEPRECATED" in caplog.text

    def test_deploy_config_migrates_tokenizer_model(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = DeployConfig(tokenizer_model="Qwen/Qwen3-8B")
        assert cfg.hf_tokenizer_name == "Qwen/Qwen3-8B"
        assert cfg.tokenizer_model is None
        assert "DEPRECATED" in caplog.text

    def test_hf_tokenizer_name_takes_precedence(self):
        cfg = SFTConfig(
            log_path="/tmp/test",
            base_model="m",
            dataset="d",
            hf_tokenizer_name="Qwen/Qwen3-8B",
        )
        assert cfg.hf_tokenizer_name == "Qwen/Qwen3-8B"

    def test_no_warning_when_using_new_name(self, caplog):
        with caplog.at_level(logging.WARNING):
            SFTConfig(
                log_path="/tmp/test",
                base_model="m",
                dataset="d",
                hf_tokenizer_name="Qwen/Qwen3-8B",
            )
        assert "DEPRECATED" not in caplog.text


class TestGradAccumDeprecation:
    """grad_accum -> batch_size migration."""

    def test_sft_config_warns_on_grad_accum(self, caplog):
        with caplog.at_level(logging.WARNING):
            SFTConfig(
                log_path="/tmp/test",
                base_model="m",
                dataset="d",
                hf_tokenizer_name="T",
                grad_accum=4,
            )
        assert "DEPRECATED" in caplog.text
        assert "grad_accum" in caplog.text

    def test_no_warning_when_grad_accum_is_default(self, caplog):
        with caplog.at_level(logging.WARNING):
            SFTConfig(
                log_path="/tmp/test",
                base_model="m",
                dataset="d",
                hf_tokenizer_name="T",
            )
        # grad_accum=1 is default, should not warn
        assert "grad_accum" not in caplog.text
