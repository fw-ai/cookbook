"""Tests for ``fireworks.training.cookbook.utils.validation.validate_preflight``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fireworks.training.cookbook.utils.validation import validate_preflight


def _cfg(**overrides):
    """Build a minimal args namespace with sensible defaults."""
    defaults = dict(
        base_model="accounts/test/models/qwen3-1p7b",
        dataset="data/test.jsonl",
        hot_load_interval=0,
        hot_load_deployment_id=None,
        create_deployment=False,
        hot_load_before_training=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Credential checks
# ---------------------------------------------------------------------------


class TestCredentialChecks:
    def test_missing_api_key_raises(self):
        with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
            validate_preflight(_cfg(), fw_api_key=None, fw_account_id="acct")

    def test_missing_account_id_raises(self):
        with pytest.raises(RuntimeError, match="FIREWORKS_ACCOUNT_ID"):
            validate_preflight(_cfg(), fw_api_key="key", fw_account_id=None)

    def test_skip_credential_check(self):
        # Should NOT raise even with missing credentials
        validate_preflight(
            _cfg(),
            fw_api_key=None,
            fw_account_id=None,
            skip_credential_check=True,
        )


# ---------------------------------------------------------------------------
# Hotload config
# ---------------------------------------------------------------------------


class TestHotloadConfig:
    def test_hot_load_interval_without_deployment_id_raises(self):
        args = _cfg(hot_load_interval=5, hot_load_deployment_id=None)
        with pytest.raises(RuntimeError, match="deployment_id"):
            validate_preflight(args, fw_api_key="k", fw_account_id="a")

    def test_create_deployment_without_deployment_id_raises(self):
        args = _cfg(create_deployment=True, hot_load_deployment_id=None)
        with pytest.raises(RuntimeError, match="deployment_id"):
            validate_preflight(args, fw_api_key="k", fw_account_id="a")

    def test_hot_load_interval_with_deployment_id_ok(self):
        args = _cfg(hot_load_interval=5, hot_load_deployment_id="dep-1")
        validate_preflight(args, fw_api_key="k", fw_account_id="a")


# ---------------------------------------------------------------------------
# Model name format
# ---------------------------------------------------------------------------


class TestModelNameFormat:
    def test_invalid_format_raises(self):
        args = _cfg(base_model="qwen3-8b")
        with pytest.raises(RuntimeError, match="Invalid base_model"):
            validate_preflight(args, fw_api_key="k", fw_account_id="a")

    def test_valid_format(self):
        args = _cfg(base_model="accounts/fireworks/models/qwen3-8b")
        validate_preflight(args, fw_api_key="k", fw_account_id="a")


# ---------------------------------------------------------------------------
# Multiple errors collected
# ---------------------------------------------------------------------------


class TestMultipleErrors:
    def test_credential_errors_collected(self):
        args = _cfg()
        with pytest.raises(RuntimeError) as exc_info:
            validate_preflight(args, fw_api_key=None, fw_account_id=None)
        msg = str(exc_info.value)
        assert "FIREWORKS_API_KEY" in msg
        assert "FIREWORKS_ACCOUNT_ID" in msg

    def test_config_errors_collected(self):
        args = _cfg(
            base_model="bad-model",
            dataset="",
            hot_load_interval=5,
            hot_load_deployment_id=None,
        )
        with pytest.raises(RuntimeError) as exc_info:
            validate_preflight(args, fw_api_key="k", fw_account_id="a")
        msg = str(exc_info.value)
        assert "Invalid base_model" in msg
        assert "Missing dataset" in msg


# ---------------------------------------------------------------------------
# Valid config passes
# ---------------------------------------------------------------------------


class TestValidConfig:
    def test_passes(self):
        args = _cfg(
            base_model="accounts/test/models/m",
            hot_load_interval=0,
        )
        validate_preflight(args, fw_api_key="key", fw_account_id="acct")
