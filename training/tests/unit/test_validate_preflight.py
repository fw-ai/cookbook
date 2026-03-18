"""Tests for ``training.utils.validation.validate_preflight``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.utils.validation import validate_preflight


def _cfg(**overrides):
    """Build a minimal args namespace with sensible defaults."""
    defaults = dict(
        base_model="accounts/test/models/qwen3-1p7b",
        dataset="data/test.jsonl",
        weight_sync_interval=0,
        hot_load_deployment_id=None,
        weight_sync_before_training=False,
        output_model_id=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Credential checks
# ---------------------------------------------------------------------------


class TestCredentialChecks:
    def test_missing_api_key_raises(self):
        with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
            validate_preflight(_cfg(), fw_api_key=None)

    def test_skip_credential_check(self):
        validate_preflight(
            _cfg(),
            fw_api_key=None,
            skip_credential_check=True,
        )


# ---------------------------------------------------------------------------
# Weight sync config
# ---------------------------------------------------------------------------


class TestWeightSyncConfig:
    def test_weight_sync_interval_without_deployment_id_ok(self):
        """Weight sync without deployment_id is fine -- setup_deployment auto-creates."""
        args = _cfg(weight_sync_interval=5, hot_load_deployment_id=None)
        validate_preflight(args, fw_api_key="k")

    def test_weight_sync_interval_with_deployment_id_ok(self):
        args = _cfg(weight_sync_interval=5, hot_load_deployment_id="dep-1")
        validate_preflight(args, fw_api_key="k")


# ---------------------------------------------------------------------------
# Model name format
# ---------------------------------------------------------------------------


class TestModelNameFormat:
    def test_invalid_format_raises(self):
        args = _cfg(base_model="qwen3-8b")
        with pytest.raises(RuntimeError, match="Invalid base_model"):
            validate_preflight(args, fw_api_key="k")

    def test_valid_format(self):
        args = _cfg(base_model="accounts/fireworks/models/qwen3-8b")
        validate_preflight(args, fw_api_key="k")

    def test_invalid_output_model_id_raises(self):
        args = _cfg(output_model_id="bad_name")
        with pytest.raises(RuntimeError, match="Invalid output_model_id"):
            validate_preflight(args, fw_api_key="k")


# ---------------------------------------------------------------------------
# Multiple errors collected
# ---------------------------------------------------------------------------


class TestMultipleErrors:
    def test_credential_errors_collected(self):
        args = _cfg()
        with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
            validate_preflight(args, fw_api_key=None)

    def test_config_errors_collected(self):
        args = _cfg(
            base_model="bad-model",
            dataset="",
            weight_sync_interval=5,
            hot_load_deployment_id=None,
            output_model_id="bad_name",
        )
        with pytest.raises(RuntimeError) as exc_info:
            validate_preflight(args, fw_api_key="k")
        msg = str(exc_info.value)
        assert "Invalid base_model" in msg
        assert "Missing dataset" in msg
        assert "Invalid output_model_id" in msg


# ---------------------------------------------------------------------------
# Valid config passes
# ---------------------------------------------------------------------------


class TestValidConfig:
    def test_passes(self):
        args = _cfg(
            base_model="accounts/test/models/m",
            weight_sync_interval=0,
        )
        validate_preflight(args, fw_api_key="key")


def test_format_sdk_error_produces_structured_output():
    from fireworks.training.sdk.errors import format_sdk_error

    msg = format_sdk_error(
        "Missing FIREWORKS_API_KEY",
        "No API key found.",
        "Set FIREWORKS_API_KEY.",
        docs_url="https://fireworks.ai/account/api-keys",
    )
    assert "Missing FIREWORKS_API_KEY" in msg
    assert "No API key found." in msg
    assert "Set FIREWORKS_API_KEY." in msg
    assert "https://fireworks.ai/account/api-keys" in msg
