"""Tests for W&B setup error handling (training.utils.logging)."""

from __future__ import annotations

import os
import sys
import types

import pytest

from training.utils.config import WandBConfig
from training.utils.logging import setup_wandb
from training.utils.runner import WandbConfigError


def _install_fake_wandb(monkeypatch, *, init_exc: Exception | None = None) -> types.ModuleType:
    """Install a minimal fake ``wandb`` module so setup_wandb runs without the dep."""
    fake = types.ModuleType("wandb")
    errors_mod = types.ModuleType("wandb.errors")

    class AuthenticationError(Exception):
        pass

    class UsageError(Exception):
        pass

    class CommError(Exception):
        pass

    errors_mod.AuthenticationError = AuthenticationError
    errors_mod.UsageError = UsageError
    errors_mod.CommError = CommError
    fake.errors = errors_mod
    fake.run = None

    def _init(**_kwargs):
        if init_exc is not None:
            raise init_exc
        fake.run = None

    fake.init = _init
    fake.define_metric = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", errors_mod)
    return fake


class TestSetupWandb:
    def test_no_entity_returns_false(self):
        assert setup_wandb(WandBConfig(), {}) is False

    def test_message_based_auth_error_raises_wandb_config_error(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "bad-key")
        _install_fake_wandb(monkeypatch, init_exc=Exception("401 Unauthorized"))
        with pytest.raises(WandbConfigError, match="authentication/configuration failed"):
            setup_wandb(WandBConfig(entity="acme", project="proj"), {})

    def test_typed_auth_error_raises_wandb_config_error(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "bad-key")
        fake = _install_fake_wandb(monkeypatch)

        def _init(**_kwargs):
            raise fake.errors.AuthenticationError("nope")

        fake.init = _init
        with pytest.raises(WandbConfigError):
            setup_wandb(WandBConfig(entity="acme"), {})

    def test_non_auth_error_propagates_unchanged(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "key")
        _install_fake_wandb(monkeypatch, init_exc=RuntimeError("disk full"))
        with pytest.raises(RuntimeError, match="disk full"):
            setup_wandb(WandBConfig(entity="acme"), {})

    def test_missing_key_uses_offline_mode(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        _install_fake_wandb(monkeypatch)
        assert setup_wandb(WandBConfig(entity="acme"), {}) is True
        assert os.environ.get("WANDB_MODE") == "offline"
