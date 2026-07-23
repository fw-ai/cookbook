"""Tests for W&B setup error handling (training.utils.logging)."""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from training.utils.config import WandBConfig
from training.utils import logging as logging_utils
from training.utils.logging import log_metrics, setup_wandb
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


def test_log_metrics_sends_the_same_plain_record_to_both_sinks(
    tmp_path,
    monkeypatch,
):
    metrics_path = tmp_path / "metrics.jsonl"
    jsonl_calls: list[tuple[str, dict]] = []
    wandb_calls: list[tuple[dict, int | None]] = []
    monkeypatch.setattr(
        logging_utils.fileio,
        "append_jsonl",
        lambda path, record: jsonl_calls.append((path, record)),
    )
    monkeypatch.setattr(
        logging_utils,
        "wandb_log",
        lambda record, step=None: wandb_calls.append((record, step)),
    )
    metrics = {
        "train/step": 3,
        "rollout/reward": 0.75,
        "train/inference_kld": float("nan"),
        "nested": {"values": (1.0, float("inf"))},
    }

    log_metrics(
        metrics,
        step=3,
        metrics_file=str(metrics_path),
    )

    expected = {
        "step": 3,
        "train/step": 3,
        "rollout/reward": 0.75,
        "train/inference_kld": None,
        "nested": {"values": [1.0, None]},
    }
    assert jsonl_calls[0][0] == str(metrics_path)
    assert jsonl_calls[0][1] == expected
    assert wandb_calls == [(expected, None)]
    assert jsonl_calls[0][1] is wandb_calls[0][0]
    assert metrics["nested"]["values"][1] == float("inf")


def test_log_metrics_writes_jsonl_when_wandb_is_disabled(tmp_path, monkeypatch):
    metrics_path = tmp_path / "metrics.jsonl"
    _install_fake_wandb(monkeypatch)

    log_metrics(
        {"rollout/step": 4, "rollout/reward": float("-inf")},
        step=4,
        metrics_file=str(metrics_path),
    )

    assert json.loads(metrics_path.read_text()) == {
        "rollout/step": 4,
        "rollout/reward": None,
        "step": 4,
    }


def test_repeated_business_steps_remain_distinct_wandb_calls(monkeypatch):
    wandb_calls: list[tuple[dict, int | None]] = []
    monkeypatch.delenv("COOKBOOK_METRICS_FILE", raising=False)
    monkeypatch.setattr(
        logging_utils,
        "wandb_log",
        lambda record, step=None: wandb_calls.append((record, step)),
    )

    log_metrics({"train/step": 3, "rollout/reward": 0.5}, step=3)
    log_metrics({"train/step": 3, "perf/overlap_ratio": 0.8}, step=3)

    assert [record for record, _ in wandb_calls] == [
        {"train/step": 3, "rollout/reward": 0.5, "step": 3},
        {"train/step": 3, "perf/overlap_ratio": 0.8, "step": 3},
    ]
    assert [transport_step for _, transport_step in wandb_calls] == [None, None]


def test_configured_jsonl_write_failure_propagates_before_wandb(monkeypatch):
    wandb_calls: list[dict] = []

    def fail_jsonl_write(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(
        logging_utils.fileio,
        "append_jsonl",
        fail_jsonl_write,
    )
    monkeypatch.setattr(logging_utils, "wandb_log", lambda record: wandb_calls.append(record))

    with pytest.raises(OSError, match="disk full"):
        log_metrics(
            {"train/step": 1},
            step=1,
            metrics_file="/metrics.jsonl",
        )

    assert wandb_calls == []


def test_wandb_finish_attaches_the_canonical_ledger(tmp_path, monkeypatch):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text('{"step":1}\n')
    fake = _install_fake_wandb(monkeypatch)
    calls: list[tuple] = []

    class FakeArtifact:
        def __init__(self, name, *, type):
            calls.append(("artifact", name, type))

        def add_file(self, path, *, name):
            calls.append(("add_file", path, name))

    fake.Artifact = FakeArtifact
    fake.run = types.SimpleNamespace(
        id="run-123",
        log_artifact=lambda artifact: calls.append(("log_artifact", artifact)),
    )
    fake.finish = lambda: calls.append(("finish",))

    logging_utils.wandb_finish(metrics_file=str(metrics_path))

    assert calls[0] == ("artifact", "run-123-metrics", "metrics")
    assert calls[1] == ("add_file", str(metrics_path), "metrics.jsonl")
    assert calls[2][0] == "log_artifact"
    assert calls[3] == ("finish",)


def test_metric_step_is_explicit_and_overrides_payload_step():
    record = logging_utils._normalize_metrics(
        {"step": 1, "train/step": 2, "rollout/step": 3},
        4,
    )

    assert record == {"step": 4, "train/step": 2, "rollout/step": 3}


def test_metric_record_requires_a_finite_explicit_step():
    with pytest.raises(ValueError, match="finite number"):
        logging_utils._normalize_metrics(
            {},
            None,
        )
    with pytest.raises(ValueError, match="finite number"):
        logging_utils._normalize_metrics(
            {},
            float("nan"),
        )


def test_sync_and_async_recipes_use_the_shared_metrics_entrypoint():
    from training.recipes import async_rl_loop, rl_loop

    assert rl_loop.log_metrics is logging_utils.log_metrics
    assert async_rl_loop.log_metrics is logging_utils.log_metrics


def test_async_producer_metrics_use_their_own_event_axis():
    assert logging_utils.ASYNC_RL_WANDB_METRIC_STEPS["producer/event"] is None
    assert logging_utils.ASYNC_RL_WANDB_METRIC_STEPS["producer/*"] == "producer/event"
    assert logging_utils.ASYNC_RL_WANDB_METRIC_STEPS["async/*"] == "rollout/step"
    assert "pipeline/*" not in logging_utils.ASYNC_RL_WANDB_METRIC_STEPS
    assert "version/*" not in logging_utils.ASYNC_RL_WANDB_METRIC_STEPS
