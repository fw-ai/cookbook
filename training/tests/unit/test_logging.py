"""Tests for W&B/MLflow setup and metric sinks (training.utils.logging)."""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from training.utils.config import LoggingConfig, MLflowConfig, WandBConfig
from training.utils import logging as logging_utils
from training.utils.logging import (
    finish_logging,
    log_metrics,
    mlflow_finish,
    mlflow_log,
    setup_logging,
    setup_mlflow,
    setup_wandb,
)
from training.utils.runner import WandbConfigError


def _install_fake_wandb(
    monkeypatch, *, init_exc: Exception | None = None
) -> types.ModuleType:
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


def _install_fake_mlflow(monkeypatch) -> types.ModuleType:
    """Install a minimal fake ``mlflow`` module recording every call."""
    fake = types.ModuleType("mlflow")
    fake.calls: list[tuple] = []
    fake._active_run = None

    def _start_run(run_name=None):
        fake.calls.append(("start_run", run_name))
        fake._active_run = types.SimpleNamespace(
            info=types.SimpleNamespace(run_id="mlrun-1")
        )

    def _end_run():
        fake.calls.append(("end_run",))
        fake._active_run = None

    fake.start_run = _start_run
    fake.end_run = _end_run
    fake.active_run = lambda: fake._active_run
    fake.set_tracking_uri = lambda uri: fake.calls.append(("set_tracking_uri", uri))
    fake.set_experiment = lambda name: fake.calls.append(("set_experiment", name))
    fake.log_params = lambda params: fake.calls.append(("log_params", params))
    fake.log_metrics = lambda metrics, step=None: fake.calls.append(
        ("log_metrics", metrics, step)
    )
    fake.log_artifact = lambda path, artifact_path=None: fake.calls.append(
        ("log_artifact", path, artifact_path)
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


class TestSetupWandb:
    def test_no_entity_returns_false(self):
        assert setup_wandb(WandBConfig(), {}) is False

    def test_message_based_auth_error_raises_wandb_config_error(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "bad-key")
        _install_fake_wandb(monkeypatch, init_exc=Exception("401 Unauthorized"))
        with pytest.raises(
            WandbConfigError, match="authentication/configuration failed"
        ):
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

    def test_offline_mode_without_entity_sets_wandb_mode_env(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        fake = _install_fake_wandb(monkeypatch)
        init_kwargs: dict = {}
        fake.init = lambda **kwargs: init_kwargs.update(kwargs)

        assert setup_wandb(WandBConfig(mode="offline", project="proj"), {}) is True

        # Offline mode is just the env var; wandb.init stays untouched.
        assert os.environ["WANDB_MODE"] == "offline"
        assert "mode" not in init_kwargs
        assert init_kwargs["entity"] is None
        assert init_kwargs["project"] == "proj"

    def test_online_mode_leaves_wandb_mode_unset(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "key")
        monkeypatch.delenv("WANDB_MODE", raising=False)
        fake = _install_fake_wandb(monkeypatch)
        init_kwargs: dict = {}
        fake.init = lambda **kwargs: init_kwargs.update(kwargs)

        assert setup_wandb(WandBConfig(entity="acme"), {}) is True

        assert "WANDB_MODE" not in os.environ
        assert "mode" not in init_kwargs


class TestSetupMlflow:
    def test_disabled_without_tracking_uri_or_experiment(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
        assert setup_mlflow(MLflowConfig(), {}) is False

    def test_enabled_via_env_tracking_uri(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        fake = _install_fake_mlflow(monkeypatch)

        assert setup_mlflow(MLflowConfig(), {}) is True

        # Env values are read natively by mlflow; no programmatic override.
        assert fake.calls == [("start_run", None)]

    def test_enabled_via_env_experiment_name(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "exp")
        fake = _install_fake_mlflow(monkeypatch)

        assert setup_mlflow(MLflowConfig(), {}) is True

        assert fake.calls == [("start_run", None)]

    def test_explicit_config_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")
        fake = _install_fake_mlflow(monkeypatch)

        assert (
            setup_mlflow(MLflowConfig(tracking_uri="http://flag:5000"), {}) is True
        )

        assert fake.calls == [
            ("set_tracking_uri", "http://flag:5000"),
            ("start_run", None),
        ]

    def test_not_installed_returns_false(self, monkeypatch):
        # sys.modules["mlflow"] = None makes `import mlflow` raise ImportError.
        monkeypatch.setitem(sys.modules, "mlflow", None)
        assert setup_mlflow(MLflowConfig(experiment_name="exp"), {}) is False

    def test_starts_run_and_logs_flat_scalar_params(self, monkeypatch):
        fake = _install_fake_mlflow(monkeypatch)

        assert (
            setup_mlflow(
                MLflowConfig(
                    tracking_uri="http://mlflow:5000",
                    experiment_name="exp",
                    run_name="run",
                ),
                {"lr": 1e-5, "algorithm": "grpo", "nested": {"x": 1}, "none": None},
            )
            is True
        )

        assert fake.calls == [
            ("set_tracking_uri", "http://mlflow:5000"),
            ("set_experiment", "exp"),
            ("start_run", "run"),
            ("log_params", {"lr": "1e-05", "algorithm": "grpo"}),
        ]

    def test_param_failure_does_not_block_setup(self, monkeypatch, caplog):
        fake = _install_fake_mlflow(monkeypatch)

        def fail_params(_params):
            raise RuntimeError("tracking server down")

        fake.log_params = fail_params

        assert setup_mlflow(MLflowConfig(experiment_name="exp"), {"lr": 1.0}) is True
        assert "Failed to log config params to MLflow" in caplog.text


def test_wandb_and_mlflow_setup_are_independent(monkeypatch):
    """Both backends can be active at once; neither gates the other."""
    monkeypatch.setenv("WANDB_API_KEY", "key")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    fake_wandb = _install_fake_wandb(monkeypatch)
    fake_wandb.init = lambda **_kw: setattr(
        fake_wandb, "run", types.SimpleNamespace(url="http://wandb/run")
    )
    _install_fake_mlflow(monkeypatch)

    assert setup_wandb(WandBConfig(entity="acme"), {}) is True
    assert setup_mlflow(MLflowConfig(experiment_name="exp"), {}) is True


def test_log_metrics_fans_out_to_wandb_and_mlflow_simultaneously(
    tmp_path, monkeypatch
):
    fake_wandb = _install_fake_wandb(monkeypatch)
    wandb_logged: list[dict] = []
    fake_wandb.run = types.SimpleNamespace()
    fake_wandb.log = lambda metrics, step=None, commit=None: wandb_logged.append(
        metrics
    )
    fake_mlflow = _install_fake_mlflow(monkeypatch)
    fake_mlflow.start_run()
    fake_mlflow.calls.clear()

    log_metrics({"train/loss": 0.5}, step=2, metrics_file=str(tmp_path / "m.jsonl"))

    assert wandb_logged == [{"train/loss": 0.5, "step": 2}]
    assert fake_mlflow.calls == [("log_metrics", {"train/loss": 0.5, "step": 2.0}, 2)]


def test_mlflow_log_skips_non_numeric_and_non_finite_values(monkeypatch):
    fake = _install_fake_mlflow(monkeypatch)
    fake.start_run()
    fake.calls.clear()

    mlflow_log(
        {
            "ok": 1,
            "half": 0.5,
            "flag": True,
            "name": "x",
            "nan": float("nan"),
            "nested": {"a": 1},
            "none": None,
        },
        step=3,
    )

    assert fake.calls == [("log_metrics", {"ok": 1.0, "half": 0.5}, 3)]


def test_mlflow_log_is_noop_without_active_run(monkeypatch):
    fake = _install_fake_mlflow(monkeypatch)

    mlflow_log({"a": 1}, step=1)

    assert fake.calls == []


def test_mlflow_log_swallows_tracking_errors(monkeypatch, caplog):
    fake = _install_fake_mlflow(monkeypatch)
    fake.start_run()

    def fail_log_metrics(*_args, **_kwargs):
        raise RuntimeError("tracking server down")

    fake.log_metrics = fail_log_metrics

    mlflow_log({"a": 1}, step=1)

    assert "Failed to log metrics to MLflow" in caplog.text


def test_log_metrics_mirrors_the_business_step_to_mlflow(tmp_path, monkeypatch):
    fake = _install_fake_mlflow(monkeypatch)
    fake.start_run()
    fake.calls.clear()

    log_metrics(
        {"train/loss": 0.5, "note": "hi"},
        step=7,
        metrics_file=str(tmp_path / "metrics.jsonl"),
    )
    # Repeated business steps stay distinct MLflow data points.
    log_metrics({"train/loss": 0.4}, step=7)

    assert fake.calls == [
        ("log_metrics", {"train/loss": 0.5, "step": 7.0}, 7),
        ("log_metrics", {"train/loss": 0.4, "step": 7.0}, 7),
    ]


def test_mlflow_finish_attaches_the_canonical_ledger(tmp_path, monkeypatch):
    fake = _install_fake_mlflow(monkeypatch)
    fake.start_run()
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text('{"step":1}\n')

    mlflow_finish(metrics_file=str(metrics_path))

    assert fake.calls[-2] == ("log_artifact", str(metrics_path), "metrics")
    assert fake.calls[-1] == ("end_run",)
    assert fake.active_run() is None


def test_mlflow_finish_is_noop_without_active_run(monkeypatch):
    fake = _install_fake_mlflow(monkeypatch)

    mlflow_finish(metrics_file="/metrics.jsonl")

    assert fake.calls == []


class TestSetupLogging:
    """The unified LoggingConfig fan-out over the per-backend setups."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="unknown logging backends"):
            setup_logging(LoggingConfig(loggers=("tensorboard",)), {})

    def test_auto_enables_nothing_by_default(self, monkeypatch):
        _install_fake_wandb(monkeypatch)
        _install_fake_mlflow(monkeypatch)

        assert setup_logging(LoggingConfig(project="proj"), {}) == []

    def test_auto_mlflow_ignores_shared_project_alone(self, monkeypatch):
        # A shared project must not spawn a local ./mlruns run in auto mode:
        # recipes set a default project on every job.
        fake_mlflow = _install_fake_mlflow(monkeypatch)

        assert setup_logging(LoggingConfig(project="proj"), {}) == []
        assert fake_mlflow.calls == []

    def test_shared_project_and_run_name_flow_to_both_backends(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "key")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        fake_wandb = _install_fake_wandb(monkeypatch)
        init_kwargs: dict = {}

        def _init(**kwargs):
            init_kwargs.update(kwargs)
            fake_wandb.run = types.SimpleNamespace(url="http://wandb/run")

        fake_wandb.init = _init
        fake_mlflow = _install_fake_mlflow(monkeypatch)

        active = setup_logging(
            LoggingConfig(project="proj", run_name="r", wandb_entity="acme"), {}
        )

        assert active == ["wandb", "mlflow"]
        assert init_kwargs["project"] == "proj"
        assert init_kwargs["name"] == "r"
        assert ("set_experiment", "proj") in fake_mlflow.calls
        assert ("start_run", "r") in fake_mlflow.calls

    def test_explicit_loggers_select_backends(self, monkeypatch):
        fake_wandb = _install_fake_wandb(monkeypatch)
        fake_mlflow = _install_fake_mlflow(monkeypatch)

        # Only mlflow selected; wandb stays off even with an entity set.
        active = setup_logging(
            LoggingConfig(loggers=("mlflow",), project="proj", wandb_entity="acme"),
            {},
        )

        assert active == ["mlflow"]
        assert fake_wandb.run is None
        assert ("start_run", None) in fake_mlflow.calls

    def test_explicit_backend_without_settings_warns(self, monkeypatch, caplog):
        _install_fake_wandb(monkeypatch)

        active = setup_logging(LoggingConfig(loggers=("wandb",)), {})

        assert active == []
        assert "logging backend 'wandb' selected but inactive" in caplog.text


def test_finish_logging_ends_every_active_backend(tmp_path, monkeypatch):
    fake_wandb = _install_fake_wandb(monkeypatch)
    wandb_calls: list[str] = []
    fake_wandb.run = types.SimpleNamespace(id="run-1")
    fake_wandb.finish = lambda: wandb_calls.append("finish")
    fake_mlflow = _install_fake_mlflow(monkeypatch)
    fake_mlflow.start_run()
    fake_mlflow.calls.clear()
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text('{"step":1}\n')

    finish_logging(metrics_file=str(metrics_path))

    assert wandb_calls == ["finish"]
    assert ("log_artifact", str(metrics_path), "metrics") in fake_mlflow.calls
    assert fake_mlflow.calls[-1] == ("end_run",)


def test_log_metrics_sends_the_same_plain_record_to_both_sinks(
    tmp_path,
    monkeypatch,
):
    metrics_path = tmp_path / "metrics.jsonl"
    jsonl_calls: list[tuple[str, dict]] = []
    wandb_calls: list[tuple[dict, int | None]] = []
    mlflow_calls: list[tuple[dict, int | None]] = []
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
    monkeypatch.setattr(
        logging_utils,
        "mlflow_log",
        lambda record, step=None: mlflow_calls.append((record, step)),
    )
    metrics = {
        "train/step": 3,
        "rollout/reward": 0.75,
        "train/inference_k3": float("nan"),
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
        "train/inference_k3": None,
        "nested": {"values": [1.0, None]},
    }
    assert jsonl_calls[0][0] == str(metrics_path)
    assert jsonl_calls[0][1] == expected
    assert wandb_calls == [(expected, None)]
    # MLflow receives the identical record, with the business step explicit.
    assert mlflow_calls == [(expected, 3)]
    assert jsonl_calls[0][1] is wandb_calls[0][0] is mlflow_calls[0][0]
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
    monkeypatch.setattr(
        logging_utils, "wandb_log", lambda record: wandb_calls.append(record)
    )

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


def test_wandb_finish_does_not_mask_run_when_trace_flush_fails(
    monkeypatch,
    caplog,
):
    def fail_trace_flush():
        raise OSError("trace disk full")

    monkeypatch.setattr(logging_utils, "flush_phase_trace", fail_trace_flush)
    fake = _install_fake_wandb(monkeypatch)
    calls: list[tuple] = []

    fake.run = types.SimpleNamespace(
        id="run-123",
    )
    fake.finish = lambda: calls.append(("finish",))

    logging_utils.wandb_finish()

    assert calls == [("finish",)]
    assert "Failed to flush client phase trace" in caplog.text


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


def test_async_eval_metrics_use_the_rollout_step_axis():
    assert logging_utils.ASYNC_RL_WANDB_METRIC_STEPS["eval/*"] == "rollout/step"
    assert logging_utils.ASYNC_RL_WANDB_METRIC_STEPS["checkpoint/*"] == "rollout/step"
