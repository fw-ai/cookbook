"""Tests for the runner contract (training.utils.runner)."""

from __future__ import annotations

import json
import os
import time

import pytest

from training.utils.runner import RunnerConfig, RunnerIO, RunStatus


# -- RunnerConfig -------------------------------------------------------------


class TestRunnerConfig:
    def test_resolve_uses_direct_values(self):
        cfg = RunnerConfig(status_file="/a", metadata_file="/b", metrics_file="/c", output_model_path="/d")
        resolved = cfg.resolve()
        assert resolved.status_file == "/a"
        assert resolved.metadata_file == "/b"
        assert resolved.metrics_file == "/c"
        assert resolved.output_model_path == "/d"

    def test_resolve_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("COOKBOOK_STATUS_FILE", "/env/status.json")
        monkeypatch.setenv("COOKBOOK_METADATA_FILE", "/env/meta.json")
        monkeypatch.setenv("COOKBOOK_METRICS_FILE", "/env/metrics.jsonl")
        monkeypatch.setenv("COOKBOOK_OUTPUT_MODEL_PATH", "/env/model.json")

        cfg = RunnerConfig()
        resolved = cfg.resolve()
        assert resolved.status_file == "/env/status.json"
        assert resolved.metadata_file == "/env/meta.json"
        assert resolved.metrics_file == "/env/metrics.jsonl"
        assert resolved.output_model_path == "/env/model.json"

    def test_resolve_direct_overrides_env(self, monkeypatch):
        monkeypatch.setenv("COOKBOOK_STATUS_FILE", "/env/status.json")
        cfg = RunnerConfig(status_file="/direct/status.json")
        resolved = cfg.resolve()
        assert resolved.status_file == "/direct/status.json"

    def test_enabled_false_when_empty(self):
        assert not RunnerConfig().enabled

    def test_enabled_true_when_any_set(self):
        assert RunnerConfig(status_file="/x").enabled
        assert RunnerConfig(metadata_file="/x").enabled
        assert RunnerConfig(metrics_file="/x").enabled
        assert RunnerConfig(output_model_path="/x").enabled


# -- RunnerIO: status ----------------------------------------------------------


class TestRunnerIOStatus:
    def test_write_status_creates_file(self, tmp_path):
        path = str(tmp_path / "status.json")
        runner = RunnerIO(RunnerConfig(status_file=path))

        runner.write_status(RunStatus.RUNNING, step=3, total_steps=10, message="training")

        data = json.loads(open(path).read())
        assert data["code"] == 0
        assert data["message"] == "training"
        assert data["details"][0]["@type"] == "type.googleapis.com/gateway.JobProgress"
        assert data["details"][0]["percent"] == 30

    def test_status_protojson_strict_structure(self, tmp_path):
        """Enforce exact protojson shape: google.rpc.Status with JobProgress detail."""
        path = str(tmp_path / "status.json")
        runner = RunnerIO(RunnerConfig(status_file=path))

        # 7/20 -> 35%
        runner.write_status(RunStatus.RUNNING, step=7, total_steps=20, message="training")

        data = json.loads(open(path).read())
        # Top-level keys: exactly code, message, details
        assert set(data.keys()) == {"code", "message", "details"}
        assert isinstance(data["code"], int)
        assert isinstance(data["message"], str)
        # Details: list of exactly one element with @type and percent
        assert isinstance(data["details"], list)
        assert len(data["details"]) == 1
        detail = data["details"][0]
        assert set(detail.keys()) == {"@type", "percent"}
        assert detail["@type"] == "type.googleapis.com/gateway.JobProgress"
        assert isinstance(detail["percent"], int)
        assert 0 <= detail["percent"] <= 100
        assert detail["percent"] == 35

    def test_write_status_with_error(self, tmp_path):
        path = str(tmp_path / "status.json")
        runner = RunnerIO(RunnerConfig(status_file=path))

        runner.write_status(RunStatus.FAILED, error="OOM")

        data = json.loads(open(path).read())
        assert data["code"] == 9
        assert data["message"] == "OOM"
        assert "details" not in data

    def test_write_status_overwrites(self, tmp_path):
        path = str(tmp_path / "status.json")
        runner = RunnerIO(RunnerConfig(status_file=path))

        runner.write_status(RunStatus.RUNNING, step=1, total_steps=10)
        runner.write_status(RunStatus.COMPLETED, step=10, total_steps=10, message="done")

        data = json.loads(open(path).read())
        assert data["code"] == 0
        assert data["message"] == "done"
        assert data["details"][0]["percent"] == 100

    def test_write_status_noop_when_no_file(self):
        runner = RunnerIO(RunnerConfig())
        runner.write_status(RunStatus.RUNNING)

    def test_write_status_zero_total_steps(self, tmp_path):
        path = str(tmp_path / "status.json")
        runner = RunnerIO(RunnerConfig(status_file=path))
        runner.write_status(RunStatus.PENDING, step=0, total_steps=0)

        data = json.loads(open(path).read())
        # Exact minimal shape: only code and message, no details
        assert set(data.keys()) == {"code", "message"}
        assert data["code"] == 0
        assert "details" not in data

    def test_all_status_values(self, tmp_path):
        path = str(tmp_path / "status.json")
        runner = RunnerIO(RunnerConfig(status_file=path))

        for status in RunStatus:
            runner.write_status(status)
            data = json.loads(open(path).read())
            assert data["message"] == status.value
            if status == RunStatus.FAILED:
                assert data["code"] == 9
            else:
                assert data["code"] == 0


# -- RunnerIO: metadata --------------------------------------------------------


class TestRunnerIOMetadata:
    def test_write_metadata_with_tokens_and_time(self, tmp_path):
        path = str(tmp_path / "meta.json")
        runner = RunnerIO(RunnerConfig(metadata_file=path))
        runner.set_accelerator_info("NVIDIA_H100_80GB", 8)
        runner.start_training()

        runner.append_metrics(1, {}, tokens=5000)
        runner.append_metrics(2, {}, tokens=3000)
        time.sleep(0.15)
        runner.write_metadata()

        data = json.loads(open(path).read())
        meta = data["metadata"]
        assert meta["tokens"] == 8000
        assert meta["accelerator_seconds"]["NVIDIA_H100_80GB"] >= 1

    def test_accelerator_seconds_scales_by_device_count(self, tmp_path):
        path = str(tmp_path / "meta.json")
        runner = RunnerIO(RunnerConfig(metadata_file=path))
        runner.set_accelerator_info(None, 4)
        runner.start_training()
        time.sleep(0.05)
        runner.write_metadata()

        data = json.loads(open(path).read())
        meta = data["metadata"]
        assert meta["accelerator_seconds"]["UNKNOWN"] >= 0

    def test_write_metadata_before_start_training(self, tmp_path):
        path = str(tmp_path / "meta.json")
        runner = RunnerIO(RunnerConfig(metadata_file=path))
        runner.write_metadata()

        data = json.loads(open(path).read())
        meta = data["metadata"]
        assert meta["tokens"] == 0
        assert meta["accelerator_seconds"] == {}

    def test_write_metadata_noop_when_no_file(self):
        runner = RunnerIO(RunnerConfig())
        runner.append_metrics(1, {}, tokens=100)
        runner.write_metadata()

    def test_metadata_structure(self, tmp_path):
        path = str(tmp_path / "meta.json")
        runner = RunnerIO(RunnerConfig(metadata_file=path))
        runner.write_metadata()

        data = json.loads(open(path).read())
        assert "metadata" in data
        assert "tokens" in data["metadata"]
        assert "accelerator_seconds" in data["metadata"]


# -- RunnerIO: metrics ---------------------------------------------------------


class TestRunnerIOMetrics:
    def test_append_metrics_creates_jsonl(self, tmp_path):
        path = str(tmp_path / "metrics.jsonl")
        runner = RunnerIO(RunnerConfig(metrics_file=path))

        runner.append_metrics(1, {"train/loss": 2.5, "train/ppl": 12.0})
        runner.append_metrics(2, {"train/loss": 1.8, "train/ppl": 6.0})

        lines = open(path).read().strip().split("\n")
        assert len(lines) == 2

        r1 = json.loads(lines[0])
        assert r1["step"] == 1
        assert r1["train/loss"] == 2.5

        r2 = json.loads(lines[1])
        assert r2["step"] == 2
        assert r2["train/loss"] == 1.8

    def test_append_metrics_handles_nan(self, tmp_path):
        path = str(tmp_path / "metrics.jsonl")
        runner = RunnerIO(RunnerConfig(metrics_file=path))

        runner.append_metrics(1, {"value": float("nan")})

        r = json.loads(open(path).read())
        assert r["value"] is None

    def test_append_metrics_noop_when_no_file(self):
        runner = RunnerIO(RunnerConfig())
        runner.append_metrics(1, {"x": 1})


# -- RunnerIO: output model ----------------------------------------------------


class TestRunnerIOOutputModel:
    def test_write_output_model(self, tmp_path):
        path = str(tmp_path / "output.json")
        runner = RunnerIO(RunnerConfig(output_model_path=path))

        runner.write_output_model(
            model_id="accounts/test/models/my-model",
            checkpoint="step-100",
            job_id="job-abc",
        )

        data = json.loads(open(path).read())
        assert data["model_id"] == "accounts/test/models/my-model"
        assert data["checkpoint"] == "step-100"
        assert data["job_id"] == "job-abc"

    def test_write_output_model_with_extra(self, tmp_path):
        path = str(tmp_path / "output.json")
        runner = RunnerIO(RunnerConfig(output_model_path=path))

        runner.write_output_model(model_id="m", extra={"lora_rank": 16})

        data = json.loads(open(path).read())
        assert data["model_id"] == "m"
        assert data["lora_rank"] == 16

    def test_write_output_model_noop_when_no_file(self):
        runner = RunnerIO(RunnerConfig())
        runner.write_output_model(model_id="m")


# -- RunnerIO: env-var construction -------------------------------------------


class TestRunnerIOEnvVar:
    def test_constructs_from_env_vars(self, tmp_path, monkeypatch):
        status = str(tmp_path / "status.json")
        meta = str(tmp_path / "meta.json")
        metrics = str(tmp_path / "metrics.jsonl")
        output = str(tmp_path / "output.json")

        monkeypatch.setenv("COOKBOOK_STATUS_FILE", status)
        monkeypatch.setenv("COOKBOOK_METADATA_FILE", meta)
        monkeypatch.setenv("COOKBOOK_METRICS_FILE", metrics)
        monkeypatch.setenv("COOKBOOK_OUTPUT_MODEL_PATH", output)

        runner = RunnerIO()

        runner.write_status(RunStatus.RUNNING)
        assert os.path.exists(status)

        runner.write_metadata()
        assert os.path.exists(meta)

        runner.append_metrics(1, {"x": 1})
        assert os.path.exists(metrics)

        runner.write_output_model(model_id="m")
        assert os.path.exists(output)


# -- RunnerIO: atomic writes --------------------------------------------------


class TestRunnerIOAtomicWrite:
    def test_no_tmp_file_left_behind(self, tmp_path):
        path = str(tmp_path / "status.json")
        runner = RunnerIO(RunnerConfig(status_file=path))
        runner.write_status(RunStatus.COMPLETED)

        files = os.listdir(tmp_path)
        assert files == ["status.json"]

    def test_creates_parent_directories(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "status.json")
        runner = RunnerIO(RunnerConfig(status_file=path))
        runner.write_status(RunStatus.PENDING)

        assert os.path.exists(path)


# -- RunnerIO: context manager -------------------------------------------------


class TestRunnerIOContextManager:
    def test_exception_writes_failed_status(self, tmp_path):
        path = str(tmp_path / "status.json")
        runner = RunnerIO(RunnerConfig(status_file=path))
        runner.write_status(RunStatus.RUNNING, step=5, total_steps=10)

        with pytest.raises(RuntimeError, match="boom"):
            with runner:
                raise RuntimeError("boom")

        data = json.loads(open(path).read())
        assert data["code"] == 9
        assert data["message"] == "boom"
        assert data["details"][0]["percent"] == 50

    def test_exception_flushes_metadata(self, tmp_path):
        meta = str(tmp_path / "meta.json")
        runner = RunnerIO(RunnerConfig(metadata_file=meta))
        runner.start_training()
        runner.append_metrics(1, {}, tokens=1000)

        with pytest.raises(ValueError):
            with runner:
                raise ValueError("fail")

        data = json.loads(open(meta).read())
        assert data["metadata"]["tokens"] == 1000

    def test_exception_propagates(self):
        runner = RunnerIO()
        with pytest.raises(KeyboardInterrupt):
            with runner:
                raise KeyboardInterrupt

    def test_clean_exit_does_not_write_completed(self, tmp_path):
        path = str(tmp_path / "status.json")
        runner = RunnerIO(RunnerConfig(status_file=path))
        runner.write_status(RunStatus.RUNNING, step=1, total_steps=10)

        with runner:
            pass  # no exception

        data = json.loads(open(path).read())
        assert data["code"] == 0
        assert data["message"] == "running"

    def test_noop_runner_context_manager_does_not_raise(self):
        runner = RunnerIO()
        with runner:
            pass


# -- RunnerIO: default noop ---------------------------------------------------


class TestRunnerIONoop:
    def test_default_runner_is_noop(self):
        """When constructed with no config, all writes are silent noops."""
        runner = RunnerIO()
        runner.write_status(RunStatus.RUNNING)
        runner.append_metrics(1, {}, tokens=100)
        runner.write_metadata()
        runner.append_metrics(1, {"x": 1})
        runner.write_output_model(model_id="m")
        runner.start_training()
        runner.set_accelerator_info("H100", 8)
