"""Unit tests for ``training.utils.checkpoints``."""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from training.utils.checkpoints import (
    DATALOADER_BASE_NAME,
    ResumeInfo,
    TrainingCheckpoints,
    validate_warm_start_config,
)


@pytest.fixture
def log_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _mock_client():
    client = MagicMock()
    client.resolve_checkpoint_path.side_effect = (
        lambda name, source_job_id=None: f"path://{source_job_id or 'self'}/{name}"
    )

    def _save_sampler(name, checkpoint_type="base"):
        result = MagicMock()
        result.snapshot_name = f"{name}-snap"
        return result

    client.save_weights_for_sampler_ext.side_effect = _save_sampler
    return client


def _mock_fw_client(rows=None):
    fw = MagicMock()
    fw.list_checkpoints.return_value = rows or []
    fw.promote_checkpoint.return_value = {"state": "READY", "kind": "HF_BASE_MODEL"}
    return fw


def _row(short_name, *, ctype, promotable, create_time):
    return {
        "name": f"accounts/a/rlorTrainerJobs/job-1/checkpoints/{short_name}",
        "createTime": create_time,
        "checkpointType": ctype,
        "promotable": promotable,
    }


def _make(log_dir, *, fw_rows=None, lora_rank=0):
    client = _mock_client()
    fw = _mock_fw_client(rows=fw_rows)
    ckpt = TrainingCheckpoints(
        client, fw, trainer_id="job-1", log_path=log_dir, lora_rank=lora_rank
    )
    return ckpt, client, fw


# -- validate_warm_start_config ------------------------------------------------


class TestValidateWarmStartConfig:
    def test_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_warm_start_config(
                warm_start_from_adapter="some/adapter",
                init_from_checkpoint="job:step-5",
                lora_rank=8,
            )

    def test_warm_start_requires_lora(self):
        with pytest.raises(ValueError, match="cfg.base_model"):
            validate_warm_start_config(
                warm_start_from_adapter="some/adapter",
                init_from_checkpoint=None,
                lora_rank=0,
            )

    def test_ok(self):
        validate_warm_start_config(
            warm_start_from_adapter=None,
            init_from_checkpoint=None,
            lora_rank=0,
        )
        validate_warm_start_config(
            warm_start_from_adapter="a",
            init_from_checkpoint=None,
            lora_rank=8,
        )


# -- resume --------------------------------------------------------------------


class TestResume:
    def test_fresh_start_when_empty(self, log_dir):
        ckpt, client, _ = _make(log_dir, fw_rows=[])
        assert ckpt.resume() is None
        client.load_state_with_optimizer.assert_not_called()

    def test_resume_newest_training_row(self, log_dir):
        rows = [
            _row("step-5", ctype="CHECKPOINT_TYPE_TRAINING", promotable=False,
                 create_time="2026-04-01T00:00:00Z"),
            _row("step-10", ctype="CHECKPOINT_TYPE_TRAINING", promotable=False,
                 create_time="2026-04-02T00:00:00Z"),
            _row("step-7-sampler", ctype="CHECKPOINT_TYPE_INFERENCE_BASE",
                 promotable=True, create_time="2026-04-03T00:00:00Z"),
        ]
        # Pre-populate dataloader.json so resume can recover data_consumed.
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, DATALOADER_BASE_NAME), "w") as f:
            json.dump({"step-5": 40, "step-10": 80}, f)

        ckpt, client, _ = _make(log_dir, fw_rows=rows)
        info = ckpt.resume()
        assert info == ResumeInfo(step=10, data_consumed=80, source_job_id="job-1")
        client.load_state_with_optimizer.assert_called_once_with("path://job-1/step-10")

    def test_resume_picks_training_lora_too(self, log_dir):
        rows = [
            _row("step-5", ctype="CHECKPOINT_TYPE_TRAINING_LORA", promotable=False,
                 create_time="2026-04-01T00:00:00Z"),
            _row("step-5-hotload", ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                 promotable=True, create_time="2026-04-01T00:05:00Z"),
        ]
        ckpt, client, _ = _make(log_dir, fw_rows=rows, lora_rank=8)
        info = ckpt.resume()
        assert info is not None
        assert info.step == 5
        client.load_state_with_optimizer.assert_called_once_with("path://job-1/step-5")

    def test_init_from_checkpoint_takes_priority(self, log_dir):
        rows = [
            _row("step-50", ctype="CHECKPOINT_TYPE_TRAINING", promotable=False,
                 create_time="2026-04-02T00:00:00Z"),
        ]
        ckpt, client, _ = _make(log_dir, fw_rows=rows)
        info = ckpt.resume(init_from_checkpoint="other-job:step-3")
        assert info == ResumeInfo(step=0, data_consumed=0, source_job_id="other-job")
        client.load_state_with_optimizer.assert_called_once_with(
            "path://other-job/step-3"
        )

    def test_warm_start_adapter_when_no_resume(self, log_dir):
        ckpt, client, _ = _make(log_dir, fw_rows=[], lora_rank=8)
        info = ckpt.resume(warm_start_from_adapter="hf/adapter")
        assert info == ResumeInfo(step=0, data_consumed=0, source_job_id=None)
        client.load_adapter.assert_called_once_with("hf/adapter")

    def test_list_failure_treated_as_fresh_start(self, log_dir):
        ckpt, client, fw = _make(log_dir)
        fw.list_checkpoints.side_effect = RuntimeError("503 Service Unavailable")
        info = ckpt.resume()
        assert info is None
        client.load_state_with_optimizer.assert_not_called()


# -- save ----------------------------------------------------------------------


class TestSave:
    def test_resumable_only_writes_dcp_and_dataloader(self, log_dir):
        ckpt, client, fw = _make(log_dir)
        ckpt.save("step-1", resumable=True, promotable=False, data_consumed=100)

        client.save_state.assert_called_once_with("step-1")
        client.save_weights_for_sampler_ext.assert_not_called()

        with open(os.path.join(log_dir, DATALOADER_BASE_NAME)) as f:
            assert json.load(f) == {"step-1": 100}

    def test_promotable_only_writes_sampler_no_dataloader(self, log_dir):
        ckpt, client, fw = _make(log_dir)
        ckpt.save("step-1", resumable=False, promotable=True)
        client.save_state.assert_not_called()
        client.save_weights_for_sampler_ext.assert_called_once_with(
            "step-1", checkpoint_type="base"
        )
        assert not os.path.exists(os.path.join(log_dir, DATALOADER_BASE_NAME))

    def test_both_writes_both(self, log_dir):
        ckpt, client, _ = _make(log_dir)
        ckpt.save("step-1", resumable=True, promotable=True, data_consumed=42)
        client.save_state.assert_called_once_with("step-1")
        client.save_weights_for_sampler_ext.assert_called_once()

    def test_neither_raises(self, log_dir):
        ckpt, _, _ = _make(log_dir)
        with pytest.raises(ValueError, match="at least one"):
            ckpt.save("step-1", resumable=False, promotable=False)

    def test_skip_if_promotable_already_exists(self, log_dir):
        existing = [
            _row("step-1", ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                 promotable=True, create_time="2026-04-01T00:00:00Z"),
        ]
        ckpt, client, _ = _make(log_dir, fw_rows=existing, lora_rank=8)
        ckpt.save("step-1", resumable=True, promotable=True, data_consumed=10)
        client.save_state.assert_called_once()
        client.save_weights_for_sampler_ext.assert_not_called()

    def test_no_skip_when_existing_not_promotable(self, log_dir):
        existing = [
            _row("step-1", ctype="CHECKPOINT_TYPE_TRAINING",
                 promotable=False, create_time="2026-04-01T00:00:00Z"),
        ]
        ckpt, client, _ = _make(log_dir, fw_rows=existing)
        ckpt.save("step-1", resumable=False, promotable=True)
        client.save_weights_for_sampler_ext.assert_called_once()

    def test_skip_check_failure_falls_back_to_save(self, log_dir):
        ckpt, client, fw = _make(log_dir)
        fw.list_checkpoints.side_effect = RuntimeError("503")
        ckpt.save("step-1", resumable=False, promotable=True)
        client.save_weights_for_sampler_ext.assert_called_once()


# -- promote_latest ------------------------------------------------------------


class TestPromoteLatest:
    def test_picks_newest_promotable(self, log_dir):
        rows = [
            _row("step-5", ctype="CHECKPOINT_TYPE_INFERENCE_BASE",
                 promotable=True, create_time="2026-04-01T00:00:00Z"),
            _row("step-10", ctype="CHECKPOINT_TYPE_INFERENCE_BASE",
                 promotable=True, create_time="2026-04-02T00:00:00Z"),
            _row("step-10-dcp", ctype="CHECKPOINT_TYPE_TRAINING",
                 promotable=False, create_time="2026-04-02T00:05:00Z"),
        ]
        ckpt, _, fw = _make(log_dir, fw_rows=rows)
        ckpt.promote_latest("my-model", "accounts/a/models/qwen3-1p7b-bf16")
        fw.promote_checkpoint.assert_called_once_with(
            "job-1",
            "step-10",
            "my-model",
            "accounts/a/models/qwen3-1p7b-bf16",
            hot_load_deployment_id=None,
        )

    def test_skips_arc_v2_because_not_promotable(self, log_dir):
        rows = [
            _row("step-5-arc", ctype="CHECKPOINT_TYPE_INFERENCE_ARC_V2",
                 promotable=False, create_time="2026-04-02T00:00:00Z"),
            _row("step-5-lora", ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                 promotable=True, create_time="2026-04-01T00:00:00Z"),
        ]
        ckpt, _, fw = _make(log_dir, fw_rows=rows)
        ckpt.promote_latest("out", "base")
        fw.promote_checkpoint.assert_called_once()
        args, _ = fw.promote_checkpoint.call_args
        assert args[1] == "step-5-lora"

    def test_errors_when_no_promotable(self, log_dir):
        rows = [
            _row("step-1", ctype="CHECKPOINT_TYPE_TRAINING",
                 promotable=False, create_time="2026-04-01T00:00:00Z"),
        ]
        ckpt, _, _ = _make(log_dir, fw_rows=rows)
        with pytest.raises(RuntimeError, match="No promotable"):
            ckpt.promote_latest("out", "base")


# -- dataloader.json bookkeeping -----------------------------------------------


class TestDataloaderJson:
    def test_bounded_history(self, log_dir):
        ckpt, _, _ = _make(log_dir)
        for i in range(1, 26):
            ckpt.save(f"step-{i}", resumable=True, promotable=False, data_consumed=i)
        with open(os.path.join(log_dir, DATALOADER_BASE_NAME)) as f:
            data = json.load(f)
        # Keep only the newest 20.
        assert len(data) == 20
        assert set(data.keys()) == {f"step-{i}" for i in range(6, 26)}
