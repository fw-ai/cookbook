"""Unit tests for checkpoint_utils -- resume state, data position, fingerprints."""

import json
import os
import tempfile

import pytest

from training.utils.checkpoint_utils import (
    ResumeState,
    resolve_resume,
    save_loop_state,
    dataset_fingerprint,
    validate_dataset,
    validate_training_shape,
    _load_last_loop_state,
    STATE_FILE,
)


@pytest.fixture
def log_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestResolveResume:
    def test_fresh_start_no_state_no_init(self, log_dir):
        state = resolve_resume(log_dir)
        assert state.step == 0
        assert state.data_consumed == 0
        assert state.dcp_name is None

    def test_resume_from_state_file(self, log_dir):
        save_loop_state(log_dir, {
            "step": 5,
            "data_consumed": 40,
            "dcp_name": "step-5",
            "dataset_fingerprint": "abc123",
            "training_shape_id": "ts-qwen3-8b-policy",
            "source_job_id": "job-abc",
        })
        state = resolve_resume(log_dir)
        assert state.step == 5
        assert state.data_consumed == 40
        assert state.dcp_name == "step-5"
        assert state.dataset_fingerprint == "abc123"
        assert state.training_shape_id == "ts-qwen3-8b-policy"
        assert state.source_job_id == "job-abc"

    def test_resume_reads_last_entry(self, log_dir):
        save_loop_state(log_dir, {"step": 2, "data_consumed": 16, "dcp_name": "step-2"})
        save_loop_state(log_dir, {"step": 4, "data_consumed": 32, "dcp_name": "step-4"})
        state = resolve_resume(log_dir)
        assert state.step == 4
        assert state.data_consumed == 32

    def test_init_from_checkpoint_simple(self, log_dir):
        state = resolve_resume(log_dir, init_from_checkpoint="step-10")
        assert state.step == 0
        assert state.data_consumed == 0
        assert state.dcp_name == "step-10"
        assert state.source_job_id is None

    def test_init_from_checkpoint_cross_job(self, log_dir):
        state = resolve_resume(log_dir, init_from_checkpoint="job-xyz:step-5")
        assert state.step == 0
        assert state.dcp_name == "step-5"
        assert state.source_job_id == "job-xyz"

    def test_init_from_checkpoint_overrides_existing_state(self, log_dir):
        save_loop_state(log_dir, {
            "step": 10,
            "data_consumed": 80,
            "dcp_name": "step-10",
        })
        state = resolve_resume(log_dir, init_from_checkpoint="other-job:step-3")
        assert state.step == 0
        assert state.data_consumed == 0
        assert state.dcp_name == "step-3"
        assert state.source_job_id == "other-job"

        last = _load_last_loop_state(log_dir)
        assert last["step"] == 0
        assert last["dcp_name"] == "step-3"

    def test_init_from_checkpoint_gcs_path(self, log_dir):
        state = resolve_resume(log_dir, init_from_checkpoint="gs://bucket/path/step-5")
        assert state.dcp_name == "gs://bucket/path/step-5"
        assert state.source_job_id is None


class TestDataConsumedSlicing:
    def test_slice_from_zero(self):
        data = list(range(100))
        state = ResumeState(data_consumed=0)
        remaining = data[state.data_consumed:]
        assert len(remaining) == 100
        assert remaining[0] == 0

    def test_slice_from_middle(self):
        data = list(range(100))
        state = ResumeState(data_consumed=40)
        remaining = data[state.data_consumed:]
        assert len(remaining) == 60
        assert remaining[0] == 40

    def test_slice_past_end(self):
        data = list(range(100))
        state = ResumeState(data_consumed=100)
        remaining = data[state.data_consumed:]
        assert len(remaining) == 0


class TestDatasetFingerprint:
    def test_consistent(self):
        rows = [{"a": 1}, {"b": 2}, {"c": 3}]
        fp1 = dataset_fingerprint(rows)
        fp2 = dataset_fingerprint(rows)
        assert fp1 == fp2
        assert len(fp1) == 12

    def test_different_data(self):
        fp1 = dataset_fingerprint([{"a": 1}])
        fp2 = dataset_fingerprint([{"a": 2}])
        assert fp1 != fp2

    def test_different_length(self):
        fp1 = dataset_fingerprint([{"a": 1}])
        fp2 = dataset_fingerprint([{"a": 1}, {"b": 2}])
        assert fp1 != fp2

    def test_empty(self):
        assert dataset_fingerprint([]) == "empty"


class TestValidateDataset:
    def test_no_warning_on_match(self, log_dir, caplog):
        validate_dataset("abc123", "abc123", 40)
        assert "Dataset changed" not in caplog.text

    def test_warning_on_mismatch(self, log_dir, caplog):
        validate_dataset("abc123", "xyz789", 40)
        assert "Dataset changed" in caplog.text

    def test_no_warning_when_saved_is_none(self, log_dir, caplog):
        validate_dataset(None, "abc123", 0)
        assert "Dataset changed" not in caplog.text


class TestValidateTrainingShape:
    def test_no_warning_on_match(self, caplog):
        validate_training_shape("ts-qwen3-8b", "ts-qwen3-8b")
        assert "Training shape changed" not in caplog.text

    def test_warning_on_mismatch(self, caplog):
        validate_training_shape("ts-qwen3-8b", "ts-qwen3-30b")
        assert "Training shape changed" in caplog.text

    def test_no_warning_when_none(self, caplog):
        validate_training_shape(None, "ts-qwen3-8b")
        assert "Training shape changed" not in caplog.text


class TestSaveLoopState:
    def test_appends_entries(self, log_dir):
        save_loop_state(log_dir, {"step": 1})
        save_loop_state(log_dir, {"step": 2})

        path = os.path.join(log_dir, STATE_FILE)
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["step"] == 1
        assert json.loads(lines[1])["step"] == 2

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as parent:
            nested = os.path.join(parent, "sub", "dir")
            save_loop_state(nested, {"step": 1})
            assert os.path.exists(os.path.join(nested, STATE_FILE))
