"""Unit tests for checkpoint_utils -- resume, save."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from training.utils.checkpoint_utils import (
    ResumeInfo,
    get_last_checkpoint,
    resolve_resume,
    save_checkpoint,
    validate_warm_start_config,
    CheckpointKind,
    CHECKPOINTS_BASE_NAME,
)
from training.utils import fileio


@pytest.fixture
def log_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _make_mock_client(job_id="test-job"):
    """Create a mock ReconnectableClient with save_state/load_state methods."""
    client = MagicMock()
    client.job_id = job_id

    def _save_state(name):
        result = MagicMock()
        result.path = name
        return result

    def _save_sampler(name, checkpoint_type="base"):
        result = MagicMock()
        result.path = f"gs://unit/sampler/{name}"
        result.snapshot_name = f"{name}-sampler"
        return result

    client.save_state.side_effect = _save_state
    client.save_weights_for_sampler_ext.side_effect = _save_sampler
    client.resolve_checkpoint_path.side_effect = lambda name, source_job_id=None: (
        f"cross_job://{source_job_id}/{name}" if source_job_id else name
    )
    return client


def _write_checkpoint(log_dir, entry):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


class TestResolveResume:
    def test_fresh_start_no_state_no_init(self, log_dir):
        client = _make_mock_client()
        result = resolve_resume(client, log_dir)
        assert result is None
        client.load_state_with_optimizer.assert_not_called()

    def test_resume_from_checkpoints_file(self, log_dir):
        _write_checkpoint(log_dir, {
            "name": "step-5",
            "step": 5,
            "data_consumed": 40,
            "state_path": "cross_job://job-abc/step-5",
            "source_job_id": "job-abc",
        })
        client = _make_mock_client()
        result = resolve_resume(client, log_dir)
        assert result is not None
        assert result.step == 5
        assert result.data_consumed == 40
        assert result.source_job_id == "job-abc"
        client.load_state_with_optimizer.assert_called_once_with("cross_job://job-abc/step-5")

    def test_resume_reads_last_entry(self, log_dir):
        _write_checkpoint(log_dir, {
            "name": "step-2", "step": 2, "data_consumed": 16,
            "state_path": "cross_job://job-1/step-2",
        })
        _write_checkpoint(log_dir, {
            "name": "step-4", "step": 4, "data_consumed": 32,
            "state_path": "cross_job://job-1/step-4",
        })
        client = _make_mock_client()
        result = resolve_resume(client, log_dir)
        assert result.step == 4
        assert result.data_consumed == 32
        client.load_state_with_optimizer.assert_called_once_with("cross_job://job-1/step-4")

    def test_init_from_checkpoint_simple(self, log_dir):
        client = _make_mock_client()
        result = resolve_resume(client, log_dir, init_from_checkpoint="step-10")
        assert result.step == 0
        assert result.data_consumed == 0
        assert result.source_job_id is None
        client.resolve_checkpoint_path.assert_called_once_with("step-10", source_job_id=None)
        client.load_state_with_optimizer.assert_called_once()

    def test_init_from_checkpoint_cross_job(self, log_dir):
        client = _make_mock_client()
        result = resolve_resume(client, log_dir, init_from_checkpoint="job-xyz:step-5")
        assert result.step == 0
        assert result.source_job_id == "job-xyz"
        client.resolve_checkpoint_path.assert_called_once_with("step-5", source_job_id="job-xyz")

    def test_init_from_checkpoint_gcs_path(self, log_dir):
        client = _make_mock_client()
        result = resolve_resume(client, log_dir, init_from_checkpoint="gs://bucket/path/step-5")
        assert result.step == 0
        assert result.source_job_id is None
        client.resolve_checkpoint_path.assert_called_once_with(
            "gs://bucket/path/step-5", source_job_id=None,
        )

    def test_warm_start_from_adapter_loads_adapter_on_fresh_start(self, log_dir):
        client = _make_mock_client()
        result = resolve_resume(
            client, log_dir, warm_start_from_adapter="gs://bucket/adapter-dir",
        )
        assert result.step == 0
        assert result.source_job_id is None
        client.load_adapter.assert_called_once_with("gs://bucket/adapter-dir")

    def test_auto_resume_wins_over_warm_start_from_adapter(self, log_dir):
        client = _make_mock_client()
        _write_checkpoint(log_dir, {"name": "step-7", "step": 7, "data_consumed": 28,
                                    "state_path": "gs://bucket/job/step-7"})
        result = resolve_resume(
            client, log_dir,
            warm_start_from_adapter="gs://bucket/adapter-dir",
        )
        assert result.step == 7
        client.load_state_with_optimizer.assert_called_once()
        client.load_adapter.assert_not_called()

    def test_init_from_checkpoint_wins_over_warm_start_from_adapter(self, log_dir):
        client = _make_mock_client()
        result = resolve_resume(
            client, log_dir,
            init_from_checkpoint="gs://bucket/dcp",
            warm_start_from_adapter="gs://bucket/adapter-dir",
        )
        assert result.step == 0
        client.load_state_with_optimizer.assert_called_once()
        client.load_adapter.assert_not_called()


class TestValidateWarmStartConfig:
    def test_both_set_rejected(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_warm_start_config(
                warm_start_from_adapter="gs://a",
                init_from_checkpoint="gs://b",
                lora_rank=16,
            )

    def test_adapter_with_zero_lora_rank_rejected(self):
        with pytest.raises(ValueError, match="lora_rank > 0"):
            validate_warm_start_config(
                warm_start_from_adapter="gs://a",
                init_from_checkpoint=None,
                lora_rank=0,
            )

    def test_adapter_with_nonzero_lora_rank_ok(self):
        validate_warm_start_config(
            warm_start_from_adapter="gs://a",
            init_from_checkpoint=None,
            lora_rank=16,
        )

    def test_all_none_ok(self):
        validate_warm_start_config(
            warm_start_from_adapter=None,
            init_from_checkpoint=None,
            lora_rank=0,
        )


class TestSaveCheckpoint:
    def test_save_state_only(self, log_dir):
        client = _make_mock_client(job_id="job-x")
        paths = save_checkpoint(client, "step-3", log_dir, {"step": 3, "data_consumed": 24})

        assert "state_path" in paths
        assert paths["state_path"] == "cross_job://job-x/step-3"
        assert "sampler_path" not in paths

        ckpt_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
        assert os.path.exists(ckpt_path)
        with open(ckpt_path) as f:
            entry = json.loads(f.readline())
        assert entry["name"] == "step-3"
        assert entry["step"] == 3
        assert entry["state_path"] == "cross_job://job-x/step-3"

    def test_save_both(self, log_dir):
        client = _make_mock_client()
        paths = save_checkpoint(client, "step-5", log_dir, {"step": 5}, kind=CheckpointKind.BOTH)

        assert "state_path" in paths
        assert "sampler_path" in paths
        assert paths["sampler_path"] == "step-5-sampler"

    def test_save_with_base_model_and_training_shape(self, log_dir):
        client = _make_mock_client(job_id="job-shape")
        save_checkpoint(
            client, "step-2", log_dir, {"step": 2},
            base_model="accounts/fireworks/models/qwen3-8b",
            training_shape="accounts/fireworks/trainingShapes/ts-qwen3-8b-policy",
        )
        ckpt_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
        with open(ckpt_path) as f:
            entry = json.loads(f.readline())
        assert entry["base_model"] == "accounts/fireworks/models/qwen3-8b"
        assert entry["training_shape"] == "accounts/fireworks/trainingShapes/ts-qwen3-8b-policy"

    def test_save_without_model_metadata_omits_fields(self, log_dir):
        client = _make_mock_client()
        save_checkpoint(client, "step-1", log_dir, {"step": 1})
        ckpt_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
        with open(ckpt_path) as f:
            entry = json.loads(f.readline())
        assert "base_model" not in entry
        assert "training_shape" not in entry

    def test_appends_entries(self, log_dir):
        client = _make_mock_client()
        save_checkpoint(client, "step-1", log_dir, {"step": 1})
        save_checkpoint(client, "step-2", log_dir, {"step": 2})

        ckpt_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
        with open(ckpt_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["step"] == 1
        assert json.loads(lines[1])["step"] == 2

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as parent:
            nested = os.path.join(parent, "sub", "dir")
            client = _make_mock_client()
            save_checkpoint(client, "step-1", nested, {"step": 1})
            assert os.path.exists(os.path.join(nested, CHECKPOINTS_BASE_NAME))


class TestRLPromptDataset:
    def test_get_batch_basic(self):
        from training.utils.data import RLPromptDataset
        rows = [{"id": i} for i in range(10)]
        ds = RLPromptDataset(rows, prompts_per_step=3)
        assert len(ds) == 4  # ceil(10/3)
        assert ds.get_batch(0) == rows[:3]
        assert ds.get_batch(1) == rows[3:6]
        assert ds.get_batch(3) == rows[9:10]  # last partial batch

    def test_empty_dataset(self):
        from training.utils.data import RLPromptDataset
        ds = RLPromptDataset([], prompts_per_step=4)
        assert len(ds) == 0

    def test_exact_division(self):
        from training.utils.data import RLPromptDataset
        rows = [{"id": i} for i in range(12)]
        ds = RLPromptDataset(rows, prompts_per_step=4)
        assert len(ds) == 3
        assert ds.get_batch(2) == rows[8:12]


class TestLogPathRequired:
    """Verify log_path is a required field on all recipe Configs (no default)."""

    def test_sft_config_requires_log_path(self):
        from training.recipes.sft_loop import Config
        with pytest.raises(TypeError, match="log_path"):
            Config()

    def test_rl_config_requires_log_path(self):
        from training.recipes.rl_loop import Config
        with pytest.raises(TypeError, match="log_path"):
            Config()

    def test_dpo_config_requires_log_path(self):
        from training.recipes.dpo_loop import Config
        with pytest.raises(TypeError, match="log_path"):
            Config()

    def test_orpo_config_requires_log_path(self):
        from training.recipes.orpo_loop import Config
        with pytest.raises(TypeError, match="log_path"):
            Config()

    def test_sft_config_accepts_log_path(self):
        from training.recipes.sft_loop import Config
        cfg = Config(log_path="/tmp/test_sft")
        assert cfg.log_path == "/tmp/test_sft"

    def test_rl_config_accepts_log_path(self):
        from training.recipes.rl_loop import Config
        cfg = Config(log_path="/tmp/test_rl")
        assert cfg.log_path == "/tmp/test_rl"

    def test_save_checkpoint_creates_log_dir(self, log_dir):
        """save_checkpoint creates log_path if it doesn't exist."""
        nested = os.path.join(log_dir, "deep", "nested")
        assert not os.path.exists(nested)
        client = _make_mock_client()
        save_checkpoint(client, "step-1", nested, {"step": 1})
        assert os.path.exists(os.path.join(nested, CHECKPOINTS_BASE_NAME))

    def test_resolve_resume_with_empty_log_dir(self, log_dir):
        """resolve_resume returns None for a fresh log_path with no checkpoints."""
        client = _make_mock_client()
        result = resolve_resume(client, log_dir)
        assert result is None
        client.load_state_with_optimizer.assert_not_called()

    def test_save_then_resume_roundtrip(self, log_dir):
        """Full roundtrip: save a checkpoint, then resume from it."""
        client = _make_mock_client(job_id="job-roundtrip")
        save_checkpoint(client, "step-3", log_dir, {
            "step": 3,
            "data_consumed": 24,
            "source_job_id": "job-roundtrip",
        })

        ckpt_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
        assert os.path.exists(ckpt_path)
        with open(ckpt_path) as f:
            entry = json.loads(f.readline())
        assert entry["name"] == "step-3"
        assert entry["step"] == 3
        assert "state_path" in entry

        client2 = _make_mock_client(job_id="job-new")
        result = resolve_resume(client2, log_dir)
        assert result is not None
        assert result.step == 3
        assert result.data_consumed == 24
        client2.load_state_with_optimizer.assert_called_once()


class TestCheckpointMetadataForPromote:
    """Verify base_model/training_shape saved in checkpoints are readable
    by the promote_checkpoint script's _resolve_checkpoint_from_jsonl."""

    def test_metadata_roundtrip_latest(self, log_dir):
        """Save checkpoints with metadata, read back the latest entry."""
        client = _make_mock_client(job_id="job-meta")
        save_checkpoint(client, "step-1", log_dir, {
            "step": 1, "data_consumed": 8, "source_job_id": "job-meta",
        }, base_model="accounts/fw/models/qwen3-8b",
           training_shape="accounts/fw/trainingShapes/ts-qwen3-8b-policy")

        save_checkpoint(client, "step-2", log_dir, {
            "step": 2, "data_consumed": 16, "source_job_id": "job-meta",
        }, base_model="accounts/fw/models/qwen3-8b",
           training_shape="accounts/fw/trainingShapes/ts-qwen3-8b-policy")

        ckpt_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
        entries = []
        with open(ckpt_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        assert len(entries) == 2
        latest = entries[-1]
        assert latest["step"] == 2
        assert latest["base_model"] == "accounts/fw/models/qwen3-8b"
        assert latest["training_shape"] == "accounts/fw/trainingShapes/ts-qwen3-8b-policy"
        assert "state_path" in latest
        assert latest["state_path"].startswith("cross_job://")

    def test_metadata_absent_for_old_checkpoints(self, log_dir):
        """Checkpoints saved without metadata lack base_model/training_shape."""
        client = _make_mock_client(job_id="job-old")
        save_checkpoint(client, "step-1", log_dir, {
            "step": 1, "source_job_id": "job-old",
        })

        ckpt_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
        with open(ckpt_path) as f:
            entry = json.loads(f.readline())
        assert "base_model" not in entry
        assert "training_shape" not in entry

    def test_select_checkpoint_by_step(self, log_dir):
        """Verify the right entry is selected when filtering by step."""
        client = _make_mock_client(job_id="job-step")
        for step in (1, 2, 3):
            save_checkpoint(client, f"step-{step}", log_dir, {
                "step": step, "source_job_id": "job-step",
            }, base_model=f"model-v{step}",
               training_shape="shape-a")

        ckpt_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
        entries = []
        with open(ckpt_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        step2 = [e for e in entries if e.get("step") == 2]
        assert len(step2) == 1
        assert step2[0]["base_model"] == "model-v2"


# ---------------------------------------------------------------------------
# GCS-path tests -- verify checkpoint_utils works with gs:// URIs by mocking
# the low-level GCS transport in fileio.
# ---------------------------------------------------------------------------


def _make_gcs_blob(initial: bytes = b""):
    """Create a mock GCS blob that stores data in memory."""
    blob = MagicMock()
    blob._data = initial

    def _exists():
        return bool(blob._data)

    def _download():
        return blob._data

    def _upload(payload):
        blob._data = payload if isinstance(payload, bytes) else payload.encode()

    blob.exists.side_effect = _exists
    blob.download_as_bytes.side_effect = _download
    blob.upload_from_string.side_effect = _upload
    return blob


class TestGcsCheckpoints:
    """Verify save_checkpoint / get_last_checkpoint / resolve_resume work
    end-to-end when log_path is a ``gs://`` URI."""

    def test_save_and_get_last(self):
        blob = _make_gcs_blob()
        with patch("training.utils.fileio._gcs_bucket_blob", return_value=(None, blob)):
            gcs_log = "gs://test-bucket/jobs/job-1/logs"
            client = _make_mock_client(job_id="job-gcs")

            save_checkpoint(client, "step-5", gcs_log, {
                "step": 5, "data_consumed": 40,
            })

            last = get_last_checkpoint(gcs_log)
            assert last is not None
            assert last["name"] == "step-5"
            assert last["step"] == 5
            assert last["state_path"].startswith("cross_job://")

    def test_get_last_returns_none_on_empty(self):
        blob = _make_gcs_blob()
        with patch("training.utils.fileio._gcs_bucket_blob", return_value=(None, blob)):
            assert get_last_checkpoint("gs://bucket/empty") is None

    def test_save_appends_multiple_entries(self):
        blob = _make_gcs_blob()
        with patch("training.utils.fileio._gcs_bucket_blob", return_value=(None, blob)):
            gcs_log = "gs://test-bucket/multi"
            client = _make_mock_client(job_id="job-m")
            save_checkpoint(client, "step-1", gcs_log, {"step": 1})
            save_checkpoint(client, "step-2", gcs_log, {"step": 2})

            records = fileio.read_jsonl(fileio.join(gcs_log, CHECKPOINTS_BASE_NAME))
            assert len(records) == 2
            assert records[0]["step"] == 1
            assert records[1]["step"] == 2

    def test_resolve_resume_from_gcs(self):
        blob = _make_gcs_blob()
        with patch("training.utils.fileio._gcs_bucket_blob", return_value=(None, blob)):
            gcs_log = "gs://test-bucket/resume-test"
            client1 = _make_mock_client(job_id="job-save")
            save_checkpoint(client1, "step-3", gcs_log, {
                "step": 3, "data_consumed": 24,
            })

            client2 = _make_mock_client(job_id="job-resume")
            result = resolve_resume(client2, gcs_log)
            assert result is not None
            assert result.step == 3
            assert result.data_consumed == 24
            client2.load_state_with_optimizer.assert_called_once()
