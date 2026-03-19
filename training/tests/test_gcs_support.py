"""Tests for GCS (gs://) path support in data.py and checkpoint_utils.py."""

from __future__ import annotations

import io
import json
import os
import tempfile
from typing import Any
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PREFERENCE_ROWS = [
    {"chosen": {"text": "good"}, "rejected": {"text": "bad"}},
    {"chosen": {"text": "yes"}, "rejected": {"text": "no"}},
]

JSONL_ROWS = [
    {"prompt": "hello", "completion": "world"},
    {"prompt": "foo", "completion": "bar"},
]


def _jsonl_bytes(rows: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r) for r in rows)


@pytest.fixture()
def local_jsonl(tmp_path: Any) -> str:
    p = tmp_path / "data.jsonl"
    p.write_text(_jsonl_bytes(JSONL_ROWS))
    return str(p)


@pytest.fixture()
def local_preference(tmp_path: Any) -> str:
    p = tmp_path / "prefs.jsonl"
    p.write_text(_jsonl_bytes(PREFERENCE_ROWS))
    return str(p)


# ---------------------------------------------------------------------------
# data.py – load_jsonl_dataset
# ---------------------------------------------------------------------------


class TestLoadJsonlDataset:
    def test_local_file(self, local_jsonl: str) -> None:
        from training.utils.data import load_jsonl_dataset

        rows = load_jsonl_dataset(local_jsonl)
        assert len(rows) == 2
        assert rows[0]["prompt"] == "hello"

    def test_local_with_max_rows(self, local_jsonl: str) -> None:
        from training.utils.data import load_jsonl_dataset

        rows = load_jsonl_dataset(local_jsonl, max_rows=1)
        assert len(rows) == 1

    def test_gs_path(self) -> None:
        from training.utils.data import load_jsonl_dataset

        content = _jsonl_bytes(JSONL_ROWS)
        fake_fh = io.StringIO(content)

        mock_open_cm = mock.MagicMock()
        mock_open_cm.__enter__ = mock.Mock(return_value=fake_fh)
        mock_open_cm.__exit__ = mock.Mock(return_value=False)

        with mock.patch("training.utils.data.fsspec") as mock_fsspec, \
             mock.patch("training.utils.data.FSSPEC_AVAILABLE", True):
            mock_fsspec.open.return_value = mock_open_cm
            rows = load_jsonl_dataset("gs://bucket/data.jsonl")

        assert len(rows) == 2
        mock_fsspec.open.assert_called_once_with("gs://bucket/data.jsonl", "r")

    def test_gs_path_without_fsspec_raises(self) -> None:
        from training.utils.data import load_jsonl_dataset

        with mock.patch("training.utils.data.FSSPEC_AVAILABLE", False):
            with pytest.raises(ImportError, match="fsspec is required"):
                load_jsonl_dataset("gs://bucket/data.jsonl")


# ---------------------------------------------------------------------------
# data.py – load_preference_dataset
# ---------------------------------------------------------------------------


class TestLoadPreferenceDataset:
    def test_local_file(self, local_preference: str) -> None:
        from training.utils.data import load_preference_dataset

        data = load_preference_dataset(local_preference)
        assert len(data) == 2
        assert data[0]["chosen"]["text"] == "good"

    def test_local_with_max_pairs(self, local_preference: str) -> None:
        from training.utils.data import load_preference_dataset

        data = load_preference_dataset(local_preference, max_pairs=1)
        assert len(data) == 1

    def test_gs_path(self) -> None:
        from training.utils.data import load_preference_dataset

        content = _jsonl_bytes(PREFERENCE_ROWS)
        fake_fh = io.StringIO(content)

        mock_open_cm = mock.MagicMock()
        mock_open_cm.__enter__ = mock.Mock(return_value=fake_fh)
        mock_open_cm.__exit__ = mock.Mock(return_value=False)

        with mock.patch("training.utils.data.fsspec") as mock_fsspec, \
             mock.patch("training.utils.data.FSSPEC_AVAILABLE", True):
            mock_fsspec.open.return_value = mock_open_cm
            data = load_preference_dataset("gs://bucket/prefs.jsonl")

        assert len(data) == 2
        mock_fsspec.open.assert_called_once_with("gs://bucket/prefs.jsonl", "r")

    def test_gs_path_without_fsspec_raises(self) -> None:
        from training.utils.data import load_preference_dataset

        with mock.patch("training.utils.data.FSSPEC_AVAILABLE", False):
            with pytest.raises(ImportError, match="fsspec is required"):
                load_preference_dataset("gs://bucket/prefs.jsonl")


# ---------------------------------------------------------------------------
# checkpoint_utils.py – save_checkpoint
# ---------------------------------------------------------------------------


def _make_mock_client(job_id: str = "job-1") -> mock.MagicMock:
    client = mock.MagicMock()
    client.job_id = job_id
    client.resolve_checkpoint_path.return_value = "/resolved/path"
    save_result = mock.MagicMock()
    save_result.snapshot_name = "snap-1"
    client.save_weights_for_sampler_ext.return_value = save_result
    return client


class TestSaveCheckpoint:
    def test_local_path(self, tmp_path: Any) -> None:
        from training.utils.checkpoint_utils import save_checkpoint, CheckpointKind

        log_dir = str(tmp_path / "logs")
        client = _make_mock_client()
        loop_state = {"step": 5, "data_consumed": 100}

        save_checkpoint(client, "ckpt-5", log_dir, loop_state, kind=CheckpointKind.STATE)

        ckpt_file = os.path.join(log_dir, "checkpoints.jsonl")
        assert os.path.exists(ckpt_file)
        with open(ckpt_file) as f:
            record = json.loads(f.readline())
        assert record["name"] == "ckpt-5"
        assert record["step"] == 5
        assert "state_path" in record

    def test_gs_path(self) -> None:
        from training.utils.checkpoint_utils import save_checkpoint, CheckpointKind

        buf = io.StringIO()
        mock_open_cm = mock.MagicMock()
        mock_open_cm.__enter__ = mock.Mock(return_value=buf)
        mock_open_cm.__exit__ = mock.Mock(return_value=False)

        client = _make_mock_client()
        loop_state = {"step": 10}

        with mock.patch("training.utils.checkpoint_utils.fsspec") as mock_fsspec, \
             mock.patch("training.utils.checkpoint_utils.FSSPEC_AVAILABLE", True):
            mock_fsspec.open.return_value = mock_open_cm
            save_checkpoint(
                client, "ckpt-10", "gs://bucket/logs", loop_state, kind=CheckpointKind.STATE,
            )

        mock_fsspec.open.assert_called_once_with(
            "gs://bucket/logs/checkpoints.jsonl", "a",
        )
        written = buf.getvalue()
        record = json.loads(written.strip())
        assert record["name"] == "ckpt-10"
        assert record["step"] == 10

    def test_gs_path_without_fsspec_raises(self) -> None:
        from training.utils.checkpoint_utils import save_checkpoint, CheckpointKind

        client = _make_mock_client()
        with mock.patch("training.utils.checkpoint_utils.FSSPEC_AVAILABLE", False):
            with pytest.raises(ImportError, match="fsspec is required"):
                save_checkpoint(
                    client, "ckpt-1", "gs://bucket/logs", {"step": 1},
                    kind=CheckpointKind.STATE,
                )

    def test_local_trailing_slash(self, tmp_path: Any) -> None:
        from training.utils.checkpoint_utils import save_checkpoint, CheckpointKind

        log_dir = str(tmp_path / "logs") + "/"
        client = _make_mock_client()
        save_checkpoint(client, "ckpt-1", log_dir, {"step": 1}, kind=CheckpointKind.STATE)

        ckpt_file = os.path.join(log_dir, "checkpoints.jsonl")
        assert os.path.exists(ckpt_file)
