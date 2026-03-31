"""Unit tests for training.utils.fileio -- transparent local / GCS I/O."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import training.utils.fileio as fileio


# ---------------------------------------------------------------------------
# Local path tests (no mock needed)
# ---------------------------------------------------------------------------


class TestLocalReadWrite:
    def test_read_bytes_missing(self, tmp_path):
        assert fileio.read_bytes(str(tmp_path / "nope")) == b""

    def test_write_and_read_bytes(self, tmp_path):
        p = str(tmp_path / "data.bin")
        fileio.write_bytes(p, b"hello")
        assert fileio.read_bytes(p) == b"hello"

    def test_write_bytes_creates_parent_dirs(self, tmp_path):
        p = str(tmp_path / "a" / "b" / "file.bin")
        fileio.write_bytes(p, b"nested")
        assert fileio.read_bytes(p) == b"nested"

    def test_write_bytes_atomic(self, tmp_path):
        p = str(tmp_path / "file.txt")
        fileio.write_bytes(p, b"first")
        fileio.write_bytes(p, b"second")
        assert not os.path.exists(p + ".tmp")
        assert fileio.read_bytes(p) == b"second"

    def test_read_text_missing(self, tmp_path):
        assert fileio.read_text(str(tmp_path / "nope")) == ""

    def test_read_text(self, tmp_path):
        p = str(tmp_path / "hello.txt")
        fileio.write_bytes(p, "café\n".encode())
        assert fileio.read_text(p) == "café\n"


class TestLocalAppend:
    def test_append_line(self, tmp_path):
        p = str(tmp_path / "log.txt")
        fileio.append_line(p, "line1\n")
        fileio.append_line(p, "line2\n")
        assert fileio.read_text(p) == "line1\nline2\n"

    def test_append_creates_parent_dirs(self, tmp_path):
        p = str(tmp_path / "sub" / "log.txt")
        fileio.append_line(p, "ok\n")
        assert fileio.read_text(p) == "ok\n"


class TestLocalJsonl:
    def test_roundtrip(self, tmp_path):
        p = str(tmp_path / "data.jsonl")
        fileio.append_jsonl(p, {"step": 1, "loss": 2.5})
        fileio.append_jsonl(p, {"step": 2, "loss": 1.8})

        records = fileio.read_jsonl(p)
        assert len(records) == 2
        assert records[0]["step"] == 1
        assert records[1]["loss"] == 1.8

    def test_read_jsonl_missing(self, tmp_path):
        assert fileio.read_jsonl(str(tmp_path / "nope.jsonl")) == []


class TestLocalWriteJson:
    def test_write_json(self, tmp_path):
        p = str(tmp_path / "out.json")
        fileio.write_json(p, {"status": "ok", "count": 42})
        data = json.loads(fileio.read_text(p))
        assert data == {"status": "ok", "count": 42}


class TestMakedirs:
    def test_local(self, tmp_path):
        d = str(tmp_path / "x" / "y")
        fileio.makedirs(d)
        assert os.path.isdir(d)

    def test_gcs_noop(self):
        fileio.makedirs("gs://bucket/prefix")


class TestJoin:
    def test_local(self):
        assert fileio.join("/a/b", "c", "d") == "/a/b/c/d"

    def test_gcs(self):
        assert fileio.join("gs://bucket/prefix", "sub", "file.txt") == "gs://bucket/prefix/sub/file.txt"

    def test_gcs_trailing_slash(self):
        assert fileio.join("gs://bucket/prefix/", "file.txt") == "gs://bucket/prefix/file.txt"


# ---------------------------------------------------------------------------
# GCS path tests (mocked google.cloud.storage)
# ---------------------------------------------------------------------------


def _make_mock_blob(data: bytes = b""):
    blob = MagicMock()
    blob.exists.return_value = bool(data)
    blob.download_as_bytes.return_value = data

    def capture_upload(payload):
        nonlocal data
        data = payload if isinstance(payload, bytes) else payload.encode()
        blob.download_as_bytes.return_value = data
        blob.exists.return_value = True

    blob.upload_from_string.side_effect = capture_upload
    return blob


class TestGcsReadWrite:
    def test_read_bytes_empty(self):
        blob = _make_mock_blob(b"")
        with patch.object(fileio, "_gcs_bucket_blob", return_value=(None, blob)):
            assert fileio.read_bytes("gs://bucket/key") == b""

    def test_write_and_read(self):
        blob = _make_mock_blob()
        with patch.object(fileio, "_gcs_bucket_blob", return_value=(None, blob)):
            fileio.write_bytes("gs://bucket/key", b"hello-gcs")
            blob.upload_from_string.assert_called_once_with(b"hello-gcs")

    def test_append_line(self):
        blob = _make_mock_blob(b"line1\n")
        with patch.object(fileio, "_gcs_bucket_blob", return_value=(None, blob)):
            fileio.append_line("gs://bucket/key", "line2\n")
            blob.upload_from_string.assert_called_once_with(b"line1\nline2\n")

    def test_append_jsonl(self):
        blob = _make_mock_blob(b'{"step":1}\n')
        with patch.object(fileio, "_gcs_bucket_blob", return_value=(None, blob)):
            fileio.append_jsonl("gs://bucket/key", {"step": 2})
            uploaded = blob.upload_from_string.call_args[0][0]
            lines = [l for l in uploaded.decode().strip().split("\n") if l]
            assert len(lines) == 2
            assert json.loads(lines[1])["step"] == 2

    def test_read_jsonl(self):
        content = b'{"a":1}\n{"a":2}\n'
        blob = _make_mock_blob(content)
        with patch.object(fileio, "_gcs_bucket_blob", return_value=(None, blob)):
            records = fileio.read_jsonl("gs://bucket/key")
            assert len(records) == 2
            assert records[0]["a"] == 1
            assert records[1]["a"] == 2
