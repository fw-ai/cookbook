"""Public data downloads must not need credentials or replace valid data on failure."""

import builtins
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training.examples.decision.kalshi_v1 import prepare_data


@pytest.fixture
def dataset(monkeypatch, tmp_path):
    body = b'{"state":"market","questions":{}}\n'
    monkeypatch.setattr(prepare_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare_data, "SPLITS", {
        "train": {
            "dataset_id": "kalshi-decision-v1-train",
            "rows": 1,
            "sha256": hashlib.sha256(body).hexdigest(),
        },
    })
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_content.return_value = [body]
    get = Mock(return_value=response)
    monkeypatch.setattr(prepare_data.requests, "get", get)
    return SimpleNamespace(body=body, response=response, get=get, path=tmp_path / "train.jsonl")


def test_default_download_needs_no_fireworks_sdk_or_key(monkeypatch, dataset):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    original_import = builtins.__import__

    def without_fireworks(name, *args, **kwargs):
        if name == "fireworks" or name.startswith("fireworks."):
            raise AssertionError("Public data download imported the Fireworks SDK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_fireworks)
    assert prepare_data.main(["--splits", "train"]) == 0
    assert dataset.path.read_bytes() == dataset.body
    dataset.get.assert_called_once_with(
        f"https://huggingface.co/datasets/{prepare_data.HF_REPO_ID}/resolve/"
        f"{prepare_data.HF_REVISION}/train.jsonl",
        stream=True, timeout=120,
    )


def test_matching_cache_skips_network(dataset):
    dataset.path.write_bytes(dataset.body)
    assert prepare_data.download_split("train", force=False) == dataset.path
    dataset.get.assert_not_called()


def test_bad_download_preserves_good_file_and_removes_partial(dataset):
    dataset.path.write_bytes(dataset.body)
    dataset.response.iter_content.return_value = [b"corrupted download\n"]
    with pytest.raises(RuntimeError, match="sha256"):
        prepare_data.download_split("train", force=True)
    assert dataset.path.read_bytes() == dataset.body
    assert not dataset.path.with_suffix(".jsonl.partial").exists()


def test_row_count_mismatch_rejects_download(dataset, monkeypatch):
    monkeypatch.setitem(prepare_data.SPLITS["train"], "rows", 2)
    with pytest.raises(RuntimeError, match="1 rows != pinned 2"):
        prepare_data.download_split("train", force=True)
    assert not dataset.path.exists()
    assert not dataset.path.with_suffix(".jsonl.partial").exists()


def test_force_download_refreshes_valid_cache(dataset):
    dataset.path.write_bytes(dataset.body)
    prepare_data.download_split("train", force=True)
    dataset.get.assert_called_once()
    assert dataset.path.read_bytes() == dataset.body


def test_unknown_split_fails_before_auth_or_network(dataset):
    assert prepare_data.main(["--splits", "typo"]) == 2
    dataset.get.assert_not_called()


def test_http_failure_cleans_partial_and_preserves_existing(dataset):
    dataset.path.write_bytes(dataset.body)
    dataset.path.with_suffix(".jsonl.partial").write_bytes(b"old interrupted download")
    dataset.response.raise_for_status.side_effect = prepare_data.requests.HTTPError("HTTP 503")
    with pytest.raises(prepare_data.requests.HTTPError):
        prepare_data.download_split("train", force=True)
    assert dataset.path.read_bytes() == dataset.body
    assert not dataset.path.with_suffix(".jsonl.partial").exists()
