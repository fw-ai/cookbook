"""Offline fixture absence is explicit; real online Hub failures remain failures."""

import pytest
from huggingface_hub import constants
from huggingface_hub.errors import LocalEntryNotFoundError
import huggingface_hub

from training.tests._v4_tokenizer_data import v4_files


@pytest.mark.parametrize("offline", [False, True])
def test_missing_pinned_fixture_skips_only_when_explicitly_offline(monkeypatch, offline):
    def missing(*args, **kwargs):
        raise LocalEntryNotFoundError("synthetic missing cache")

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    expected = pytest.skip.Exception if offline else LocalEntryNotFoundError
    with pytest.raises(expected):
        v4_files.__wrapped__()
