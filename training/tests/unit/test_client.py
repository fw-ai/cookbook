from __future__ import annotations

from fireworks.training.sdk.client import GradAccNormalization
import pytest

from training.utils.client import ReconnectableClient


class _FakeFuture:
    def __init__(self):
        self.timeout = None

    def result(self, timeout=None):
        self.timeout = timeout
        return {"ok": True, "timeout": timeout}


class _FakeInnerClient:
    def __init__(self):
        self.calls = []
        self.future = _FakeFuture()

    def optim_step(self, params, **kwargs):
        self.calls.append((params, kwargs))
        return self.future


def _make_client(inner: _FakeInnerClient) -> ReconnectableClient:
    client = object.__new__(ReconnectableClient)
    client._client = inner
    client._default_timeout = 123
    return client


def test_optim_step_converts_legacy_string_normalization():
    inner = _FakeInnerClient()
    client = _make_client(inner)

    result = client.optim_step("adam", grad_accumulation_normalization="num_loss_tokens")

    assert result == {"ok": True, "timeout": 123}
    assert inner.calls == [
        (
            "adam",
            {"grad_accumulation_normalization": GradAccNormalization.NUM_LOSS_TOKENS},
        )
    ]


def test_optim_step_accepts_enum_normalization():
    inner = _FakeInnerClient()
    client = _make_client(inner)

    client.optim_step(
        "adam",
        grad_accumulation_normalization=GradAccNormalization.NUM_SEQUENCES,
    )

    assert inner.calls == [
        (
            "adam",
            {"grad_accumulation_normalization": GradAccNormalization.NUM_SEQUENCES},
        )
    ]


def test_optim_step_rejects_unknown_normalization():
    inner = _FakeInnerClient()
    client = _make_client(inner)

    with pytest.raises(ValueError, match="Unknown grad_accumulation_normalization"):
        client.optim_step("adam", grad_accumulation_normalization="not-a-real-mode")
