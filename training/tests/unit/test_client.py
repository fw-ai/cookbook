from __future__ import annotations

import threading

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
        self.holder = None
        self.sampler_calls = []

    def optim_step(self, params, **kwargs):
        self.calls.append((params, kwargs))
        return self.future

    def save_weights_for_sampler_ext(self, name, checkpoint_type=None):
        self.sampler_calls.append((name, checkpoint_type))
        return {"name": name, "checkpoint_type": checkpoint_type}


class _BlockingSamplerInner(_FakeInnerClient):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def save_weights_for_sampler_ext(self, name, checkpoint_type=None):
        self.sampler_calls.append((name, checkpoint_type))
        self.started.set()
        self.release.wait()
        return {"name": name, "checkpoint_type": checkpoint_type}


def _make_client(inner: _FakeInnerClient) -> ReconnectableClient:
    client = object.__new__(ReconnectableClient)
    client._client = inner
    client._default_timeout = 123
    client._closed = False
    client._endpoint = None
    client._job_id = "job-test"
    return client


class _FakeCleanupFuture:
    def __init__(self):
        self.timeout = None

    def result(self, timeout=None):
        self.timeout = timeout
        return None


class _FakeTelemetry:
    def __init__(self):
        self.flushed = 0
        self.drained = 0
        self.drain_result = True
        self.stopped = 0

    def _trigger_flush(self):
        self.flushed += 1

    def _wait_until_drained_sync(self):
        self.drained += 1
        return self.drain_result

    def stop(self):
        self.stopped += 1


class _FakeHolder:
    def __init__(self):
        self.future = _FakeCleanupFuture()
        self._telemetry = _FakeTelemetry()

    async def _async_cleanup(self):
        return None

    def get_telemetry(self):
        return self._telemetry

    def run_coroutine_threadsafe(self, coro):
        try:
            coro.close()
        except Exception:
            pass
        return self.future


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


def test_save_weights_for_sampler_ext_returns_inner_result():
    inner = _FakeInnerClient()
    client = _make_client(inner)

    result = client.save_weights_for_sampler_ext(
        "step-2",
        checkpoint_type="base",
        timeout=5,
    )

    assert result == {"name": "step-2", "checkpoint_type": "base"}
    assert inner.sampler_calls == [("step-2", "base")]


def test_save_weights_for_sampler_ext_honors_timeout():
    inner = _BlockingSamplerInner()
    client = _make_client(inner)

    with pytest.raises(TimeoutError, match="did not complete within 0.01s"):
        client.save_weights_for_sampler_ext(
            "step-3",
            checkpoint_type="base",
            timeout=0.01,
        )

    assert inner.started.wait(0.2)
    inner.release.set()


def test_close_drains_telemetry_and_stops_holder_cleanup():
    inner = _FakeInnerClient()
    inner.holder = _FakeHolder()
    client = _make_client(inner)

    client.close(timeout=7.5)

    assert client._closed is True
    assert client._client is None
    assert inner.holder._telemetry.flushed == 1
    assert inner.holder._telemetry.drained == 1
    assert inner.holder.future.timeout == 7.5
    assert inner.holder._telemetry.stopped == 1

    client.close(timeout=1.0)
    assert inner.holder._telemetry.flushed == 1
    assert inner.holder._telemetry.drained == 1
    assert inner.holder._telemetry.stopped == 1
