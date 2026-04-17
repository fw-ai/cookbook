from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import tinker
from fireworks.training.sdk.client import GradAccNormalization

import training.utils.client as client_mod
from training.utils.client import ReconnectableClient, _retry_on_transient_not_found


def _make_not_found(message: str = "Trainer job not found or not running") -> tinker.NotFoundError:
    """Build a real ``tinker.NotFoundError`` instance for tests."""
    response = MagicMock()
    response.status_code = 404
    response.headers = {}
    return tinker.NotFoundError(message=message, response=response, body=None)


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

    def optim_step(self, params, **kwargs):
        self.calls.append((params, kwargs))
        return self.future


def _make_client(
    inner: _FakeInnerClient,
    *,
    rlor_mgr: object | None = None,
) -> ReconnectableClient:
    client = object.__new__(ReconnectableClient)
    client._client = inner
    client._default_timeout = 123
    client._closed = False
    client._endpoint = None
    client._job_id = "job-test"
    client._rlor_mgr = rlor_mgr
    return client


class _FakeRlorMgr:
    def __init__(self, states):
        self.states = list(states)
        self.calls = 0

    def get(self, job_id):
        self.calls += 1
        if not self.states:
            return {"state": "JOB_STATE_RUNNING"}
        state = self.states.pop(0)
        return {"state": state}


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


# -- Verified-retry behaviour -------------------------------------------------


def test_retry_succeeds_immediately_when_no_404(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda _: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    result = _retry_on_transient_not_found(
        fn, is_running=lambda: True, job_id="job-test"
    )

    assert result == "ok"
    assert calls["n"] == 1


def test_retry_recovers_after_transient_404_when_still_running(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda _: None)
    n = {"v": 0}

    def fn():
        n["v"] += 1
        if n["v"] < 3:
            raise _make_not_found()
        return {"ok": True}

    result = _retry_on_transient_not_found(
        fn, is_running=lambda: True, job_id="job-test"
    )

    assert result == {"ok": True}
    assert n["v"] == 3


def test_retry_fails_fast_when_job_no_longer_running(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda _: None)
    n = {"v": 0}

    def fn():
        n["v"] += 1
        raise _make_not_found()

    with pytest.raises(tinker.NotFoundError):
        _retry_on_transient_not_found(
            fn, is_running=lambda: False, job_id="job-test"
        )

    assert n["v"] == 1


def test_retry_fails_fast_when_control_plane_probe_errors(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda _: None)
    n = {"v": 0}

    def fn():
        n["v"] += 1
        raise _make_not_found()

    def boom():
        raise RuntimeError("control plane down")

    with pytest.raises(tinker.NotFoundError):
        _retry_on_transient_not_found(
            fn, is_running=boom, job_id="job-test"
        )

    assert n["v"] == 1


def test_retry_exhausts_when_404_persists(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda _: None)
    n = {"v": 0}

    def fn():
        n["v"] += 1
        raise _make_not_found()

    with pytest.raises(tinker.NotFoundError):
        _retry_on_transient_not_found(
            fn, is_running=lambda: True, job_id="job-test"
        )

    assert n["v"] == client_mod._NOT_FOUND_MAX_RETRIES + 1


def test_retry_propagates_non_404_immediately(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda _: None)
    n = {"v": 0}

    def fn():
        n["v"] += 1
        raise RuntimeError("not a 404")

    with pytest.raises(RuntimeError, match="not a 404"):
        _retry_on_transient_not_found(
            fn, is_running=lambda: True, job_id="job-test"
        )

    assert n["v"] == 1


def test_forward_backward_retries_only_when_running(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda _: None)

    class _Inner:
        def __init__(self):
            self.calls = 0
            self.holder = None

        def forward_backward(self, data, loss_fn, loss_fn_config=None):
            self.calls += 1

            class _F:
                def __init__(self, calls):
                    self._calls = calls

                def result(self, timeout=None):
                    if self._calls < 2:
                        raise _make_not_found()
                    return {"ok": True}

            return _F(self.calls)

    inner = _Inner()
    rlor_mgr = _FakeRlorMgr(["JOB_STATE_RUNNING"])
    client = _make_client(inner, rlor_mgr=rlor_mgr)

    result = client.forward_backward([{"x": 1}], "cross_entropy")

    assert result == {"ok": True}
    assert inner.calls == 2
    assert rlor_mgr.calls == 1


def test_forward_backward_does_not_retry_when_job_deleted(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda _: None)

    class _Inner:
        def __init__(self):
            self.calls = 0
            self.holder = None

        def forward_backward(self, data, loss_fn, loss_fn_config=None):
            self.calls += 1

            class _F:
                def result(self, timeout=None):
                    raise _make_not_found()

            return _F()

    inner = _Inner()
    rlor_mgr = _FakeRlorMgr(["JOB_STATE_FAILED"])
    client = _make_client(inner, rlor_mgr=rlor_mgr)

    with pytest.raises(tinker.NotFoundError):
        client.forward_backward([{"x": 1}], "cross_entropy")

    assert inner.calls == 1
    assert rlor_mgr.calls == 1


# -- Data-plane warmup --------------------------------------------------------


def test_warmup_returns_after_required_consecutive_successes(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda _: None)

    class _Inner:
        def __init__(self):
            self.holder = None
            self.calls = 0

        def get_info(self):
            self.calls += 1
            return {"ok": True}

    inner = _Inner()
    client = _make_client(inner)

    client._wait_for_data_plane_ready(
        timeout_s=10.0, required_successes=3,
    )

    assert inner.calls == 3


def test_warmup_recovers_after_transient_404s(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda _: None)

    class _Inner:
        def __init__(self):
            self.holder = None
            self.calls = 0

        def get_info(self):
            self.calls += 1
            if self.calls <= 2:
                raise _make_not_found()
            return {"ok": True}

    inner = _Inner()
    client = _make_client(inner)

    client._wait_for_data_plane_ready(
        timeout_s=10.0, required_successes=2,
    )

    assert inner.calls == 4


def test_warmup_raises_on_persistent_404_after_timeout(monkeypatch):
    fake_now = {"t": 0.0}

    def fake_monotonic():
        return fake_now["t"]

    def fake_sleep(s):
        fake_now["t"] += s

    monkeypatch.setattr(client_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(client_mod.time, "sleep", fake_sleep)

    class _Inner:
        def __init__(self):
            self.holder = None
            self.calls = 0

        def get_info(self):
            self.calls += 1
            raise _make_not_found()

    inner = _Inner()
    client = _make_client(inner)

    with pytest.raises(TimeoutError, match="gateway route did not stabilise"):
        client._wait_for_data_plane_ready(
            timeout_s=5.0, required_successes=2,
        )

    assert inner.calls >= 1


def test_warmup_skips_on_non_404_error(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda _: None)

    class _Inner:
        def __init__(self):
            self.holder = None
            self.calls = 0

        def get_info(self):
            self.calls += 1
            raise RuntimeError("real failure")

    inner = _Inner()
    client = _make_client(inner)

    client._wait_for_data_plane_ready(
        timeout_s=10.0, required_successes=2,
    )

    assert inner.calls == 1
