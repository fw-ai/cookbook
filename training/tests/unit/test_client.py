from __future__ import annotations

from unittest.mock import MagicMock

from fireworks.training.sdk.client import GradAccNormalization
import pytest
import tinker

from training.utils.client import ReconnectableClient, _retry_on_not_found


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


def _make_client(inner: _FakeInnerClient) -> ReconnectableClient:
    client = object.__new__(ReconnectableClient)
    client._client = inner
    client._default_timeout = 123
    client._closed = False
    client._endpoint = None
    client._job_id = "job-test"
    client._service = None
    client._owns_service = True
    client._lora_rank = 0
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


# ---------------------------------------------------------------------------
# _retry_on_not_found tests
# ---------------------------------------------------------------------------


def _make_not_found_error(message: str = "Trainer job not found or not running"):
    """Create a ``tinker.NotFoundError`` with minimal mocking."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.json.return_value = {
        "error": {"message": message, "param": "jobId", "code": "NOT_FOUND"}
    }
    mock_response.headers = {}
    return tinker.NotFoundError(
        message=message,
        response=mock_response,
        body={"error": {"message": message}},
    )


def test_retry_on_not_found_succeeds_immediately():
    fn = MagicMock(return_value="ok")
    result = _retry_on_not_found(fn, timeout=60)
    assert result == "ok"
    assert fn.call_count == 1


def test_retry_on_not_found_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("training.utils.client.time.sleep", lambda _: None)
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _make_not_found_error()
        return "recovered"

    result = _retry_on_not_found(fn, timeout=60)
    assert result == "recovered"
    assert call_count == 3


def test_retry_on_not_found_exhausts_retries(monkeypatch):
    monkeypatch.setattr("training.utils.client.time.sleep", lambda _: None)
    monkeypatch.setattr("training.utils.client._NOT_FOUND_MAX_RETRIES", 2)

    fn = MagicMock(side_effect=_make_not_found_error())
    with pytest.raises(tinker.NotFoundError):
        _retry_on_not_found(fn, timeout=60)
    assert fn.call_count == 3  # initial + 2 retries


def test_retry_on_not_found_propagates_other_errors():
    fn = MagicMock(side_effect=ValueError("some other error"))
    with pytest.raises(ValueError, match="some other error"):
        _retry_on_not_found(fn, timeout=60)
    assert fn.call_count == 1


def test_forward_backward_retries_not_found(monkeypatch):
    """ReconnectableClient.forward_backward retries transient 404s."""
    monkeypatch.setattr("training.utils.client.time.sleep", lambda _: None)
    call_count = 0

    class _RetryFuture:
        def result(self, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise _make_not_found_error()
            return {"ok": True}

    class _RetryInnerClient:
        def __init__(self):
            self.holder = None

        def forward_backward(self, data, loss_fn, loss_fn_config=None):
            return _RetryFuture()

    inner = _RetryInnerClient()
    client = _make_client(inner)
    client._client = inner

    result = client.forward_backward(["datum"], "cross_entropy")
    assert result == {"ok": True}
    assert call_count == 2
