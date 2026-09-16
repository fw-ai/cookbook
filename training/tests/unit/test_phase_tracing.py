from __future__ import annotations

import asyncio
import gc
import json
import sys
import threading
import types
from pathlib import Path

import pytest

from training.utils import phase_tracing
from training.utils.phase_tracing import (
    configure_phase_tracing,
    flush_phase_trace,
    phase_span,
)


@pytest.fixture(autouse=True)
def reset_phase_tracing(monkeypatch):
    monkeypatch.delenv("COOKBOOK_TRACE_FILE", raising=False)
    monkeypatch.delenv("COOKBOOK_OTEL_ENABLED", raising=False)
    phase_tracing._reset_phase_tracing_for_tests()
    yield
    phase_tracing._reset_phase_tracing_for_tests()


def _duration_events(payload: dict) -> list[dict]:
    return [event for event in payload["traceEvents"] if event["ph"] == "X"]


def test_disabled_phase_span_is_a_noop() -> None:
    with phase_span("disabled") as span:
        assert span is None

    assert flush_phase_trace() is None


def test_trace_file_environment_lazily_enables_export(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "trace.json"
    monkeypatch.setenv("COOKBOOK_TRACE_FILE", str(trace_path))

    with phase_span("environment-enabled"):
        pass

    assert flush_phase_trace() == str(trace_path)
    assert _duration_events(json.loads(trace_path.read_text()))[0]["name"] == (
        "environment-enabled"
    )


def test_nested_spans_export_parent_relationship(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    recorder = configure_phase_tracing(trace_path)
    assert recorder is not None

    with phase_span("outer", attributes={"step": 3}) as outer:
        assert outer is not None
        with phase_span("inner", category="train") as inner:
            assert inner is not None
            inner.set_attribute("tokens", 128)

    assert flush_phase_trace() == str(trace_path)
    payload = json.loads(trace_path.read_text())
    events = {event["name"]: event for event in _duration_events(payload)}

    assert payload["schema_version"] == 1
    assert payload["displayTimeUnit"] == "ms"
    assert isinstance(payload["trace_origin_unix_ns"], int)
    assert payload["dropped_events"] == 0
    assert {
        "name",
        "cat",
        "ph",
        "ts",
        "dur",
        "pid",
        "tid",
        "args",
    } <= set(events["outer"])
    assert events["outer"]["args"]["step"] == 3
    assert events["inner"]["cat"] == "train"
    assert events["inner"]["args"]["tokens"] == 128
    assert (
        events["inner"]["args"]["parent_span_id"] == events["outer"]["args"]["span_id"]
    )
    assert events["outer"]["ts"] <= events["inner"]["ts"]


def test_exception_closes_span_and_records_error_type(tmp_path: Path) -> None:
    recorder = configure_phase_tracing(tmp_path / "trace.json")
    assert recorder is not None

    with pytest.raises(ValueError, match="bad phase"):
        with phase_span("failing"):
            raise ValueError("bad phase")

    event = _duration_events(recorder.payload())[0]
    assert event["name"] == "failing"
    assert event["args"]["error.type"] == "ValueError"


def test_async_tasks_get_distinct_perfetto_lanes(tmp_path: Path) -> None:
    recorder = configure_phase_tracing(tmp_path / "trace.json")
    assert recorder is not None

    async def run() -> None:
        release = asyncio.Event()
        entered = 0
        all_entered = asyncio.Event()

        async def worker(name: str) -> None:
            nonlocal entered
            with phase_span(name, category="rollout"):
                entered += 1
                if entered == 2:
                    all_entered.set()
                await release.wait()

        tasks = [
            asyncio.create_task(worker("rollout-a"), name="rollout-a"),
            asyncio.create_task(worker("rollout-b"), name="rollout-b"),
        ]
        await all_entered.wait()
        release.set()
        await asyncio.gather(*tasks)

    asyncio.run(run())
    events = _duration_events(recorder.payload())

    assert {event["name"] for event in events} == {"rollout-a", "rollout-b"}
    assert len({event["tid"] for event in events}) == 2


def test_sequential_async_tasks_do_not_reuse_stale_lanes(tmp_path: Path) -> None:
    recorder = configure_phase_tracing(tmp_path / "trace.json")
    assert recorder is not None

    async def run() -> None:
        async def worker(index: int) -> None:
            with phase_span(f"rollout-{index}"):
                pass

        for index in range(50):
            task = asyncio.create_task(worker(index), name=f"rollout-{index}")
            await task
            del task
            gc.collect()

    asyncio.run(run())
    assert len({event["tid"] for event in _duration_events(recorder.payload())}) == 50


def test_event_limit_also_bounds_async_lane_metadata(tmp_path: Path) -> None:
    recorder = configure_phase_tracing(
        tmp_path / "trace.json",
        max_events=2,
    )
    assert recorder is not None

    async def run() -> None:
        async def worker(index: int) -> None:
            with phase_span(f"rollout-{index}"):
                await asyncio.sleep(0)

        await asyncio.gather(*(worker(index) for index in range(10)))

    asyncio.run(run())
    payload = recorder.payload()
    metadata = [event for event in payload["traceEvents"] if event["ph"] == "M"]

    assert len(metadata) <= 2
    assert len(_duration_events(payload)) == 2
    assert payload["dropped_events"] == 8


def test_later_concurrent_flush_cannot_be_overwritten_by_stale_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "nested" / "trace.json"
    recorder = configure_phase_tracing(trace_path)
    assert recorder is not None
    with phase_span("before"):
        pass

    errors: list[BaseException] = []
    first_snapshot_ready = threading.Event()
    release_first_flush = threading.Event()
    second_flush_done = threading.Event()
    original_payload = recorder.payload
    payload_calls = 0
    payload_calls_lock = threading.Lock()

    def controlled_payload() -> dict:
        nonlocal payload_calls
        payload = original_payload()
        with payload_calls_lock:
            payload_calls += 1
            call_number = payload_calls
        if call_number == 1:
            first_snapshot_ready.set()
            assert release_first_flush.wait(timeout=2)
        return payload

    monkeypatch.setattr(recorder, "payload", controlled_payload)

    def flush(done: threading.Event | None = None) -> None:
        try:
            recorder.flush()
        except BaseException as error:
            errors.append(error)
        finally:
            if done is not None:
                done.set()

    first = threading.Thread(target=flush)
    first.start()
    assert first_snapshot_ready.wait(timeout=2)

    with phase_span("after"):
        pass
    second = threading.Thread(target=flush, args=(second_flush_done,))
    second.start()

    # The second flush must wait behind the first instead of writing a newer
    # snapshot that the first flush can subsequently overwrite.
    assert not second_flush_done.wait(timeout=0.05)
    release_first_flush.set()
    first.join()
    second.join()

    assert not errors
    assert {
        event["name"] for event in _duration_events(json.loads(trace_path.read_text()))
    } == {
        "before",
        "after",
    }
    assert list(trace_path.parent.glob(f".{trace_path.name}.*.tmp")) == []


def test_broken_default_otel_provider_does_not_break_perfetto(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    fake_otel = types.ModuleType("opentelemetry")

    class BrokenTrace:
        @staticmethod
        def get_tracer(_name):
            raise RuntimeError("provider broken")

    fake_otel.trace = BrokenTrace()
    monkeypatch.setitem(sys.modules, "opentelemetry", fake_otel)
    monkeypatch.setenv("COOKBOOK_OTEL_ENABLED", "1")

    recorder = configure_phase_tracing(tmp_path / "trace.json")
    assert recorder is not None
    with phase_span("still-recorded"):
        pass

    assert _duration_events(recorder.payload())[0]["name"] == "still-recorded"
    assert "OpenTelemetry tracer initialization failed" in caplog.text


def test_optional_otel_tracer_mirrors_span_without_dependency(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []

    class FakeOtelSpan:
        def set_attribute(self, name, value):
            calls.append(("attribute", name, value))

    class FakeSpanManager:
        def __init__(self):
            self.span = FakeOtelSpan()

        def __enter__(self):
            calls.append(("enter",))
            return self.span

        def __exit__(self, exc_type, exc, traceback):
            calls.append(("exit", exc_type))
            return False

    class FakeTracer:
        def start_as_current_span(self, name, *, attributes):
            calls.append(("start", name, attributes))
            return FakeSpanManager()

    configure_phase_tracing(
        tmp_path / "trace.json",
        otel_tracer=FakeTracer(),
    )
    with phase_span("harness", attributes={"step": 5}) as span:
        assert span is not None
        span.set_attribute("tool", "pytest")

    assert calls[0] == ("start", "harness", {"step": 5})
    assert ("attribute", "tool", "pytest") in calls
    assert calls[-1] == ("exit", None)
