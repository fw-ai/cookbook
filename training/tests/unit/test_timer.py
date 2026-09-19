from __future__ import annotations

import time

from training.utils import phase_tracing
from training.utils.phase_tracing import configure_phase_tracing
from training.utils.timer import Timer, elapsed_timer, flush_timing, wall_timer


def test_wall_timer_measures_without_recording_step_metric(monkeypatch) -> None:
    timer = Timer()
    timer.reset()
    timestamps = iter((10.0, 12.5))
    monkeypatch.setattr(time, "perf_counter", lambda: next(timestamps))

    with wall_timer() as span:
        assert span.elapsed == 0.0

    assert span.elapsed == 2.5
    assert timer.log_dict() == {}


def test_elapsed_timer_preserves_metric_and_emits_phase_span(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("COOKBOOK_TRACE_FILE", raising=False)
    phase_tracing._reset_phase_tracing_for_tests()
    configure_phase_tracing(tmp_path / "trace.json")

    with elapsed_timer("fwd_bwd"):
        pass

    metrics = flush_timing()
    recorder = phase_tracing.get_phase_trace_recorder()
    assert recorder is not None
    events = [
        event for event in recorder.payload()["traceEvents"] if event["ph"] == "X"
    ]

    assert metrics.keys() == {"perf/fwd_bwd_time"}
    assert metrics["perf/fwd_bwd_time"] >= 0
    assert [event["name"] for event in events] == ["fwd_bwd"]

    phase_tracing._reset_phase_tracing_for_tests()


def test_named_wall_timer_emits_span_without_step_metric(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("COOKBOOK_TRACE_FILE", raising=False)
    phase_tracing._reset_phase_tracing_for_tests()
    recorder = configure_phase_tracing(tmp_path / "trace.json")
    assert recorder is not None

    with wall_timer(
        "evaluation",
        category="evaluation",
        attributes={"step": 2},
    ):
        pass

    events = [
        event for event in recorder.payload()["traceEvents"] if event["ph"] == "X"
    ]
    assert flush_timing() == {}
    assert events[0]["name"] == "evaluation"
    assert events[0]["cat"] == "evaluation"
    assert events[0]["args"]["step"] == 2

    phase_tracing._reset_phase_tracing_for_tests()
