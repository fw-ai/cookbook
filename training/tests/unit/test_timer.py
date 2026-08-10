from __future__ import annotations

import time

from training.utils.timer import Timer, wall_timer


def test_wall_timer_measures_without_recording_step_metric(monkeypatch) -> None:
    timer = Timer()
    timer.reset()
    timestamps = iter((10.0, 12.5))
    monkeypatch.setattr(time, "perf_counter", lambda: next(timestamps))

    with wall_timer() as span:
        assert span.elapsed == 0.0

    assert span.elapsed == 2.5
    assert timer.log_dict() == {}
