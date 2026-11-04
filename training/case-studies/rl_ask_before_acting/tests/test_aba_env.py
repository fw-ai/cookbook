"""Offline contract tests for the Ask Before Acting environment (no key, no network)."""

from pathlib import Path as _TestPath
import sys as _test_sys
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1]))

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aba_env import generate_tasks, score_episode  # noqa: E402

TASK = {"action": "book_meeting",
        "spec": {"title": "roadmap planning", "day": "Friday", "start_time": "09:00", "duration_minutes": 30}}


def call(**args):
    return {"name": "book_meeting", "arguments": {"title": "roadmap planning", "day": "Friday",
                                                  "start_time": "09:00", "duration_minutes": 30, **args}}


def test_equivalent_spellings_match():
    assert score_episode(TASK, call(day="on Friday", start_time="9:00 AM", duration_minutes="30 minutes"), 0, 1)["right_args"]


def test_wrong_value_containing_the_right_number_does_not_match():
    assert not score_episode(TASK, call(start_time="not 09:00; use 15:00"), 0, 1)["right_args"]
    assert not score_episode(TASK, call(duration_minutes="30 hours"), 0, 1)["right_args"]
    assert not score_episode(TASK, call(start_time="15:00"), 0, 1)["right_args"]


def test_vague_time_phrase_agrees_with_hidden_hour():
    for t in generate_tasks(600, 0):
        f = "start_time" if "start_time" in t["spec"] else "time" if "time" in t["spec"] else None
        if f and f in t["vague"]:
            hour = int(t["spec"][f].split(":")[0])
            req = t["request"]
            if hour < 12:
                assert "afternoon" not in req and "lunch" not in req, req
            else:
                assert "morning" not in req and "first thing" not in req, req
