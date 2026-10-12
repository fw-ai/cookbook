"""Unit tests for ``RawRowCursor`` (clamping, resume ambiguity, recording)."""

from __future__ import annotations

from training.tests._dataloader_test_helpers import load_cursor

RawRowCursor = load_cursor()


# ---------------------------------------------------------------------------
# RawRowCursor: resume() — clamping, ambiguity, fallback
# ---------------------------------------------------------------------------


def test_resume_with_persisted_value_sets_cursor():
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(5)
    assert cursor.value == 5


def test_resume_clamps_persisted_past_max():
    cursor = RawRowCursor(max_rows=3)
    cursor.resume(99)
    assert cursor.value == 3


def test_resume_clamps_negative_persisted_to_zero():
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(-5)
    assert cursor.value == 0


def test_resume_with_none_uses_fallback():
    """``persisted=None`` (no dataloader.json) → cursor takes ``fallback``."""
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(None, fallback=4)
    assert cursor.value == 4


def test_resume_with_zero_persisted_and_zero_fallback_trusts_zero():
    """Fresh start (both 0) is trusted, not routed through fallback."""
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(0, fallback=0)
    assert cursor.value == 0


def test_resume_with_zero_persisted_and_nonzero_fallback_takes_fallback():
    """Legacy ckpt signal: ``persisted=0`` with non-zero fallback → step-derived path."""
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(0, fallback=4)
    assert cursor.value == 4


def test_resume_prefers_persisted_over_fallback_when_both_nonzero():
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(7, fallback=4)
    assert cursor.value == 7


def test_resume_clamps_fallback_past_max():
    cursor = RawRowCursor(max_rows=3)
    cursor.resume(None, fallback=10)
    assert cursor.value == 3


def test_resume_without_max_rows_does_not_clamp():
    """Multi-epoch SFT/DPO cursors run unbounded."""
    cursor = RawRowCursor()  # max_rows=None
    cursor.resume(10_000)
    assert cursor.value == 10_000


# ---------------------------------------------------------------------------
# RawRowCursor: record() — accumulation, clamping, defensive paths
# ---------------------------------------------------------------------------


def test_record_advances_cursor():
    cursor = RawRowCursor(max_rows=10)
    cursor.record(3)
    cursor.record(5)
    assert cursor.value == 8


def test_record_clamps_at_max_rows():
    cursor = RawRowCursor(max_rows=3)
    cursor.record(99)
    assert cursor.value == 3


def test_record_zero_is_noop():
    cursor = RawRowCursor(max_rows=10)
    cursor.record(0)
    assert cursor.value == 0


def test_record_negative_does_not_rewind():
    cursor = RawRowCursor(max_rows=10)
    cursor.record(5)
    cursor.record(-3)
    assert cursor.value == 5


def test_record_after_resume_accumulates():
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(2)
    cursor.record(3)
    assert cursor.value == 5


def test_record_without_max_rows_grows_unbounded():
    cursor = RawRowCursor()
    cursor.record(1_000_000)
    assert cursor.value == 1_000_000
