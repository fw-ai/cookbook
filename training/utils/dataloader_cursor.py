"""Cumulative raw-row cursor shared across SFT/DPO/RL/IGPO recipes."""

from __future__ import annotations


class RawRowCursor:
    """Cumulative raw source-row cursor shared across training runs.

    This is the same accounting unit as ``ResumeInfo.row_cursor``: raw rows
    pulled from the source dataset, including drops and sample failures.
    Set ``max_rows`` to clamp (e.g. RL's pre-multiplied row list); leave
    ``None`` for cursors that grow across epochs (SFT/DPO multi-epoch).
    """

    def __init__(self, *, max_rows: int | None = None) -> None:
        self._max = max_rows
        self._value = 0

    @property
    def value(self) -> int:
        return self._value

    def resume(self, row_cursor: int) -> None:
        """Set the exact cursor resolved by ``TrainingCheckpoints.resume``."""
        self._value = self._clamp(row_cursor)

    def record(self, raw_rows: int) -> None:
        """Advance cursor by ``raw_rows`` (clamped). Negative is a no-op."""
        if raw_rows <= 0:
            return
        self._value = self._clamp(self._value + raw_rows)

    def _clamp(self, v: int) -> int:
        v = max(0, v)
        if self._max is not None:
            v = min(v, self._max)
        return v
