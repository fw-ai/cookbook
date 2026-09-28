"""Signal-driven termination of training recipes."""

from __future__ import annotations


class TerminatedBySignal(SystemExit):
    """SIGINT or SIGTERM stopped a recipe.

    It unwinds exactly like ``SystemExit`` so recipe cleanup runs; the type lets
    callers tell an external stop (for example a cancelled CI job) from a failure.
    """

    def __init__(self, signal_name: str) -> None:
        super().__init__(f"Terminated by {signal_name}")
        self.signal_name = signal_name
