"""Process-local accounting for harness trials that never reach a trajectory.

A completed trajectory carries its own TITO summary, so its timing reaches
WandB through the per-step trajectory merge. Attempts that fail — a broken
provider stream, a whole-trial-bound firing, a materialization error — produce
no trajectory at all, so without this accumulator a step that burned half its
wall clock on doomed trials looks exactly like a clean one.

The recorders are called from the client event loop (the rollout retry loop and
the trial driver), never from the artifact worker process: a counter
incremented in a child process would be discarded with that process. Values
are drained once per optimizer step by ``publish_tito_sidecar_metrics``.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping

_COUNTER_ROOT = "tito/trial/"
_FAILED_PHASE_ROOT = "tito/trial_failed/"
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")

_LOCK = threading.Lock()
_COUNTERS: dict[str, float] = {}
_DISTRIBUTIONS: dict[str, list[float]] = {}


def _slug(reason: str) -> str:
    value = _SLUG_PATTERN.sub("_", reason.strip().lower()).strip("_")
    return value or "unknown"


def _increment(name: str, amount: float = 1.0) -> None:
    with _LOCK:
        _COUNTERS[name] = _COUNTERS.get(name, 0.0) + amount


def _observe(name: str, value: float) -> None:
    with _LOCK:
        _DISTRIBUTIONS.setdefault(name, []).append(float(value))


def record_trial_attempt() -> None:
    """One Harbor trial attempt started (retries count separately)."""
    _increment(f"{_COUNTER_ROOT}attempts")


def record_trial_retry(reason: str) -> None:
    """A transient attempt failure that will be retried on a fresh trial."""
    _increment(f"{_COUNTER_ROOT}retries")
    _increment(f"{_COUNTER_ROOT}retry_reason/{_slug(reason)}")


def record_trial_discarded(reason: str) -> None:
    """A task abandoned after exhausting its attempts, or a fatal attempt."""
    _increment(f"{_COUNTER_ROOT}discarded")
    _increment(f"{_COUNTER_ROOT}discard_reason/{_slug(reason)}")


def record_trial_timeout() -> None:
    """The whole-trial bound fired: a silent hang became a retryable failure."""
    _increment(f"{_COUNTER_ROOT}whole_trial_bound_firings")


def record_failed_trial_wall(seconds: float) -> None:
    """Wall clock burned by an attempt that produced no trajectory."""
    _observe(f"{_FAILED_PHASE_ROOT}trial_wall_seconds", seconds)


def record_failed_trial_phases(phases: Mapping[str, float]) -> None:
    """Harbor phase brackets of a trial that ran but yielded no trajectory."""
    for phase, seconds in phases.items():
        _observe(f"{_FAILED_PHASE_ROOT}{phase}", float(seconds))


def drain() -> dict[str, float]:
    """Return and clear everything recorded since the previous drain."""
    with _LOCK:
        counters = _COUNTERS
        distributions = _DISTRIBUTIONS
        _reset_locked()
    drained: dict[str, float] = dict(counters)
    for name, values in distributions.items():
        drained[f"{name}_count"] = float(len(values))
        drained[f"{name}_sum"] = float(sum(values))
        drained[f"{name}_min"] = float(min(values))
        drained[f"{name}_max"] = float(max(values))
        drained[f"{name}_mean"] = float(sum(values) / len(values))
    return drained


def _reset_locked() -> None:
    global _COUNTERS, _DISTRIBUTIONS
    _COUNTERS = {}
    _DISTRIBUTIONS = {}


def reset() -> None:
    """Drop everything recorded so far (test isolation)."""
    with _LOCK:
        _reset_locked()


__all__ = [
    "drain",
    "record_failed_trial_phases",
    "record_failed_trial_wall",
    "record_trial_attempt",
    "record_trial_discarded",
    "record_trial_retry",
    "record_trial_timeout",
    "reset",
]
