"""Read-only local artifact observations; these do not prove process liveness."""

from __future__ import annotations

from pathlib import Path
import time


def pending_trial_inventory(
    trials_dir: str | Path, *, now: float | None = None
) -> list[dict[str, str | float]]:
    """List unfinished trials and approximate phase ages without reading secrets.

    A compact COMPLETE means agent output was collected, not that the verifier
    finished. Its local modification time approximates the start of the
    verification/finalization wait. config.json's modification time approximates
    trial creation. Copied/touched artifacts can change these estimates; verify
    remote processes before declaring a stall or taking any recovery action.
    """
    observed_at = time.time() if now is None else now
    pending = []
    for trial in Path(trials_dir).glob("*"):
        if trial.name.startswith("e2b-template-") or not trial.is_dir():
            continue
        if (trial / "result.json").exists():
            continue
        try:
            created_at = (trial / "config.json").stat().st_mtime
            complete = trial / "artifacts/tito/compact/COMPLETE"
            if complete.is_file():
                phase = "verification_or_finalization"
                phase_at = complete.stat().st_mtime
            else:
                phase = "agent_or_setup"
                phase_at = created_at
        except FileNotFoundError:
            # A trial can be created/finalized/pruned concurrently with a poll.
            continue
        pending.append(
            {
                "trial": trial.name,
                "phase": phase,
                "trial_age_s": max(0.0, observed_at - created_at),
                "phase_age_s": max(0.0, observed_at - phase_at),
            }
        )
    return sorted(pending, key=lambda item: float(item["phase_age_s"]), reverse=True)
