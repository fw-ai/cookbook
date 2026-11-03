"""Offline reward curriculum state and completion accounting."""
from __future__ import annotations
import json
from pathlib import Path


def initial_phase(mode: str, resume_from: str = "", state_path: str = "") -> str:
    if mode not in ("auto", "ramp", "strict"):
        raise ValueError(f"Unknown reward phase: {mode}")
    if mode != "auto":
        return mode  # An explicit fixed phase is a deliberate override.
    if not resume_from:
        return "ramp"
    if not state_path:
        raise ValueError("Auto-mode resume requires --resume-reward-state from the saved checkpoint; do not guess the old phase.")
    state = json.loads(Path(state_path).read_text())
    if state.get("version") != 1 or state.get("checkpoint") != resume_from or state.get("phase") not in ("ramp", "strict"):
        raise ValueError("Reward state does not match this resume checkpoint")
    return state["phase"]


def save_phase_state(path: Path, checkpoint: str, phase: str) -> None:
    if phase not in ("ramp", "strict"):
        raise ValueError("Invalid reward phase")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"version": 1, "checkpoint": checkpoint, "phase": phase}) + "\n")
    temporary.replace(path)


def completed_score(score: tuple, truncated: bool) -> tuple:
    """Only completed answers count as e-free or answered; retain diagnostic counts."""
    if truncated:
        return (0.0, False, False, *score[3:])
    return score
