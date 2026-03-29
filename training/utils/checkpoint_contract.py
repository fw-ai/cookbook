"""Reusable checkpoints.jsonl data contract helpers.

This module defines the canonical, append-only checkpoint entry shape used by
all cookbook training loops (SFT, DPO/ORPO, and RFT).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from tinker_cookbook.checkpoint_utils import CHECKPOINTS_BASE_NAME

RunnerKind = Literal["sft", "dpo", "rft"]


@dataclass
class LoopState:
    """Loop cursor persisted in each checkpoint entry."""

    epoch: int = 0
    batch_index: int = 0
    global_step: int = 0
    algorithm_payload: dict[str, Any] = field(default_factory=dict)


def normalize_loop_state(loop_state: dict[str, Any] | None, *, step: int) -> dict[str, Any]:
    """Return a fully populated loop_state object with backward-safe defaults."""
    state = dict(loop_state or {})
    return {
        "epoch": int(state.get("epoch", 0)),
        "batch_index": int(state.get("batch_index", 0)),
        "global_step": int(state.get("global_step", step)),
        "algorithm_payload": dict(state.get("algorithm_payload", {})),
    }


def build_checkpoint_entry(
    *,
    name: str,
    step: int,
    data_consumed: int,
    source_job_id: str | None,
    state_path: str | None,
    sampler_path: str | None = None,
    runner_kind: RunnerKind | None = None,
    loop_state: dict[str, Any] | None = None,
    optimizer_resume_supported: bool = True,
    safe_to_resume: bool = True,
    pause_boundary: str = "optimizer_step",
    base_model: str | None = None,
    training_shape: str | None = None,
) -> dict[str, Any]:
    """Construct a checkpoint ledger entry in canonical shape."""
    entry: dict[str, Any] = {
        "name": name,
        "step": step,
        "data_consumed": data_consumed,
        "source_job_id": source_job_id,
        "loop_state": normalize_loop_state(loop_state, step=step),
        "optimizer_resume_supported": bool(optimizer_resume_supported),
        "safe_to_resume": bool(safe_to_resume),
        "pause_boundary": pause_boundary,
    }
    if runner_kind is not None:
        entry["runner_kind"] = runner_kind
    if state_path is not None:
        entry["state_path"] = state_path
    if sampler_path is not None:
        entry["sampler_path"] = sampler_path
    if base_model is not None:
        entry["base_model"] = base_model
    if training_shape is not None:
        entry["training_shape"] = training_shape
    return entry


def append_checkpoint_entry(log_path: str, entry: dict[str, Any]) -> None:
    """Append one checkpoint entry to checkpoints.jsonl (append-only)."""
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, CHECKPOINTS_BASE_NAME), "a") as f:
        f.write(json.dumps(entry) + "\n")
