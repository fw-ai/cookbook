"""Checkpoint and reconnect utilities.

Two resume paths:

* **DCP checkpoint resume** (``resolve_resume`` / ``checkpoints.jsonl``):
  Heavy-weight.  Loads full optimizer + weight state into a *new* trainer
  job via ``load_state_with_optimizer``.  Use when the remote trainer has
  died and a new job must be created.

* **Client-side reconnect** (``save_reconnect_state`` / ``load_reconnect_state``):
  Light-weight.  The remote trainer is still alive — only the client
  script crashed.  Reads a local JSON state file, verifies the trainer is
  healthy, and resumes the training loop from where it left off with no
  DCP load.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from training.utils import ReconnectableClient
from enum import Enum
from typing import Any

from tinker_cookbook.checkpoint_utils import (
    get_last_checkpoint,
    CHECKPOINTS_BASE_NAME,
)

logger = logging.getLogger(__name__)

def get_sampler_checkpoint_id(save_result: Any) -> str:
    """Extract the promotable sampler checkpoint id from a save result."""
    snapshot_name = getattr(save_result, "snapshot_name", None)
    if snapshot_name:
        return snapshot_name

    path = getattr(save_result, "path", None)
    if not path:
        raise ValueError("save_weights_for_sampler_ext() returned no checkpoint identifier")

    return str(path).rstrip("/").rsplit("/", 1)[-1]

# -- Resume info ---------------------------------------------------------------


@dataclass
class ResumeInfo:
    """Resolved resume state returned by ``resolve_resume``."""

    step: int = 0
    data_consumed: int = 0
    source_job_id: str | None = None


def _parse_cross_job(spec: str) -> tuple[str | None, str]:
    """Parse ``"job_id:checkpoint_name"`` or a plain path/name."""
    if ":" in spec and not spec.startswith(("gs://", "/")):
        job_id, name = spec.split(":", 1)
        return job_id, name
    return None, spec


def resolve_resume(
    client: Any,
    log_path: str,
    init_from_checkpoint: str | None = None,
) -> ResumeInfo | None:
    """Determine resume state from ``checkpoints.jsonl``.

    Returns ``None`` for a completely fresh start (no checkpoint to load).
    When a checkpoint is found or *init_from_checkpoint* is set, the
    weights + optimizer state are loaded into *client* before returning.
    """
    if init_from_checkpoint:
        source_job_id, dcp_name = _parse_cross_job(init_from_checkpoint)
        path = client.resolve_checkpoint_path(dcp_name, source_job_id=source_job_id)
        logger.info("Fresh start with pretrained weights: %s", path)
        t0 = time.time()
        client.load_state_with_optimizer(path)
        logger.info("Checkpoint loaded (%.1fs)", time.time() - t0)
        return ResumeInfo(step=0, data_consumed=0, source_job_id=source_job_id)

    last = get_last_checkpoint(log_path)
    if last is not None:
        logger.info("Resuming from checkpoints.jsonl: %s", last)
        t0 = time.time()
        client.load_state_with_optimizer(last["state_path"])
        logger.info("Checkpoint loaded: %s (%.1fs)", last["state_path"], time.time() - t0)
        return ResumeInfo(
            step=last.get("step", 0),
            data_consumed=last.get("data_consumed", 0),
            source_job_id=last.get("source_job_id"),
        )

    logger.info("Fresh start (no checkpoint)")
    return None


# -- Checkpoint save -----------------------------------------------------------

class CheckpointKind(str, Enum):
    STATE = "state"
    SAMPLER = "sampler"
    BOTH = "both"

def save_checkpoint(
    client: ReconnectableClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: CheckpointKind = CheckpointKind.STATE,
    *,
    base_model: str | None = None,
    training_shape: str | None = None,
) -> dict[str, str]:
    """Save a checkpoint using tinker_cookbook's ``checkpoints.jsonl`` format.

    *kind* can be ``CheckpointKind.STATE`` (optimizer + weights), ``CheckpointKind.SAMPLER`` (weights
    only for inference), or ``CheckpointKind.BOTH``.

    *base_model* and *training_shape* are persisted into the checkpoint
    entry so that downstream tools (e.g. ``train_promote_checkpoint.py``)
    can reconstruct the trainer without requiring the user to pass them
    again manually.

    The ``state_path`` stored is resolved to a cross-job checkpoint
    reference at save time, so any future trainer job can load it
    directly without additional resolution.
    """
    paths: dict[str, str] = {}
    if kind in (CheckpointKind.STATE, CheckpointKind.BOTH):
        client.save_state(name)
        paths["state_path"] = client.resolve_checkpoint_path(
            name, source_job_id=client.job_id,
        )
    if kind in (CheckpointKind.SAMPLER, CheckpointKind.BOTH):
        save_result = client.save_weights_for_sampler_ext(name, checkpoint_type="base")
        paths["sampler_path"] = get_sampler_checkpoint_id(save_result)

    full_dict = {"name": name, **loop_state, **paths}
    if base_model:
        full_dict["base_model"] = base_model
    if training_shape:
        full_dict["training_shape"] = training_shape
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, CHECKPOINTS_BASE_NAME), "a") as f:
        f.write(json.dumps(full_dict) + "\n")
    logger.info("Saved checkpoint: %s", full_dict)
    return paths


# -- Client-side reconnect state -----------------------------------------------


@dataclass
class ReconnectState:
    """Client-side loop progress for lightweight reconnect.

    Saved to a local JSON file after each training step.  When the
    *client script* crashes but the remote trainer pod is still alive,
    this state lets the loop resume without the expensive DCP
    ``load_state_with_optimizer`` call.

    **This is NOT a DCP checkpoint.**  If the trainer itself died, use
    ``resolve_resume`` with ``checkpoints.jsonl`` instead.
    """

    step: int
    data_consumed: int
    policy_job_id: str
    reference_job_id: str | None = None
    deployment_id: str | None = None
    base_model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    """Arbitrary extra state recipes can stash (e.g. epoch number)."""


def save_reconnect_state(path: str, state: ReconnectState) -> None:
    """Atomically write *state* to *path*.

    Uses write-to-tmp + ``os.replace`` so a crash mid-write never
    leaves a corrupt file.
    """
    data = asdict(state)
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def load_reconnect_state(path: str) -> ReconnectState | None:
    """Read a reconnect state file.  Returns ``None`` if the file does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # Drop fields that aren't in the dataclass (e.g. timestamp).
    known = {f.name for f in ReconnectState.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known}
    return ReconnectState(**filtered)


def try_client_reconnect(
    path: str,
    rlor_mgr: Any,
) -> ReconnectState:
    """Verify the saved trainer is still healthy and return reconnect state.

    Raises ``FileNotFoundError`` if the state file is missing, or
    ``RuntimeError`` if the trainer is no longer alive (with guidance to
    use DCP checkpoint resume instead).
    """
    state = load_reconnect_state(path)
    if state is None:
        raise FileNotFoundError(
            f"Reconnect state file not found: {path}\n"
            "Cannot reconnect — no previous client state saved."
        )

    # Check job state via control plane.
    job_id = state.policy_job_id
    try:
        job = rlor_mgr.get(job_id)
    except Exception as e:
        raise RuntimeError(
            f"Cannot query trainer job {job_id}: {e}\n"
            "The trainer may have been deleted. "
            "Use DCP checkpoint resume (init_from_checkpoint or checkpoints.jsonl) instead."
        ) from e

    job_state = job.get("state", "")
    if job_state != "JOB_STATE_RUNNING":
        raise RuntimeError(
            f"Trainer job {job_id} is in state '{job_state}', not RUNNING.\n"
            "The trainer has died since the client disconnected.\n"
            "Use DCP checkpoint resume (init_from_checkpoint or "
            "checkpoints.jsonl) instead."
        )

    # Verify endpoint is actually responsive.
    base_url = rlor_mgr._get_trainer_gateway_url(job_id)
    if not rlor_mgr._check_healthz(base_url):
        raise RuntimeError(
            f"Trainer job {job_id} is RUNNING but /healthz failed.\n"
            "The trainer may be restarting or unhealthy.\n"
            "Use DCP checkpoint resume (init_from_checkpoint or "
            "checkpoints.jsonl) instead."
        )

    # Check reference trainer if present.
    if state.reference_job_id:
        ref_url = rlor_mgr._get_trainer_gateway_url(state.reference_job_id)
        if not rlor_mgr._check_healthz(ref_url):
            raise RuntimeError(
                f"Reference trainer {state.reference_job_id} is not healthy.\n"
                "Use DCP checkpoint resume instead."
            )

    logger.info(
        "Client reconnect OK: step=%d, policy=%s, ref=%s",
        state.step, state.policy_job_id, state.reference_job_id,
    )
    return state
