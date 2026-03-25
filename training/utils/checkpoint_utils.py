"""Checkpoint utilities using tinker_cookbook's checkpoints.jsonl format.

Reading uses ``get_last_checkpoint`` from tinker_cookbook directly.
Writing (``save_checkpoint``) and resume (``resolve_resume``) are
implemented locally for Fireworks RLOR compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
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
    can auto-detect them without requiring the user to pass them manually.

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
