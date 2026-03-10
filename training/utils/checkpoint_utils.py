"""Checkpoint utilities using tinker_cookbook's checkpoints.jsonl format.

Checkpoint state is persisted in ``checkpoints.jsonl`` (same format as
``tinker_cookbook.checkpoint_utils``).  Reading uses ``get_last_checkpoint``
imported directly from tinker_cookbook; writing uses a sync
``save_checkpoint`` that follows the same schema.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from tinker_cookbook.checkpoint_utils import (
    get_last_checkpoint,
    CHECKPOINTS_BASE_NAME,
)

logger = logging.getLogger(__name__)


# -- Resume info ---------------------------------------------------------------


@dataclass
class ResumeInfo:
    """Resolved resume state returned by ``resolve_resume``."""

    step: int = 0
    data_consumed: int = 0
    dataset_fingerprint: str | None = None
    training_shape_id: str | None = None
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
            dataset_fingerprint=last.get("dataset_fingerprint"),
            training_shape_id=last.get("training_shape_id"),
            source_job_id=last.get("source_job_id"),
        )

    logger.info("Fresh start (no checkpoint)")
    return None


# -- Checkpoint save -----------------------------------------------------------


def save_checkpoint(
    client: Any,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: str = "state",
) -> dict[str, str]:
    """Save a checkpoint using tinker_cookbook's ``checkpoints.jsonl`` format.

    *kind* can be ``"state"`` (optimizer + weights), ``"sampler"`` (weights
    only for inference), or ``"both"``.

    The ``state_path`` stored is resolved to a cross-job checkpoint
    reference at save time, so any future trainer job can load it
    directly without additional resolution.
    """
    paths: dict[str, str] = {}
    if kind in ("state", "both"):
        client.save_state(name)
        paths["state_path"] = client.resolve_checkpoint_path(
            name, source_job_id=client.job_id,
        )
    if kind in ("sampler", "both"):
        paths["sampler_path"] = client.save_weights_for_sampler_ext(name).path

    full_dict = {"name": name, **loop_state, **paths}
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, CHECKPOINTS_BASE_NAME), "a") as f:
        f.write(json.dumps(full_dict) + "\n")
    logger.info("Saved checkpoint: %s", full_dict)
    return paths


# -- Dataset fingerprint -------------------------------------------------------


def dataset_fingerprint(rows: list[dict]) -> str:
    """Short hash of row count + first/last row content."""
    if not rows:
        return "empty"
    content = (
        f"{len(rows)}:"
        f"{json.dumps(rows[0], sort_keys=True)}:"
        f"{json.dumps(rows[-1], sort_keys=True)}"
    )
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def validate_dataset(
    saved_fingerprint: str | None,
    current_fingerprint: str,
    data_consumed: int,
) -> None:
    """Warn if the dataset changed between checkpoint save and resume."""
    if saved_fingerprint and saved_fingerprint != current_fingerprint:
        logger.warning(
            "Dataset changed since checkpoint! "
            "fingerprint: saved=%s current=%s. "
            "data_consumed=%d may point to different data.",
            saved_fingerprint,
            current_fingerprint,
            data_consumed,
        )


# -- Training shape validation -------------------------------------------------


def validate_training_shape(
    saved_shape_id: str | None,
    current_shape_id: str | None,
) -> None:
    """Warn if the training shape changed between checkpoint save and resume."""
    if saved_shape_id and current_shape_id and saved_shape_id != current_shape_id:
        logger.warning(
            "Training shape changed! saved=%s current=%s. "
            "Model parallelism may be incompatible.",
            saved_shape_id,
            current_shape_id,
        )
