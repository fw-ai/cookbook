"""Checkpoint utilities using tinker_cookbook's checkpoints.jsonl format.

Reading uses ``get_last_checkpoint`` from tinker_cookbook directly.
Writing (``save_checkpoint``) and resume (``resolve_resume``) are
implemented locally for Fireworks RLOR compatibility.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from training.utils import ReconnectableClient

from tinker_cookbook.checkpoint_utils import (
    get_last_checkpoint,
    CHECKPOINTS_BASE_NAME,
)
from training.utils.checkpoint_contract import (
    RunnerKind,
    append_checkpoint_entry,
    build_checkpoint_entry,
    normalize_loop_state,
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
    loop_state: dict[str, Any] = field(default_factory=dict)
    runner_kind: RunnerKind | None = None
    optimizer_resume_supported: bool = True
    safe_to_resume: bool = True
    pause_boundary: str = "optimizer_step"


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
        return ResumeInfo(
            step=0,
            data_consumed=0,
            source_job_id=source_job_id,
            loop_state=normalize_loop_state(loop_state=None, step=0),
        )

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
            loop_state=normalize_loop_state(
                loop_state=last.get("loop_state"),
                step=last.get("step", 0),
            ),
            runner_kind=last.get("runner_kind"),
            optimizer_resume_supported=bool(last.get("optimizer_resume_supported", True)),
            safe_to_resume=bool(last.get("safe_to_resume", True)),
            pause_boundary=last.get("pause_boundary", "optimizer_step"),
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
    runner_kind: RunnerKind | None = None,
    checkpoint_loop_state: dict[str, Any] | None = None,
    optimizer_resume_supported: bool = True,
    safe_to_resume: bool = True,
    pause_boundary: str = "optimizer_step",
    sampler_name: str | None = None,
    state_path_override: str | None = None,
    sampler_path_override: str | None = None,
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
    loop_fields = dict(loop_state)
    step = int(loop_fields.pop("step", 0))
    data_consumed = int(loop_fields.pop("data_consumed", 0))
    source_job_id = loop_fields.pop("source_job_id", None)
    if source_job_id is None:
        source_job_id = getattr(client, "job_id", None)
    nested_loop_state = loop_fields.pop("loop_state", None)
    if checkpoint_loop_state is None:
        checkpoint_loop_state = nested_loop_state

    paths: dict[str, str] = {}
    if kind in (CheckpointKind.STATE, CheckpointKind.BOTH):
        if state_path_override is not None:
            paths["state_path"] = state_path_override
        else:
            client.save_state(name)
            paths["state_path"] = client.resolve_checkpoint_path(
                name,
                source_job_id=source_job_id,
            )
    if kind in (CheckpointKind.SAMPLER, CheckpointKind.BOTH):
        if sampler_path_override is not None:
            paths["sampler_path"] = sampler_path_override
        else:
            sampler_checkpoint_name = sampler_name or name
            save_sampler_fn = getattr(client, "save_weights_for_sampler_ext", None)
            if save_sampler_fn is None and hasattr(client, "inner"):
                save_sampler_fn = getattr(client.inner, "save_weights_for_sampler_ext", None)
            if save_sampler_fn is None:
                raise AttributeError("Client does not support save_weights_for_sampler_ext")
            save_result = save_sampler_fn(
                sampler_checkpoint_name,
                checkpoint_type="base",
            )
            paths["sampler_path"] = get_sampler_checkpoint_id(save_result)

    full_dict = build_checkpoint_entry(
        name=name,
        runner_kind=runner_kind,
        step=step,
        data_consumed=data_consumed,
        source_job_id=source_job_id,
        state_path=paths.get("state_path"),
        sampler_path=paths.get("sampler_path"),
        loop_state=checkpoint_loop_state,
        optimizer_resume_supported=optimizer_resume_supported,
        safe_to_resume=safe_to_resume,
        pause_boundary=pause_boundary,
        base_model=base_model,
        training_shape=training_shape,
    )
    for extra_key, extra_value in loop_fields.items():
        if extra_key not in full_dict:
            full_dict[extra_key] = extra_value
    append_checkpoint_entry(log_path=log_path, entry=full_dict)
    logger.info("Saved checkpoint: %s", full_dict)
    return paths
