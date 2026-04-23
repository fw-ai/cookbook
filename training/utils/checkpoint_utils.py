"""Checkpoint utilities -- resume, save, and GCS-transparent I/O.

All file access goes through ``training.utils.fileio`` so the same code
works identically against local paths and ``gs://`` URIs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import training.utils.fileio as fileio
from training.utils.client import ReconnectableClient

CHECKPOINTS_BASE_NAME = "checkpoints.jsonl"

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


def get_last_checkpoint(log_path: str) -> dict[str, Any] | None:
    """Return the last valid entry from ``checkpoints.jsonl``, or None.

    Works transparently for both local directories and ``gs://`` URIs.
    """
    ckpt_file = fileio.join(log_path, CHECKPOINTS_BASE_NAME)
    records = fileio.read_jsonl(ckpt_file)
    return records[-1] if records else None


def validate_warm_start_config(
    *,
    warm_start_from_adapter: str | None,
    init_from_checkpoint: str | None,
    lora_rank: int,
) -> None:
    """Validate cross-field constraints on warm-start inputs.

    Raises ``ValueError`` if combinations are invalid.
    """
    if warm_start_from_adapter and init_from_checkpoint:
        raise ValueError(
            "warm_start_from_adapter and init_from_checkpoint are mutually exclusive"
        )
    if warm_start_from_adapter and lora_rank == 0:
        raise ValueError("warm_start_from_adapter requires lora_rank > 0")


def resolve_resume(
    client: Any,
    log_path: str,
    init_from_checkpoint: str | None = None,
    warm_start_from_adapter: str | None = None,
) -> ResumeInfo | None:
    """Determine resume state.

    Priority:
      1. ``init_from_checkpoint`` — explicit DCP load (weights + optimizer).
      2. Last entry in ``log_path/checkpoints.jsonl`` — auto-resume.
      3. ``warm_start_from_adapter`` — HF PEFT adapter (weights-only).
      4. Fresh start.

    Returns ``None`` only for case 4. Higher-priority paths load into
    *client* before returning.
    """
    if init_from_checkpoint:
        source_job_id, dcp_name = _parse_cross_job(init_from_checkpoint)
        path = client.resolve_checkpoint_path(dcp_name, source_job_id=source_job_id)
        logger.info(
            "Starting at step 0 with weights loaded from %s (no resume — step counter resets)",
            path,
        )
        t0 = time.time()
        client.load_state_with_optimizer(path)
        logger.info("Checkpoint loaded (%.1fs)", time.time() - t0)
        return ResumeInfo(step=0, data_consumed=0, source_job_id=source_job_id)

    last = get_last_checkpoint(log_path)
    if last:
        logger.info("Resuming from checkpoints.jsonl: %s", last)
        t0 = time.time()
        client.load_state_with_optimizer(last["state_path"])
        logger.info("Checkpoint loaded: %s (%.1fs)", last["state_path"], time.time() - t0)
        return ResumeInfo(
            step=last.get("step", 0),
            data_consumed=last.get("data_consumed", 0),
            source_job_id=last.get("source_job_id"),
        )

    if warm_start_from_adapter:
        logger.info("Fresh start with HF adapter: %s", warm_start_from_adapter)
        t0 = time.time()
        client.load_adapter(warm_start_from_adapter)
        logger.info("Adapter loaded (%.1fs)", time.time() - t0)
        return ResumeInfo(step=0, data_consumed=0, source_job_id=None)

    logger.info("Starting at step 0 from base model (no checkpoint)")
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
    entry so that downstream tools (e.g. ``examples/tools/promote_checkpoint.py``)
    can auto-detect them without requiring the user to pass them manually.

    The ``state_path`` stored is resolved to a cross-job checkpoint
    reference at save time, so any future trainer job can load it
    directly without additional resolution.
    """
    paths: dict[str, str] = {}
    t0 = time.time()
    if kind in (CheckpointKind.STATE, CheckpointKind.BOTH):
        logger.info("Saving DCP state checkpoint '%s'...", name)
        client.save_state(name)
        logger.info("DCP state checkpoint '%s' saved (%.1fs)", name, time.time() - t0)
        paths["state_path"] = client.resolve_checkpoint_path(
            name, source_job_id=client.job_id,
        )
    if kind in (CheckpointKind.SAMPLER, CheckpointKind.BOTH):
        t1 = time.time()
        logger.info("Saving sampler checkpoint '%s'...", name)
        save_result = client.save_weights_for_sampler_ext(name, checkpoint_type="base")
        paths["sampler_path"] = get_sampler_checkpoint_id(save_result)
        logger.info("Sampler checkpoint '%s' saved (%.1fs)", name, time.time() - t1)

    full_dict = {"name": name, **loop_state, **paths}
    if base_model:
        full_dict["base_model"] = base_model
    if training_shape:
        full_dict["training_shape"] = training_shape
    fileio.makedirs(log_path)
    fileio.append_jsonl(fileio.join(log_path, CHECKPOINTS_BASE_NAME), full_dict)
    logger.info("Checkpoint '%s' complete (%.1fs total): %s", name, time.time() - t0, full_dict)
    return paths


def save_dcp_checkpoint_if_due(
    client: ReconnectableClient,
    *,
    step: int,
    rollouts_completed: int,
    dcp_interval: int,
    log_path: str,
    resume_info: ResumeInfo | None,
    prompt_groups_per_step: int,
    policy_job_id: str,
    base_model: str,
    training_shape: str | None,
) -> None:
    """Save a STATE checkpoint when ``rollouts_completed`` hits ``dcp_interval``."""
    if dcp_interval <= 0 or rollouts_completed <= 0 or rollouts_completed % dcp_interval != 0:
        return
    data_consumed = (resume_info.data_consumed if resume_info else 0) + (
        rollouts_completed * prompt_groups_per_step
    )
    save_checkpoint(
        client,
        f"step-{step}",
        log_path,
        {"step": step, "data_consumed": data_consumed, "source_job_id": policy_job_id},
        kind=CheckpointKind.STATE,
        base_model=base_model,
        training_shape=training_shape,
    )


def save_final_checkpoint_and_promote(
    client: ReconnectableClient,
    rlor_mgr: Any,
    runner: Any,
    *,
    global_step: int,
    step_offset: int,
    ppo_n_minibatches: int,
    prompt_groups_per_step: int,
    resume_info: ResumeInfo | None,
    log_path: str,
    base_model: str,
    training_shape: str | None,
    policy_job_id: str,
    output_model_id: str | None,
) -> None:
    """Save a BOTH checkpoint at end-of-run and (optionally) promote it."""
    try:
        rollouts_this_run = (global_step - step_offset) // max(1, ppo_n_minibatches)
        data_consumed = (resume_info.data_consumed if resume_info else 0) + (
            rollouts_this_run * prompt_groups_per_step
        )
        cp_name = f"step-{global_step}"
        paths = save_checkpoint(
            client,
            cp_name,
            log_path,
            {"step": global_step, "data_consumed": data_consumed, "source_job_id": policy_job_id},
            kind=CheckpointKind.BOTH,
            base_model=base_model,
            training_shape=training_shape,
        )
        if output_model_id:
            rlor_mgr.promote_checkpoint(
                policy_job_id, paths["sampler_path"], output_model_id, base_model,
            )
            runner.write_output_model(
                model_id=output_model_id, checkpoint=cp_name, job_id=policy_job_id,
            )
    except Exception as e:
        logger.warning("Failed to save final checkpoint: %s", e)
