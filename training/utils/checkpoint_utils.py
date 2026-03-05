"""Checkpoint utilities -- single source of truth for resume state.

All resume logic flows through ``local_checkpoint_state.jsonl``:

- **Continuing a run**: last entry has ``dcp_name``, ``step``, ``rows_consumed``.
- **Fresh start with pretrained weights**: ``init_from_dcp`` writes an
  initial entry with ``step=0, rows_consumed=0``, then loads the DCP.
- **Completely fresh start**: no entries, no DCP load.

No ``ResumeConfig``, no regex step parsing, no dual paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

STATE_FILE = "local_checkpoint_state.jsonl"


# -- Resume state --------------------------------------------------------------


@dataclass
class ResumeState:
    """Resolved resume state -- the single source of truth."""

    step: int = 0
    data_consumed: int = 0
    dcp_name: str | None = None
    dataset_fingerprint: str | None = None
    training_shape_id: str | None = None
    source_job_id: str | None = None


def resolve_resume(
    log_path: str,
    init_from_dcp: str | None = None,
) -> ResumeState:
    """Determine resume state from ``local_checkpoint_state.jsonl``.

    Priority:
    1. If ``local_checkpoint_state.jsonl`` has entries, resume from the last one.
    2. If empty/missing but *init_from_dcp* is set, start fresh
       with DCP weights (step=0, rows_consumed=0).
    3. Otherwise, completely fresh start (no DCP).

    *init_from_dcp* supports cross-job format ``"job_id:checkpoint_name"``.
    """
    last = _load_last_loop_state(log_path)
    if last is not None:
        logger.info("Resuming from local_checkpoint_state.jsonl: %s", last)
        return ResumeState(
            step=last.get("step", 0),
            data_consumed=last.get("data_consumed", 0),
            dcp_name=last.get("dcp_name"),
            dataset_fingerprint=last.get("dataset_fingerprint"),
            training_shape_id=last.get("training_shape_id"),
            source_job_id=last.get("source_job_id"),
        )

    if init_from_dcp:
        source_job_id = None
        dcp_name = init_from_dcp
        if ":" in init_from_dcp and not init_from_dcp.startswith(("gs://", "/")):
            source_job_id, dcp_name = init_from_dcp.split(":", 1)

        logger.info(
            "Fresh start with pretrained weights: dcp=%s source_job=%s",
            dcp_name,
            source_job_id,
        )
        initial = ResumeState(
            step=0,
            dcp_name=dcp_name,
            source_job_id=source_job_id,
        )
        save_loop_state(log_path, _state_to_dict(initial))
        return initial

    logger.info("Fresh start (no checkpoint)")
    return ResumeState()


def load_dcp(client: Any, state: ResumeState) -> float:
    """Load DCP checkpoint if *state.dcp_name* is set.

    Resolves cross-job references and loads model weights + optimizer.
    Returns the load time in seconds (0.0 if no checkpoint was loaded).
    """
    if state.dcp_name is None:
        return 0.0

    checkpoint_ref = client.resolve_checkpoint_path(
        state.dcp_name,
        source_job_id=state.source_job_id,
    )
    logger.info("Loading DCP checkpoint: %s", checkpoint_ref)
    t0 = time.time()
    client.load_state_with_optimizer(checkpoint_ref)
    elapsed = time.time() - t0
    logger.info("DCP checkpoint loaded: %s (%.1fs)", state.dcp_name, elapsed)
    return elapsed


# -- Loop state persistence ----------------------------------------------------


def save_loop_state(log_path: str, loop_state: dict[str, Any]) -> None:
    """Append a loop-state entry to ``local_checkpoint_state.jsonl``."""
    os.makedirs(log_path, exist_ok=True)
    path = os.path.join(log_path, STATE_FILE)
    with open(path, "a") as f:
        f.write(json.dumps(loop_state) + "\n")
    logger.info("Saved loop state: %s", loop_state)


def _load_last_loop_state(log_path: str) -> dict[str, Any] | None:
    """Read the most recent entry from ``local_checkpoint_state.jsonl``."""
    path = os.path.join(log_path, STATE_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def _state_to_dict(state: ResumeState) -> dict[str, Any]:
    """Serialize a ResumeState to a dict for local_checkpoint_state.jsonl."""
    return {
        "step": state.step,
        "data_consumed": state.data_consumed,
        "dcp_name": state.dcp_name,
        "dataset_fingerprint": state.dataset_fingerprint,
        "training_shape_id": state.training_shape_id,
        "source_job_id": state.source_job_id,
    }


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


# -- Checkpoint availability ---------------------------------------------------


def verify_checkpoint_available(client: Any, dcp_name: str) -> bool:
    """Check that a DCP checkpoint exists on the trainer."""
    try:
        checkpoints, _ = client.list_checkpoints()
        if dcp_name in checkpoints:
            logger.info(
                "Checkpoint '%s' available (all: %s)", dcp_name, checkpoints,
            )
            return True
        logger.warning(
            "Checkpoint '%s' not found. Available: %s", dcp_name, checkpoints,
        )
        return False
    except Exception as e:
        logger.warning("Could not list checkpoints: %s. Proceeding anyway.", e)
        return True


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
