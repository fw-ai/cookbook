"""Checkpoint resume helpers."""

from __future__ import annotations

import re
import logging

from fireworks.training.cookbook.utils.client import ReconnectableClient
from fireworks.training.cookbook.utils.config import ResumeConfig

logger = logging.getLogger(__name__)


def setup_resume(
    client: ReconnectableClient,
    resume: ResumeConfig,
) -> tuple[int, str | None]:
    """Load a checkpoint and return (step_offset, checkpoint_name)."""
    if not resume.resume_from:
        return 0, None

    checkpoint_ref = client.resolve_checkpoint_path(
        resume.resume_from,
        source_job_id=resume.resume_job_id,
    )
    logger.info("Loading checkpoint: %s", checkpoint_ref)
    client.load_state_with_optimizer(checkpoint_ref).result(timeout=1800)
    logger.info("Checkpoint loaded: %s", resume.resume_from)

    if resume.step_offset is not None:
        logger.info("Step offset (explicit): %d", resume.step_offset)
        return resume.step_offset, resume.resume_from

    step_offset = 0
    step_match = re.search(r"step-(\d+)", resume.resume_from)
    if step_match:
        step_offset = int(step_match.group(1))
        logger.warning(
            "Inferred step_offset=%d from checkpoint name '%s'. "
            "Set ResumeConfig.step_offset explicitly to avoid this heuristic.",
            step_offset,
            resume.resume_from,
        )
    else:
        logger.warning(
            "Could not infer step offset from '%s'. Starting from step 0.",
            resume.resume_from,
        )
    return step_offset, resume.resume_from
