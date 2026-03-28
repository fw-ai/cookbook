"""Pipeline-parallel batch recommendation.

Replicates the server-side ``local_batch_size`` calculation so the client
can align its batch dimensions for zero padding waste and minimal bubble.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fireworks.training.sdk import TrainingShapeProfile

logger = logging.getLogger(__name__)

DEFAULT_MAX_BATCH_SIZE_TOKENS = 65536


@dataclass
class PPBatchRecommendation:
    """Recommended batch configuration for optimal pipeline-parallel efficiency.

    Returned by :func:`compute_pp_recommendation`.  Pass these values to
    the recipe config (``group_size``, ``prompts_per_step``) to minimize
    PP bubble and padding waste.
    """

    local_batch_size: int
    """Server-side PP batch size (n_microbatches with microbatch_size=1)."""
    pp_degree: int
    bubble_ratio: float
    """Per-execution bubble fraction: ``(pp_degree - 1) / (local_batch_size + pp_degree - 1)``."""
    recommended_group_size: int
    """Ideal ``group_size`` = ``local_batch_size`` (fills one PP batch exactly)."""
    recommended_prompts_per_step: int
    """Number of prompts per optimizer step so that
    ``group_size * prompts_per_step >= local_batch_size``."""


def compute_pp_recommendation(
    profile: TrainingShapeProfile,
    group_size: int,
    max_batch_size_tokens: int = DEFAULT_MAX_BATCH_SIZE_TOKENS,
) -> PPBatchRecommendation:
    """Compute optimal batch config for pipeline-parallel training.

    TODO: This recommendation needs improvement -- it doesn't account for
    grad accumulation normalization, variable sequence lengths, or the
    interaction between PP bubble ratio and batch partitioning strategy.

    Replicates the server-side ``local_batch_size`` calculation from
    ``torchtitan_trainer_factory.py`` so the client can align its batch
    dimensions for zero padding waste and minimal bubble.

    The training is mathematically equivalent regardless of how items are
    partitioned across ``forward_backward_custom`` calls -- the gradient
    accumulated before ``optim_step`` is identical.
    """
    pp = max(1, profile.pipeline_parallelism)
    max_seq = max(1, profile.max_supported_context_length)
    local_batch_size = max(pp, max_batch_size_tokens // max_seq)

    if pp <= 1:
        return PPBatchRecommendation(
            local_batch_size=local_batch_size,
            pp_degree=1,
            bubble_ratio=0.0,
            recommended_group_size=group_size,
            recommended_prompts_per_step=1,
        )

    bubble = (pp - 1) / (local_batch_size + pp - 1)

    if group_size >= local_batch_size:
        prompts_per_step = 1
    else:
        prompts_per_step = max(1, local_batch_size // group_size)

    return PPBatchRecommendation(
        local_batch_size=local_batch_size,
        pp_degree=pp,
        bubble_ratio=bubble,
        recommended_group_size=local_batch_size,
        recommended_prompts_per_step=prompts_per_step,
    )
