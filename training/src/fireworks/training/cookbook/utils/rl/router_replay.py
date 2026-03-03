"""Router Replay (R3) alignment utilities for MoE models.

Aligns routing matrices from inference to training model_input positions so
the same expert routing decisions are replayed during training.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def build_r3_routing_matrices(
    routing_matrices: Optional[List[str]],
    prompt_len: int,
    model_input_len: int,
    completion_only: bool = False,
) -> Optional[List[str]]:
    """Build routing matrices aligned to model_input positions.

    Supports echo mode (full sequence) and legacy mode (completion-only,
    padded with empty strings for prompt positions).

    Args:
        completion_only: If True, blank out prompt-position routing matrices
            so the server uses its own routing for prompt tokens and only
            replays inference routing for completion tokens.
    """
    if not routing_matrices:
        return None

    rm = list(routing_matrices)

    if len(rm) != model_input_len:
        expected = model_input_len - (prompt_len - 1)
        if len(rm) != expected:
            logger.warning(
                "R3: routing_matrices length (%d) != expected (%d). "
                "prompt_len=%d, model_input_len=%d. Alignment may be off.",
                len(rm),
                expected,
                prompt_len,
                model_input_len,
            )
        rm = [""] * (prompt_len - 1) + rm
        rm = rm[:model_input_len]

    if completion_only:
        prefix_len = max(0, prompt_len - 1)
        rm[:prefix_len] = [""] * prefix_len

    return rm
