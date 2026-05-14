"""Router Replay (R3) alignment utilities for MoE models.

Aligns routing matrices from inference to training model_input positions so
the same expert routing decisions are replayed during training.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class R3RoutingAlignment:
    source_len: int
    prompt_len: int
    model_input_len: int
    completion_only_expected: int
    kind: str
    aligned_len: int
    source_non_empty: int
    aligned_non_empty: int


def _non_empty_count(values: list[str]) -> int:
    return sum(1 for value in values if value)


def describe_r3_routing_alignment(
    routing_matrices: Optional[List[str]],
    prompt_len: int,
    model_input_len: int,
    completion_only: bool = False,
) -> R3RoutingAlignment:
    """Describe how R3 routing matrices align before sending them to Tinker.

    This mirrors ``build_r3_routing_matrices`` without hiding invalid-short
    inputs. It is intended for diagnostics so we can distinguish source
    convention bugs from trainer-side replay issues.
    """
    rm = list(routing_matrices or [])
    source_len = len(rm)
    prefix_len = max(0, prompt_len - 1)
    completion_only_expected = max(0, model_input_len - prefix_len)

    if source_len == 0:
        kind = "missing"
    elif source_len == model_input_len:
        kind = "full"
    elif source_len == completion_only_expected:
        kind = "completion_only"
    elif source_len > model_input_len:
        kind = "overlong"
    elif source_len < completion_only_expected:
        kind = "invalid_short"
    else:
        kind = "invalid_between_completion_and_full"

    aligned = list(rm)
    if aligned and len(aligned) != model_input_len:
        aligned = [""] * prefix_len + aligned
        aligned = aligned[:model_input_len]

    if completion_only and aligned:
        aligned[:prefix_len] = [""] * prefix_len

    return R3RoutingAlignment(
        source_len=source_len,
        prompt_len=prompt_len,
        model_input_len=model_input_len,
        completion_only_expected=completion_only_expected,
        kind=kind,
        aligned_len=len(aligned),
        source_non_empty=_non_empty_count(rm),
        aligned_non_empty=_non_empty_count(aligned),
    )


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

    alignment = describe_r3_routing_alignment(
        routing_matrices,
        prompt_len=prompt_len,
        model_input_len=model_input_len,
        completion_only=completion_only,
    )
    rm = list(routing_matrices)

    if len(rm) != model_input_len:
        expected = model_input_len - (prompt_len - 1)
        if len(rm) != expected:
            logger.warning(
                "R3 alignment mismatch before Tinker: kind=%s source_len=%d "
                "model_input_len=%d completion_only_expected=%d prompt_len=%d "
                "completion_only=%s source_non_empty=%d aligned_len=%d "
                "aligned_non_empty=%d. Alignment may be off.",
                alignment.kind,
                alignment.source_len,
                alignment.model_input_len,
                alignment.completion_only_expected,
                alignment.prompt_len,
                completion_only,
                alignment.source_non_empty,
                alignment.aligned_len,
                alignment.aligned_non_empty,
            )
        rm = [""] * (prompt_len - 1) + rm
        rm = rm[:model_input_len]

    if completion_only:
        prefix_len = max(0, prompt_len - 1)
        rm[:prefix_len] = [""] * prefix_len

    return rm
