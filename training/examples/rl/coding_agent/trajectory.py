"""Token-level trajectory merge for the black-box coding-agent shim.

Coding-agent-specific reconstruction: the shim records one :class:`TurnRecord`
per engine call (the exact prompt token ids it rendered + the raw output token
ids + per-token logprobs); :func:`merge_turns` replays a chain of turns into
one flat ``(prompt_ids, response_ids, loss_mask, rollout_log_probs)`` segment:
assistant-generated tokens get ``loss_mask=1``; rendered scaffolding / tool
results / user context between turns get ``0``.

This is a *re-render + repair* model rather than the cookbook's strict
append-only assert
(:mod:`training.utils.rl.rollout.assembler` /
:mod:`training.utils.rl.rollout.message`): a black-box agent re-renders the
whole conversation each turn, so a later prompt may not token-match an earlier
raw output (e.g. a renderer that strips ``<think>``).  When that happens we
mask/drop the unstitched tail instead of crashing.

The reusable turn-matching primitive (:func:`common_prefix_len`, plus the
new/append/wipe classifier) lives in
:mod:`training.utils.rl.rollout.turn_matching`; this module only does the
example's token stitching.  A run that never compacted or dispatched a
sub-agent yields one segment; :class:`TurnSegment` / :func:`merge_turn_segments`
let a run that does (compaction ``wipe`` + sub-agent excursions) fan out into
multiple disjoint segments.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from training.utils.rl.rollout.turn_matching import common_prefix_len

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """Exact token snapshot for one assistant generation."""

    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class TokenSegment:
    """One training segment assembled from an agent trajectory."""

    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]
    rollout_log_probs: list[float] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class TurnSegment:
    """A frozen group of turns before token-level merge."""

    turns: list[TurnRecord]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


def make_turn_segment(
    turns: list[TurnRecord],
    *,
    kind: str = "",
    metadata: dict[str, Any] | None = None,
) -> TurnSegment:
    """Freeze ``turns`` into a :class:`TurnSegment`, tagging ``segment_kind``."""
    frozen_turns = list(turns)
    segment_metadata = dict(metadata or {})
    if kind:
        segment_metadata.setdefault("segment_kind", kind)
    segment_metadata.setdefault("finish_reason", frozen_turns[-1].finish_reason if frozen_turns else "")
    return TurnSegment(turns=frozen_turns, metadata=segment_metadata)


def _output_log_probs(turn: TurnRecord) -> list[float]:
    if len(turn.output_log_probs) == len(turn.output_ids):
        return list(turn.output_log_probs)
    logger.warning(
        "[trajectory] turn logprob length mismatch; zeroing output logprobs (%d ids, %d logprobs)",
        len(turn.output_ids), len(turn.output_log_probs),
    )
    return [0.0] * len(turn.output_ids)


def merge_turns(turns: list[TurnRecord], *, metadata: dict[str, Any] | None = None) -> TokenSegment | None:
    """Replay turn records into one linear training segment.

    The first turn's prompt becomes the segment prompt. Later turn prompts are
    stitched against ``prompt + response_so_far``. Any new prompt suffix is
    non-model context and receives loss mask 0. If a later prompt diverges
    inside a previous model output, the retained prefix of that whole output
    turn is also masked out, because partial token matches are not a faithful
    training target for that turn.
    """
    if not turns:
        return None

    prompt_ids = list(turns[0].prompt_ids)
    response_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    output_spans: list[tuple[int, int]] = []

    for i, turn in enumerate(turns):
        if i > 0:
            if turn.prompt_ids[: len(prompt_ids)] != prompt_ids:
                logger.warning("[trajectory] merge prompt base changed; starting segment from drifted prompt")
                prompt_ids = list(turn.prompt_ids)
                response_ids = []
                loss_mask = []
                rollout_log_probs = []
                output_spans = []
            else:
                prompt_suffix = turn.prompt_ids[len(prompt_ids):]
                matched_len = common_prefix_len(response_ids, prompt_suffix)
                if matched_len < len(response_ids):
                    logger.warning(
                        "[trajectory] merge prefix drift; truncating %d unstitched response tokens",
                        len(response_ids) - matched_len,
                    )
                    for start, end in output_spans:
                        if start < matched_len < end:
                            loss_mask[start:matched_len] = [0] * (matched_len - start)
                            rollout_log_probs[start:matched_len] = [0.0] * (matched_len - start)
                    response_ids = response_ids[:matched_len]
                    loss_mask = loss_mask[:matched_len]
                    rollout_log_probs = rollout_log_probs[:matched_len]
                    output_spans = [
                        (start, min(end, matched_len)) for start, end in output_spans if start < matched_len
                    ]

                context_tail = prompt_suffix[matched_len:]
                response_ids.extend(context_tail)
                loss_mask.extend([0] * len(context_tail))
                rollout_log_probs.extend([0.0] * len(context_tail))

        output_start = len(response_ids)
        response_ids.extend(turn.output_ids)
        loss_mask.extend([1] * len(turn.output_ids))
        rollout_log_probs.extend(_output_log_probs(turn))
        output_spans.append((output_start, len(response_ids)))

    rollout_log_probs = [lp if m else 0.0 for lp, m in zip(rollout_log_probs, loss_mask, strict=True)]

    return TokenSegment(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        metadata=dict(metadata or {}),
    )


def merge_turn_segments(
    segments: list[TurnSegment],
    *,
    max_context_tokens: int = 0,
) -> list[TokenSegment]:
    """Merge frozen turn segments and drop empty or oversized outputs."""
    out: list[TokenSegment] = []
    for turn_segment in segments:
        token_segment = merge_turns(turn_segment.turns, metadata=turn_segment.metadata)
        if token_segment is None:
            continue
        total_tokens = len(token_segment.prompt_ids) + len(token_segment.response_ids)
        if token_segment.response_ids and (max_context_tokens <= 0 or total_tokens <= max_context_tokens):
            out.append(token_segment)
    return out
