"""Typed :class:`Trajectory` dataclass and adapter for the async RL loop.

Design contract for ``trajectory_to_prompt_group``:

1. **No reward computation.** ``trajectory.rewards`` MUST be populated by the
   caller; the adapter raises if it is ``None`` or length-mismatched.
2. **No logprob alignment.** Each segment's ``inference_logprobs`` MUST have
   the same length as its ``tokens``.  The adapter raises on mismatch rather
   than silently padding.
3. **Pure data → data.** No I/O, no sampler calls, no graders.

Multi-turn rollouts are represented as one outer list per completion and an
inner list of :class:`CompletionSegment` (one segment per generation call).
Segments carry a per-call ``version`` so a future decoupled-IS correction can
re-weight stale turns.  :class:`PromptGroup` itself is a flat per-completion
dataclass today, so the adapter drops per-segment metadata when packing —
the raw per-segment information stays on the :class:`Trajectory` for logging
and later-layer consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import tinker

from training.utils.data import compute_advantages
from training.utils.rl.losses import PromptGroup

__all__ = [
    "CompletionSegment",
    "Trajectory",
    "trajectory_to_prompt_group",
]


@dataclass
class CompletionSegment:
    """One generation call's output.

    Single-turn rollout: one segment per completion.
    Multi-turn / tool-using rollout: a list of segments per completion; any
    intermediate messages (tool results, user replies) live outside this type
    and are stitched back together by the caller before rendering.
    """

    tokens: List[int]
    """Completion tokens only (no prompt)."""

    inference_logprobs: List[float]
    """Per-token logprobs aligned with ``tokens``.  Must be len(tokens)."""

    version: int
    """Deployment version at the time this segment was generated."""

    finish_reason: str = "unknown"
    text: str = ""
    routing_matrices: list | None = None
    """Optional MoE routing matrices for router replay."""


@dataclass
class Trajectory:
    """Output of one rollout call.  Pure data; consumed by the adapter."""

    prompt_tokens: List[int]
    """Prompt token ids shared by all completions in this trajectory."""

    completions: List[List[CompletionSegment]]
    """Outer: one entry per completion.  Inner: one entry per generation call."""

    rewards: List[float]
    """One reward per completion.  Required -- the adapter raises if lengths
    disagree with ``completions``."""

    prompt_messages: list[dict] | None = None
    """Original chat messages (for trajectory logging)."""

    row_meta: dict | None = None
    """Dataset row metadata (for trajectory logging)."""


def _validate(traj: Trajectory) -> None:
    if traj.rewards is None:
        raise ValueError(
            "Trajectory.rewards must be populated before calling "
            "trajectory_to_prompt_group. The adapter does not compute rewards."
        )
    if len(traj.rewards) != len(traj.completions):
        raise ValueError(
            f"Trajectory.rewards length ({len(traj.rewards)}) does not match "
            f"completions length ({len(traj.completions)})."
        )
    for ci, segments in enumerate(traj.completions):
        if not segments:
            raise ValueError(f"Completion {ci} has no segments.")
        for si, seg in enumerate(segments):
            if len(seg.inference_logprobs) != len(seg.tokens):
                raise ValueError(
                    f"Completion {ci} segment {si}: "
                    f"len(inference_logprobs)={len(seg.inference_logprobs)} "
                    f"!= len(tokens)={len(seg.tokens)}. The adapter does not "
                    "align logprobs; callers must supply them pre-aligned."
                )


def trajectory_to_prompt_group(
    traj: Trajectory,
    *,
    advantage_fn: Callable[[List[float]], List[float]] = compute_advantages,
    with_reference: bool = False,
    persist_raw: bool = False,
) -> PromptGroup | None:
    """Pack one :class:`Trajectory` into a :class:`PromptGroup`.

    Fails loud on missing rewards or unaligned logprobs (see module docstring).
    Returns ``None`` if no completion produced at least one usable token.
    """
    _validate(traj)

    prompt_ids = list(traj.prompt_tokens)
    prompt_len = len(prompt_ids)

    policy_data: List[tinker.Datum] = []
    reference_data: List[tinker.Datum] = []
    adv_filtered: List[float] = []
    rewards_filtered: List[float] = []
    inf_logprobs_aligned: List[List[float]] = []
    completion_lens: List[int] = []
    truncated: List[bool] = []
    completion_texts: List[str] = []

    advantages = advantage_fn(list(traj.rewards))

    for ci, segments in enumerate(traj.completions):
        comp_tokens: List[int] = []
        comp_lp: List[float] = []
        for seg in segments:
            comp_tokens.extend(seg.tokens)
            comp_lp.extend(seg.inference_logprobs)

        if not comp_tokens:
            continue

        full_tokens = prompt_ids + comp_tokens
        if len(full_tokens) < 2:
            continue
        model_input_len = len(full_tokens) - 1

        policy_datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(full_tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=full_tokens[1:], dtype="int64", shape=[model_input_len],
                ),
            },
        )
        policy_data.append(policy_datum)

        if with_reference:
            reference_data.append(
                tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(full_tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData(
                            data=full_tokens[1:], dtype="int64", shape=[model_input_len],
                        ),
                    },
                )
            )

        # Align inf_logprobs to model_input_len: left-pad with zeros for the
        # prompt positions that did not generate a token.  This is NOT
        # adjusting the user-provided per-token values -- it is placing them
        # at the correct training-step indices.  The user-facing contract
        # (per-token logprobs aligned with per-token tokens) is preserved.
        response_start = max(0, prompt_len - 1)
        aligned = [0.0] * response_start + list(comp_lp)
        if len(aligned) < model_input_len:
            aligned = aligned + [0.0] * (model_input_len - len(aligned))
        elif len(aligned) > model_input_len:
            aligned = aligned[:model_input_len]

        inf_logprobs_aligned.append(aligned)
        adv_filtered.append(advantages[ci])
        rewards_filtered.append(traj.rewards[ci])
        completion_lens.append(len(comp_tokens))
        truncated.append(segments[-1].finish_reason == "length")
        completion_texts.append("".join(seg.text for seg in segments))

    if not policy_data:
        return None

    return PromptGroup(
        data=policy_data,
        ref_data=reference_data,
        advantages=adv_filtered,
        ref_logprobs=None,
        prompt_len=prompt_len,
        rewards=rewards_filtered,
        inf_logprobs=inf_logprobs_aligned,
        completion_lens=completion_lens,
        truncated=truncated,
        prompt=traj.prompt_messages if persist_raw else None,
        completions=completion_texts if persist_raw else None,
        row_meta=dict(traj.row_meta) if persist_raw and traj.row_meta else None,
    )
