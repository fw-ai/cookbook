"""Flat, mask-native rollout sample contract.

The user's ``rollout_fn(sample_prompt)`` hands back one
:class:`RolloutSample` per call -- one trajectory.  Each sample is three parallel lists
(``tokens``, ``logprobs``, ``loss_mask``) plus a scalar reward.  Multi-turn
rollouts flatten into the same shape: turn boundaries are implicit in
``loss_mask`` transitions (0 on prompts / env feedback / tool responses,
1 on assistant-generated tokens).

This matches the user-facing contract used by AReaL and slime.  Group
assembly (collecting N samples per row and computing GRPO-style
advantages) happens framework-side via :class:`GroupAssembler` -- see
:mod:`training.utils.rl.rollout.group_assembler`.

:class:`Rollout` is the internal group representation that the
:func:`rollout_to_prompt_group` adapter consumes after the assembler
joins N samples by row.  The adapter emits ``tinker.Datum`` objects
with a per-token ``loss_mask`` in ``loss_fn_inputs``; the existing
loss kernels already honour this (see ``_get_loss_mask`` in
``training/utils/rl/common.py``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List

import tinker

from training.utils.data import compute_advantages
from training.utils.rl.losses import PromptGroup
from training.utils.rl.router_replay import build_r3_routing_matrices

logger = logging.getLogger(__name__)

__all__ = [
    "RolloutSample",
    "Rollout",
    "rollout_to_prompt_group",
]


@dataclass
class RolloutSample:
    """One completion's flat, trainer-ready data.

    The three parallel lists MUST have identical length.  ``loss_mask``
    is ``1`` on assistant-generated positions (trained on) and ``0``
    everywhere else (prompt, user messages, tool responses, env feedback
    injected between turns).  ``logprobs`` is the per-token inference
    logprob aligned with ``tokens``; use ``0.0`` on non-generated
    positions since they carry no training signal.
    """

    tokens: List[int]
    logprobs: List[float]
    loss_mask: List[int]
    reward: float
    routing_matrices: List[str] | None = None
    """Optional per-token MoE routing matrices captured from the inference
    deployment (R3 / Router Replay).  When set, the adapter aligns them to
    ``tokens[:-1]`` via :func:`build_r3_routing_matrices` and threads them
    through ``tinker.ModelInput.from_ints(routing_matrices=...)`` so the
    training step replays the same expert routing decisions made at
    inference time.  Length should match ``tokens`` (echo mode) or the
    completion suffix (legacy mode); the alignment helper handles both."""
    finish_reason: str = "stop"
    text: str = ""
    """Decoded assistant output.  For logging only; not consumed by the
    adapter or the trainer."""


@dataclass
class Rollout:
    """One row's worth of completions (one GRPO group)."""

    samples: List[RolloutSample]
    row_meta: dict | None = None


def _validate(rollout: Rollout) -> None:
    if not rollout.samples:
        raise ValueError("Rollout.samples is empty")
    for i, s in enumerate(rollout.samples):
        n = len(s.tokens)
        if len(s.logprobs) != n or len(s.loss_mask) != n:
            raise ValueError(
                f"Sample {i}: tokens/logprobs/loss_mask length mismatch "
                f"({n} / {len(s.logprobs)} / {len(s.loss_mask)}). All three "
                "lists must be the same length.",
            )
        if n < 2:
            raise ValueError(f"Sample {i}: tokens must have length >= 2.")
        if not any(m > 0 for m in s.loss_mask):
            raise ValueError(
                f"Sample {i}: loss_mask is all zeros -- no tokens would be "
                "trained on.  Set loss_mask=1 on assistant-generated "
                "positions.",
            )


def rollout_to_prompt_group(
    rollout: Rollout,
    *,
    advantage_fn: Callable[[List[float]], List[float]] = compute_advantages,
    with_reference: bool = False,
    router_replay_completion_only: bool = False,
) -> PromptGroup | None:
    """Pack one :class:`Rollout` into a :class:`PromptGroup`.

    Per-token ``loss_mask`` flows through ``tinker.Datum.loss_fn_inputs``
    so the existing kernels (``_get_loss_mask`` in
    ``training/utils/rl/common.py``) mask prompt / env / tool tokens
    correctly without any trainer-side changes.

    When a sample carries ``routing_matrices``, they are aligned to the
    model_input shape via :func:`build_r3_routing_matrices` and passed
    through ``tinker.ModelInput.from_ints(routing_matrices=...)`` so MoE
    expert routing is replayed at training time (R3).
    ``router_replay_completion_only=True`` zeroes out prompt-position
    routing matrices so only completion-token routing is replayed (the
    server picks its own routing for prompt tokens).  Reference-side
    datums never carry routing matrices.

    Returns ``None`` when the group has no samples; raises on structural
    issues (mismatched lengths, empty samples, all-zero loss mask).
    """
    _validate(rollout)

    rewards = [s.reward for s in rollout.samples]
    advantages = list(advantage_fn(list(rewards)))

    # Drop on non-finite advantages (e.g., GRPO z-score on a length-1 group).
    # Don't precheck on sample count -- REINFORCE (cpp=1, lambda r: r) is valid.
    if any(not math.isfinite(a) for a in advantages):
        logger.warning(
            "rollout_to_prompt_group: dropping rollout (N=%d) -- "
            "advantage_fn produced non-finite advantages %r.  This "
            "typically happens when the default GRPO-style "
            "``compute_advantages`` z-score normalizer runs on a "
            "single-sample group (std of a length-1 tensor is "
            "undefined).  For REINFORCE-style runs with "
            "completions_per_prompt=1, pass a single-sample-safe "
            "advantage_fn (e.g. ``lambda r: r``).",
            len(rollout.samples), advantages,
        )
        return None

    policy_data: List[tinker.Datum] = []
    reference_data: List[tinker.Datum] = []
    inf_logprobs_aligned: List[List[float]] = []
    completion_lens: List[int] = []
    truncated: List[bool] = []
    per_sample_prompt_lens: List[int] = []

    # Datum predicts tokens[1:] from tokens[:-1]; shift both loss_mask and
    # logprobs to match target positions.
    for s in rollout.samples:
        n = len(s.tokens)
        target_len = n - 1

        target_tokens = s.tokens[1:]
        target_mask = s.loss_mask[1:]
        target_logprobs = s.logprobs[1:]

        # Per-sample prompt boundary: index of the first assistant
        # (loss_mask=1) token.  Heterogeneous rollouts (multi-turn,
        # tool branches) can have different prefix lengths per sample,
        # so the single ``PromptGroup.prompt_len`` is wrong for them.
        sample_prompt_len = next(
            (i for i, m in enumerate(s.loss_mask) if m > 0),
            0,
        )
        per_sample_prompt_lens.append(sample_prompt_len)

        rm = None
        if s.routing_matrices is not None:
            rm = build_r3_routing_matrices(
                s.routing_matrices,
                prompt_len=sample_prompt_len,
                model_input_len=target_len,
                completion_only=router_replay_completion_only,
            )

        policy_data.append(tinker.Datum(
            model_input=tinker.ModelInput.from_ints(s.tokens[:-1], routing_matrices=rm),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=target_tokens, dtype="int64", shape=[target_len],
                ),
                "weights": tinker.TensorData(
                    data=target_mask, dtype="int64", shape=[target_len],
                ),
            },
        ))

        if with_reference:
            reference_data.append(tinker.Datum(
                model_input=tinker.ModelInput.from_ints(s.tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=target_tokens, dtype="int64", shape=[target_len],
                    ),
                    "loss_mask": tinker.TensorData(
                        data=target_mask, dtype="int64", shape=[target_len],
                    ),
                },
            ))

        inf_logprobs_aligned.append(target_logprobs)
        completion_lens.append(sum(1 for m in s.loss_mask if m > 0))
        truncated.append(s.finish_reason == "length")

    # ``prompt_len`` is the legacy scalar (back-compat); ``prompt_lens``
    # is the authoritative per-sample list for heterogeneous rollouts.
    return PromptGroup(
        data=policy_data,
        ref_data=reference_data,
        advantages=list(advantages),
        ref_logprobs=None,
        prompt_len=per_sample_prompt_lens[0],
        rewards=rewards,
        inf_logprobs=inf_logprobs_aligned,
        completion_lens=completion_lens,
        truncated=truncated,
        prompt=None,
        completions=None,
        row_meta=dict(rollout.row_meta) if rollout.row_meta else None,
        prompt_lens=per_sample_prompt_lens,
    )
