"""Flat, mask-native rollout contract for the async RL loop.

The user's ``rollout_fn`` hands back one :class:`Rollout` per row of the
dataset.  A :class:`Rollout` is a list of :class:`RolloutSample` -- one per
completion in the group (N samples per row for a GRPO-style objective).
Each sample is three parallel lists (``tokens``, ``logprobs``,
``loss_mask``) plus a scalar reward.  Multi-turn rollouts flatten into the
same shape: turn boundaries are implicit in ``loss_mask`` transitions
(0 on prompts / env feedback / tool responses, 1 on assistant-generated
tokens).

This matches the user-facing contract used by AReaL, slime, and miles.

The adapter :func:`rollout_to_prompt_group` translates one :class:`Rollout`
into the trainer's :class:`PromptGroup`, emitting ``tinker.Datum`` objects
with a per-token ``loss_mask`` in ``loss_fn_inputs``.  The existing loss
kernels already honour this (see ``_get_loss_mask`` in
``training/utils/rl/common.py``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, List

import tinker

from training.utils.data import compute_advantages
from training.utils.rl.losses import PromptGroup

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
    versions: List[int] | None = None
    """Optional per-token deployment version.  Used by decoupled-IS
    corrections to re-weight stale tokens; ignored by today's losses.
    When provided, ``len(versions) == len(tokens)``."""
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
        if s.versions is not None and len(s.versions) != n:
            raise ValueError(
                f"Sample {i}: versions length {len(s.versions)} != "
                f"tokens length {n}."
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
) -> PromptGroup | None:
    """Pack one :class:`Rollout` into a :class:`PromptGroup`.

    Per-token ``loss_mask`` flows through ``tinker.Datum.loss_fn_inputs``
    so the existing kernels (``_get_loss_mask`` in
    ``training/utils/rl/common.py``) mask prompt / env / tool tokens
    correctly without any trainer-side changes.

    Returns ``None`` when the group has no samples; raises on structural
    issues (mismatched lengths, empty samples, all-zero loss mask).
    """
    _validate(rollout)

    rewards = [s.reward for s in rollout.samples]
    advantages = list(advantage_fn(list(rewards)))

    # Validate the computed advantages instead of pre-rejecting
    # singleton groups by sample count.  REINFORCE-style async RL is
    # a legitimate single-sample objective (``completions_per_prompt=1``);
    # users who run it supply a custom ``advantage_fn`` such as
    # ``lambda r: r`` (raw reward as advantage) for which N=1 is
    # well-defined.  An earlier ``len(samples) < 2`` precheck silently
    # dropped every such rollout and made the recipe make no training
    # progress despite advertising REINFORCE support.
    #
    # The protection that precheck was meant to provide is still
    # necessary: the default GRPO-style ``compute_advantages``
    # z-score-normalizes by ``torch.std(rewards)``, which is NaN on
    # a length-1 tensor; that NaN would flow into the loss kernel
    # and poison the training step.  Validating ``advantages`` after
    # the fn runs preserves that protection (NaN/inf outputs trigger
    # a drop) WITHOUT presuming what advantage_fn the caller picked.
    if any(not math.isfinite(a) for a in advantages):
        logger.warning(
            "rollout_to_prompt_group: dropping rollout (N=%d) — "
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

        policy_data.append(tinker.Datum(
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

    # ``prompt_len`` is a single scalar on PromptGroup for back-compat
    # with consumers that don't yet read ``prompt_lens``; we populate it
    # with the first sample's prompt boundary.  The authoritative
    # per-sample list is ``prompt_lens``, and ``combine_prompt_groups``
    # prefers it when set so heterogeneous rollouts (multi-turn / tool
    # branches whose samples have different prefix lengths) slice each
    # sample at its own prompt boundary.
    return PromptGroup(
        data=policy_data,
        ref_data=reference_data,
        advantages=list(advantages),
        ref_logprobs=None,
        prompt_len=per_sample_prompt_lens[0] if per_sample_prompt_lens else 0,
        rewards=rewards,
        inf_logprobs=inf_logprobs_aligned,
        completion_lens=completion_lens,
        truncated=truncated,
        prompt=None,
        completions=None,
        row_meta=dict(rollout.row_meta) if rollout.row_meta else None,
        prompt_lens=per_sample_prompt_lens,
    )
