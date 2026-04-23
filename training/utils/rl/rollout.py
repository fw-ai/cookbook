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

from dataclasses import dataclass, field
from typing import Callable, List

import tinker

from training.utils.data import compute_advantages
from training.utils.rl.losses import PromptGroup

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
    advantages = advantage_fn(list(rewards))

    policy_data: List[tinker.Datum] = []
    reference_data: List[tinker.Datum] = []
    inf_logprobs_aligned: List[List[float]] = []
    completion_lens: List[int] = []
    truncated: List[bool] = []

    # Datum predicts tokens[1:] from tokens[:-1]; shift both loss_mask and
    # logprobs to match target positions.
    for s in rollout.samples:
        n = len(s.tokens)
        target_len = n - 1

        target_tokens = s.tokens[1:]
        target_mask = s.loss_mask[1:]
        target_logprobs = s.logprobs[1:]

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

    # ``prompt_len`` is a single scalar on PromptGroup today.  We keep it
    # for back-compat but the per-token loss_mask is what the loss kernels
    # actually consume: the mask correctly identifies which positions to
    # train on regardless of how many prompt / env tokens appear.
    first_assistant = next(
        (i for i, m in enumerate(rollout.samples[0].loss_mask) if m > 0),
        0,
    )

    return PromptGroup(
        data=policy_data,
        ref_data=reference_data,
        advantages=list(advantages),
        ref_logprobs=None,
        prompt_len=first_assistant,
        rewards=rewards,
        inf_logprobs=inf_logprobs_aligned,
        completion_lens=completion_lens,
        truncated=truncated,
        prompt=None,
        completions=None,
        row_meta=dict(rollout.row_meta) if rollout.row_meta else None,
    )
