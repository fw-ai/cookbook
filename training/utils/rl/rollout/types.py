"""Flat, mask-native rollout run contract.

The user's ``rollout_fn(sample_prompt)`` hands back one
:class:`RolloutRun` per call -- one trajectory.  Each run contains one or
more :class:`RolloutSample` segments.  A segment is three parallel lists
(``tokens``, ``logprobs``, ``loss_mask``) plus the run reward.  Multi-turn
rollouts can either flatten into one segment or emit one segment per
trainable assistant span.  The trajectory-level reward is shared by every
segment in the same run.

This matches the user-facing contract used by AReaL and slime.  Group
assembly (collecting N rollout runs per row and computing GRPO-style
advantages) happens framework-side via :class:`GroupAssembler` -- see
:mod:`training.utils.rl.rollout.group_assembler`.

:class:`Rollout` is the internal group representation that the
:func:`rollout_to_prompt_group` adapter consumes after the assembler
joins N runs by row.  The adapter computes advantages over run rewards,
broadcasts each run advantage to that run's segments, and emits
``tinker.Datum`` objects with a per-token ``loss_mask`` in
``loss_fn_inputs``; the existing loss kernels already honour this (see
``_get_loss_mask`` in ``training/utils/rl/common.py``).
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
from training.utils.supervised import build_multimodal_policy_datum

logger = logging.getLogger(__name__)

__all__ = [
    "RolloutSample",
    "RolloutRun",
    "Rollout",
    "rollout_to_prompt_group",
]


@dataclass
class RolloutSample:
    """One trainable segment's flat, trainer-ready data.

    The three parallel lists MUST have identical length.  ``loss_mask``
    is ``1`` on assistant-generated positions (trained on) and ``0``
    everywhere else (prompt, user messages, tool responses, env feedback
    injected between turns).  ``logprobs`` carries per-token
    ``rollout_logprobs`` after sampling temperature/masks, aligned with
    ``tokens``; use ``0.0`` on non-generated positions since they carry no
    training signal.
    """

    tokens: List[int]
    logprobs: List[float]
    loss_mask: List[int]
    reward: float
    prompt_model_input: tinker.ModelInput | None = None
    """When set, :func:`rollout_to_prompt_group` builds the trainer datum via
    :func:`training.utils.supervised.build_multimodal_policy_datum` (chunked
    ``model_input`` with image pointers).  ``tokens`` / ``logprobs`` /
    ``loss_mask`` are text-only parallels for metrics and logprob alignment."""
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
    raw_logprobs: List[float] | None = None
    """Optional raw model logprobs aligned with ``tokens`` for observability.

    ``logprobs`` remains the rollout/sampling logprob source used by loss
    ratios and TIS. When present, ``raw_logprobs`` is packed into
    ``PromptGroup.raw_inf_logprobs`` for optional train/inference drift metrics.
    It never replaces behavior logprobs in the loss or TIS.
    """


@dataclass
class RolloutRun:
    """One rollout function invocation, also called one trajectory.

    ``segments`` contains one or more trainable spans from the same
    trajectory.  All segments in a run must carry the same scalar reward;
    the adapter computes one advantage for the run and broadcasts it to
    every segment before training.
    """

    segments: List[RolloutSample]
    run_id: str | None = None
    metadata: dict | None = None


@dataclass
class Rollout:
    """One row's worth of rollout runs (one GRPO group)."""

    runs: List[RolloutRun]
    row_meta: dict | None = None


def _completion_tokens_from_sample(sample: RolloutSample) -> List[int]:
    """Return assistant tokens from a flat segment (``loss_mask==1``)."""
    return [int(t) for t, m in zip(sample.tokens, sample.loss_mask) if m > 0]


def _completion_logprobs_from_sample(sample: RolloutSample) -> List[float]:
    """Return per-completion ``rollout_logprobs`` (``loss_mask==1``)."""
    return [float(lp) for lp, m in zip(sample.logprobs, sample.loss_mask) if m > 0]


def _completion_raw_logprobs_from_sample(sample: RolloutSample) -> List[float] | None:
    """Return per-completion raw logprobs when present."""
    if sample.raw_logprobs is None:
        return None
    return [float(lp) for lp, m in zip(sample.raw_logprobs, sample.loss_mask) if m > 0]


def _align_multimodal_inf_logprobs(
    completion_lps: List[float],
    shifted_weights: List[float],
) -> List[float]:
    """Map completion-only inference logprobs into expanded target space.

    Canonical multimodal datums use one shared shifted coordinate space for
    ``target_tokens``, weights, forward logprobs, and built-in loss inputs.
    Samplers return completion-only logprobs, so scatter those values onto the
    trained positions and fill prompt/image positions with zeros. GRPO/TIS can
    then slice the result with ``prompt_lens`` without changing coordinates.
    """
    active_indices = [i for i, w in enumerate(shifted_weights) if w > 0]
    if len(completion_lps) != len(active_indices):
        raise ValueError(
            "multimodal inference logprobs misaligned with datum weights "
            f"(got {len(completion_lps)} completion logprobs, "
            f"expected {len(active_indices)} trained positions)."
        )
    aligned = [0.0] * len(shifted_weights)
    for idx, lp in zip(active_indices, completion_lps):
        aligned[idx] = lp
    return aligned


def _validate(rollout: Rollout) -> None:
    if not rollout.runs:
        raise ValueError("Rollout.runs is empty")
    for run_index, run in enumerate(rollout.runs):
        if not run.segments:
            raise ValueError(f"Run {run_index}: segments is empty")
        _run_reward(run, run_index)
        for segment_index, segment in enumerate(run.segments):
            _validate_segment(run_index, segment_index, segment)


def _run_reward(run: RolloutRun, run_index: int) -> float:
    reward = float(run.segments[0].reward)
    for segment_index, segment in enumerate(run.segments[1:], start=1):
        if float(segment.reward) != reward:
            raise ValueError(
                f"Run {run_index}: segment {segment_index} reward "
                f"({segment.reward}) differs from segment 0 reward ({reward}). "
                "A rollout run has one trajectory-level reward; duplicate it "
                "onto every segment.",
            )
    return reward


def _validate_segment(
    run_index: int,
    segment_index: int,
    segment: RolloutSample,
) -> None:
    def _validate_optional_logprobs(values: List[float] | None, *, n: int) -> None:
        if values is not None and len(values) != n:
            raise ValueError(
                f"Run {run_index} segment {segment_index}: "
                "raw_logprobs length mismatch "
                f"({len(values)} / {n}). When set, raw_logprobs must align "
                "with tokens."
            )

    if segment.prompt_model_input is not None:
        n = len(segment.tokens)
        if len(segment.logprobs) != n or len(segment.loss_mask) != n:
            raise ValueError(
                f"Run {run_index} segment {segment_index}: multimodal "
                "tokens/logprobs/loss_mask mismatch "
                f"({n} / {len(segment.logprobs)} / {len(segment.loss_mask)})."
            )
        _validate_optional_logprobs(segment.raw_logprobs, n=n)
        if n < 2:
            raise ValueError(
                f"Run {run_index} segment {segment_index}: tokens must have "
                "length >= 2.",
            )
        if not any(m > 0 for m in segment.loss_mask):
            raise ValueError(
                f"Run {run_index} segment {segment_index}: loss_mask is all "
                "zeros for multimodal segment.",
            )
        if not _completion_tokens_from_sample(segment):
            raise ValueError(
                f"Run {run_index} segment {segment_index}: multimodal segment "
                "has no completion tokens.",
            )
        return

    n = len(segment.tokens)
    if len(segment.logprobs) != n or len(segment.loss_mask) != n:
        raise ValueError(
            f"Run {run_index} segment {segment_index}: "
            "tokens/logprobs/loss_mask length mismatch "
            f"({n} / {len(segment.logprobs)} / {len(segment.loss_mask)}). "
            "All three lists must be the same length.",
        )
    _validate_optional_logprobs(segment.raw_logprobs, n=n)
    if n < 2:
        raise ValueError(
            f"Run {run_index} segment {segment_index}: tokens must have "
            "length >= 2.",
        )
    if not any(mask_value > 0 for mask_value in segment.loss_mask):
        raise ValueError(
            f"Run {run_index} segment {segment_index}: loss_mask is all zeros "
            "-- no tokens would be trained on. Set loss_mask=1 on "
            "assistant-generated positions.",
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

    Returns ``None`` when the group has no runs; raises on structural
    issues (mismatched lengths, empty runs, all-zero loss mask).
    """
    _validate(rollout)

    rewards = [
        _run_reward(run, run_index)
        for run_index, run in enumerate(rollout.runs)
    ]
    advantages = list(advantage_fn(list(rewards)))
    if len(advantages) != len(rewards):
        raise ValueError(
            "advantage_fn must return one advantage per rollout run "
            f"(got {len(advantages)} for {len(rewards)} rewards).",
        )

    # Drop on non-finite advantages (e.g., GRPO z-score on a length-1 group).
    # Don't precheck on run count -- REINFORCE (cpp=1, lambda r: r) is valid.
    if any(not math.isfinite(a) for a in advantages):
        logger.warning(
            "rollout_to_prompt_group: dropping rollout (N=%d) -- "
            "advantage_fn produced non-finite advantages %r.  This "
            "typically happens when the default GRPO-style "
            "``compute_advantages`` z-score normalizer runs on a "
            "single-run group (std of a length-1 tensor is "
            "undefined).  For REINFORCE-style runs with "
            "completions_per_prompt=1, pass a single-run-safe "
            "advantage_fn (e.g. ``lambda r: r``).",
            len(rollout.runs), advantages,
        )
        return None

    policy_data: List[tinker.Datum] = []
    reference_data: List[tinker.Datum] = []
    inf_logprobs_aligned: List[List[float]] = []
    raw_inf_logprobs_aligned: List[List[float]] = []
    completion_lens: List[int] = []
    truncated: List[bool] = []
    per_sample_prompt_lens: List[int] = []

    # Datum predicts tokens[1:] from tokens[:-1]; shift both loss_mask and
    # logprobs to match target positions.
    segment_advantages: List[float] = []
    for run, run_advantage in zip(rollout.runs, advantages):
        for s in run.segments:
            segment_advantages.append(run_advantage)

            if s.prompt_model_input is not None:
                completion_tokens = _completion_tokens_from_sample(s)
                datum = build_multimodal_policy_datum(
                    s.prompt_model_input,
                    completion_tokens,
                )
                reference_model_input = datum.model_input
                target_tokens = [
                    int(x) for x in datum.loss_fn_inputs["target_tokens"].data
                ]
                target_len = len(target_tokens)
                target_mask = [
                    float(x) for x in datum.loss_fn_inputs["weights"].data
                ]
                if not (
                    target_len == len(target_mask) == datum.model_input.length
                ):
                    raise ValueError(
                        "multimodal datum must use canonical expanded coordinates "
                        "for model_input, target_tokens, and weights "
                        f"({datum.model_input.length} / {target_len} / "
                        f"{len(target_mask)})."
                    )
                target_logprobs = _align_multimodal_inf_logprobs(
                    _completion_logprobs_from_sample(s), target_mask,
                )
                completion_raw_logprobs = _completion_raw_logprobs_from_sample(s)
                target_raw_logprobs = (
                    _align_multimodal_inf_logprobs(completion_raw_logprobs, target_mask)
                    if completion_raw_logprobs is not None
                    else []
                )

                # ``run_loss_loop`` uses ``response_start = prompt_len - 1`` on
                # shifted datum weights.  The text path records the first active
                # index in the *unshifted* loss_mask; map shifted weights the
                # same way (+1) so multimodal GRPO/TIS slices align.
                shifted_first_active = next(
                    (i for i, w in enumerate(target_mask) if w > 0),
                    0,
                )
                sample_prompt_len = shifted_first_active + 1
                per_sample_prompt_lens.append(sample_prompt_len)

                if s.routing_matrices is not None:
                    rm = build_r3_routing_matrices(
                        s.routing_matrices,
                        prompt_len=sample_prompt_len,
                        model_input_len=len(target_mask),
                        completion_only=router_replay_completion_only,
                    )
                    if rm is not None:
                        datum = tinker.Datum(
                            model_input=datum.model_input.model_copy(
                                update={"routing_matrices": rm}
                            ),
                            loss_fn_inputs=datum.loss_fn_inputs,
                        )

                policy_data.append(datum)

                if with_reference:
                    mask_len = len(target_mask)
                    reference_data.append(tinker.Datum(
                        model_input=reference_model_input,
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData(
                                data=target_tokens,
                                dtype="int64",
                                shape=[target_len],
                            ),
                            "loss_mask": tinker.TensorData(
                                data=target_mask,
                                dtype="float32",
                                shape=[mask_len],
                            ),
                        },
                    ))

                inf_logprobs_aligned.append(target_logprobs)
                raw_inf_logprobs_aligned.append(target_raw_logprobs)
                completion_lens.append(sum(1 for w in target_mask if w > 0))
                truncated.append(s.finish_reason == "length")
                continue

            n = len(s.tokens)
            target_len = n - 1

            target_tokens = s.tokens[1:]
            target_mask = s.loss_mask[1:]
            target_logprobs = s.logprobs[1:]
            target_raw_logprobs = s.raw_logprobs[1:] if s.raw_logprobs is not None else []

            # Per-segment prompt boundary: index of the first assistant
            # (loss_mask=1) token.  Heterogeneous rollouts (multi-turn,
            # tool branches) can have different prefix lengths per segment,
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
                model_input=tinker.ModelInput.from_ints(
                    s.tokens[:-1], routing_matrices=rm,
                ),
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
            raw_inf_logprobs_aligned.append(target_raw_logprobs)
            completion_lens.append(sum(1 for m in s.loss_mask if m > 0))
            truncated.append(s.finish_reason == "length")

    # ``prompt_len`` is the legacy scalar (back-compat); ``prompt_lens``
    # is the authoritative per-segment list for heterogeneous rollouts.
    return PromptGroup(
        data=policy_data,
        ref_data=reference_data,
        advantages=segment_advantages,
        ref_logprobs=None,
        prompt_len=per_sample_prompt_lens[0],
        rewards=rewards,
        inf_logprobs=inf_logprobs_aligned,
        raw_inf_logprobs=raw_inf_logprobs_aligned,
        completion_lens=completion_lens,
        truncated=truncated,
        prompt=None,
        completions=None,
        row_meta=dict(rollout.row_meta) if rollout.row_meta else None,
        prompt_lens=per_sample_prompt_lens,
    )
