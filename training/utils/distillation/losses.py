"""Server-side datums for sampled-token on-policy distillation (OPD).

The student samples a response, the teacher scores the same tokens, and each
response token gets the dense reward

    teacher_logprob - sampling_logprob

as its advantage. Tinker's built-in ``importance_sampling`` loss computes
``-exp(current_logprob - sampling_logprob) * advantage`` per token, so at the
rollout policy its value and gradient equal the sampled reverse
``KL(student || teacher)``, and the importance ratio corrects for any
rollout/current-policy drift.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import tinker

from training.utils.distillation.sampling import (
    align_completion_logprobs,
    align_response_logprobs,
)

logger = logging.getLogger(__name__)


@dataclass
class OPDPromptGroup:
    """The scored rollouts for one prompt.

    ``teacher_logprobs`` and ``sampling_logprobs`` are aligned to each datum's
    ``target_tokens``; ``prompt_len`` is the student prompt length.
    """

    data: list[tinker.Datum]
    teacher_logprobs: list[list[float]]
    sampling_logprobs: list[list[float]]
    prompt_len: int


@dataclass(frozen=True)
class OPDInputMetrics:
    """Summary metrics computed while building OPD server-side datums."""

    active_tokens: int
    sampled_reverse_kl_sum: float
    opd_advantage_sum: float
    opd_abs_advantage_sum: float
    teacher_minus_sampling_logprob_sum: float
    abs_logprob_gap_sum: float
    teacher_nll_sum: float
    sampling_nll_sum: float

    def as_dict(self) -> dict[str, float]:
        denom = max(self.active_tokens, 1)
        sampled_reverse_kl = self.sampled_reverse_kl_sum / denom
        return {
            "opd_active_tokens": float(self.active_tokens),
            "opd_sampled_reverse_kl": sampled_reverse_kl,
            "opd_advantage": self.opd_advantage_sum / denom,
            "opd_abs_advantage": self.opd_abs_advantage_sum / denom,
            "opd_student_logprob_minus_teacher_logprob": sampled_reverse_kl,
            "opd_teacher_logprob_minus_student_logprob": (
                self.teacher_minus_sampling_logprob_sum / denom
            ),
            "opd_abs_logprob_gap": self.abs_logprob_gap_sum / denom,
            "opd_teacher_nll": self.teacher_nll_sum / denom,
            "opd_sampling_nll": self.sampling_nll_sum / denom,
        }


def build_opd_prompt_group(
    sampled: Sequence[Any],
    teacher_logprobs: Sequence[list[float] | None],
) -> OPDPromptGroup | None:
    """Pair the student's samples for one prompt with their teacher scores.

    Samples without a teacher score or with incomplete logprobs are skipped;
    returns ``None`` when nothing is left.
    """
    group = OPDPromptGroup(data=[], teacher_logprobs=[], sampling_logprobs=[], prompt_len=0)
    for sample, teacher_lp in zip(sampled, teacher_logprobs, strict=True):
        if teacher_lp is None:
            continue
        if not sample.sampling_logprobs:
            logger.warning("Skipping OPD sample without student sampling logprobs")
            continue
        target_tokens = list(sample.full_tokens[1:])
        aligned_sampling = align_completion_logprobs(
            list(sample.sampling_logprobs),
            prompt_len=sample.prompt_len,
            target_len=len(target_tokens),
            echoed=getattr(sample, "logprobs_echoed", False),
        )
        aligned_teacher = align_response_logprobs(
            teacher_lp,
            prompt_len=sample.prompt_len,
            target_len=len(target_tokens),
        )
        if aligned_sampling is None or aligned_teacher is None:
            logger.warning("Skipping OPD sample with incomplete student or teacher logprobs")
            continue

        group.data.append(
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(list(sample.full_tokens[:-1])),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=target_tokens,
                        dtype="int64",
                        shape=[len(target_tokens)],
                    ),
                },
            )
        )
        group.teacher_logprobs.append(aligned_teacher)
        group.sampling_logprobs.append(aligned_sampling)
        group.prompt_len = sample.prompt_len
    return group if group.data else None


def combine_opd_prompt_groups(
    groups: Sequence[OPDPromptGroup],
) -> tuple[list[tinker.Datum], list[list[float]], list[int], list[list[float]]]:
    """Flatten OPD prompt groups into ``(data, teacher_lp, prompt_lens, sampling_lp)``."""
    data: list[tinker.Datum] = []
    teacher_logprobs: list[list[float]] = []
    prompt_lens: list[int] = []
    sampling_logprobs: list[list[float]] = []

    for group_idx, group in enumerate(groups):
        n = len(group.data)
        if len(group.teacher_logprobs) != n:
            raise ValueError(
                f"Group {group_idx}: teacher_logprobs length ({len(group.teacher_logprobs)}) "
                f"does not match data length ({n})."
            )
        if len(group.sampling_logprobs) != n:
            raise ValueError(
                f"Group {group_idx}: sampling_logprobs length ({len(group.sampling_logprobs)}) "
                f"does not match data length ({n})."
            )

        data.extend(group.data)
        teacher_logprobs.extend(group.teacher_logprobs)
        prompt_lens.extend([group.prompt_len] * n)
        sampling_logprobs.extend(group.sampling_logprobs)

    return data, teacher_logprobs, prompt_lens, sampling_logprobs


def build_opd_datums(
    groups: Sequence[OPDPromptGroup],
    *,
    loss_scale: float = 1.0,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """``importance_sampling`` datums and ``opd_*`` metrics for one step's groups."""
    data, teacher_logprobs, prompt_lens, sampling_logprobs = combine_opd_prompt_groups(groups)
    return build_opd_server_datums(
        data,
        teacher_logprobs,
        sampling_logprobs,
        prompt_lens,
        loss_scale=loss_scale,
    )


def _pad_or_trim(values: Sequence[float], length: int) -> list[float]:
    result = [float(v) for v in values[:length]]
    if len(result) < length:
        result.extend([0.0] * (length - len(result)))
    return result


def _loss_mask_for_datum(datum: tinker.Datum, length: int) -> list[float]:
    mask = datum.loss_fn_inputs.get("loss_mask")
    if mask is None:
        return [1.0] * length
    return _pad_or_trim(mask.data, length)


def _require_lengths_match(name: str, values: Iterable[object], expected: int) -> list[object]:
    result = list(values)
    if len(result) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(result)}.")
    return result


def _require_active_logprobs(
    name: str,
    values: Sequence[float],
    active_positions: Sequence[int],
    *,
    datum_idx: int,
) -> None:
    if not active_positions:
        return
    required_len = max(active_positions) + 1
    if len(values) < required_len:
        raise ValueError(
            f"Datum {datum_idx}: {name} has length {len(values)}, "
            f"but active OPD tokens require at least {required_len} logprobs."
        )


def build_opd_server_datums(
    data: Sequence[tinker.Datum],
    teacher_logprobs: Sequence[Sequence[float]],
    sampling_logprobs: Sequence[Sequence[float]],
    prompt_lens: Sequence[int],
    *,
    loss_scale: float = 1.0,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Build datums for Tinker's server-side ``importance_sampling`` loss.

    Args:
        data: Training datums containing ``target_tokens`` and model inputs.
        teacher_logprobs: Teacher logprobs aligned to ``target_tokens``.
        sampling_logprobs: Student rollout logprobs aligned to ``target_tokens``.
        prompt_lens: Full prompt token count per datum.  The first response
            token has logprob index ``prompt_len - 1``.
        loss_scale: Optional scalar multiplier for the OPD dense reward.

    Returns:
        ``(server_datums, metrics)``.  ``server_datums`` contain exactly the
        fields required by Tinker's built-in RL losses:
        ``target_tokens``, ``logprobs`` (sampling logprobs), and
        ``advantages`` (teacher minus sampling on response tokens).
    """
    n = len(data)
    teacher_logprobs = _require_lengths_match("teacher_logprobs", teacher_logprobs, n)
    sampling_logprobs = _require_lengths_match("sampling_logprobs", sampling_logprobs, n)
    prompt_lens = _require_lengths_match("prompt_lens", prompt_lens, n)

    server_datums: list[tinker.Datum] = []
    active_tokens = 0
    sampled_reverse_kl_sum = 0.0
    opd_advantage_sum = 0.0
    opd_abs_advantage_sum = 0.0
    teacher_minus_sampling_logprob_sum = 0.0
    abs_logprob_gap_sum = 0.0
    teacher_nll_sum = 0.0
    sampling_nll_sum = 0.0

    for idx, datum in enumerate(data):
        target_data = datum.loss_fn_inputs.get("target_tokens")
        if target_data is None:
            raise ValueError(f"Datum {idx} is missing loss_fn_inputs['target_tokens'].")

        target_tokens = list(target_data.data)
        target_len = len(target_tokens)
        response_start = max(0, int(prompt_lens[idx]) - 1)
        raw_teacher_lp = [float(v) for v in teacher_logprobs[idx]]
        raw_sampling_lp = [float(v) for v in sampling_logprobs[idx]]
        loss_mask = _loss_mask_for_datum(datum, target_len)
        active_positions = [
            pos for pos in range(response_start, target_len) if loss_mask[pos] > 0.0
        ]
        _require_active_logprobs(
            "teacher_logprobs",
            raw_teacher_lp,
            active_positions,
            datum_idx=idx,
        )
        _require_active_logprobs(
            "sampling_logprobs",
            raw_sampling_lp,
            active_positions,
            datum_idx=idx,
        )
        teacher_lp = _pad_or_trim(raw_teacher_lp, target_len)
        sampling_lp = _pad_or_trim(raw_sampling_lp, target_len)

        advantages = [0.0] * target_len
        for pos in active_positions:
            logprob_gap = (teacher_lp[pos] - sampling_lp[pos]) * loss_mask[pos]
            advantage = logprob_gap * loss_scale
            advantages[pos] = advantage
            active_tokens += 1
            sampled_reverse_kl_sum += (sampling_lp[pos] - teacher_lp[pos]) * loss_mask[pos]
            opd_advantage_sum += advantage
            opd_abs_advantage_sum += abs(advantage)
            teacher_minus_sampling_logprob_sum += logprob_gap
            abs_logprob_gap_sum += abs(logprob_gap)
            teacher_nll_sum += -teacher_lp[pos] * loss_mask[pos]
            sampling_nll_sum += -sampling_lp[pos] * loss_mask[pos]

        server_datums.append(
            tinker.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=target_tokens,
                        dtype="int64",
                        shape=[target_len],
                    ),
                    "logprobs": tinker.TensorData(
                        data=sampling_lp,
                        dtype="float32",
                        shape=[target_len],
                    ),
                    "advantages": tinker.TensorData(
                        data=advantages,
                        dtype="float32",
                        shape=[target_len],
                    ),
                },
            )
        )

    metrics = OPDInputMetrics(
        active_tokens=active_tokens,
        sampled_reverse_kl_sum=sampled_reverse_kl_sum,
        opd_advantage_sum=opd_advantage_sum,
        opd_abs_advantage_sum=opd_abs_advantage_sum,
        teacher_minus_sampling_logprob_sum=teacher_minus_sampling_logprob_sum,
        abs_logprob_gap_sum=abs_logprob_gap_sum,
        teacher_nll_sum=teacher_nll_sum,
        sampling_nll_sum=sampling_nll_sum,
    )
    return server_datums, metrics.as_dict()
