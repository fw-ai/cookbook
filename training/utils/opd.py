"""Utilities for sampled-token on-policy distillation (OPD).

The Tinker server already has the primitive OPD needs: the built-in
``importance_sampling`` loss computes

    -exp(current_logprob - sampling_logprob) * advantage

per token.  Sampled-token OPD uses the teacher/student log-ratio as the dense
reward, so the per-token advantage is simply

    teacher_logprob - sampling_logprob

for response tokens.  These helpers build server-side datums that encode that
reward in ``loss_fn_inputs["advantages"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import tinker


@dataclass
class OPDPromptGroup:
    """Processed rollouts for one prompt in sampled-token OPD.

    This mirrors the small part of ``PromptGroup`` needed by the shared async
    runner, but names the OPD-specific tensors directly instead of treating
    teacher logprobs as RLHF reference logprobs.
    """

    data: list[tinker.Datum]
    teacher_logprobs: list[list[float]]
    sampling_logprobs: list[list[float]]
    prompt_len: int
    rewards: list[float]
    completion_lens: list[int] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)


@dataclass(frozen=True)
class OPDInputMetrics:
    """Summary metrics computed while building OPD server-side datums."""

    active_tokens: int
    sampled_reverse_kl_sum: float
    opd_advantage_sum: float
    teacher_nll_sum: float
    sampling_nll_sum: float

    def as_dict(self) -> dict[str, float]:
        denom = max(self.active_tokens, 1)
        return {
            "opd_active_tokens": float(self.active_tokens),
            "opd_sampled_reverse_kl": self.sampled_reverse_kl_sum / denom,
            "opd_advantage": self.opd_advantage_sum / denom,
            "opd_teacher_nll": self.teacher_nll_sum / denom,
            "opd_sampling_nll": self.sampling_nll_sum / denom,
        }


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


def combine_opd_prompt_groups(
    groups: Sequence[OPDPromptGroup],
) -> tuple[list[tinker.Datum], list[list[float]], list[int], list[list[float]]]:
    """Flatten OPD prompt groups into arrays for one server-side loss call."""
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
    teacher_nll_sum = 0.0
    sampling_nll_sum = 0.0

    for idx, datum in enumerate(data):
        target_data = datum.loss_fn_inputs.get("target_tokens")
        if target_data is None:
            raise ValueError(f"Datum {idx} is missing loss_fn_inputs['target_tokens'].")

        target_tokens = list(target_data.data)
        target_len = len(target_tokens)
        response_start = max(0, int(prompt_lens[idx]) - 1)
        teacher_lp = _pad_or_trim(teacher_logprobs[idx], target_len)
        sampling_lp = _pad_or_trim(sampling_logprobs[idx], target_len)
        loss_mask = _loss_mask_for_datum(datum, target_len)

        advantages = [0.0] * target_len
        for pos in range(response_start, target_len):
            if loss_mask[pos] <= 0.0:
                continue
            advantage = (teacher_lp[pos] - sampling_lp[pos]) * loss_scale * loss_mask[pos]
            advantages[pos] = advantage
            active_tokens += 1
            sampled_reverse_kl_sum += (sampling_lp[pos] - teacher_lp[pos]) * loss_mask[pos]
            opd_advantage_sum += advantage
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
        teacher_nll_sum=teacher_nll_sum,
        sampling_nll_sum=sampling_nll_sum,
    )
    return server_datums, metrics.as_dict()
