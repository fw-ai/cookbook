"""Distillation utilities for sampled reverse KL and top-K SDFT helpers.

Sampled reverse KL uses student rollouts and teacher logprobs on the sampled
tokens. The backend estimator is Tinker's built-in ``importance_sampling`` loss,
which computes

    -exp(current_logprob - sampling_logprob) * advantage

per token. The per-token advantage is the sampled reverse-KL reward

    teacher_logprob - sampling_logprob

for response tokens. These helpers build server-side datums that encode that
reward in ``loss_fn_inputs["advantages"]``.

Top-K SDFT forward KL uses teacher top-K distributions returned by the teacher
inference deployment and the built-in ``cross_entropy`` path over ``[N, K]``
soft targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

import tinker

from training.utils.distillation.sampling import TopKDist


class DistillMode(str, Enum):
    """Distillation objective mode."""

    SAMPLED_REVERSE_KL = "sampled_reverse_kl"
    TOPK_FORWARD_KL = "topk_forward_kl"


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
    teacher_topk: list[list[TopKDist]] = field(default_factory=list)


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
        opd_advantage = self.opd_advantage_sum / denom
        abs_logprob_gap = self.opd_abs_advantage_sum / denom
        teacher_minus_sampling_logprob = self.teacher_minus_sampling_logprob_sum / denom
        raw_abs_logprob_gap = self.abs_logprob_gap_sum / denom
        return {
            "opd_active_tokens": float(self.active_tokens),
            "opd_sampled_reverse_kl": sampled_reverse_kl,
            "opd_advantage": opd_advantage,
            "opd_abs_advantage": abs_logprob_gap,
            "opd_student_logprob_minus_teacher_logprob": sampled_reverse_kl,
            "opd_teacher_logprob_minus_student_logprob": teacher_minus_sampling_logprob,
            "opd_abs_logprob_gap": raw_abs_logprob_gap,
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


def combine_topk_prompt_groups(
    groups: Sequence[OPDPromptGroup],
) -> tuple[list[tinker.Datum], list[list[TopKDist]], list[int]]:
    """Flatten prompt groups for inference-source top-K SDFT training."""
    student_data: list[tinker.Datum] = []
    teacher_topk: list[list[TopKDist]] = []
    student_prompt_lens: list[int] = []

    for group_idx, group in enumerate(groups):
        n = len(group.data)
        if len(group.teacher_topk) != n:
            raise ValueError(
                f"Group {group_idx}: teacher_topk length "
                f"({len(group.teacher_topk)}) does not match data length ({n})."
            )

        student_data.extend(group.data)
        teacher_topk.extend(group.teacher_topk)
        student_prompt_lens.extend([group.prompt_len] * n)

    return student_data, teacher_topk, student_prompt_lens


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
        for pos in range(response_start, target_len):
            if loss_mask[pos] <= 0.0:
                continue
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


# ---------------------------------------------------------------------------
# Multi-target teacher routing (one student, N frozen teachers, routed per prompt)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherConfig:
    """One teacher in a multi-target distillation run.

    Args:
        model: Inference model id (or base-model resource to auto-deploy).
        route_value: Dataset route value for this teacher. If unset, rows route
            by ``model``.
        deployment_id: Optional explicit frozen-teacher deployment id.
        deployment_shape: Optional deployment shape for this teacher. When
            unset, the recipe uses the run-level teacher deployment shape, or
            lets the deployment API choose a compatible shape for heterogeneous
            teachers.
        blend_weight: Non-negative SDFT blend weight. Only used by
            ``TOPK_FORWARD_KL`` multi-teacher blending; sampled reverse-KL OPD
            still routes each prompt to one teacher.
        teacher_messages_key: Dataset key holding this teacher's privileged
            prompt messages (falls back to the row's own messages).
        tokenizer_model: Optional tokenizer identifier for validation. When
            set, it must match the student deployment tokenizer.
    """

    model: str
    route_value: str | None = None
    deployment_id: str | None = None
    deployment_shape: str | None = None
    teacher_messages_key: str = "teacher_messages"
    tokenizer_model: str | None = None
    blend_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.model, str):
            raise TypeError("TeacherConfig.model must be a string.")
        if not self.model:
            raise ValueError("TeacherConfig.model is required.")
        if self.route_value is not None and not isinstance(self.route_value, str):
            raise TypeError("TeacherConfig.route_value must be a string or None.")
        if self.route_value == "":
            raise ValueError("TeacherConfig.route_value must be non-empty when set.")
        if self.deployment_shape is not None and not isinstance(self.deployment_shape, str):
            raise TypeError("TeacherConfig.deployment_shape must be a string or None.")
        if self.deployment_shape == "":
            raise ValueError("TeacherConfig.deployment_shape must be non-empty when set.")
        if not isinstance(self.teacher_messages_key, str):
            raise TypeError("TeacherConfig.teacher_messages_key must be a string.")
        if not self.teacher_messages_key:
            raise ValueError("TeacherConfig.teacher_messages_key must be non-empty.")
        if self.tokenizer_model is not None and not isinstance(self.tokenizer_model, str):
            raise TypeError("TeacherConfig.tokenizer_model must be a string or None.")
        if not isinstance(self.blend_weight, (float, int)) or isinstance(
            self.blend_weight, bool
        ):
            raise TypeError("TeacherConfig.blend_weight must be a number.")
        if not math.isfinite(float(self.blend_weight)) or float(self.blend_weight) < 0.0:
            raise ValueError("TeacherConfig.blend_weight must be a non-negative finite number.")

    @property
    def routing_value(self) -> str:
        return self.route_value or self.model


@dataclass
class MultiTeacherConfig:
    """Multi-teacher config.

    ``SAMPLED_REVERSE_KL`` routes each prompt to one teacher by ``route_key``.
    ``TOPK_FORWARD_KL`` scores each rollout with all teachers and blends their
    sparse top-K distributions with probability-union blending.
    """

    teachers: list[TeacherConfig] = field(default_factory=list)
    route_key: str = "teacher"
    """Row key whose value selects a teacher routing value."""

    def __post_init__(self) -> None:
        if not self.teachers:
            raise ValueError("MultiTeacherConfig requires at least one teacher.")
        if not isinstance(self.route_key, str):
            raise TypeError("MultiTeacherConfig.route_key must be a string.")
        if not self.route_key:
            raise ValueError("MultiTeacherConfig.route_key must be non-empty.")
        route_values = [teacher.routing_value for teacher in self.teachers]
        if len(set(route_values)) != len(route_values):
            raise ValueError(f"Duplicate teacher route values in MultiTeacherConfig: {route_values}")


# ---------------------------------------------------------------------------
# Top-K distribution helpers
# ---------------------------------------------------------------------------


def teacher_topk_from_row(
    row: dict,
    *,
    ids_key: str = "teacher_topk_ids",
    logprobs_key: str = "teacher_topk_logprobs",
) -> list[TopKDist] | None:
    """Parse dataset-stored teacher top-K into per-response-position entries.

    This is a deferred offline hook. The online distillation loop does not read
    these fields today.
    """
    ids_rows = row.get(ids_key)
    logprob_rows = row.get(logprobs_key)
    if ids_rows is None and logprob_rows is None:
        return None
    if not isinstance(ids_rows, list) or not isinstance(logprob_rows, list):
        raise TypeError(f"{ids_key} and {logprobs_key} must both be lists.")
    if len(ids_rows) != len(logprob_rows):
        raise ValueError(
            f"{ids_key} ({len(ids_rows)}) and {logprobs_key} "
            f"({len(logprob_rows)}) must have the same number of positions."
        )

    topk_by_pos: list[TopKDist] = []
    for position, (token_ids, logprobs) in enumerate(zip(ids_rows, logprob_rows, strict=True)):
        if not isinstance(token_ids, list) or not isinstance(logprobs, list):
            raise TypeError(
                f"{ids_key}[{position}] and {logprobs_key}[{position}] must both be lists."
            )
        topk_by_pos.append(
            TopKDist(
                token_ids=[int(token_id) for token_id in token_ids],
                logprobs=[float(logprob) for logprob in logprobs],
            )
        )
    return topk_by_pos


def _renormalize_topk(dist: TopKDist) -> tuple[list[int], list[float]]:
    """Return ``(ids, probs)`` where ``probs`` sums to 1 over ``dist``."""
    if not dist.logprobs:
        return [], []

    max_logprob = max(dist.logprobs)
    probs = [math.exp(logprob - max_logprob) for logprob in dist.logprobs]
    normalizer = sum(probs)
    if normalizer <= 0.0:
        raise ValueError("Top-K probabilities must have positive total mass.")
    return list(dist.token_ids), [prob / normalizer for prob in probs]


def blend_teacher_topk(
    per_teacher: Sequence[tuple[TopKDist, float]],
    *,
    top_k: int,
) -> TopKDist:
    """Blend several teachers' top-K distributions into one top-K distribution.

    ``per_teacher`` contains ``(dist, weight)`` pairs for the same response
    position. Probability mass is mixed over the union of candidate token ids,
    truncated to ``top_k``, and returned as logprobs.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    total_weight = 0.0
    pooled_probs: dict[int, float] = {}
    for dist, weight in per_teacher:
        if weight < 0.0:
            raise ValueError("Teacher blend weights must be non-negative.")
        if weight == 0.0:
            continue
        if not dist.token_ids:
            continue
        token_ids, probs = _renormalize_topk(dist)
        total_weight += weight
        for token_id, prob in zip(token_ids, probs, strict=True):
            pooled_probs[token_id] = pooled_probs.get(token_id, 0.0) + weight * prob

    if total_weight <= 0.0:
        raise ValueError("Teacher blend weights must have positive total mass.")
    if not pooled_probs:
        return TopKDist(token_ids=[], logprobs=[])

    mixed = {
        token_id: prob / total_weight
        for token_id, prob in pooled_probs.items()
        if prob > 0.0
    }
    ranked = sorted(mixed.items(), key=lambda item: item[1], reverse=True)[:top_k]
    normalizer = sum(prob for _, prob in ranked)
    if normalizer <= 0.0:
        return TopKDist(token_ids=[], logprobs=[])

    token_ids = [token_id for token_id, _ in ranked]
    logprobs = [math.log(prob / normalizer) for _, prob in ranked]
    return TopKDist(token_ids=token_ids, logprobs=logprobs)


def build_topk_datum(
    model_input: tinker.ModelInput,
    topk_by_pos: Sequence[TopKDist | None],
    *,
    target_len: int,
    prompt_len: int,
    top_k: int,
) -> tinker.Datum:
    """Build one ``[N, K]`` datum for top-K distillation modes.

    Per position, ``target_tokens`` stores the source K token ids and
    ``weights`` stores the target distribution renormalized over those ids.
    ``None`` entries and padding slots get zero weights.
    """
    if target_len < 0:
        raise ValueError("target_len must be non-negative.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    response_start = max(0, prompt_len - 1)
    target_token_rows = [[0] * top_k for _ in range(target_len)]
    weight_rows = [[0.0] * top_k for _ in range(target_len)]

    for response_offset, dist in enumerate(topk_by_pos):
        target_pos = response_start + response_offset
        if target_pos >= target_len:
            break
        if dist is None:
            continue

        ranked = sorted(
            zip(dist.token_ids, dist.logprobs, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        limited_dist = TopKDist(
            token_ids=[token_id for token_id, _ in ranked],
            logprobs=[logprob for _, logprob in ranked],
        )
        token_ids, probs = _renormalize_topk(limited_dist)
        for slot, (token_id, prob) in enumerate(
            zip(token_ids, probs, strict=True)
        ):
            target_token_rows[target_pos][slot] = int(token_id)
            weight_rows[target_pos][slot] = float(prob)

    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=[token_id for row in target_token_rows for token_id in row],
                dtype="int64",
                shape=[target_len, top_k],
            ),
            "weights": tinker.TensorData(
                data=[weight for row in weight_rows for weight in row],
                dtype="float32",
                shape=[target_len, top_k],
            ),
        },
    )


def _tensor_shape(tensor: object, *, name: str, datum_idx: int) -> list[int]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise ValueError(f"Datum {datum_idx}: {name} is missing shape.")
    return [int(dim) for dim in shape]


def _target_len(datum: tinker.Datum, *, datum_idx: int) -> int:
    target = datum.loss_fn_inputs.get("target_tokens")
    if target is None:
        raise ValueError(f"Datum {datum_idx} is missing loss_fn_inputs['target_tokens'].")
    return len(target.data)


@dataclass
class _TopKForwardKLMetrics:
    active_positions: int = 0
    active_slots: int = 0
    entropy_sum: float = 0.0
    topk_masses: list[float] = field(default_factory=list)

    def add_position(self, logprobs: Sequence[float], probs: Sequence[float]) -> None:
        self.active_positions += 1
        self.active_slots += sum(1 for prob in probs if prob > 0.0)
        self.entropy_sum += -sum(prob * math.log(prob) for prob in probs if prob > 0.0)
        self.topk_masses.append(sum(math.exp(logprob) for logprob in logprobs))

    def as_dict(self, *, top_k: int) -> dict[str, float]:
        return {
            "sdft_active_positions": float(self.active_positions),
            "sdft_active_slots": float(self.active_slots),
            "sdft_top_k": float(top_k),
            "sdft_teacher_topk_entropy": (
                self.entropy_sum / self.active_positions if self.active_positions > 0 else 0.0
            ),
            "sdft_teacher_topk_mass_min": min(self.topk_masses) if self.topk_masses else 0.0,
            "sdft_teacher_topk_mass_max": max(self.topk_masses) if self.topk_masses else 0.0,
        }


def build_topk_forward_kl_datums(
    student_data: Sequence[tinker.Datum],
    teacher_topk: Sequence[Sequence[TopKDist]],
    student_prompt_lens: Sequence[int],
    *,
    top_k: int,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Build student soft-target datums from inference teacher top-K outputs.

    The teacher inference deployment returns top-K ids at each response
    position. Student training gathers at those same ids with
    ``target_tokens=[N,K]`` and ``weights=[N,K]`` for builtin cross-entropy
    forward KL.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    n = len(student_data)
    student_prompt_values = _require_lengths_match("student_prompt_lens", student_prompt_lens, n)
    teacher_topk_values = _require_lengths_match("teacher_topk", teacher_topk, n)
    student_prompt_lens = [int(value) for value in student_prompt_values]

    sdft_datums: list[tinker.Datum] = []
    metrics = _TopKForwardKLMetrics()

    for datum_idx, (student_datum, topk_by_response_pos) in enumerate(
        zip(student_data, teacher_topk_values, strict=True)
    ):
        student_target_len = _target_len(student_datum, datum_idx=datum_idx)
        student_response_start = max(0, student_prompt_lens[datum_idx] - 1)
        response_len = max(0, student_target_len - student_response_start)
        topk_by_response_pos = list(topk_by_response_pos)
        if len(topk_by_response_pos) < response_len:
            raise ValueError(
                f"Datum {datum_idx}: teacher inference top-K returned "
                f"{len(topk_by_response_pos)} response positions, expected {response_len}."
            )

        for position, dist in enumerate(topk_by_response_pos[:response_len]):
            if len(dist.token_ids) < top_k:
                raise ValueError(
                    f"Datum {datum_idx}, response position {position}: teacher inference "
                    f"top-K returned {len(dist.token_ids)} candidates, expected {top_k}."
                )
            if len(dist.token_ids) > top_k:
                dist = TopKDist(
                    token_ids=list(dist.token_ids[:top_k]),
                    logprobs=list(dist.logprobs[:top_k]),
                )
                topk_by_response_pos[position] = dist
            _ids, probs = _renormalize_topk(dist)
            metrics.add_position(dist.logprobs, probs)

        sdft_datums.append(
            build_topk_datum(
                student_datum.model_input,
                topk_by_response_pos[:response_len],
                target_len=student_target_len,
                prompt_len=student_prompt_lens[datum_idx],
                top_k=top_k,
            )
        )

    return sdft_datums, metrics.as_dict(top_k=top_k)
