"""Objective-specific train-batch builders for distillation recipes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

import tinker

from training.utils.distillation import (
    DistillMode,
    OPDPromptGroup,
    TopKDist,
    blend_teacher_topk,
    build_opd_server_datums,
    build_topk_forward_kl_datums,
    combine_opd_prompt_groups,
    combine_topk_prompt_groups,
)
from training.utils.distillation.sampling import (
    _align_completion_logprobs,
    _align_response_logprobs,
    _build_teacher_scoring_tokens,
)

_TeacherLogprobCallable = Callable[[int, list[int], int, int], Awaitable[list[float] | None]]
_TeacherTopKCallable = Callable[
    [int, list[int], int, int, int],
    Awaitable[list[TopKDist] | None],
]
_WarningCallable = Callable[..., None]


@dataclass(frozen=True)
class TeacherSourceContext:
    prompt_tokens: list[int]
    weight: float = 1.0


@dataclass(frozen=True)
class TeacherScoringFns:
    logprobs: _TeacherLogprobCallable
    topk: _TeacherTopKCallable


TeacherScores = list[float] | list[TopKDist] | None


@dataclass(frozen=True)
class DistillationTrainBatch:
    datums: list[tinker.Datum]
    loss_name: str
    loss_fn_config: dict[str, Any] | None
    input_metrics: dict[str, float]
    shape_record: dict[str, list[int]] | None = None


@dataclass(frozen=True)
class TopKForwardKLInputs:
    student_data: list[tinker.Datum]
    teacher_topk: list[list[TopKDist]]
    student_prompt_lens: list[int]


@dataclass(frozen=True)
class DistillationStepSummary:
    active_tokens: int
    log_message: str
    log_args: tuple[Any, ...]
    json_metrics: dict[str, Any]


@dataclass(frozen=True)
class DistillationObjectiveSettings:
    mode: DistillMode
    top_k: int
    has_multi_teacher: bool
    max_top_logprobs: int = 5


class DistillationObjective(Protocol):
    mode: DistillMode

    async def collect_teacher_scores(
        self,
        sampled: Sequence[Any],
        teacher_sources: Sequence[TeacherSourceContext],
        scorer: TeacherScoringFns,
    ) -> list[TeacherScores]: ...

    def build_prompt_group(
        self,
        sampled: Sequence[Any],
        teacher_scores: Sequence[TeacherScores],
        teacher_sources: Sequence[TeacherSourceContext],
        *,
        warning: _WarningCallable,
    ) -> OPDPromptGroup | None: ...

    def build_train_batch(
        self,
        prompt_groups: list[OPDPromptGroup],
        *,
        step: int,
        include_shape_record: bool,
    ) -> DistillationTrainBatch: ...

    def summarize_step(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
    ) -> DistillationStepSummary: ...


@dataclass(frozen=True)
class SampledReverseKLObjective:
    loss_scale: float
    server_loss_config: dict[str, Any] | None
    mode: DistillMode = DistillMode.SAMPLED_REVERSE_KL

    async def collect_teacher_scores(
        self,
        sampled: Sequence[Any],
        teacher_sources: Sequence[TeacherSourceContext],
        scorer: TeacherScoringFns,
    ) -> list[TeacherScores]:
        if len(teacher_sources) != 1:
            raise ValueError("SAMPLED_REVERSE_KL expects exactly one routed teacher source.")
        teacher_prompt_tokens = teacher_sources[0].prompt_tokens
        teacher_scores: list[TeacherScores] = [None] * len(sampled)
        teacher_task_indices: list[int] = []
        teacher_tasks = []
        for idx, sample in enumerate(sampled):
            teacher_scoring_tokens = _build_teacher_scoring_tokens(
                list(teacher_prompt_tokens),
                sample.full_tokens,
                student_prompt_len=sample.prompt_len,
            )
            if teacher_scoring_tokens is None:
                continue
            scoring_tokens, response_len = teacher_scoring_tokens
            teacher_task_indices.append(idx)
            teacher_tasks.append(
                scorer.logprobs(0, scoring_tokens, len(teacher_prompt_tokens), response_len)
            )
        if teacher_tasks:
            teacher_results = await asyncio.gather(*teacher_tasks)
            for idx, teacher_logprobs in zip(teacher_task_indices, teacher_results, strict=True):
                teacher_scores[idx] = teacher_logprobs
        return teacher_scores

    def build_prompt_group(
        self,
        sampled: Sequence[Any],
        teacher_scores: Sequence[TeacherScores],
        teacher_sources: Sequence[TeacherSourceContext],
        *,
        warning: _WarningCallable,
    ) -> OPDPromptGroup | None:
        if len(teacher_sources) != 1:
            raise ValueError("SAMPLED_REVERSE_KL expects exactly one routed teacher source.")
        teacher_prompt_tokens = teacher_sources[0].prompt_tokens
        prompt_group_builder = _prompt_group_builder(sampled, teacher_scores)
        for sample, teacher_logprobs in zip(sampled, teacher_scores, strict=True):
            sample_inputs = _sample_inputs(sample, teacher_prompt_tokens)
            if sample_inputs is None or teacher_logprobs is None:
                continue
            if not _is_float_list(teacher_logprobs):
                raise TypeError("SAMPLED_REVERSE_KL teacher scores must be scalar logprobs.")
            if not sample.sampling_logprobs:
                warning("Skipping OPD sample without student sampling logprobs")
                continue

            aligned_sampling_logprobs = _align_completion_logprobs(
                list(sample.sampling_logprobs),
                prompt_len=sample.prompt_len,
                target_len=sample_inputs.target_len,
                echoed=getattr(sample, "logprobs_echoed", False),
            )
            if aligned_sampling_logprobs is None:
                warning("Skipping OPD sample with incomplete student logprobs")
                continue
            aligned_teacher_logprobs = _align_response_logprobs(
                teacher_logprobs,
                prompt_len=sample.prompt_len,
                target_len=sample_inputs.target_len,
            )
            if aligned_teacher_logprobs is None:
                warning("Skipping OPD sample with incomplete teacher logprobs")
                continue

            prompt_group_builder.add_policy_sample(
                sample,
                sample_inputs.policy_datum,
                teacher_logprobs=aligned_teacher_logprobs,
                sampling_logprobs=aligned_sampling_logprobs,
            )
        return prompt_group_builder.build()

    def build_train_batch(
        self,
        prompt_groups: list[OPDPromptGroup],
        *,
        step: int,
        include_shape_record: bool,
    ) -> DistillationTrainBatch:
        return build_sampled_reverse_kl_train_batch(
            prompt_groups,
            loss_scale=self.loss_scale,
            server_loss_config=self.server_loss_config,
        )

    def summarize_step(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
    ) -> DistillationStepSummary:
        return _summarize_sampled_reverse_kl_step(metrics, step=step)


@dataclass(frozen=True)
class TopKForwardKLObjective:
    top_k: int
    mode: DistillMode = DistillMode.TOPK_FORWARD_KL
    multi_teacher: bool = False

    async def collect_teacher_scores(
        self,
        sampled: Sequence[Any],
        teacher_sources: Sequence[TeacherSourceContext],
        scorer: TeacherScoringFns,
    ) -> list[TeacherScores]:
        if not teacher_sources:
            raise ValueError("TOPK_FORWARD_KL requires at least one teacher source.")
        if not self.multi_teacher and len(teacher_sources) != 1:
            raise ValueError("TOPK_FORWARD_KL expects exactly one teacher source.")

        teacher_scores: list[TeacherScores] = [None] * len(sampled)
        teacher_task_indices: list[int] = []
        teacher_tasks = []
        teacher_tasks_per_sample: list[int] = []
        for sample_idx, sample in enumerate(sampled):
            sample_task_count = 0
            for source_idx, source in enumerate(teacher_sources):
                teacher_scoring_tokens = _build_teacher_scoring_tokens(
                    source.prompt_tokens,
                    sample.full_tokens,
                    student_prompt_len=sample.prompt_len,
                )
                if teacher_scoring_tokens is None:
                    continue
                scoring_tokens, response_len = teacher_scoring_tokens
                teacher_task_indices.append(sample_idx)
                teacher_tasks.append(
                    scorer.topk(
                        source_idx,
                        scoring_tokens,
                        len(source.prompt_tokens),
                        response_len,
                        self.top_k,
                    )
                )
                sample_task_count += 1
            teacher_tasks_per_sample.append(sample_task_count)

        if not teacher_tasks:
            return teacher_scores

        task_results = await asyncio.gather(*teacher_tasks)
        per_sample_results: list[list[list[TopKDist] | None]] = [
            [] for _ in sampled
        ]
        for sample_idx, result in zip(teacher_task_indices, task_results, strict=True):
            per_sample_results[sample_idx].append(result)

        for sample_idx, results in enumerate(per_sample_results):
            if len(results) != teacher_tasks_per_sample[sample_idx] or not results:
                continue
            if any(result is None for result in results):
                continue
            teacher_scores[sample_idx] = (
                _blend_topk_results(
                    [result for result in results if result is not None],
                    weights=[source.weight for source in teacher_sources],
                    top_k=self.top_k,
                )
                if self.multi_teacher
                else results[0]
            )
        return teacher_scores

    def build_prompt_group(
        self,
        sampled: Sequence[Any],
        teacher_scores: Sequence[TeacherScores],
        teacher_sources: Sequence[TeacherSourceContext],
        *,
        warning: _WarningCallable,
    ) -> OPDPromptGroup | None:
        prompt_group_builder = _prompt_group_builder(sampled, teacher_scores)
        source_prompt_tokens = teacher_sources[0].prompt_tokens if teacher_sources else []
        for sample, topk_by_pos in zip(sampled, teacher_scores, strict=True):
            if topk_by_pos is None:
                warning("Skipping SDFT sample without teacher inference top-K")
                continue
            if not _is_topk_list(topk_by_pos):
                raise TypeError("TOPK_FORWARD_KL teacher scores must be top-K distributions.")
            sample_inputs = _sample_inputs(sample, source_prompt_tokens)
            if sample_inputs is None:
                continue
            prompt_group_builder.add_topk_sample(
                sample,
                sample_inputs.policy_datum,
                teacher_topk=topk_by_pos,
            )
        return prompt_group_builder.build()

    def build_train_batch(
        self,
        prompt_groups: list[OPDPromptGroup],
        *,
        step: int,
        include_shape_record: bool,
    ) -> DistillationTrainBatch:
        inputs = build_topk_forward_kl_inputs(prompt_groups)
        return build_topk_forward_kl_train_batch(
            inputs,
            top_k=self.top_k,
            include_shape_record=include_shape_record,
        )

    def summarize_step(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
    ) -> DistillationStepSummary:
        return _summarize_topk_forward_kl_step(metrics, step=step, top_k=self.top_k)


def build_sampled_reverse_kl_train_batch(
    prompt_groups: list[OPDPromptGroup],
    *,
    loss_scale: float,
    server_loss_config: dict[str, Any] | None,
) -> DistillationTrainBatch:
    student_data, teacher_logprobs, prompt_lens, sampling_logprobs = combine_opd_prompt_groups(
        prompt_groups
    )
    datums, input_metrics = build_opd_server_datums(
        student_data,
        teacher_logprobs,
        sampling_logprobs,
        prompt_lens,
        loss_scale=loss_scale,
    )
    return DistillationTrainBatch(
        datums=datums,
        loss_name="importance_sampling",
        loss_fn_config=server_loss_config,
        input_metrics=input_metrics,
    )


def build_topk_forward_kl_inputs(
    prompt_groups: list[OPDPromptGroup],
) -> TopKForwardKLInputs:
    student_data, teacher_topk, student_prompt_lens = combine_topk_prompt_groups(prompt_groups)
    return TopKForwardKLInputs(
        student_data=student_data,
        teacher_topk=teacher_topk,
        student_prompt_lens=student_prompt_lens,
    )


def build_topk_forward_kl_train_batch(
    inputs: TopKForwardKLInputs,
    *,
    top_k: int,
    include_shape_record: bool,
) -> DistillationTrainBatch:
    datums, input_metrics = build_topk_forward_kl_datums(
        inputs.student_data,
        inputs.teacher_topk,
        inputs.student_prompt_lens,
        top_k=top_k,
    )
    return DistillationTrainBatch(
        datums=datums,
        loss_name="cross_entropy",
        loss_fn_config=None,
        input_metrics=input_metrics,
        shape_record=_sdft_shape_record(inputs.teacher_topk, datums, top_k=top_k)
        if include_shape_record and datums
        else None,
    )


def summarize_distillation_step(
    distill_mode: DistillMode,
    metrics: Mapping[str, Any],
    *,
    step: int,
    top_k: int,
) -> DistillationStepSummary:
    return _STEP_SUMMARIZERS[distill_mode](metrics, step=step, top_k=top_k)


def create_distillation_objective(
    settings: DistillationObjectiveSettings,
    *,
    loss_scale: float,
    server_loss_config: dict[str, Any] | None,
) -> DistillationObjective:
    validate_distillation_objective_settings(settings)
    return _OBJECTIVE_FACTORIES[settings.mode](
        settings,
        loss_scale=loss_scale,
        server_loss_config=server_loss_config,
    )


def validate_distillation_objective_settings(settings: DistillationObjectiveSettings) -> None:
    if settings.top_k <= 0:
        raise ValueError("Config.sdft_top_k must be positive.")
    if settings.max_top_logprobs < 0:
        raise ValueError("max_top_logprobs must be non-negative.")
    _OBJECTIVE_VALIDATORS[settings.mode](settings)


def _sdft_shape_record(
    teacher_topk: Sequence[Sequence[TopKDist]],
    train_datums: list[tinker.Datum],
    *,
    top_k: int,
) -> dict[str, list[int]]:
    if not train_datums:
        raise ValueError("SDFT shape logging requires at least one train datum.")
    active_positions = sum(len(row) for row in teacher_topk)
    return {
        "top_k_logprobs_shape": [active_positions, top_k],
        "top_k_indices_shape": [active_positions, top_k],
        "target_tokens_shape": list(train_datums[0].loss_fn_inputs["target_tokens"].shape),
        "weights_shape": list(train_datums[0].loss_fn_inputs["weights"].shape),
    }


@dataclass(frozen=True)
class _SampleInputs:
    policy_datum: tinker.Datum
    teacher_datum: tinker.Datum
    target_len: int


@dataclass
class _PromptGroupBuilder:
    prompt_len: int
    data: list[tinker.Datum] = field(default_factory=list)
    teacher_logprobs: list[list[float]] = field(default_factory=list)
    sampling_logprobs: list[list[float]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    completion_lens: list[int] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    teacher_topk: list[list[TopKDist]] = field(default_factory=list)

    def add_policy_sample(
        self,
        sample: Any,
        policy_datum: tinker.Datum,
        *,
        teacher_logprobs: list[float],
        sampling_logprobs: list[float],
    ) -> None:
        self.data.append(policy_datum)
        self.teacher_logprobs.append(teacher_logprobs)
        self.sampling_logprobs.append(sampling_logprobs)
        self._add_common_sample_fields(sample)

    def add_topk_sample(
        self,
        sample: Any,
        policy_datum: tinker.Datum,
        *,
        teacher_topk: list[TopKDist],
    ) -> None:
        self.data.append(policy_datum)
        self.teacher_logprobs.append([])
        self.sampling_logprobs.append([])
        self.teacher_topk.append(teacher_topk)
        self._add_common_sample_fields(sample)

    def build(self) -> OPDPromptGroup | None:
        if not self.data:
            return None
        return OPDPromptGroup(
            data=self.data,
            teacher_logprobs=self.teacher_logprobs,
            sampling_logprobs=self.sampling_logprobs,
            prompt_len=self.prompt_len,
            rewards=self.rewards,
            completion_lens=self.completion_lens,
            truncated=self.truncated,
            teacher_topk=self.teacher_topk,
        )

    def _add_common_sample_fields(self, sample: Any) -> None:
        self.rewards.append(0.0)
        self.completion_lens.append(sample.completion_len)
        self.truncated.append(sample.finish_reason == "length")


def _prompt_group_builder(
    sampled: Sequence[Any],
    teacher_scores: Sequence[TeacherScores],
) -> _PromptGroupBuilder:
    if not sampled:
        raise ValueError("sampled must not be empty.")
    if len(sampled) != len(teacher_scores):
        raise ValueError(
            f"teacher_scores must have length {len(sampled)}, got {len(teacher_scores)}."
        )
    return _PromptGroupBuilder(sampled[0].prompt_len)


def _is_float_list(values: object) -> bool:
    return isinstance(values, list) and all(
        isinstance(value, (float, int)) and not isinstance(value, bool)
        for value in values
    )


def _is_topk_list(values: object) -> bool:
    return isinstance(values, list) and all(isinstance(value, TopKDist) for value in values)


def _blend_topk_results(
    per_teacher_topk: Sequence[Sequence[TopKDist]],
    *,
    weights: Sequence[float],
    top_k: int,
) -> list[TopKDist]:
    if len(per_teacher_topk) != len(weights):
        raise ValueError(
            f"per_teacher_topk length ({len(per_teacher_topk)}) must match weights "
            f"length ({len(weights)})."
        )
    if not per_teacher_topk:
        return []
    response_len = len(per_teacher_topk[0])
    for teacher_idx, teacher_topk in enumerate(per_teacher_topk):
        if len(teacher_topk) != response_len:
            raise ValueError(
                f"Teacher {teacher_idx} returned {len(teacher_topk)} top-K positions, "
                f"expected {response_len}."
            )

    blended: list[TopKDist] = []
    for pos in range(response_len):
        blended.append(
            blend_teacher_topk(
                [
                    (teacher_topk[pos], float(weight))
                    for teacher_topk, weight in zip(per_teacher_topk, weights, strict=True)
                ],
                top_k=top_k,
            )
        )
    return blended


def _sample_inputs(sample: Any, teacher_prompt_tokens: Sequence[int]) -> _SampleInputs | None:
    teacher_scoring_tokens = _build_teacher_scoring_tokens(
        list(teacher_prompt_tokens),
        sample.full_tokens,
        student_prompt_len=sample.prompt_len,
    )
    if teacher_scoring_tokens is None:
        return None
    scoring_tokens, _response_len = teacher_scoring_tokens
    target_tokens = sample.full_tokens[1:]
    policy_datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(sample.full_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[len(target_tokens)],
            ),
        },
    )
    teacher_target_tokens = scoring_tokens[1:]
    teacher_datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(scoring_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=teacher_target_tokens,
                dtype="int64",
                shape=[len(teacher_target_tokens)],
            ),
        },
    )
    return _SampleInputs(
        policy_datum=policy_datum,
        teacher_datum=teacher_datum,
        target_len=len(target_tokens),
    )


def _summarize_sampled_reverse_kl_step(
    metrics: Mapping[str, Any],
    *,
    step: int,
) -> DistillationStepSummary:
    sampled_kl = metrics.get("train/opd_sampled_reverse_kl", 0.0)
    active_tokens = int(metrics.get("train/opd_active_tokens", 0.0))
    return DistillationStepSummary(
        active_tokens=active_tokens,
        log_message="Step %d | sampled reverse KL: %.4f | OPD advantage: %.4f | tokens=%d",
        log_args=(
            step,
            sampled_kl,
            metrics.get("train/opd_advantage", 0.0),
            active_tokens,
        ),
        json_metrics={
            "opd_sampled_reverse_kl": sampled_kl,
            "opd_advantage": metrics.get("train/opd_advantage", 0.0),
            "active_tokens": active_tokens,
        },
    )


def _summarize_topk_forward_kl_step(
    metrics: Mapping[str, Any],
    *,
    step: int,
    top_k: int,
) -> DistillationStepSummary:
    active_tokens = int(
        metrics.get(
            "train/sdft_active_positions",
            metrics.get("train/response_tokens", 0.0),
        )
    )
    return DistillationStepSummary(
        active_tokens=active_tokens,
        log_message="Step %d | SDFT forward KL top_k=%d | positions=%d | slots=%d",
        log_args=(
            step,
            top_k,
            int(metrics.get("train/sdft_active_positions", 0.0)),
            int(metrics.get("train/sdft_active_slots", 0.0)),
        ),
        json_metrics={
            "sdft_top_k": top_k,
            "sdft_active_positions": metrics.get("train/sdft_active_positions", 0.0),
            "sdft_active_slots": metrics.get("train/sdft_active_slots", 0.0),
            "loss_sum": metrics.get("train/loss:sum", 0.0),
            "response_tokens": metrics.get("train/response_tokens", 0.0),
        },
    )


def _summarize_sampled_reverse_kl_step_from_dispatch(
    metrics: Mapping[str, Any],
    *,
    step: int,
    top_k: int,
) -> DistillationStepSummary:
    return _summarize_sampled_reverse_kl_step(metrics, step=step)


def _validate_sampled_reverse_kl_settings(settings: DistillationObjectiveSettings) -> None:
    return None


def _validate_topk_forward_kl_settings(settings: DistillationObjectiveSettings) -> None:
    _validate_topk_source_settings(settings)


def _validate_topk_source_settings(settings: DistillationObjectiveSettings) -> None:
    if settings.top_k > settings.max_top_logprobs:
        raise ValueError(
            "Config.sdft_top_k exceeds the inference top_logprobs limit "
            f"({settings.max_top_logprobs})."
        )


def _sampled_reverse_kl_factory(
    settings: DistillationObjectiveSettings,
    *,
    loss_scale: float,
    server_loss_config: dict[str, Any] | None,
) -> DistillationObjective:
    return SampledReverseKLObjective(loss_scale=loss_scale, server_loss_config=server_loss_config)


def _topk_forward_kl_factory(
    settings: DistillationObjectiveSettings,
    *,
    loss_scale: float,
    server_loss_config: dict[str, Any] | None,
) -> DistillationObjective:
    return TopKForwardKLObjective(top_k=settings.top_k, multi_teacher=settings.has_multi_teacher)


_OBJECTIVE_VALIDATORS = {
    DistillMode.SAMPLED_REVERSE_KL: _validate_sampled_reverse_kl_settings,
    DistillMode.TOPK_FORWARD_KL: _validate_topk_forward_kl_settings,
}

_OBJECTIVE_FACTORIES = {
    DistillMode.SAMPLED_REVERSE_KL: _sampled_reverse_kl_factory,
    DistillMode.TOPK_FORWARD_KL: _topk_forward_kl_factory,
}

_STEP_SUMMARIZERS = {
    DistillMode.SAMPLED_REVERSE_KL: _summarize_sampled_reverse_kl_step_from_dispatch,
    DistillMode.TOPK_FORWARD_KL: _summarize_topk_forward_kl_step,
}
