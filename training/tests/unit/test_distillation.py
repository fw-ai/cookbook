"""Unit tests for sampled-token OPD helpers."""

from __future__ import annotations

import asyncio
import math
from math import exp
from pathlib import Path
from types import SimpleNamespace

import pytest
import tinker

from training.recipes.distillation_loop import (
    Config,
    _default_teacher_deployment_id,
    _is_base_model_resource,
    _resolve_teacher_runtime,
    _resolve_teacher_specs,
    _teacher_deployment_shape_for_spec,
    _teacher_deployment_id_for_spec,
    _teacher_metric_slug,
    _validate_teacher_tokenizers,
    main as run_opd_main,
)
from training.utils import (
    CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO,
    DeployConfig,
    RunnerConfig,
    TrainerConfig,
)
from training.utils.distillation import (
    DistillMode,
    MultiTeacherConfig,
    OPDPromptGroup,
    TeacherConfig,
    blend_teacher_topk,
    build_opd_server_datums,
    build_topk_forward_kl_datums,
    build_topk_datum,
    combine_opd_prompt_groups,
    combine_topk_prompt_groups,
    teacher_topk_from_row,
)
from training.utils.distillation.eval import (
    evaluate_teacher_trace_logprob_gap,
    extract_final_answer,
    expected_final_answer,
    make_teacher_trace_logprob_gap_eval,
    normalize_final_answer,
    validate_privileged_opd_dataset,
    validate_opd_trace_result,
)
from training.utils.distillation.objectives import (
    DistillationObjectiveSettings,
    TeacherScoringFns,
    TeacherSourceContext,
    build_topk_forward_kl_inputs,
    build_topk_forward_kl_train_batch,
    create_distillation_objective,
    summarize_distillation_step,
)
from training.utils.distillation.sampling import (
    TopKDist,
    _align_completion_logprobs,
    _align_response_logprobs,
    _build_teacher_scoring_tokens,
    _extract_teacher_topk,
    _extract_scored_token_logprobs,
    _score_teacher_topk,
    _slice_response_logprobs,
    _teacher_messages_for_row,
    _tokenize_teacher_prompt,
)

MAX_CONTEXT_LEN = 262_144


def _build_qwen_opd_config(
    *,
    training_shape_id: str | None = None,
    log_path: str = "/tmp/opd",
    metrics_file: str | None = None,
) -> Config:
    return Config(
        log_path=log_path,
        base_model="accounts/fireworks/models/qwen3p5-9b",
        teacher_model="accounts/fireworks/models/qwen3p5-9b",
        teacher_deployment_id="distillation-teacher-qwen3p5-9b-unit",
        dataset="/tmp/opd_math.jsonl",
        trainer=TrainerConfig(training_shape_id=training_shape_id),
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3.5-9B"),
        weight_sync_interval=1,
        runner=RunnerConfig(metrics_file=metrics_file),
        step_eval=make_teacher_trace_logprob_gap_eval(min_pre_final_tokens=4),
        step_eval_interval=1,
        eval_before_training=True,
        lora_rank=0,
        learning_rate=2e-5,
        max_rows=8,
        epochs=25,
        prompt_groups_per_step=8,
        completions_per_prompt=4,
        max_completion_tokens=1024,
        temperature=1.0,
        max_seq_len=None if training_shape_id else MAX_CONTEXT_LEN,
        save_final_checkpoint=False,
    )


def _passing_trace_logprob_eval() -> dict[str, float]:
    return {
        "eval/opd_trace_teacher_nll": 0.1,
        "eval/opd_trace_student_nll": 0.2,
        "eval/opd_trace_student_minus_teacher_nll": 0.1,
        "eval/opd_trace_pre_final_teacher_nll": 0.1,
        "eval/opd_trace_pre_final_student_nll": 0.2,
        "eval/opd_trace_pre_final_student_minus_teacher_nll": 0.1,
        "eval/opd_trace_final_teacher_nll": 0.1,
        "eval/opd_trace_final_student_nll": 0.2,
        "eval/opd_trace_final_student_minus_teacher_nll": 0.1,
        "eval/opd_trace_teacher_final_accuracy": 1.0,
        "eval/opd_trace_student_generation_accuracy": 1.0,
        "eval/opd_trace_examples": 8.0,
        "eval/opd_trace_tokens": 32.0,
        "eval/opd_trace_pre_final_tokens": 24.0,
        "eval/opd_trace_final_tokens": 8.0,
    }


def _server_importance_sampling_loss(
    current_logprobs: list[float],
    sampling_logprobs: list[float],
    advantages: list[float],
) -> float:
    return -sum(
        exp(current_lp - sampling_lp) * advantage
        for current_lp, sampling_lp, advantage in zip(
            current_logprobs,
            sampling_logprobs,
            advantages,
            strict=True,
        )
    )


def test_topk_dist_rejects_mismatched_or_invalid_fields() -> None:
    with pytest.raises(ValueError, match="same length"):
        TopKDist(token_ids=[1], logprobs=[-0.1, -0.2])
    with pytest.raises(TypeError, match="integers"):
        TopKDist(token_ids=[True], logprobs=[-0.1])
    with pytest.raises(ValueError, match="non-negative"):
        TopKDist(token_ids=[-1], logprobs=[-0.1])


def test_build_topk_datum_places_renormalized_weights_on_response_positions() -> None:
    datum = build_topk_datum(
        _datum().model_input,
        [
            TopKDist(
                token_ids=[101, 102, 103],
                logprobs=[math.log(0.2), math.log(0.3), math.log(0.5)],
            ),
            None,
            TopKDist(token_ids=[201], logprobs=[-3.0]),
        ],
        target_len=4,
        prompt_len=2,
        top_k=2,
    )

    assert datum.loss_fn_inputs["target_tokens"].shape == [4, 2]
    assert datum.loss_fn_inputs["target_tokens"].data == [
        0,
        0,
        103,
        102,
        0,
        0,
        201,
        0,
    ]
    assert datum.loss_fn_inputs["weights"].data == pytest.approx(
        [
            0.0,
            0.0,
            0.625,
            0.375,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
    )


def _teacher_topk_output() -> SimpleNamespace:
    return SimpleNamespace()


def _teacher_inference_topk() -> list[TopKDist]:
    return [
        TopKDist(token_ids=[101, 102], logprobs=[math.log(0.8), math.log(0.2)]),
        TopKDist(token_ids=[202, 201], logprobs=[math.log(0.75), math.log(0.25)]),
    ]


def test_combine_topk_prompt_groups_flattens_inference_topk_rows() -> None:
    student = _datum()
    teacher_topk = _teacher_inference_topk()
    group = OPDPromptGroup(
        data=[student],
        teacher_logprobs=[[]],
        sampling_logprobs=[[]],
        prompt_len=2,
        rewards=[0.0],
        teacher_topk=[teacher_topk],
    )

    student_data, teacher_rows, student_prompt_lens = combine_topk_prompt_groups([group])

    assert student_data == [student]
    assert teacher_rows == [teacher_topk]
    assert student_prompt_lens == [2]


def test_build_topk_forward_kl_datums_uses_inference_response_support() -> None:
    student = _datum(tokens=[10, 11, 40, 41])

    datums, metrics = build_topk_forward_kl_datums(
        [student],
        [_teacher_inference_topk()],
        student_prompt_lens=[2],
        top_k=2,
    )

    assert len(datums) == 1
    assert datums[0].loss_fn_inputs["target_tokens"].shape == [3, 2]
    assert datums[0].loss_fn_inputs["target_tokens"].data == [
        0,
        0,
        101,
        102,
        202,
        201,
    ]
    assert datums[0].loss_fn_inputs["weights"].data == pytest.approx(
        [
            0.0,
            0.0,
            0.8,
            0.2,
            0.75,
            0.25,
        ]
    )
    assert metrics["sdft_active_positions"] == pytest.approx(2.0)
    assert metrics["sdft_active_slots"] == pytest.approx(4.0)
    assert metrics["sdft_top_k"] == pytest.approx(2.0)
    assert metrics["sdft_teacher_topk_mass_min"] == pytest.approx(1.0)
    assert metrics["sdft_teacher_topk_mass_max"] == pytest.approx(1.0)
    assert metrics["sdft_teacher_topk_entropy"] > 0.0


def test_build_topk_forward_kl_datums_rejects_missing_teacher_topk() -> None:
    with pytest.raises(ValueError, match="teacher inference top-K returned"):
        build_topk_forward_kl_datums(
            [_datum()],
            [[]],
            student_prompt_lens=[1],
            top_k=2,
        )


def test_build_topk_forward_kl_datums_rejects_short_teacher_topk_slots() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        build_topk_forward_kl_datums(
            [_datum(tokens=[10, 11, 40])],
            [[TopKDist(token_ids=[101], logprobs=[0.0])]],
            student_prompt_lens=[2],
            top_k=2,
        )


def test_topk_forward_kl_objective_builds_cross_entropy_batch() -> None:
    student = _datum(tokens=[10, 11, 40, 41])
    teacher_topk = _teacher_inference_topk()
    group = OPDPromptGroup(
        data=[student],
        teacher_logprobs=[[]],
        sampling_logprobs=[[]],
        prompt_len=2,
        rewards=[0.0],
        teacher_topk=[teacher_topk],
    )

    inputs = build_topk_forward_kl_inputs([group])
    batch = build_topk_forward_kl_train_batch(
        inputs,
        top_k=2,
        include_shape_record=True,
    )

    assert inputs.teacher_topk == [teacher_topk]
    assert batch.loss_name == "cross_entropy"
    assert batch.loss_fn_config is None
    assert batch.datums[0].loss_fn_inputs["target_tokens"].shape == [3, 2]
    assert batch.shape_record == {
        "top_k_logprobs_shape": [2, 2],
        "top_k_indices_shape": [2, 2],
        "target_tokens_shape": [3, 2],
        "weights_shape": [3, 2],
    }


def test_topk_forward_kl_strategy_scores_inference_top_logprobs() -> None:
    topk_calls: list[tuple[int, list[int], int, int, int]] = []

    async def score_logprobs(
        source_idx: int,
        scoring_tokens: list[int],
        prompt_len: int,
        response_len: int,
    ) -> list[float] | None:
        raise AssertionError("TOPK_FORWARD_KL should not request scalar logprobs")

    async def score_topk(
        source_idx: int,
        scoring_tokens: list[int],
        prompt_len: int,
        response_len: int,
        top_k: int,
    ) -> list[TopKDist] | None:
        topk_calls.append((source_idx, scoring_tokens, prompt_len, response_len, top_k))
        return _teacher_inference_topk()

    objective = create_distillation_objective(
        DistillationObjectiveSettings(
            mode=DistillMode.TOPK_FORWARD_KL,
            top_k=2,
            has_multi_teacher=False,
            max_top_logprobs=5,
        ),
        loss_scale=1.0,
        server_loss_config=None,
    )
    sources = [TeacherSourceContext(prompt_tokens=[20, 21, 22, 23])]
    teacher_scores = asyncio.run(
        objective.collect_teacher_scores(
            [SimpleNamespace(**_sample_kwargs(tokens=[10, 11, 40, 41], prompt_len=2))],
            sources,
            TeacherScoringFns(logprobs=score_logprobs, topk=score_topk),
        )
    )
    group = objective.build_prompt_group(
        [SimpleNamespace(**_sample_kwargs(tokens=[10, 11, 40, 41], prompt_len=2))],
        teacher_scores,
        sources,
        warning=lambda *args, **kwargs: None,
    )
    assert group is not None
    batch = objective.build_train_batch([group], step=3, include_shape_record=False)

    assert topk_calls == [(0, [20, 21, 22, 23, 40, 41], 4, 2, 2)]
    assert batch.loss_name == "cross_entropy"
    assert batch.datums[0].loss_fn_inputs["weights"].shape == [3, 2]


def test_topk_forward_kl_strategy_blends_multi_teacher_sources() -> None:
    samples = [SimpleNamespace(**_sample_kwargs(tokens=[10, 11, 40], prompt_len=2))]
    topk_calls: list[tuple[int, list[int], int, int, int]] = []

    async def score_logprobs(
        source_idx: int,
        scoring_tokens: list[int],
        prompt_len: int,
        response_len: int,
    ) -> list[float] | None:
        raise AssertionError("multi-teacher SDFT should not request scalar logprobs")

    async def score_topk(
        source_idx: int,
        scoring_tokens: list[int],
        prompt_len: int,
        response_len: int,
        top_k: int,
    ) -> list[TopKDist] | None:
        topk_calls.append((source_idx, scoring_tokens, prompt_len, response_len, top_k))
        if source_idx == 0:
            return [TopKDist(token_ids=[101, 102], logprobs=[math.log(0.8), math.log(0.2)])]
        return [TopKDist(token_ids=[102, 103], logprobs=[math.log(0.5), math.log(0.5)])]

    objective = create_distillation_objective(
        DistillationObjectiveSettings(
            mode=DistillMode.TOPK_FORWARD_KL,
            top_k=2,
            has_multi_teacher=True,
            max_top_logprobs=5,
        ),
        loss_scale=1.0,
        server_loss_config=None,
    )
    sources = [
        TeacherSourceContext(prompt_tokens=[20, 21]),
        TeacherSourceContext(prompt_tokens=[30, 31]),
    ]
    teacher_scores = asyncio.run(
        objective.collect_teacher_scores(
            samples,
            sources,
            TeacherScoringFns(logprobs=score_logprobs, topk=score_topk),
        )
    )
    group = objective.build_prompt_group(
        samples,
        teacher_scores,
        sources,
        warning=lambda *args, **kwargs: None,
    )
    assert group is not None
    batch = objective.build_train_batch([group], step=0, include_shape_record=True)

    assert topk_calls == [
        (0, [20, 21, 40], 2, 1, 2),
        (1, [30, 31, 40], 2, 1, 2),
    ]
    assert batch.datums[0].loss_fn_inputs["target_tokens"].data == [0, 0, 101, 102]
    assert batch.datums[0].loss_fn_inputs["weights"].data == pytest.approx(
        [0.0, 0.0, 0.4 / 0.75, 0.35 / 0.75]
    )


def test_topk_forward_kl_strategy_uses_teacher_blend_weights() -> None:
    samples = [SimpleNamespace(**_sample_kwargs(tokens=[10, 11, 40], prompt_len=2))]

    async def score_logprobs(
        source_idx: int,
        scoring_tokens: list[int],
        prompt_len: int,
        response_len: int,
    ) -> list[float] | None:
        raise AssertionError("multi-teacher SDFT should not request scalar logprobs")

    async def score_topk(
        source_idx: int,
        scoring_tokens: list[int],
        prompt_len: int,
        response_len: int,
        top_k: int,
    ) -> list[TopKDist] | None:
        if source_idx == 0:
            return [TopKDist(token_ids=[101, 102], logprobs=[math.log(0.8), math.log(0.2)])]
        return [TopKDist(token_ids=[102, 103], logprobs=[math.log(0.5), math.log(0.5)])]

    objective = create_distillation_objective(
        DistillationObjectiveSettings(
            mode=DistillMode.TOPK_FORWARD_KL,
            top_k=2,
            has_multi_teacher=True,
            max_top_logprobs=5,
        ),
        loss_scale=1.0,
        server_loss_config=None,
    )
    teacher_scores = asyncio.run(
        objective.collect_teacher_scores(
            samples,
            [
                TeacherSourceContext(prompt_tokens=[20, 21], weight=2.0),
                TeacherSourceContext(prompt_tokens=[30, 31], weight=1.0),
            ],
            TeacherScoringFns(logprobs=score_logprobs, topk=score_topk),
        )
    )
    group = objective.build_prompt_group(
        samples,
        teacher_scores,
        [TeacherSourceContext(prompt_tokens=[20, 21])],
        warning=lambda *args, **kwargs: None,
    )
    assert group is not None
    batch = objective.build_train_batch([group], step=0, include_shape_record=False)

    assert batch.datums[0].loss_fn_inputs["target_tokens"].data == [0, 0, 101, 102]
    assert batch.datums[0].loss_fn_inputs["weights"].data == pytest.approx(
        [0.0, 0.0, 0.64, 0.36]
    )


def test_summarize_distillation_step_reports_topk_forward_kl_metrics() -> None:
    summary = summarize_distillation_step(
        DistillMode.TOPK_FORWARD_KL,
        {
            "train/sdft_active_positions": 7.0,
            "train/sdft_active_slots": 140.0,
            "train/loss:sum": 12.5,
            "train/response_tokens": 140.0,
        },
        step=2,
        top_k=20,
    )

    assert summary.active_tokens == 7
    assert summary.json_metrics == {
        "sdft_top_k": 20,
        "sdft_active_positions": 7.0,
        "sdft_active_slots": 140.0,
        "loss_sum": 12.5,
        "response_tokens": 140.0,
    }


def test_topk_forward_kl_summary_falls_back_to_response_tokens() -> None:
    summary = summarize_distillation_step(
        DistillMode.TOPK_FORWARD_KL,
        {
            "train/sdft_active_slots": 140.0,
            "train/loss:sum": 12.5,
            "train/response_tokens": 11.0,
        },
        step=2,
        top_k=20,
    )

    assert summary.active_tokens == 11


def test_teacher_topk_from_row_rejects_mismatched_position_counts() -> None:
    with pytest.raises(ValueError, match="same number of positions"):
        teacher_topk_from_row(
            {
                "teacher_topk_ids": [[1], [2]],
                "teacher_topk_logprobs": [[-0.1]],
            }
        )


def test_blend_teacher_topk_mixes_probability_mass_and_rejects_empty_weight() -> None:
    blended = blend_teacher_topk(
        [
            (
                TopKDist(
                    token_ids=[1, 2],
                    logprobs=[math.log(0.8), math.log(0.2)],
                ),
                2.0,
            ),
            (
                TopKDist(
                    token_ids=[2, 3],
                    logprobs=[math.log(0.5), math.log(0.5)],
                ),
                1.0,
            ),
        ],
        top_k=2,
    )

    assert blended.token_ids == [1, 2]
    assert [math.exp(logprob) for logprob in blended.logprobs] == pytest.approx(
        [0.64, 0.36]
    )
    blended_with_empty = blend_teacher_topk(
        [
            (TopKDist(token_ids=[], logprobs=[]), 2.0),
            (TopKDist(token_ids=[7], logprobs=[0.0]), 1.0),
        ],
        top_k=1,
    )
    assert blended_with_empty.token_ids == [7]
    assert [math.exp(logprob) for logprob in blended_with_empty.logprobs] == pytest.approx(
        [1.0]
    )

    with pytest.raises(ValueError, match="positive total mass"):
        blend_teacher_topk(
            [(TopKDist(token_ids=[1], logprobs=[0.0]), 0.0)],
            top_k=1,
        )


def test_extract_teacher_topk_uses_candidate_ids_and_tokenizer_fallback() -> None:
    tokenizer = _FakeTokenizer(
        {
            "A": [201],
            "B": [102],
            "multi": [301, 302],
        }
    )
    response = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"logprob": 0.0},
                        {"logprob": -9.0, "token_id": 10},
                        {
                            "logprob": -0.1,
                            "token_id": 42,
                            "top_logprobs": [
                                {"token_id": 101, "logprob": math.log(0.6)},
                                {"token": "B", "logprob": math.log(0.4)},
                                {"token": "multi", "logprob": -99.0},
                            ],
                        },
                        {
                            "logprob": -0.3,
                            "token": "A",
                            "top_logprobs": [{"token": "A", "logprob": -0.3}],
                        },
                    ],
                },
            }
        ]
    }

    topk_by_pos = _extract_teacher_topk(
        response,
        prompt_len=2,
        response_len=2,
        target_len=3,
        tokenizer=tokenizer,
    )

    assert topk_by_pos == [
        TopKDist(token_ids=[101, 102], logprobs=[math.log(0.6), math.log(0.4)]),
        TopKDist(token_ids=[201], logprobs=[-0.3]),
    ]


def test_extract_teacher_topk_rejects_missing_top_logprobs() -> None:
    response = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"logprob": 0.0},
                        {"logprob": -0.1, "token_id": 42},
                    ],
                },
            }
        ]
    }

    assert (
        _extract_teacher_topk(
            response,
            prompt_len=1,
            response_len=1,
            target_len=1,
        )
        is None
    )


def test_score_teacher_topk_requests_top_logprobs_and_aligns_response_window() -> None:
    sampler = _FakeScoringSampler(
        top_logprobs_by_token={
            2: [{"token_id": 20, "logprob": -0.2}, {"token_id": 21, "logprob": -1.2}],
            3: [{"token_id": 30, "logprob": -0.3}, {"token_id": 31, "logprob": -1.3}],
        }
    )

    topk_by_pos = asyncio.run(
        _score_teacher_topk(
            sampler,
            [1, 2, 3],
            prompt_len=1,
            response_len=2,
            top_logprobs=2,
            http_timeout=30,
        )
    )

    assert topk_by_pos == [
        TopKDist(token_ids=[20, 21], logprobs=[-0.2, -1.2]),
        TopKDist(token_ids=[30, 31], logprobs=[-0.3, -1.3]),
    ]
    assert sampler.calls[0][1]["top_logprobs"] == 2


def test_score_teacher_topk_fails_when_inference_returns_too_few_candidates() -> None:
    sampler = _FakeScoringSampler(
        top_logprobs_by_token={
            2: [{"token_id": 20, "logprob": -0.2}],
        }
    )

    with pytest.raises(ValueError, match="fewer candidates"):
        asyncio.run(
            _score_teacher_topk(
                sampler,
                [1, 2],
                prompt_len=1,
                response_len=1,
                top_logprobs=2,
                http_timeout=30,
            )
        )


def _datum(
    *,
    tokens: list[int] | None = None,
    loss_mask: list[float] | None = None,
) -> tinker.Datum:
    if tokens is None:
        tokens = [10, 11, 12, 13, 14]
    target_tokens = tokens[1:]
    inputs = {
        "target_tokens": tinker.TensorData(
            data=target_tokens,
            dtype="int64",
            shape=[len(target_tokens)],
        ),
    }
    if loss_mask is not None:
        inputs["loss_mask"] = tinker.TensorData(
            data=loss_mask,
            dtype="float32",
            shape=[len(loss_mask)],
        )

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs=inputs,
    )


def _sample_kwargs(
    *,
    tokens: list[int],
    prompt_len: int,
) -> dict[str, object]:
    return {
        "full_tokens": tokens,
        "prompt_len": prompt_len,
        "completion_len": max(0, len(tokens) - prompt_len),
        "finish_reason": "stop",
        "inference_logprobs": [-0.1] * max(0, len(tokens) - prompt_len),
        "logprobs_echoed": False,
    }


class _FakeTokenizer:
    def __init__(self, token_map: dict[str, list[int]]):
        self.token_map = token_map

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        tokens: list[int] = []
        for message in messages:
            tokens.extend(self.token_map[message["content"]])
        return tokens

    def encode(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return self.token_map[text]


class _FakeScoringSampler:
    def __init__(
        self,
        response_logprob: float = -0.1,
        response_logprobs_by_token: dict[int, float] | None = None,
        top_logprobs_by_token: dict[int, list[dict[str, object]]] | None = None,
        tokenizer: _FakeTokenizer | None = None,
        sample_text: str = "",
        sample_completion_tokens: list[int] | None = None,
    ):
        self.response_logprob = response_logprob
        self.response_logprobs_by_token = response_logprobs_by_token or {}
        self.top_logprobs_by_token = top_logprobs_by_token or {}
        self.tokenizer = tokenizer
        self.sample_text = sample_text
        self.sample_completion_tokens = sample_completion_tokens or []
        self.calls = []
        self.sample_calls = []

    async def async_completions_stream(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        content = [{"logprob": 0.0}]
        for token_id in prompt[1:]:
            token_id = int(token_id)
            slot = {
                "logprob": self.response_logprobs_by_token.get(token_id, self.response_logprob),
                "token_id": token_id,
            }
            if token_id in self.top_logprobs_by_token:
                slot["top_logprobs"] = self.top_logprobs_by_token[token_id]
            content.append(slot)
        return {"choices": [{"logprobs": {"content": content}}]}, {}

    async def sample_with_tokens(self, messages, **kwargs):
        if self.tokenizer is None:
            raise RuntimeError("fake sampler needs a tokenizer to sample")
        self.sample_calls.append((messages, kwargs))
        prompt_tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        completion_tokens = list(self.sample_completion_tokens)
        return [
            SimpleNamespace(
                text=self.sample_text,
                full_tokens=list(prompt_tokens) + completion_tokens,
                prompt_len=len(prompt_tokens),
                completion_len=len(completion_tokens),
                finish_reason="stop",
                inference_logprobs=None,
                logprobs_echoed=False,
            )
        ]


def test_build_opd_server_datums_encodes_teacher_minus_sampling_advantages() -> None:
    datum = _datum()
    teacher_lp = [[-9.0, -0.2, -1.0, -0.3]]
    sampling_lp = [[-8.0, -0.5, -0.8, -0.7]]

    server_datums, metrics = build_opd_server_datums(
        [datum],
        teacher_lp,
        sampling_lp,
        prompt_lens=[2],
        loss_scale=2.0,
    )

    inputs = server_datums[0].loss_fn_inputs
    assert inputs["target_tokens"].data == [11, 12, 13, 14]
    assert inputs["logprobs"].data == pytest.approx(sampling_lp[0])
    assert inputs["advantages"].data == pytest.approx([0.0, 0.6, -0.4, 0.8])
    assert metrics["opd_active_tokens"] == pytest.approx(3.0)
    assert metrics["opd_sampled_reverse_kl"] == pytest.approx((-0.3 + 0.2 - 0.4) / 3)
    assert metrics["opd_advantage"] == pytest.approx((0.6 - 0.4 + 0.8) / 3)
    assert metrics["opd_abs_advantage"] == pytest.approx((0.6 + 0.4 + 0.8) / 3)
    assert metrics["opd_student_logprob_minus_teacher_logprob"] == metrics["opd_sampled_reverse_kl"]
    assert metrics["opd_teacher_logprob_minus_student_logprob"] == pytest.approx((0.3 - 0.2 + 0.4) / 3)
    assert metrics["opd_abs_logprob_gap"] == pytest.approx((0.3 + 0.2 + 0.4) / 3)


def test_opd_datums_make_server_loss_equal_sampled_reverse_kl_at_rollout_policy() -> None:
    datum = _datum()
    teacher_lp = [[-9.0, -0.2, -1.0, -0.3]]
    sampling_lp = [[-8.0, -0.5, -0.8, -0.7]]

    server_datums, _metrics = build_opd_server_datums(
        [datum],
        teacher_lp,
        sampling_lp,
        prompt_lens=[2],
    )

    inputs = server_datums[0].loss_fn_inputs
    old_lp = inputs["logprobs"].data
    advantages = inputs["advantages"].data
    loss = _server_importance_sampling_loss(old_lp, old_lp, advantages)

    active_positions = [1, 2, 3]
    expected_sampled_reverse_kl = sum(
        sampling_lp[0][pos] - teacher_lp[0][pos] for pos in active_positions
    )
    assert loss == pytest.approx(expected_sampled_reverse_kl)

    pos = 1
    eps = 1e-6
    plus_lp = list(old_lp)
    minus_lp = list(old_lp)
    plus_lp[pos] += eps
    minus_lp[pos] -= eps
    finite_difference_grad = (
        _server_importance_sampling_loss(plus_lp, old_lp, advantages)
        - _server_importance_sampling_loss(minus_lp, old_lp, advantages)
    ) / (2 * eps)
    assert finite_difference_grad == pytest.approx(sampling_lp[0][pos] - teacher_lp[0][pos])


def test_build_opd_server_datums_folds_loss_mask_into_advantages() -> None:
    datum = _datum(loss_mask=[1.0, 1.0, 0.0, 1.0])

    server_datums, metrics = build_opd_server_datums(
        [datum],
        teacher_logprobs=[[-1.0, -0.1, -0.1, -0.1]],
        sampling_logprobs=[[-1.0, -0.5, -0.5, -0.5]],
        prompt_lens=[2],
    )

    assert server_datums[0].loss_fn_inputs["advantages"].data == pytest.approx(
        [0.0, 0.4, 0.0, 0.4]
    )
    assert metrics["opd_active_tokens"] == pytest.approx(2.0)


def test_combine_opd_prompt_groups_uses_explicit_teacher_and_sampling_fields() -> None:
    datum = _datum()
    group = OPDPromptGroup(
        data=[datum],
        teacher_logprobs=[[-0.1, -0.2, -0.3, -0.4]],
        sampling_logprobs=[[-0.5, -0.6, -0.7, -0.8]],
        prompt_len=2,
        rewards=[0.0],
    )

    data, teacher_logprobs, prompt_lens, sampling_logprobs = combine_opd_prompt_groups([group])

    assert data == [datum]
    assert teacher_logprobs == group.teacher_logprobs
    assert prompt_lens == [2]
    assert sampling_logprobs == group.sampling_logprobs


def test_combine_opd_prompt_groups_rejects_mismatched_teacher_scores() -> None:
    group = OPDPromptGroup(
        data=[_datum()],
        teacher_logprobs=[],
        sampling_logprobs=[[-0.5, -0.6, -0.7, -0.8]],
        prompt_len=2,
        rewards=[0.0],
    )

    with pytest.raises(ValueError, match="teacher_logprobs length"):
        combine_opd_prompt_groups([group])


def test_build_opd_server_datums_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="teacher_logprobs must have length 1"):
        build_opd_server_datums(
            [_datum()],
            teacher_logprobs=[],
            sampling_logprobs=[[-0.1, -0.2, -0.3, -0.4]],
            prompt_lens=[2],
        )


def test_build_opd_server_datums_rejects_missing_active_teacher_logprobs() -> None:
    with pytest.raises(ValueError, match="teacher_logprobs has length 2"):
        build_opd_server_datums(
            [_datum()],
            teacher_logprobs=[[-1.0, -0.1]],
            sampling_logprobs=[[-1.0, -0.5, -0.5, -0.5]],
            prompt_lens=[2],
        )


def test_build_opd_server_datums_rejects_missing_active_sampling_logprobs() -> None:
    with pytest.raises(ValueError, match="sampling_logprobs has length 2"):
        build_opd_server_datums(
            [_datum()],
            teacher_logprobs=[[-1.0, -0.1, -0.1, -0.1]],
            sampling_logprobs=[[-1.0, -0.5]],
            prompt_lens=[2],
        )


def test_align_completion_logprobs_pads_prompt_prefix_for_non_echo_generation() -> None:
    aligned = _align_completion_logprobs(
        [-0.4, -0.5],
        prompt_len=3,
        target_len=4,
        echoed=False,
    )

    assert aligned == [0.0, 0.0, -0.4, -0.5]


def test_align_completion_logprobs_rejects_missing_active_generation_logprobs() -> None:
    assert (
        _align_completion_logprobs(
            [-0.4],
            prompt_len=3,
            target_len=4,
            echoed=False,
        )
        is None
    )


def test_align_completion_logprobs_trims_echoed_logprobs() -> None:
    aligned = _align_completion_logprobs(
        [-0.1, -0.2, -0.3, -0.4, -0.5],
        prompt_len=3,
        target_len=4,
        echoed=True,
    )

    assert aligned == [-0.1, -0.2, -0.3, -0.4]


def test_align_response_logprobs_places_teacher_scores_on_student_response_tokens() -> None:
    assert _align_response_logprobs(
        [-0.4, -0.5],
        prompt_len=3,
        target_len=4,
    ) == [0.0, 0.0, -0.4, -0.5]


def test_extract_scored_token_logprobs_drops_unconditional_and_extra_token() -> None:
    response = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"logprob": 0.0},
                        {"logprob": -0.1},
                        {"logprob": -0.2},
                        {"logprob": -0.3},
                        {"logprob": -99.0},
                    ],
                },
            }
        ]
    }

    assert _extract_scored_token_logprobs(response, target_len=3) == [-0.1, -0.2, -0.3]


def test_slice_response_logprobs_extracts_teacher_completion_span() -> None:
    assert _slice_response_logprobs(
        [-0.1, -0.2, -0.3, -0.4],
        prompt_len=3,
        response_len=2,
    ) == [-0.3, -0.4]


def test_teacher_scoring_tokens_use_privileged_prompt_and_sampled_response() -> None:
    row = {
        "messages": [{"role": "user", "content": "student"}],
        "teacher_messages": [
            {"role": "system", "content": "privileged"},
            {"role": "user", "content": "student"},
        ],
    }
    student_messages = [{"role": "user", "content": "student"}]
    tokenizer = _FakeTokenizer(
        {
            "privileged": [900, 901],
            "student": [10, 11],
        }
    )

    teacher_messages = _teacher_messages_for_row(row, student_messages)
    teacher_prompt_tokens = _tokenize_teacher_prompt(tokenizer, teacher_messages)
    scoring_tokens = _build_teacher_scoring_tokens(
        teacher_prompt_tokens,
        [10, 11, 40, 41],
        student_prompt_len=2,
    )

    assert teacher_messages == row["teacher_messages"]
    assert scoring_tokens == ([900, 901, 10, 11, 40, 41], 2)


def test_teacher_messages_fallback_to_student_prompt() -> None:
    student_messages = [{"role": "user", "content": "student"}]

    assert _teacher_messages_for_row({"messages": student_messages}, student_messages) == student_messages


def test_teacher_messages_use_explicit_teacher_key() -> None:
    student_messages = [{"role": "user", "content": "student"}]
    row = {
        "teacher_messages": [{"role": "user", "content": "generic"}],
        "math_teacher_messages": [{"role": "user", "content": "math"}],
    }

    assert _teacher_messages_for_row(
        row,
        student_messages,
        teacher_messages_key="math_teacher_messages",
    ) == [{"role": "user", "content": "math"}]


def test_explicit_teacher_messages_key_falls_back_to_student_prompt() -> None:
    student_messages = [{"role": "user", "content": "student"}]
    row = {"teacher_messages": [{"role": "user", "content": "generic"}]}

    assert _teacher_messages_for_row(
        row,
        student_messages,
        teacher_messages_key="math_teacher_messages",
    ) == student_messages


def test_teacher_model_resource_detection() -> None:
    assert _is_base_model_resource("accounts/fireworks/models/qwen3p5-9b")
    assert not _is_base_model_resource("accounts/test/deployments/distillation-teacher-qwen3p5-9b")
    assert not _is_base_model_resource("accounts/test/deployedModels/qwen3p5-9b")


class _FakeTeacherRuntimeService:
    def __init__(self) -> None:
        self.direct_calls: list[dict] = []
        self.deployment_calls: list[dict] = []

    def create_deployment_sampler_for_model(
        self,
        model: str,
        *,
        tokenizer,
        inference_url: str | None = None,
    ):
        self.direct_calls.append(
            {
                "model": model,
                "tokenizer": tokenizer,
                "inference_url": inference_url,
            }
        )
        return SimpleNamespace(model=model)

    def create_inference_deployment_sampler(
        self,
        config,
        *,
        timeout_s: float,
        cleanup_on_close: str | None,
        tokenizer,
    ):
        self.deployment_calls.append(
            {
                "config": config,
                "timeout_s": timeout_s,
                "cleanup_on_close": cleanup_on_close,
                "tokenizer": tokenizer,
            }
        )
        return SimpleNamespace(model=f"accounts/test/deployments/{config.deployment_id}")


def test_resolve_teacher_runtime_uses_existing_inference_model_directly() -> None:
    cfg = Config(
        log_path="/tmp/opd",
        teacher_model="accounts/test/deployments/teacher-existing",
        teacher_inference_url="https://inference.teacher",
    )
    service = _FakeTeacherRuntimeService()

    runtime = _resolve_teacher_runtime(
        cfg=cfg,
        teacher_specs=_resolve_teacher_specs(cfg),
        service=service,
        tokenizer="tok",
        base_url="https://api.test",
        cancel_on_exit=True,
    )

    assert len(service.direct_calls) == 1
    assert service.direct_calls[0] == {
        "model": "accounts/test/deployments/teacher-existing",
        "tokenizer": "tok",
        "inference_url": "https://inference.teacher",
    }
    assert service.deployment_calls == []
    assert runtime.primary.resolved_model == "accounts/test/deployments/teacher-existing"
    assert runtime.primary.sampler.model == "accounts/test/deployments/teacher-existing"
    assert runtime.route_key == "teacher"


def test_resolve_teacher_runtime_reuses_duplicate_base_model_deployment() -> None:
    teacher_model = "accounts/fireworks/models/qwen3p5-9b"
    cfg = Config(
        log_path="/tmp/opd",
        teacher_model="",
        multi_teacher=MultiTeacherConfig(
            teachers=[
                TeacherConfig(model=teacher_model, route_value="math"),
                TeacherConfig(model=teacher_model, route_value="code"),
            ],
            route_key="teacher_route",
        ),
        teacher_deployment_id="ignored-for-multi-teacher",
        teacher_replica_count=2,
        teacher_deployment_timeout_s=123,
    )
    service = _FakeTeacherRuntimeService()

    runtime = _resolve_teacher_runtime(
        cfg=cfg,
        teacher_specs=_resolve_teacher_specs(cfg),
        service=service,
        tokenizer="tok",
        base_url="https://api.test",
        cancel_on_exit=True,
    )

    assert service.direct_calls == []
    assert len(service.deployment_calls) == 1
    call = service.deployment_calls[0]
    assert call["timeout_s"] == 123
    assert call["cleanup_on_close"] == CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO
    assert call["tokenizer"] == "tok"
    assert call["config"].base_model == teacher_model
    assert call["config"].min_replica_count == 2
    assert call["config"].max_replica_count == 2
    assert call["config"].enable_hot_load is False
    assert call["config"].hot_load_bucket_type is None
    assert call["config"].for_training is True
    assert runtime.is_multi_teacher
    assert runtime.route_key == "teacher_route"
    assert sorted(runtime.route_to_entry) == ["code", "math"]
    assert {id(entry.sampler) for entry in runtime.entries} == {id(runtime.primary.sampler)}


def test_default_teacher_deployment_id_is_stable_and_safe() -> None:
    assert (
        _default_teacher_deployment_id("accounts/fireworks/models/Qwen_3.5_9B")
        == "distillation-teacher-qwen-3-5-9b"
    )


def test_multi_teacher_default_deployment_ids_include_full_model_identity() -> None:
    first = _teacher_deployment_id_for_spec(
        TeacherConfig(model="accounts/first/models/shared-name"),
        single_teacher=False,
        single_teacher_deployment_id=None,
    )
    second = _teacher_deployment_id_for_spec(
        TeacherConfig(model="accounts/second/models/shared-name"),
        single_teacher=False,
        single_teacher_deployment_id=None,
    )

    assert first != second


def test_multi_teacher_metric_slugs_include_full_model_identity() -> None:
    assert _teacher_metric_slug("accounts/first/models/shared-name") != _teacher_metric_slug(
        "accounts/second/models/shared-name"
    )


def test_resolve_teacher_specs_allows_multi_teacher_without_single_teacher_model() -> None:
    cfg = Config(
        log_path="/tmp/opd",
        teacher_model="",
        multi_teacher=MultiTeacherConfig(
            teachers=[TeacherConfig(model="accounts/fireworks/models/math-teacher")]
        ),
    )

    assert _resolve_teacher_specs(cfg) == cfg.multi_teacher.teachers


def test_resolve_teacher_specs_requires_single_teacher_model() -> None:
    with pytest.raises(ValueError, match="teacher_model is required"):
        _resolve_teacher_specs(Config(log_path="/tmp/opd", teacher_model=""))


def test_multi_teacher_config_rejects_empty_teachers() -> None:
    with pytest.raises(ValueError, match="requires at least one teacher"):
        MultiTeacherConfig()


def test_teacher_config_routing_value_defaults_to_model() -> None:
    spec = TeacherConfig(model="accounts/fireworks/models/math-teacher")

    assert spec.routing_value == "accounts/fireworks/models/math-teacher"


def test_teacher_config_routing_value_uses_route_value() -> None:
    spec = TeacherConfig(
        model="accounts/fireworks/models/qwen3p5-35b-a3b",
        route_value="math-teacher",
    )

    assert spec.routing_value == "math-teacher"


def test_teacher_config_rejects_empty_route_value() -> None:
    with pytest.raises(ValueError, match="route_value"):
        TeacherConfig(
            model="accounts/fireworks/models/math-teacher",
            route_value="",
        )


def test_multi_teacher_config_allows_duplicate_models_with_distinct_routes() -> None:
    cfg = MultiTeacherConfig(
        teachers=[
            TeacherConfig(
                model="accounts/fireworks/models/qwen3p5-35b-a3b",
                route_value="math-teacher",
            ),
            TeacherConfig(
                model="accounts/fireworks/models/qwen3p5-35b-a3b",
                route_value="code-teacher",
            ),
        ]
    )

    assert [teacher.routing_value for teacher in cfg.teachers] == ["math-teacher", "code-teacher"]


def test_multi_teacher_config_rejects_duplicate_route_values() -> None:
    with pytest.raises(ValueError, match="Duplicate teacher route values"):
        MultiTeacherConfig(
            teachers=[
                TeacherConfig(
                    model="accounts/fireworks/models/math-teacher",
                    route_value="shared-route",
                ),
                TeacherConfig(
                    model="accounts/fireworks/models/code-teacher",
                    route_value="shared-route",
                ),
            ]
        )


def test_teacher_config_rejects_empty_deployment_shape() -> None:
    with pytest.raises(ValueError, match="deployment_shape"):
        TeacherConfig(
            model="accounts/fireworks/models/math-teacher",
            deployment_shape="",
        )


def test_teacher_config_rejects_invalid_blend_weight() -> None:
    with pytest.raises(ValueError, match="blend_weight"):
        TeacherConfig(
            model="accounts/fireworks/models/math-teacher",
            blend_weight=-1.0,
        )
    with pytest.raises(ValueError, match="blend_weight"):
        TeacherConfig(
            model="accounts/fireworks/models/math-teacher",
            blend_weight=math.inf,
        )
    with pytest.raises(TypeError, match="blend_weight"):
        TeacherConfig(
            model="accounts/fireworks/models/math-teacher",
            blend_weight=True,
        )


def test_teacher_deployment_shape_prefers_per_teacher_override() -> None:
    cfg = Config(
        log_path="/tmp/opd",
        base_model="accounts/fireworks/models/student",
        teacher_deployment_shape="shape-run",
    )
    spec = TeacherConfig(
        model="accounts/fireworks/models/teacher",
        deployment_shape="shape-teacher",
    )

    assert (
        _teacher_deployment_shape_for_spec(
            spec,
            cfg,
        )
        == "shape-teacher"
    )


def test_teacher_deployment_shape_uses_run_level_override() -> None:
    cfg = Config(
        log_path="/tmp/opd",
        base_model="accounts/fireworks/models/student",
        teacher_deployment_shape="shape-run",
    )
    spec = TeacherConfig(model="accounts/fireworks/models/teacher")

    assert (
        _teacher_deployment_shape_for_spec(
            spec,
            cfg,
        )
        == "shape-run"
    )


def test_teacher_deployment_shape_leaves_same_base_default_to_sdk() -> None:
    cfg = Config(log_path="/tmp/opd", base_model="accounts/fireworks/models/student")
    spec = TeacherConfig(model="accounts/fireworks/models/student")

    assert (
        _teacher_deployment_shape_for_spec(
            spec,
            cfg,
        )
        is None
    )


def test_teacher_deployment_shape_lets_api_choose_for_heterogeneous_teacher() -> None:
    cfg = Config(log_path="/tmp/opd", base_model="accounts/fireworks/models/student")
    spec = TeacherConfig(model="accounts/fireworks/models/teacher")

    assert (
        _teacher_deployment_shape_for_spec(
            spec,
            cfg,
        )
        is None
    )


def test_validate_teacher_tokenizers_rejects_declared_mismatch() -> None:
    with pytest.raises(ValueError, match="tokenizer_model must match"):
        _validate_teacher_tokenizers(
            [TeacherConfig(model="accounts/fireworks/models/code-teacher", tokenizer_model="tok-b")],
            student_tokenizer_model="tok-a",
        )


def test_qwen_opd_config_can_be_full_param_and_privileged() -> None:
    cfg = _build_qwen_opd_config(training_shape_id=None)

    assert cfg.lora_rank == 0
    assert cfg.trainer.training_shape_id is None
    assert cfg.base_model == cfg.teacher_model
    assert cfg.step_eval is not None
    assert cfg.step_eval_interval == 1
    assert cfg.eval_before_training is True
    assert cfg.max_seq_len == MAX_CONTEXT_LEN
    assert cfg.prompt_groups_per_step == cfg.max_rows
    assert cfg.weight_sync_interval == 1


def test_qwen_opd_config_lets_pinned_shape_own_context_length() -> None:
    cfg = _build_qwen_opd_config(
        training_shape_id="accounts/fireworks/trainingShapes/qwen3p5-9b-256k"
    )

    assert cfg.trainer.training_shape_id == "accounts/fireworks/trainingShapes/qwen3p5-9b-256k"
    assert cfg.max_seq_len is None


def test_opd_rejects_stale_weight_sync_windows() -> None:
    cfg = _build_qwen_opd_config()
    cfg.weight_sync_interval = 2

    with pytest.raises(ValueError, match="weight_sync_interval"):
        run_opd_main(cfg)


def test_topk_forward_kl_rejects_top_k_above_inference_limit() -> None:
    cfg = _build_qwen_opd_config()
    cfg.distill_mode = DistillMode.TOPK_FORWARD_KL
    cfg.sdft_top_k = 20

    with pytest.raises(ValueError, match="inference top_logprobs limit"):
        run_opd_main(cfg)


def test_teacher_trace_eval_normalizes_expected_answers() -> None:
    row = {"expected_answer": " 5. "}

    assert expected_final_answer(row) == "5"
    assert normalize_final_answer("  5.  ") == "5"
    assert extract_final_answer("Reasoning...\nFinal: 5.") == "5"
    assert extract_final_answer("Final: 5\nActually, continue") is None


def test_validate_privileged_opd_dataset_rejects_bad_teacher_prompt() -> None:
    rows = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Solve using normal reasoning. Do not discuss the output format. "
                        "End with exactly one line: Final: <answer>."
                    ),
                },
                {"role": "user", "content": "What is the hidden result for case A?"},
            ],
            "teacher_messages": [
                {
                    "role": "system",
                    "content": (
                        "Use the privileged worked solution. "
                        "End with exactly one line: Final: <answer>."
                    ),
                },
                {"role": "user", "content": "Privileged worked solution: case A is 5."},
            ],
            "expected_answer": "5",
        }
    ]

    with pytest.raises(ValueError, match="placeholder final-answer"):
        validate_privileged_opd_dataset(rows)


def test_teacher_trace_logprob_eval_scores_reasoning_trace(tmp_path) -> None:
    eval_file = tmp_path / "teacher_trace_eval.jsonl"
    tokenizer = _FakeTokenizer(
        {
            "student": [10, 11],
            "teacher": [20, 21],
            "Add 2 and 3.\nFinal: ": [30, 31, 32, 33],
            "Add 2 and 3.\nFinal: 5": [30, 31, 32, 33, 40],
            "Now I know it.\nFinal: 5": [50, 51, 52, 40],
            "5": [40],
        }
    )
    teacher_trace = "Add 2 and 3.\nFinal: 5"
    student_generation = "Now I know it.\nFinal: 5"

    metrics = evaluate_teacher_trace_logprob_gap(
        {
            "config": _build_qwen_opd_config(),
            "dataset": [
                {
                    "messages": [{"role": "user", "content": "student"}],
                    "teacher_messages": [{"role": "user", "content": "teacher"}],
                    "expected_answer": "5",
                }
            ],
            "teacher_sampler": _FakeScoringSampler(
                response_logprob=-0.1,
                response_logprobs_by_token={40: -0.05},
                tokenizer=tokenizer,
                sample_text=teacher_trace,
                sample_completion_tokens=[30, 31, 32, 33, 40],
            ),
            "student_sampler": _FakeScoringSampler(
                response_logprob=-0.5,
                response_logprobs_by_token={40: -0.4},
                tokenizer=tokenizer,
                sample_text=student_generation,
                sample_completion_tokens=[50, 51, 52, 40],
            ),
            "tokenizer": tokenizer,
            "global_step": 3,
            "max_seq_len": MAX_CONTEXT_LEN,
        },
        trace_log_path=eval_file,
        min_pre_final_tokens=4,
    )

    assert metrics["eval/opd_trace_teacher_nll"] == pytest.approx((0.1 * 4 + 0.05) / 5)
    assert metrics["eval/opd_trace_student_nll"] == pytest.approx((0.5 * 4 + 0.4) / 5)
    assert metrics["eval/opd_trace_pre_final_student_minus_teacher_nll"] == pytest.approx(0.4)
    assert metrics["eval/opd_trace_final_student_minus_teacher_nll"] == pytest.approx(0.35)
    assert metrics["eval/opd_trace_teacher_final_accuracy"] == pytest.approx(1.0)
    assert metrics["eval/opd_trace_student_generation_accuracy"] == pytest.approx(1.0)
    assert metrics["eval/opd_trace_pre_final_tokens"] == pytest.approx(4.0)
    assert metrics["eval/opd_trace_final_tokens"] == pytest.approx(1.0)
    records = [line for line in eval_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 1
    assert '"step":3' in records[0]
    assert '"teacher_generated_final":"5"' in records[0]


def test_teacher_trace_logprob_eval_routes_multi_teacher_rows() -> None:
    math_teacher = "accounts/fireworks/models/math-teacher"
    code_teacher = "accounts/fireworks/models/code-teacher"
    tokenizer = _FakeTokenizer(
        {
            "student-a": [10, 11],
            "student-b": [12, 13],
            "math": [20, 21],
            "code": [30, 31],
            "Trace\nFinal: ": [40, 41],
            "Student\nFinal: ": [50, 51],
            "5": [42],
        }
    )
    teacher_trace = "Trace\nFinal: 5"
    math_sampler = _FakeScoringSampler(
        response_logprob=-0.1,
        tokenizer=tokenizer,
        sample_text=teacher_trace,
        sample_completion_tokens=[40, 41, 42],
    )
    code_sampler = _FakeScoringSampler(
        response_logprob=-0.2,
        tokenizer=tokenizer,
        sample_text=teacher_trace,
        sample_completion_tokens=[40, 41, 42],
    )
    student_sampler = _FakeScoringSampler(
        response_logprob=-0.5,
        tokenizer=tokenizer,
        sample_text="Student\nFinal: 5",
        sample_completion_tokens=[50, 51, 42],
    )

    metrics = evaluate_teacher_trace_logprob_gap(
        {
            "config": _build_qwen_opd_config(),
            "dataset": [
                {
                    "messages": [{"role": "user", "content": "student-a"}],
                    "math_teacher_messages": [{"role": "user", "content": "math"}],
                    "teacher": math_teacher,
                    "expected_answer": "5",
                },
                {
                    "messages": [{"role": "user", "content": "student-b"}],
                    "code_teacher_messages": [{"role": "user", "content": "code"}],
                    "teacher": code_teacher,
                    "expected_answer": "5",
                },
            ],
            "teacher_sampler": math_sampler,
            "teacher_samplers": {
                math_teacher: math_sampler,
                code_teacher: code_sampler,
            },
            "teacher_messages_keys": {
                math_teacher: "math_teacher_messages",
                code_teacher: "code_teacher_messages",
            },
            "teacher_route_key": "teacher",
            "is_multi_teacher": True,
            "student_sampler": student_sampler,
            "tokenizer": tokenizer,
            "global_step": 3,
            "max_seq_len": MAX_CONTEXT_LEN,
        },
        min_pre_final_tokens=2,
    )

    assert metrics["eval/opd_trace_examples"] == pytest.approx(2.0)
    assert math_sampler.sample_calls[0][0] == [{"role": "user", "content": "math"}]
    assert code_sampler.sample_calls[0][0] == [{"role": "user", "content": "code"}]


def test_teacher_trace_logprob_eval_uses_single_teacher_custom_messages_key() -> None:
    tokenizer = _FakeTokenizer(
        {
            "student": [10, 11],
            "generic": [12, 13],
            "math": [20, 21],
            "Trace\nFinal: ": [40, 41],
            "Student\nFinal: ": [50, 51],
            "5": [42],
        }
    )
    teacher_sampler = _FakeScoringSampler(
        response_logprob=-0.1,
        tokenizer=tokenizer,
        sample_text="Trace\nFinal: 5",
        sample_completion_tokens=[40, 41, 42],
    )
    student_sampler = _FakeScoringSampler(
        response_logprob=-0.5,
        tokenizer=tokenizer,
        sample_text="Student\nFinal: 5",
        sample_completion_tokens=[50, 51, 42],
    )

    metrics = evaluate_teacher_trace_logprob_gap(
        {
            "config": _build_qwen_opd_config(),
            "dataset": [
                {
                    "messages": [{"role": "user", "content": "student"}],
                    "teacher_messages": [{"role": "user", "content": "generic"}],
                    "math_teacher_messages": [{"role": "user", "content": "math"}],
                    "expected_answer": "5",
                }
            ],
            "teacher_sampler": teacher_sampler,
            "teacher_messages_keys": {
                "math-teacher": "math_teacher_messages",
            },
            "is_multi_teacher": False,
            "student_sampler": student_sampler,
            "tokenizer": tokenizer,
            "max_seq_len": MAX_CONTEXT_LEN,
        },
        min_pre_final_tokens=2,
    )

    assert metrics["eval/opd_trace_examples"] == pytest.approx(1.0)
    assert teacher_sampler.sample_calls[0][0] == [{"role": "user", "content": "math"}]


def test_teacher_trace_logprob_eval_skips_unconfigured_multi_teacher_routes() -> None:
    math_teacher = "accounts/fireworks/models/math-teacher"
    tokenizer = _FakeTokenizer(
        {
            "student-a": [10, 11],
            "math": [20, 21],
            "Trace\nFinal: ": [40, 41],
            "Student\nFinal: ": [50, 51],
            "5": [42],
        }
    )
    math_sampler = _FakeScoringSampler(
        response_logprob=-0.1,
        tokenizer=tokenizer,
        sample_text="Trace\nFinal: 5",
        sample_completion_tokens=[40, 41, 42],
    )
    student_sampler = _FakeScoringSampler(
        response_logprob=-0.5,
        tokenizer=tokenizer,
        sample_text="Student\nFinal: 5",
        sample_completion_tokens=[50, 51, 42],
    )

    metrics = evaluate_teacher_trace_logprob_gap(
        {
            "config": _build_qwen_opd_config(),
            "dataset": [
                {
                    "messages": [{"role": "user", "content": "student-a"}],
                    "math_teacher_messages": [{"role": "user", "content": "math"}],
                    "teacher": math_teacher,
                    "expected_answer": "5",
                },
                {
                    "messages": [{"role": "user", "content": "student-b"}],
                    "teacher": "accounts/fireworks/models/code-teacher",
                },
                {
                    "messages": [{"role": "user", "content": "student-c"}],
                },
            ],
            "teacher_sampler": math_sampler,
            "teacher_samplers": {
                math_teacher: math_sampler,
            },
            "teacher_messages_keys": {
                math_teacher: "math_teacher_messages",
            },
            "teacher_route_key": "teacher",
            "is_multi_teacher": True,
            "student_sampler": student_sampler,
            "tokenizer": tokenizer,
            "max_seq_len": MAX_CONTEXT_LEN,
        },
        min_pre_final_tokens=2,
    )

    assert metrics["eval/opd_trace_examples"] == pytest.approx(1.0)
    assert len(math_sampler.sample_calls) == 1
    assert len(student_sampler.sample_calls) == 1


def test_opd_trace_result_validation_requires_opd_signal(tmp_path) -> None:
    cfg = _build_qwen_opd_config(metrics_file=str(tmp_path / "metrics.jsonl"))
    cfg.epochs = 1
    Path(cfg.runner.metrics_file).write_text(
        "\n".join(
            [
                '{"step":0,"eval/opd_trace_teacher_nll":0.1,'
                '"eval/opd_trace_student_minus_teacher_nll":1.0}',
                '{"step":1,"train/opd_active_tokens":4,"train/opd_abs_advantage":0.25}',
                '{"step":1,"eval/opd_trace_teacher_nll":0.1,'
                '"eval/opd_trace_student_minus_teacher_nll":0.5}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    validate_opd_trace_result(
        cfg,
        {
            "steps": 1,
            "max_seq_len": MAX_CONTEXT_LEN,
            "eval": _passing_trace_logprob_eval(),
        },
        min_max_seq_len=MAX_CONTEXT_LEN,
        min_teacher_final_accuracy=1.0,
        min_student_generation_accuracy=1.0,
        min_trace_gap_improvement=0.2,
    )

    Path(cfg.runner.metrics_file).write_text(
        "\n".join(
            [
                '{"step":0,"eval/opd_trace_teacher_nll":0.1,'
                '"eval/opd_trace_student_minus_teacher_nll":1.0}',
                '{"step":1,"train/opd_active_tokens":4,"train/opd_abs_advantage":0.0}',
                '{"step":1,"eval/opd_trace_teacher_nll":0.1,'
                '"eval/opd_trace_student_minus_teacher_nll":0.5}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="teacher/student logprob delta"):
        validate_opd_trace_result(
            cfg,
            {
                "steps": 1,
                "max_seq_len": MAX_CONTEXT_LEN,
                "eval": _passing_trace_logprob_eval(),
            },
            min_max_seq_len=MAX_CONTEXT_LEN,
            min_teacher_final_accuracy=1.0,
            min_student_generation_accuracy=1.0,
            min_trace_gap_improvement=0.2,
        )


def test_opd_trace_result_validation_requires_trace_logprob_eval(tmp_path) -> None:
    cfg = _build_qwen_opd_config(metrics_file=str(tmp_path / "metrics.jsonl"))
    cfg.epochs = 1
    Path(cfg.runner.metrics_file).write_text(
        "\n".join(
            [
                '{"step":0,"eval/opd_trace_teacher_nll":0.1,'
                '"eval/opd_trace_student_minus_teacher_nll":1.0}',
                '{"step":1,"train/opd_active_tokens":4,"train/opd_abs_advantage":0.25}',
                '{"step":1,"eval/opd_trace_teacher_nll":0.1,'
                '"eval/opd_trace_student_minus_teacher_nll":0.5}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="teacher-trace eval metric"):
        validate_opd_trace_result(
            cfg,
            {
                "steps": 1,
                "max_seq_len": MAX_CONTEXT_LEN,
                "eval": {},
            },
        )


def test_opd_trace_result_validation_requires_gap_improvement(tmp_path) -> None:
    cfg = _build_qwen_opd_config(metrics_file=str(tmp_path / "metrics.jsonl"))
    cfg.epochs = 8
    train_records = [
        f'{{"step":{step},"train/opd_active_tokens":4,'
        f'"train/opd_abs_advantage":1.0,"train/opd_sampled_reverse_kl":1.0}}'
        for step in range(1, 9)
    ]
    eval_records = [
        f'{{"step":{step},"eval/opd_trace_teacher_nll":0.1,'
        f'"eval/opd_trace_student_minus_teacher_nll":1.0}}'
        for step in range(0, 9)
    ]
    Path(cfg.runner.metrics_file).write_text(
        "\n".join([*eval_records, *train_records]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="logprob gap did not improve"):
        validate_opd_trace_result(
            cfg,
            {
                "steps": 8,
                "max_seq_len": MAX_CONTEXT_LEN,
                "eval": _passing_trace_logprob_eval(),
            },
            min_train_gap_improvement=0.2,
            min_trace_gap_improvement=0.2,
        )
