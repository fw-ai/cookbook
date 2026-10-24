"""Unit tests for sampled-token on-policy distillation."""

from __future__ import annotations

import asyncio
from math import exp
from types import SimpleNamespace

import pytest
import training.recipes.distillation_loop as distillation_loop
from fireworks.training import sdk
from training.recipes.distillation_loop import Config
from training.recipes.distillation_loop import main as run_opd_main
from training.utils import (
    CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO,
    DeployConfig,
    TrainerConfig,
)
from training.utils.distillation import (
    OPDPromptGroup,
    build_opd_datums,
    build_opd_prompt_group,
    build_opd_server_datums,
    combine_opd_prompt_groups,
)
from training.utils.distillation.eval import (
    EvalContext,
    evaluate_teacher_trace_logprob_gap,
    expected_final_answer,
    extract_final_answer,
    normalize_final_answer,
    validate_opd_trace_result,
    validate_privileged_opd_dataset,
)
from training.utils.distillation.sampling import (
    _extract_scored_token_logprobs,
    _slice_response_logprobs,
    align_completion_logprobs,
    align_response_logprobs,
    completions_from_sample_response,
    score_teacher_logprobs,
    score_teacher_logprobs_with_sampling_client,
    teacher_messages_for_row,
    tokenize_teacher_prompt,
)
from training.utils.distillation.teacher import (
    create_teacher_sampler,
    default_teacher_deployment_id,
    is_base_model_resource,
)

MAX_CONTEXT_LEN = 262_144


def _build_qwen_opd_config(
    *,
    training_shape_id: str | None = None,
    log_path: str = "/tmp/opd",
) -> Config:
    return Config(
        log_path=log_path,
        base_model="accounts/fireworks/models/qwen3p5-9b",
        teacher_model="accounts/fireworks/models/qwen3p5-9b",
        teacher_deployment_id="distillation-teacher-qwen3p5-9b-unit",
        dataset="/tmp/opd_math.jsonl",
        trainer=TrainerConfig(training_shape_id=training_shape_id),
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3.5-9B"),
        eval_every=1,
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


def _datum(
    *,
    tokens: list[int] | None = None,
    loss_mask: list[float] | None = None,
) -> sdk.Datum:
    if tokens is None:
        tokens = [10, 11, 12, 13, 14]
    target_tokens = tokens[1:]
    inputs = {
        "target_tokens": sdk.TensorData(
            data=target_tokens,
            dtype="int64",
            shape=[len(target_tokens)],
        ),
    }
    if loss_mask is not None:
        inputs["loss_mask"] = sdk.TensorData(
            data=loss_mask,
            dtype="float32",
            shape=[len(loss_mask)],
        )

    return sdk.Datum(
        model_input=sdk.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs=inputs,
    )


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
        tokenizer: _FakeTokenizer | None = None,
        sample_text: str = "",
        sample_completion_tokens: list[int] | None = None,
    ):
        self.response_logprob = response_logprob
        self.response_logprobs_by_token = response_logprobs_by_token or {}
        self.tokenizer = tokenizer
        self.sample_text = sample_text
        self.sample_completion_tokens = sample_completion_tokens or []
        self.calls = []
        self.sample_calls = []
        self.closed = False

    def close(self):
        self.closed = True

    async def async_completions_stream(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        content = [{"logprob": 0.0}]
        for token_id in prompt[1:]:
            token_id = int(token_id)
            content.append(
                {
                    "logprob": self.response_logprobs_by_token.get(token_id, self.response_logprob),
                    "token_id": token_id,
                }
            )
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
                sampling_logprobs=None,
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
    expected_sampled_reverse_kl = sum(sampling_lp[0][pos] - teacher_lp[0][pos] for pos in active_positions)
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

    assert server_datums[0].loss_fn_inputs["advantages"].data == pytest.approx([0.0, 0.4, 0.0, 0.4])
    assert metrics["opd_active_tokens"] == pytest.approx(2.0)


def test_combine_opd_prompt_groups_uses_explicit_teacher_and_sampling_fields() -> None:
    datum = _datum()
    group = OPDPromptGroup(
        data=[datum],
        teacher_logprobs=[[-0.1, -0.2, -0.3, -0.4]],
        sampling_logprobs=[[-0.5, -0.6, -0.7, -0.8]],
        prompt_len=2,
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
    aligned = align_completion_logprobs(
        [-0.4, -0.5],
        prompt_len=3,
        target_len=4,
        echoed=False,
    )

    assert aligned == [0.0, 0.0, -0.4, -0.5]


def test_align_completion_logprobs_rejects_missing_active_generation_logprobs() -> None:
    assert (
        align_completion_logprobs(
            [-0.4],
            prompt_len=3,
            target_len=4,
            echoed=False,
        )
        is None
    )


def test_align_completion_logprobs_trims_echoed_logprobs() -> None:
    aligned = align_completion_logprobs(
        [-0.1, -0.2, -0.3, -0.4, -0.5],
        prompt_len=3,
        target_len=4,
        echoed=True,
    )

    assert aligned == [-0.1, -0.2, -0.3, -0.4]


def test_align_response_logprobs_places_teacher_scores_on_student_response_tokens() -> None:
    assert align_response_logprobs(
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


def test_build_opd_datums_flattens_groups_into_server_datums() -> None:
    group = OPDPromptGroup(
        data=[_datum(), _datum()],
        teacher_logprobs=[[-9.0, -0.2, -1.0, -0.3]] * 2,
        sampling_logprobs=[[-8.0, -0.5, -0.8, -0.7]] * 2,
        prompt_len=2,
    )

    datums, metrics = build_opd_datums([group], loss_scale=2.0)

    assert len(datums) == 2
    assert datums[1].loss_fn_inputs["advantages"].data == pytest.approx([0.0, 0.6, -0.4, 0.8])
    assert metrics["opd_active_tokens"] == pytest.approx(6.0)


def _student_sample(
    tokens: list[int],
    *,
    prompt_len: int,
    sampling_logprobs: list[float] | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        full_tokens=tokens,
        prompt_len=prompt_len,
        sampling_logprobs=sampling_logprobs,
        logprobs_echoed=False,
    )


def test_build_opd_prompt_group_aligns_student_and_teacher_logprobs() -> None:
    sample = _student_sample([10, 11, 40, 41], prompt_len=2, sampling_logprobs=[-0.4, -0.5])

    group = build_opd_prompt_group([sample], [[-0.1, -0.2]])

    assert group is not None
    assert group.prompt_len == 2
    assert group.data[0].loss_fn_inputs["target_tokens"].data == [11, 40, 41]
    assert group.data[0].model_input.to_ints() == [10, 11, 40]
    assert group.sampling_logprobs == [[0.0, -0.4, -0.5]]
    assert group.teacher_logprobs == [[0.0, -0.1, -0.2]]


def test_build_opd_prompt_group_skips_unscored_or_incomplete_samples() -> None:
    samples = [
        _student_sample([10, 11, 40, 41], prompt_len=2, sampling_logprobs=[-0.4, -0.5]),
        _student_sample([10, 11, 40, 41], prompt_len=2, sampling_logprobs=None),
        _student_sample([10, 11, 40, 41], prompt_len=2, sampling_logprobs=[-0.4]),
    ]

    assert build_opd_prompt_group(samples, [None, [-0.1, -0.2], [-0.1, -0.2]]) is None


def test_score_teacher_logprobs_scores_sampled_response_under_teacher_prompt() -> None:
    row = {
        "messages": [{"role": "user", "content": "student"}],
        "teacher_messages": [
            {"role": "system", "content": "privileged"},
            {"role": "user", "content": "student"},
        ],
    }
    student_messages = [{"role": "user", "content": "student"}]
    tokenizer = _FakeTokenizer({"privileged": [900, 901], "student": [10, 11]})
    teacher = _FakeScoringSampler(response_logprob=-0.1, response_logprobs_by_token={41: -0.3})
    samples = [
        _student_sample([10, 11, 40, 41], prompt_len=2, sampling_logprobs=[-0.4, -0.5]),
        _student_sample([10, 11], prompt_len=2, sampling_logprobs=[]),
    ]

    teacher_messages = teacher_messages_for_row(row, student_messages)
    teacher_prompt = tokenize_teacher_prompt(tokenizer, teacher_messages)
    scores = asyncio.run(score_teacher_logprobs(teacher, teacher_prompt, samples, http_timeout=5))

    assert teacher_messages == row["teacher_messages"]
    assert scores == [pytest.approx([-0.1, -0.3]), None]
    assert len(teacher.calls) == 1
    prompt, kwargs = teacher.calls[0]
    assert prompt == [900, 901, 10, 11, 40, 41]
    assert kwargs["echo"] is True and kwargs["logprobs"] is True
    assert "top_logprobs" not in kwargs


def test_teacher_messages_fallback_to_student_prompt() -> None:
    student_messages = [{"role": "user", "content": "student"}]

    assert teacher_messages_for_row({"messages": student_messages}, student_messages) == student_messages


def test_teacher_messages_accept_privileged_messages_alias() -> None:
    student_messages = [{"role": "user", "content": "student"}]
    row = {"privileged_messages": [{"role": "user", "content": "privileged"}]}

    assert teacher_messages_for_row(row, student_messages) == row["privileged_messages"]


def test_teacher_model_resource_detection() -> None:
    assert is_base_model_resource("accounts/fireworks/models/qwen3p5-9b")
    assert not is_base_model_resource("accounts/test/deployments/distillation-teacher-qwen3p5-9b")
    assert not is_base_model_resource("accounts/test/deployedModels/qwen3p5-9b")


class _FakeTeacherService:
    def __init__(self) -> None:
        self.direct_calls: list[dict] = []
        self.deployment_calls: list[dict] = []

    def create_deployment_sampler_for_model(self, model: str, *, tokenizer, inference_url=None):
        self.direct_calls.append({"model": model, "tokenizer": tokenizer, "inference_url": inference_url})
        return SimpleNamespace(model=model)

    def create_inference_deployment_sampler(self, config, *, timeout_s, cleanup_on_close, tokenizer):
        self.deployment_calls.append(
            {
                "config": config,
                "timeout_s": timeout_s,
                "cleanup_on_close": cleanup_on_close,
                "tokenizer": tokenizer,
            }
        )
        return SimpleNamespace(model=f"accounts/test/deployments/{config.deployment_id}")


def test_create_teacher_sampler_uses_existing_inference_model_directly() -> None:
    service = _FakeTeacherService()

    sampler = create_teacher_sampler(
        service,
        "accounts/test/deployments/teacher-existing",
        tokenizer="tok",
        inference_url="https://api.test",
        cleanup_on_close=True,
    )

    assert service.direct_calls == [
        {
            "model": "accounts/test/deployments/teacher-existing",
            "tokenizer": "tok",
            "inference_url": "https://api.test",
        }
    ]
    assert service.deployment_calls == []
    assert sampler.model == "accounts/test/deployments/teacher-existing"


def test_create_teacher_sampler_deploys_frozen_base_model() -> None:
    teacher_model = "accounts/fireworks/models/qwen3p5-9b"
    service = _FakeTeacherService()

    create_teacher_sampler(
        service,
        teacher_model,
        tokenizer="tok",
        inference_url="https://api.test",
        deployment_shape="shape-teacher",
        replica_count=2,
        timeout_s=123,
        cleanup_on_close=True,
    )

    assert service.direct_calls == []
    [call] = service.deployment_calls
    assert call["timeout_s"] == 123
    assert call["cleanup_on_close"] == CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO
    assert call["tokenizer"] == "tok"
    config = call["config"]
    assert config.deployment_id == "distillation-teacher-qwen3p5-9b"
    assert config.base_model == teacher_model
    assert config.deployment_shape == "shape-teacher"
    assert config.min_replica_count == 2
    assert config.max_replica_count == 2
    assert config.enable_hot_load is False
    assert config.hot_load_bucket_type is None
    assert config.for_training is True


def test_create_teacher_sampler_keeps_teacher_deployment_without_cleanup() -> None:
    service = _FakeTeacherService()

    create_teacher_sampler(
        service,
        "accounts/fireworks/models/qwen3p5-9b",
        tokenizer="tok",
        inference_url="https://api.test",
        deployment_id="my-teacher",
    )

    [call] = service.deployment_calls
    assert call["config"].deployment_id == "my-teacher"
    assert call["config"].deployment_shape is None
    assert call["cleanup_on_close"] is None


def test_default_teacher_deployment_id_is_stable_and_safe() -> None:
    assert default_teacher_deployment_id("accounts/fireworks/models/Qwen_3.5_9B") == "distillation-teacher-qwen-3-5-9b"


def test_qwen_opd_config_can_be_full_param_and_privileged() -> None:
    cfg = _build_qwen_opd_config(training_shape_id=None)

    assert cfg.lora_rank == 0
    assert cfg.trainer.training_shape_id is None
    assert cfg.base_model == cfg.teacher_model
    assert cfg.eval_every == 1
    assert cfg.max_seq_len == MAX_CONTEXT_LEN
    assert cfg.prompt_groups_per_step == cfg.max_rows


def test_qwen_opd_config_lets_pinned_shape_own_context_length() -> None:
    cfg = _build_qwen_opd_config(training_shape_id="accounts/fireworks/trainingShapes/qwen3p5-9b-256k")

    assert cfg.trainer.training_shape_id == "accounts/fireworks/trainingShapes/qwen3p5-9b-256k"
    assert cfg.max_seq_len is None


def test_main_requires_teacher_model() -> None:
    cfg = _build_qwen_opd_config()
    cfg.teacher_model = ""

    with pytest.raises(ValueError, match="teacher_model is required"):
        run_opd_main(cfg)


def test_main_rejects_non_positive_teacher_replicas() -> None:
    cfg = _build_qwen_opd_config()
    cfg.teacher_replica_count = 0

    with pytest.raises(ValueError, match="teacher_replica_count"):
        run_opd_main(cfg)


def test_main_samples_scores_trains_and_hotloads_each_step(monkeypatch) -> None:
    events: list[str] = []
    logged: list[dict] = []
    eval_steps: list[int] = []
    tokenizer = _FakeTokenizer({"q0": [10, 11], "q1": [12, 13], "q2": [14, 15], "privileged": [9]})

    class _Student:
        model = "accounts/test/deployments/student"

        def close(self):
            events.append("student-closed")

        def drain_metrics(self):
            return []

        async def sample_with_tokens(self, messages, *, n, **_kwargs):
            prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            events.append(f"sample:{messages[-1]['content']}")
            return [
                _student_sample(prompt + [40, 41], prompt_len=len(prompt), sampling_logprobs=[-0.4, -0.5])
                for _ in range(n)
            ]

    teacher = _FakeScoringSampler(response_logprob=-0.1)
    teacher.model = "accounts/test/deployments/teacher"

    class _Policy:
        lora_rank = 0

        def save_weights_for_sampler(self, name, checkpoint_type=None):
            return SimpleNamespace(path=f"/{name}")

        def forward_backward(self, datums, loss_fn, *, loss_fn_config=None):
            events.append(f"fwd-bwd:{len(datums)}:{loss_fn}:{loss_fn_config['ratio_log_cap']}")
            return SimpleNamespace(metrics={"loss:sum": 1.0})

        def optim_step(self, _params, **_kwargs):
            events.append("optim")
            return SimpleNamespace(metrics={})

    class _Service:
        trainer_job_id = "trainer"
        deployment_id = "deployment"
        max_context_length = 4096

        def create_training_client(self, *_args, **_kwargs):
            return _Policy()

        def create_deployment_sampler(self, **_kwargs):
            return _Student()

        def create_inference_deployment_sampler(self, config, **_kwargs):
            events.append(f"teacher-deployment:{config.deployment_id}")
            return teacher

        def hotload_sampler_snapshot(self, path):
            events.append(f"hotload:{path}")

        def close(self):
            events.append("close")

    class _Checkpoints:
        def __init__(self, *_args, **_kwargs):
            pass

        def resume(self, **_kwargs):
            return None

        def save(self, name, **kwargs):
            events.append(f"checkpoint:{name}:{kwargs['data_consumed']}")

    async def eval_fn(context: EvalContext) -> dict[str, float]:
        assert context.teacher_sampler is teacher
        eval_steps.append(context.step)
        return {"eval/score": float(context.step)}

    rows = [
        {"messages": [{"role": "user", "content": "q0"}]},
        {
            "messages": [{"role": "user", "content": "q1"}],
            "teacher_messages": [{"role": "user", "content": "privileged"}],
        },
        {"messages": [{"role": "user", "content": "q2"}]},
    ]
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    for name in ("setup_wandb", "validate_config", "log_metrics_json", "wandb_finish", "log_run_resources"):
        monkeypatch.setattr(distillation_loop, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(distillation_loop, "load_deployment_tokenizer", lambda *_args: tokenizer)
    monkeypatch.setattr(distillation_loop, "load_jsonl_dataset", lambda *_args: list(rows))
    monkeypatch.setattr(distillation_loop, "build_service_client", lambda **_kwargs: _Service())
    monkeypatch.setattr(
        distillation_loop,
        "ReconnectableClient",
        SimpleNamespace(from_training_client=lambda client, **_kwargs: client),
    )
    monkeypatch.setattr(distillation_loop, "TrainingCheckpoints", _Checkpoints)
    monkeypatch.setattr(distillation_loop, "log_metrics", lambda metrics, *, step: logged.append(dict(metrics)))

    cfg = _build_qwen_opd_config()
    cfg.epochs = 1
    cfg.prompt_groups_per_step = 2
    cfg.completions_per_prompt = 2
    cfg.save_final_checkpoint = True
    result = run_opd_main(cfg, eval_fn=eval_fn)

    assert result["steps"] == 2
    assert result["eval"] == {"eval/score": 2.0}
    assert eval_steps == [0, 1, 2]
    assert events[0] == "teacher-deployment:distillation-teacher-qwen3p5-9b-unit"
    assert events.index("hotload:/step-0-base") < events.index("sample:q0")
    assert events.index("fwd-bwd:4:importance_sampling:20.0") < events.index("hotload:/step-1")
    assert events.index("hotload:/step-1") < events.index("sample:q2")
    assert events.count("optim") == 2
    # The loop owns both samplers it created and closes them on exit.
    assert "student-closed" in events
    assert teacher.closed
    assert "checkpoint:step-2:3" in events
    # The privileged row is scored under the teacher prompt, not the student one.
    assert [9, 40, 41] in [prompt for prompt, _kwargs in teacher.calls]
    train_records = [record for record in logged if "train/opd_active_tokens" in record]
    assert [record["train/step"] for record in train_records] == [1, 2]
    assert train_records[0]["train/opd_active_tokens"] == pytest.approx(8.0)
    assert train_records[0]["train/loss:sum"] == pytest.approx(1.0)


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
                    "content": ("Use the privileged worked solution. " "End with exactly one line: Final: <answer>."),
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

    metrics = asyncio.run(
        evaluate_teacher_trace_logprob_gap(
            EvalContext(
                config=_build_qwen_opd_config(),
                rows=[
                    {
                        "messages": [{"role": "user", "content": "student"}],
                        "teacher_messages": [{"role": "user", "content": "teacher"}],
                        "expected_answer": "5",
                    }
                ],
                teacher_sampler=_FakeScoringSampler(
                    response_logprob=-0.1,
                    response_logprobs_by_token={40: -0.05},
                    tokenizer=tokenizer,
                    sample_text=teacher_trace,
                    sample_completion_tokens=[30, 31, 32, 33, 40],
                ),
                student_sampler=_FakeScoringSampler(
                    response_logprob=-0.5,
                    response_logprobs_by_token={40: -0.4},
                    tokenizer=tokenizer,
                    sample_text=student_generation,
                    sample_completion_tokens=[50, 51, 52, 40],
                ),
                tokenizer=tokenizer,
                step=3,
                max_seq_len=MAX_CONTEXT_LEN,
            ),
            trace_log_path=eval_file,
            min_pre_final_tokens=4,
        )
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


def test_opd_trace_result_validation_requires_opd_signal(tmp_path) -> None:
    cfg = _build_qwen_opd_config()
    metrics_file = tmp_path / "metrics.jsonl"
    cfg.epochs = 1
    metrics_file.write_text(
        "\n".join(
            [
                '{"step":0,"eval/opd_trace_teacher_nll":0.1,' '"eval/opd_trace_student_minus_teacher_nll":1.0}',
                '{"step":1,"train/opd_active_tokens":4,"train/opd_abs_advantage":0.25}',
                '{"step":1,"eval/opd_trace_teacher_nll":0.1,' '"eval/opd_trace_student_minus_teacher_nll":0.5}',
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
        metrics_file=metrics_file,
        min_max_seq_len=MAX_CONTEXT_LEN,
        min_teacher_final_accuracy=1.0,
        min_student_generation_accuracy=1.0,
        min_trace_gap_improvement=0.2,
    )

    metrics_file.write_text(
        "\n".join(
            [
                '{"step":0,"eval/opd_trace_teacher_nll":0.1,' '"eval/opd_trace_student_minus_teacher_nll":1.0}',
                '{"step":1,"train/opd_active_tokens":4,"train/opd_abs_advantage":0.0}',
                '{"step":1,"eval/opd_trace_teacher_nll":0.1,' '"eval/opd_trace_student_minus_teacher_nll":0.5}',
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
            metrics_file=metrics_file,
            min_max_seq_len=MAX_CONTEXT_LEN,
            min_teacher_final_accuracy=1.0,
            min_student_generation_accuracy=1.0,
            min_trace_gap_improvement=0.2,
        )


def test_opd_trace_result_validation_requires_trace_logprob_eval(tmp_path) -> None:
    cfg = _build_qwen_opd_config()
    metrics_file = tmp_path / "metrics.jsonl"
    cfg.epochs = 1
    metrics_file.write_text(
        "\n".join(
            [
                '{"step":0,"eval/opd_trace_teacher_nll":0.1,' '"eval/opd_trace_student_minus_teacher_nll":1.0}',
                '{"step":1,"train/opd_active_tokens":4,"train/opd_abs_advantage":0.25}',
                '{"step":1,"eval/opd_trace_teacher_nll":0.1,' '"eval/opd_trace_student_minus_teacher_nll":0.5}',
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
            metrics_file=metrics_file,
        )


def test_opd_trace_result_validation_requires_gap_improvement(tmp_path) -> None:
    cfg = _build_qwen_opd_config()
    metrics_file = tmp_path / "metrics.jsonl"
    cfg.epochs = 8
    train_records = [
        f'{{"step":{step},"train/opd_active_tokens":4,'
        f'"train/opd_abs_advantage":1.0,"train/opd_sampled_reverse_kl":1.0}}'
        for step in range(1, 9)
    ]
    eval_records = [
        f'{{"step":{step},"eval/opd_trace_teacher_nll":0.1,' f'"eval/opd_trace_student_minus_teacher_nll":1.0}}'
        for step in range(0, 9)
    ]
    metrics_file.write_text(
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
            metrics_file=metrics_file,
            min_train_gap_improvement=0.2,
            min_trace_gap_improvement=0.2,
        )


# -- Serverless (Tinker sampling client) scoring and the K3 example -------------


class _Done:
    """A completed future, as Tinker clients return."""

    def __init__(self, value):
        self.value = value

    def result(self, timeout=None):
        return self.value


class _LogprobSampler:
    """compute_logprobs: ``None`` for the first token, then a fixed logprob."""

    def __init__(self, logprob: float = -0.1):
        self.logprob = logprob
        self.scored: list[list[int]] = []
        self.closed = False

    def compute_logprobs(self, model_input):
        tokens = model_input.to_ints()
        self.scored.append(tokens)
        return _Done([None] + [self.logprob] * (len(tokens) - 1))

    def close(self):
        self.closed = True


def _sample_response(*sequences: tuple[list[int], list[float] | None]) -> SimpleNamespace:
    return SimpleNamespace(
        sequences=[
            SimpleNamespace(tokens=tokens, logprobs=logprobs, stop_reason="stop") for tokens, logprobs in sequences
        ]
    )


def test_completions_from_sample_response_prepends_the_student_prompt() -> None:
    completions = completions_from_sample_response([10, 11], _sample_response(([40, 41], [-0.4, -0.5]), ([42], None)))

    assert [c.full_tokens for c in completions] == [[10, 11, 40, 41], [10, 11, 42]]
    assert [c.prompt_len for c in completions] == [2, 2]
    assert [c.completion_len for c in completions] == [2, 1]
    assert completions[0].sampling_logprobs == [-0.4, -0.5]
    assert completions[1].sampling_logprobs is None
    assert completions[0].logprobs_echoed is False


def test_score_teacher_logprobs_with_sampling_client_slices_response() -> None:
    teacher = _LogprobSampler(logprob=-0.2)
    sampled = completions_from_sample_response([10, 11], _sample_response(([40, 41], [-0.4, -0.5]), ([], [])))

    scores = score_teacher_logprobs_with_sampling_client(teacher, [900, 901, 902], sampled, timeout=5)

    assert scores == [[-0.2, -0.2], None]
    assert teacher.scored == [[900, 901, 902, 40, 41]]
    group = build_opd_prompt_group(sampled, scores)
    assert group is not None
    assert group.teacher_logprobs == [[0.0, -0.2, -0.2]]
    assert group.sampling_logprobs == [[0.0, -0.4, -0.5]]


def test_serverless_k3_example_distills_from_step0_teacher(monkeypatch, tmp_path) -> None:
    from training.examples.distillation.serverless_k3_privileged import (
        train_serverless_k3_privileged as example,
    )

    events: list[str] = []
    teacher = _LogprobSampler(logprob=-0.1)
    students: list[SimpleNamespace] = []

    class Student:
        def __init__(self):
            self.closed = False
            students.append(self)

        def sample(self, *, prompt, num_samples, sampling_params):
            events.append(f"sample:{num_samples}")
            return _Done(_sample_response(*[([40, 41], [-0.5, -0.5])] * num_samples))

        def close(self):
            self.closed = True

    class TrainingClient:
        def save_weights_for_sampler(self, name):
            events.append(f"save:{name}")
            return _Done(SimpleNamespace(path=f"snap/{name}"))

        def forward_backward(self, datums, loss_fn):
            events.append(f"fwd-bwd:{len(datums)}:{loss_fn}")
            self.datums = datums
            return _Done(SimpleNamespace(metrics={"loss:sum": 1.0}))

        def optim_step(self, adam):
            events.append("optim")
            return _Done(SimpleNamespace(metrics={}))

    training_client = TrainingClient()

    class Service:
        def __init__(self, *, api_key, base_url):
            assert base_url == "https://api.example.test/training/v1/serverless"

        def create_lora_training_client(self, *, base_model, rank):
            return training_client

        def create_sampling_client(self, *, model_path, tokenizer):
            return teacher if model_path == "snap/sdft-teacher-step0" else Student()

        def close(self):
            events.append("service-close")

    class Renderer:
        def build_generation_prompt(self, messages):
            # System turn -> [900], each user turn -> [10, 11].
            tokens = [900 if m["role"] == "system" else 10 for m in messages] + [11]
            return sdk.ModelInput.from_ints(tokens)

        def get_stop_sequences(self):
            return []

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://api.example.test")
    monkeypatch.setattr(example, "FiretitanServiceClient", Service)
    monkeypatch.setattr(example, "load_tokenizer", lambda name: object())
    monkeypatch.setattr(example, "get_renderer", lambda name, tokenizer: Renderer())
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text('{"question": "q1"}\n{"question": "q2"}\n')

    cfg = example.Config(
        log_path=str(tmp_path / "logs"),
        dataset=str(dataset),
        steps=2,
        prompt_groups_per_step=2,
        completions_per_prompt=3,
    )
    result = example.main(cfg)

    assert events == [
        "save:sdft-teacher-step0",
        "save:sdft-student-0001",
        "sample:3",
        "sample:3",
        "fwd-bwd:6:importance_sampling",
        "optim",
        "save:sdft-student-0002",
        "sample:3",
        "sample:3",
        "fwd-bwd:6:importance_sampling",
        "optim",
        "save:sdft-student-final",
        "service-close",
    ]
    # The teacher scores the student's tokens under the privileged system turn.
    assert teacher.scored[0] == [900, 10, 11, 40, 41]
    assert teacher.closed and all(student.closed for student in students)
    # advantage = teacher (-0.1) - sampling (-0.5) on the two response tokens.
    datum = training_client.datums[0]
    assert datum.loss_fn_inputs["target_tokens"].data == [11, 40, 41]
    assert datum.loss_fn_inputs["advantages"].data == pytest.approx([0.0, 0.4, 0.4])
    assert result["final_snapshot"] == "snap/sdft-student-final"
    records = (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()
    assert len(records) == 2
    assert '"train/opd_sampled_reverse_kl"' in records[0]
