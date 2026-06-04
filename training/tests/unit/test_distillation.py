"""Unit tests for sampled-token OPD helpers."""

from __future__ import annotations

import threading
from math import exp
from pathlib import Path
from types import SimpleNamespace

import pytest
import tinker

from training.recipes.distillation_loop import (
    Config,
    _default_teacher_deployment_id,
    _is_base_model_resource,
    _request_frozen_teacher_deployment,
    _resolve_teacher_specs,
    _teacher_deployment_shape_for_spec,
    _teacher_deployment_id_for_spec,
    _teacher_metric_slug,
    _teacher_top_logprobs,
    _validate_teacher_tokenizers,
    _wait_frozen_teacher_deployments,
    main as run_opd_main,
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
from training.utils.distillation.sampling import (
    _align_completion_logprobs,
    _align_response_logprobs,
    _build_teacher_scoring_tokens,
    _extract_scored_token_logprobs,
    _slice_response_logprobs,
    _teacher_messages_for_row,
    _tokenize_teacher_prompt,
)
from training.utils import DeployConfig, RunnerConfig, TrainerConfig
from training.utils.distillation import (
    MultiTeacherConfig,
    OPDPromptGroup,
    TeacherConfig,
    build_opd_server_datums,
    combine_opd_prompt_groups,
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

    async def async_completions_stream(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        content = [{"logprob": 0.0}]
        content.extend(
            {"logprob": self.response_logprobs_by_token.get(int(token_id), self.response_logprob)}
            for token_id in prompt[1:]
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
                logprobs_echoed=False,
            )
        ]


class _FakeResponse:
    status_code = 200

    def __init__(self, payload=None):
        self._payload = payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeDeployMgr:
    account_id = "acct"
    api_key = "fake-api-key"
    base_url = "https://api.fireworks.ai"
    additional_headers = {"x-test": "yes"}

    def __init__(self, existing=None):
        self.existing = existing
        self.created_config = None
        self.waited = []

    def get(self, deployment_id):
        self.get_id = deployment_id
        return self.existing

    def create_or_get(self, config):
        self.created_config = config
        return SimpleNamespace(
            deployment_id=config.deployment_id,
            state="CREATING",
            inference_model=None,
        )

    def _parse_deployment_info(self, deployment_id, payload):
        return SimpleNamespace(
            deployment_id=deployment_id,
            state="CREATING",
            inference_model=None,
        )

    def wait_for_ready(self, deployment_id, timeout_s):
        self.waited.append((deployment_id, timeout_s))
        return SimpleNamespace(inference_model=f"accounts/acct/deployedModels/{deployment_id}")


class _FakeCleanup:
    def __init__(self):
        self.deployments = []

    def __call__(self, deployment_id):
        self.deployments.append(deployment_id)


class _BlockingWaitDeployMgr(_FakeDeployMgr):
    def __init__(self, expected_waiters: int):
        super().__init__()
        self.expected_waiters = expected_waiters
        self.active_waiters = 0
        self.max_active_waiters = 0
        self.all_waiters_active = threading.Event()
        self.wait_lock = threading.Lock()

    def wait_for_ready(self, deployment_id, timeout_s):
        with self.wait_lock:
            self.active_waiters += 1
            self.max_active_waiters = max(self.max_active_waiters, self.active_waiters)
            if self.active_waiters == self.expected_waiters:
                self.all_waiters_active.set()

        self.all_waiters_active.wait(timeout=timeout_s)

        with self.wait_lock:
            self.active_waiters -= 1
        return super().wait_for_ready(deployment_id, timeout_s)


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
    assert not _is_base_model_resource("accounts/pyroworks/deployments/distillation-teacher-qwen3p5-9b")
    assert not _is_base_model_resource("accounts/pyroworks/deployedModels/qwen3p5-9b")


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


def test_request_frozen_teacher_deployment_requests_frozen_deployment() -> None:
    deploy_mgr = _FakeDeployMgr()
    cleanup = _FakeCleanup()

    request = _request_frozen_teacher_deployment(
        deploy_mgr,
        base_model="accounts/fireworks/models/qwen3p5-9b",
        deployment_id="distillation-teacher-unit",
        deployment_shape="accounts/fireworks/deploymentShapes/qwen/versions/v1",
        replica_count=2,
        cleanup=cleanup,
    )

    assert request.deployment_id == "distillation-teacher-unit"
    assert deploy_mgr.created_config is not None
    assert deploy_mgr.created_config.deployment_id == "distillation-teacher-unit"
    assert deploy_mgr.created_config.base_model == "accounts/fireworks/models/qwen3p5-9b"
    assert deploy_mgr.created_config.enable_hot_load is False
    assert deploy_mgr.created_config.hot_load_bucket_type is None
    assert deploy_mgr.created_config.min_replica_count == 2
    assert deploy_mgr.created_config.max_replica_count == 2
    assert deploy_mgr.created_config.deployment_shape == "accounts/fireworks/deploymentShapes/qwen/versions/v1"
    assert cleanup.deployments == ["distillation-teacher-unit"]


def test_request_frozen_teacher_reuses_existing_deployment_without_cleanup() -> None:
    deploy_mgr = _FakeDeployMgr(existing=SimpleNamespace(state="READY"))
    cleanup = _FakeCleanup()

    request = _request_frozen_teacher_deployment(
        deploy_mgr,
        base_model="accounts/fireworks/models/qwen3p5-9b",
        deployment_id="distillation-teacher-existing",
        deployment_shape="shape-v1",
        replica_count=1,
        cleanup=cleanup,
    )

    assert request.deployment_id == "distillation-teacher-existing"
    assert request.info is deploy_mgr.existing
    assert deploy_mgr.created_config is None
    assert cleanup.deployments == []


def test_wait_frozen_teacher_deployments_waits_in_parallel() -> None:
    deploy_mgr = _BlockingWaitDeployMgr(expected_waiters=2)
    requests = {
        "accounts/fireworks/models/math-teacher": SimpleNamespace(deployment_id="math-teacher"),
        "accounts/fireworks/models/code-teacher": SimpleNamespace(deployment_id="code-teacher"),
    }

    resolved = _wait_frozen_teacher_deployments(deploy_mgr, requests, timeout_s=1.0)

    assert deploy_mgr.max_active_waiters == 2
    assert resolved == {
        "accounts/fireworks/models/math-teacher": "accounts/acct/deployedModels/math-teacher",
        "accounts/fireworks/models/code-teacher": "accounts/acct/deployedModels/code-teacher",
    }


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


def test_teacher_top_logprobs_uses_per_teacher_override() -> None:
    spec = TeacherConfig(model="accounts/fireworks/models/math-teacher", top_logprobs=3)

    assert _teacher_top_logprobs(spec, default_top_logprobs=1) == 3


def test_teacher_top_logprobs_rejects_negative_default() -> None:
    with pytest.raises(ValueError, match="teacher_top_logprobs"):
        _teacher_top_logprobs(TeacherConfig(model="accounts/fireworks/models/math-teacher"), -1)


def test_teacher_config_rejects_negative_top_logprobs() -> None:
    with pytest.raises(ValueError, match="top_logprobs"):
        TeacherConfig(model="accounts/fireworks/models/math-teacher", top_logprobs=-1)


def test_teacher_config_rejects_empty_deployment_shape() -> None:
    with pytest.raises(ValueError, match="deployment_shape"):
        TeacherConfig(
            model="accounts/fireworks/models/math-teacher",
            deployment_shape="",
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
            student_deployment_shape="shape-student",
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
            student_deployment_shape="shape-student",
        )
        == "shape-run"
    )


def test_teacher_deployment_shape_reuses_student_shape_for_same_base_model() -> None:
    cfg = Config(log_path="/tmp/opd", base_model="accounts/fireworks/models/student")
    spec = TeacherConfig(model="accounts/fireworks/models/student")

    assert (
        _teacher_deployment_shape_for_spec(
            spec,
            cfg,
            student_deployment_shape="shape-student",
        )
        == "shape-student"
    )


def test_teacher_deployment_shape_lets_api_choose_for_heterogeneous_teacher() -> None:
    cfg = Config(log_path="/tmp/opd", base_model="accounts/fireworks/models/student")
    spec = TeacherConfig(model="accounts/fireworks/models/teacher")

    assert (
        _teacher_deployment_shape_for_spec(
            spec,
            cfg,
            student_deployment_shape="shape-student",
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
