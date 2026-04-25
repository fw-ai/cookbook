"""Unit tests for sampled-token OPD helpers."""

from __future__ import annotations

from math import exp
from pathlib import Path
from types import SimpleNamespace

import pytest
import tinker
from fireworks import omit

from training.recipes.opd_loop import (
    Config,
    _default_teacher_deployment_id,
    _is_base_model_resource,
    _make_teacher_deployment_provisioner,
    _request_frozen_teacher_deployment,
)
from training.utils.opd_eval import (
    evaluate_teacher_trace_logprob_gap,
    extract_final_answer,
    expected_final_answer,
    make_teacher_trace_logprob_gap_eval,
    normalize_final_answer,
    validate_privileged_opd_dataset,
    validate_opd_trace_result,
)
from training.utils.opd_sampling import (
    _align_completion_logprobs,
    _align_response_logprobs,
    _build_teacher_scoring_tokens,
    _extract_scored_token_logprobs,
    _slice_response_logprobs,
    _teacher_messages_for_row,
    _tokenize_teacher_prompt,
)
from training.utils import DeployConfig, InfraConfig, RunnerConfig, WeightSyncConfig
from training.utils.opd import (
    OPDPromptGroup,
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
        teacher_deployment_id="opd-teacher-qwen3p5-9b-unit",
        dataset="/tmp/opd_math.jsonl",
        infra=InfraConfig(training_shape_id=training_shape_id),
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3.5-9B"),
        weight_sync=WeightSyncConfig(weight_sync_interval=1),
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

    def deployment(self, deployment_id, action="delete"):
        self.deployments.append((deployment_id, action))


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
    assert inputs["logprobs"].data == sampling_lp[0]
    assert inputs["advantages"].data == pytest.approx([0.0, 0.6, -0.4, 0.8])
    assert metrics["opd_active_tokens"] == pytest.approx(3.0)
    assert metrics["opd_sampled_reverse_kl"] == pytest.approx((-0.3 + 0.2 - 0.4) / 3)
    assert metrics["opd_advantage"] == pytest.approx((0.6 - 0.4 + 0.8) / 3)
    assert metrics["opd_abs_advantage"] == pytest.approx((0.6 + 0.4 + 0.8) / 3)
    assert metrics["opd_student_logprob_minus_teacher_logprob"] == metrics["opd_sampled_reverse_kl"]
    assert metrics["opd_teacher_logprob_minus_student_logprob"] == metrics["opd_advantage"]
    assert metrics["opd_abs_logprob_gap"] == metrics["opd_abs_advantage"]


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


def test_teacher_model_resource_detection() -> None:
    assert _is_base_model_resource("accounts/fireworks/models/qwen3p5-9b")
    assert not _is_base_model_resource("accounts/pyroworks/deployments/opd-teacher-qwen3p5-9b")
    assert not _is_base_model_resource("accounts/pyroworks/deployedModels/qwen3p5-9b")


def test_default_teacher_deployment_id_is_stable_and_safe() -> None:
    assert (
        _default_teacher_deployment_id("accounts/fireworks/models/Qwen_3.5_9B")
        == "opd-teacher-qwen-3-5-9b"
    )


def test_teacher_deployment_provisioner_requests_frozen_deployment_and_waits(monkeypatch) -> None:
    create_calls = []

    class _FakeFireworks:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.deployments = self
            self.deployment_shape_versions = self

        def list(self, **kwargs):
            return [
                SimpleNamespace(
                    model_dump=lambda by_alias=True: {
                        "snapshot": {"acceleratorType": "NVIDIA_B200_180GB"}
                    }
                )
            ]

        def get(self, **kwargs):
            return SimpleNamespace(
                model_dump=lambda by_alias=True: {
                    "snapshot": {"acceleratorType": "NVIDIA_B200_180GB"}
                }
            )

        def create(self, **kwargs):
            create_calls.append((self.kwargs, kwargs))
            return SimpleNamespace(
                name="ignored",
                state="CREATING",
                hot_load_bucket_url=None,
                model_dump=lambda by_alias=True: {
                    "name": "ignored",
                    "state": "CREATING",
                }
            )

    monkeypatch.setattr("training.utils.infra.Fireworks", _FakeFireworks)
    deploy_mgr = _FakeDeployMgr()
    cleanup = _FakeCleanup()
    cfg = Config(
        log_path="/tmp/opd",
        teacher_model="accounts/fireworks/models/qwen3p5-9b",
        teacher_deployment_id="opd-teacher-unit",
        teacher_replica_count=2,
        teacher_deployment_timeout_s=123,
    )

    model_out = {}
    provisioner = _make_teacher_deployment_provisioner(
        cfg,
        deploy_mgr,
        cleanup=cleanup,
        model_out=model_out,
    )
    resource = provisioner("accounts/fireworks/deploymentShapes/qwen/versions/v1")

    assert resource is not None
    label, wait_fn = resource
    assert label == "teacher_deployment"
    assert deploy_mgr.created_config is None
    assert len(create_calls) == 1
    client_kwargs, create_kwargs = create_calls[0]
    assert client_kwargs == {
        "api_key": "fake-api-key",
        "account_id": "acct",
        "base_url": "https://api.fireworks.ai",
        "default_headers": {"x-test": "yes"},
    }
    assert create_kwargs == {
        "account_id": "acct",
        "deployment_id": "opd-teacher-unit",
        "base_model": "accounts/fireworks/models/qwen3p5-9b",
        "enable_hot_load": False,
        "min_replica_count": 2,
        "max_replica_count": 2,
        "deployment_shape": "accounts/fireworks/deploymentShapes/qwen/versions/v1",
        "accelerator_type": omit,
        "placement": {"region": "US_OHIO_1"},
        "skip_shape_validation": omit,
        "disable_speculative_decoding": True,
        "timeout": 60,
    }
    assert cleanup.deployments == [("opd-teacher-unit", "scale_to_zero")]
    wait_fn()
    assert model_out["teacher_model"] == "accounts/acct/deployedModels/opd-teacher-unit"
    assert deploy_mgr.waited == [("opd-teacher-unit", 123)]


def test_request_frozen_teacher_reuses_existing_deployment_without_cleanup() -> None:
    deploy_mgr = _FakeDeployMgr(existing=SimpleNamespace(state="READY"))
    cleanup = _FakeCleanup()
    cfg = Config(log_path="/tmp/opd")

    request = _request_frozen_teacher_deployment(
        deploy_mgr,
        infra_cfg=cfg.infra,
        base_model="accounts/fireworks/models/qwen3p5-9b",
        deployment_id="opd-teacher-existing",
        deployment_shape="shape-v1",
        replica_count=1,
        cleanup=cleanup,
    )

    assert request.deployment_id == "opd-teacher-existing"
    assert request.info is deploy_mgr.existing
    assert deploy_mgr.created_config is None
    assert cleanup.deployments == []


def test_qwen_opd_config_can_be_full_param_and_privileged() -> None:
    cfg = _build_qwen_opd_config(training_shape_id=None)

    assert cfg.lora_rank == 0
    assert cfg.infra.training_shape_id is None
    assert cfg.base_model == cfg.teacher_model
    assert cfg.step_eval is not None
    assert cfg.step_eval_interval == 1
    assert cfg.eval_before_training is True
    assert cfg.max_seq_len == MAX_CONTEXT_LEN
    assert cfg.prompt_groups_per_step == cfg.max_rows
    assert cfg.weight_sync.weight_sync_interval == 1


def test_qwen_opd_config_lets_pinned_shape_own_context_length() -> None:
    cfg = _build_qwen_opd_config(
        training_shape_id="accounts/fireworks/trainingShapes/qwen3p5-9b-256k"
    )

    assert cfg.infra.training_shape_id == "accounts/fireworks/trainingShapes/qwen3p5-9b-256k"
    assert cfg.max_seq_len is None


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
