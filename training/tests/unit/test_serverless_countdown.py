from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

import training.examples.serverless_rl.countdown_rl as countdown_rl
from training.examples.serverless_rl.countdown_rl import (
    MAX_PROMOTABLE_CHECKPOINT_NAME_LEN,
    Config,
    _account_from_session,
    _carve_out_eval_rows,
    _control_plane_base_url,
    _find_promotable,
    _group_relative_advantages,
    _is_truncated,
    _mean_policy_sample_logprob_gap,
    _mean_policy_sample_k3,
    _serverless_base_url,
    _step_from_checkpoint_name,
    _validate_config,
    _validate_length,
    _validate_resume_reference,
)


def test_serverless_defaults_use_supported_model_and_matching_tokenizer():
    cfg = Config()
    assert cfg.base_model == "accounts/fireworks/models/kimi-k3"
    assert cfg.tokenizer_model == "moonshotai/Kimi-K3"
    # renderer_name defaults to "" and is auto-resolved from the tokenizer.
    assert cfg.renderer_name == ""
    assert cfg.max_seq_len > 0
    assert cfg.lora_rank > 0
    assert cfg.router_replay is True
    assert cfg.eval_prompt_groups == 16
    assert cfg.eval_interval == 5
    assert cfg.eval_group_size == 8


def test_renderer_name_auto_resolves_from_tokenizer():
    from training.examples.serverless_rl.countdown_rl import resolve_renderer_name

    assert resolve_renderer_name("moonshotai/Kimi-K3", "") == "kimi_k3"
    assert resolve_renderer_name("deepseek-ai/DeepSeek-V4-Flash-0731", "") == "deepseek_v4"
    # An explicit override is honored.
    assert resolve_renderer_name("moonshotai/Kimi-K3", "kimi_k3_disable_thinking") == "kimi_k3_disable_thinking"


@pytest.mark.parametrize("max_seq_len", [0, -1])
def test_serverless_rejects_invalid_sequence_bound(max_seq_len):
    cfg = Config(max_seq_len=max_seq_len)
    with pytest.raises(ValueError, match="max_seq_len > 0"):
        _validate_config(cfg)


def test_serverless_rejects_non_lora_config():
    cfg = Config(lora_rank=0)
    with pytest.raises(ValueError, match="lora_rank > 0"):
        _validate_config(cfg)


def test_serverless_rejects_negative_dcp_save_interval():
    with pytest.raises(ValueError, match="dcp_save_interval"):
        _validate_config(Config(dcp_save_interval=-1))


def test_serverless_allows_disabled_dcp_saves():
    _validate_config(Config(dcp_save_interval=0))


def test_serverless_rejects_eval_without_carved_rows():
    with pytest.raises(ValueError, match="eval_prompt_groups"):
        _validate_config(Config(eval_prompt_groups=0))


def test_carved_eval_rows_are_deterministic_and_disjoint():
    rows = [{"id": i} for i in range(20)]
    train_a, eval_a, indices_a = _carve_out_eval_rows(rows, 4, 7)
    train_b, eval_b, indices_b = _carve_out_eval_rows(rows, 4, 7)

    assert (train_a, eval_a, indices_a) == (train_b, eval_b, indices_b)
    assert len(train_a) == 16
    assert len(eval_a) == 4
    assert {row["id"] for row in train_a}.isdisjoint(row["id"] for row in eval_a)
    assert [rows[index] for index in indices_a] == eval_a


def test_carved_eval_rows_must_leave_training_data():
    with pytest.raises(ValueError, match="leave at least one training row"):
        _carve_out_eval_rows([{"id": 0}], 1, 0)


def test_truncation_falls_back_to_exact_token_cap_without_finish_reason():
    assert _is_truncated(SimpleNamespace(tokens=[1, 2], finish_reason=""), 2) is True
    assert _is_truncated(SimpleNamespace(tokens=[1], finish_reason=""), 2) is False


def test_promotion_rejects_overlong_final_checkpoint_name():
    with pytest.raises(ValueError, match="too long to promote"):
        _validate_config(
            Config(
                final_checkpoint_name="x" * (MAX_PROMOTABLE_CHECKPOINT_NAME_LEN + 1),
                output_model_id="countdown-model",
            )
        )


def test_promotion_validates_output_model_id():
    with pytest.raises(ValueError, match="lowercase"):
        _validate_config(Config(output_model_id="Bad_Model_ID"))


def test_serverless_length_accepts_exact_bound():
    _validate_length("prompt", 1024, 1024)


def test_serverless_length_rejects_overflow():
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        _validate_length("prompt", 1025, 1024)


def test_serverless_base_url_appends_suffix():
    assert _serverless_base_url("https://api.fireworks.ai") == "https://api.fireworks.ai/training/v1/serverless"
    assert _serverless_base_url("https://api.fireworks.ai/training/v1") == "https://api.fireworks.ai/training/v1/serverless"
    assert _serverless_base_url("https://api.fireworks.ai/training/v1/serverless") == "https://api.fireworks.ai/training/v1/serverless"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.fireworks.ai",
        "https://api.fireworks.ai/",
        "https://api.fireworks.ai/training/v1",
        "https://api.fireworks.ai/training/v1/serverless",
    ],
)
def test_control_plane_base_url_strips_training_suffixes(base_url):
    assert _control_plane_base_url(base_url) == "https://api.fireworks.ai"


@pytest.mark.parametrize(
    "reference",
    ["training/run-abc/cd-state-0002", "cd-state-0001"],
)
def test_validate_resume_reference_accepts_numbered_dcp(reference):
    _validate_resume_reference(reference)


@pytest.mark.parametrize(
    "reference",
    ["cd-final", "cd-state", "", "run-x/", "cd-sample-0002", "training/run-abc/cd-sample-0002"],
)
def test_validate_resume_reference_rejects_sampler_or_unnumbered_names(reference):
    with pytest.raises(ValueError, match="numbered training checkpoint"):
        _validate_resume_reference(reference)


def test_validate_config_uses_configured_training_checkpoint_prefix():
    _validate_config(Config(resume_from="training/run-abc/custom-state-0012", training_checkpoint_name="custom-state"))

    with pytest.raises(ValueError, match="custom-state"):
        _validate_config(Config(resume_from="training/run-abc/cd-state-0012", training_checkpoint_name="custom-state"))


def test_step_from_checkpoint_name_recovers_completed_steps():
    assert _step_from_checkpoint_name("training/run-abc/cd-state-0012") == 12
    assert _step_from_checkpoint_name("cd-final") == 0


def test_account_from_session_parses_account():
    assert _account_from_session("accounts/training/trainingSessions/ts-1") == "training"
    assert _account_from_session(None) is None


def test_resume_reference_uses_session_account_without_control_plane_lookup(monkeypatch):
    class _UnexpectedClient:
        def __init__(self, **kwargs):
            raise AssertionError(f"unexpected FireworksClient construction: {kwargs}")

    monkeypatch.setattr(countdown_rl, "FireworksClient", _UnexpectedClient)
    runner = object.__new__(countdown_rl.ServerlessCountdownRL)
    runner.cfg = Config()
    runner.session_name = "accounts/training/trainingSessions/ts-1"
    runner.run_id = "run-abc"

    assert runner._resume_reference("cd-state-0002") == "training/run-abc/cd-state-0002"


def test_resume_reference_falls_back_to_control_plane_account(monkeypatch):
    clients = []

    class _FallbackClient:
        account_id = "fallback-account"

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            clients.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(countdown_rl, "FireworksClient", _FallbackClient)
    runner = object.__new__(countdown_rl.ServerlessCountdownRL)
    runner.cfg = Config(api_key="fw-test", api_base_url="https://api.fireworks.ai/training/v1/serverless")
    runner.session_name = None
    runner.run_id = "run-abc"

    assert runner._resume_reference("cd-state-0002") == "fallback-account/run-abc/cd-state-0002"
    assert clients[0].kwargs == {"api_key": "fw-test", "base_url": "https://api.fireworks.ai"}
    assert clients[0].closed is True


def test_resume_reference_rejects_missing_run_id():
    runner = object.__new__(countdown_rl.ServerlessCountdownRL)
    runner.cfg = Config()
    runner.session_name = "accounts/training/trainingSessions/ts-1"
    runner.run_id = None

    with pytest.raises(RuntimeError, match="run_id"):
        runner._resume_reference("cd-state-0002")


def test_find_promotable_matches_run_prefixed_sampler_checkpoint():
    checkpoints = [
        {"name": "accounts/a/trainingSessions/s/checkpoints/run-aaa-cd-final-deadbeef", "promotable": True},
        {"name": "accounts/a/trainingSessions/s/checkpoints/cd-state-0002", "promotable": False},
    ]
    match = _find_promotable(checkpoints, "cd-final", "run-aaa")
    assert match is not None
    assert match["promotable"] is True


def test_group_relative_advantages_standardizes():
    advs = _group_relative_advantages([0.0, 1.0])
    assert advs[0] < 0 < advs[1]
    assert abs(advs[0] + advs[1]) < 1e-6


def test_group_relative_advantages_single_sample_is_zero():
    assert _group_relative_advantages([0.7]) == [0.0]


def test_group_relative_advantages_uses_true_std_below_one():
    """Countdown rewards live in [0, 1], so group std is < 1; GRPO must divide by
    the true std, not a 1.0 floor (which would reduce it to mean-centering)."""
    # std of [0.0, 0.2] is 0.1414 (< 1). If floored to 1.0 the spread would be ~0.2.
    advs = _group_relative_advantages([0.0, 0.2])
    assert advs[1] - advs[0] > 1.0  # standardized spread, not the raw 0.2


def test_group_relative_advantages_constant_group_is_zero_not_nan():
    advs = _group_relative_advantages([0.5, 0.5, 0.5])
    assert all(a == 0.0 for a in advs)


class _TensorData:
    def __init__(self, data: list[float]) -> None:
        self.data = data


def test_k1_reads_dict_and_object_outputs_plus_tensordata_inputs():
    """forward_backward may return dict or object rows; Datum stores TensorData."""
    datum = SimpleNamespace(
        loss_fn_inputs={
            "logprobs": _TensorData([-0.1, -0.2, -2.0]),
            "target_tokens": _TensorData([0.0, 0.0, 7.0]),
        }
    )
    dict_fb = SimpleNamespace(loss_fn_outputs=[{"logprobs": [0.0, 0.0, -1.0]}])
    object_fb = SimpleNamespace(
        loss_fn_outputs=[SimpleNamespace(logprobs=_TensorData([0.0, 0.0, -1.0]))]
    )
    assert _mean_policy_sample_logprob_gap([datum], dict_fb) == pytest.approx(1.0)
    assert _mean_policy_sample_logprob_gap([datum], object_fb) == pytest.approx(1.0)


def test_k1_returns_none_when_logprobs_are_absent():
    datum = SimpleNamespace(loss_fn_inputs={"logprobs": [0.0], "target_tokens": [1.0]})
    assert _mean_policy_sample_logprob_gap([datum], SimpleNamespace(loss_fn_outputs=[{}])) is None
    assert _mean_policy_sample_logprob_gap([datum], SimpleNamespace(loss_fn_outputs=[])) is None


def test_k3_reads_response_logprobs_and_is_non_negative():
    datum = SimpleNamespace(
        loss_fn_inputs={
            "logprobs": _TensorData([-0.1, -0.2, -2.0, -3.0]),
            "target_tokens": _TensorData([0.0, 0.0, 7.0, 8.0]),
        }
    )
    fb = SimpleNamespace(loss_fn_outputs=[{"logprobs": [9.0, 9.0, -1.0, -3.5]}])
    expected = ((math.exp(1.0) - 1.0 - 1.0) + (math.exp(-0.5) - 1.0 + 0.5)) / 2

    assert _mean_policy_sample_k3([datum], fb) == pytest.approx(expected)
    assert _mean_policy_sample_k3([datum], fb) >= 0.0


def test_k3_returns_none_when_logprobs_are_absent():
    datum = SimpleNamespace(loss_fn_inputs={"logprobs": [0.0], "target_tokens": [1.0]})
    assert _mean_policy_sample_k3([datum], SimpleNamespace(loss_fn_outputs=[{}])) is None
    assert _mean_policy_sample_k3([datum], SimpleNamespace(loss_fn_outputs=[])) is None


@pytest.mark.parametrize(
    ("logger", "position", "metric"),
    [
        (countdown_rl._log_wandb_step, 3, {"kld/mean_k3": 0.125}),
        (countdown_rl._log_wandb_eval, 5, {"eval/raw_reward": 0.75}),
    ],
)
def test_wandb_rows_commit_immediately_without_explicit_internal_step(
    monkeypatch, logger, position, metric
):
    wandb = ModuleType("wandb")
    wandb.run = object()
    wandb.log = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    logger(position, metric)

    wandb.log.assert_called_once_with({"train/step": position, **metric})


def test_reward_plot_uses_checkpoint_resolved_base_model(monkeypatch, tmp_path):
    figure = MagicMock()
    axes = MagicMock()
    matplotlib = ModuleType("matplotlib")
    matplotlib.use = MagicMock()
    pyplot = ModuleType("matplotlib.pyplot")
    pyplot.subplots = MagicMock(return_value=(figure, axes))
    pyplot.close = MagicMock()
    matplotlib.pyplot = pyplot
    wandb = ModuleType("wandb")
    wandb.run = None
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    runner = object.__new__(countdown_rl.ServerlessCountdownRL)
    runner.cfg = Config(base_model="accounts/fireworks/models/stale-cli-model")
    runner.base_model = "accounts/fireworks/models/checkpoint-model"
    runner.run_dir = tmp_path
    runner._plot(
        [
            {
                "step": 2,
                "rollout/raw_reward": 0.5,
                "rollout/filtered_reward": 0.4,
            }
        ]
    )

    axes.set_title.assert_called_once_with(
        "Serverless Countdown RL (accounts/fireworks/models/checkpoint-model, importance_sampling)"
    )


def test_countdown_r3_keeps_each_route_matrix_with_its_sequence(monkeypatch, tmp_path):
    """Variable-length samples must not reuse the last sequence's R3 payload."""

    class _Immediate:
        def __init__(self, value):
            self.value = value

        def result(self, timeout=None):
            del timeout
            return self.value

    long_seq = SimpleNamespace(
        tokens=[101, 102, 103, 104, 105],
        logprobs=[-0.1] * 5,
        routing_matrices=["long-route"] * 5,
    )
    short_seq = SimpleNamespace(
        tokens=[201, 202, 203],
        logprobs=[-0.2] * 3,
        routing_matrices=["short-route"] * 3,
    )
    sample_response = SimpleNamespace(sequences=[long_seq, short_seq])

    class _Sampler:
        def sample(self, **kwargs):
            del kwargs
            return _Immediate(sample_response)

        def close(self):
            return None

    class _Service:
        def create_sampling_client(self, **kwargs):
            del kwargs
            return _Sampler()

    captured_datums = []

    class _TrainingClient:
        def save_weights_for_sampler(self, name):
            del name
            return _Immediate(SimpleNamespace(path="tinker://snapshot"))

        def forward_backward(self, datums, loss_name):
            assert loss_name == "importance_sampling"
            captured_datums.extend(datums)
            outputs = [
                {"logprobs": [0.0] * datum.model_input.length}
                for datum in datums
            ]
            return _Immediate(
                SimpleNamespace(
                    metrics={"loss:sum": 1.0, "response_tokens": 8},
                    loss_fn_outputs=outputs,
                )
            )

        def optim_step(self, params):
            del params
            return _Immediate(None)

    prompt = countdown_rl.tinker.ModelInput.from_ints([7, 8, 9])

    class _Renderer:
        def build_generation_prompt(self, messages):
            del messages
            return prompt

        def get_stop_sequences(self):
            return []

        def parse_response(self, tokens):
            return ("long" if tokens[0] == 101 else "short", None)

    monkeypatch.setattr(countdown_rl, "get_text_content", lambda content: content)
    monkeypatch.setattr(
        countdown_rl,
        "composite_reward",
        lambda content, ground_truth: 1.0 if content == "long" else 0.0,
    )

    runner = object.__new__(countdown_rl.ServerlessCountdownRL)
    runner.cfg = Config(
        prompt_groups_per_step=1,
        group_size=2,
        prompt_concurrency=1,
        max_sample_tokens=8,
        max_seq_len=64,
    )
    runner.training_client = _TrainingClient()
    runner.service = _Service()
    runner.tokenizer = None
    runner.renderer = _Renderer()
    runner.router_replay_enabled = True
    runner.metrics_path = tmp_path / "metrics.jsonl"
    runner._next_batch = lambda: [{"messages": [], "ground_truth": 0}]

    rec = runner._step(0)

    assert rec["train/trained"] is True
    assert rec["kld/mean_k3"] == rec["train/inference_k3"]
    assert rec["rollout/truncated_samples"] == 0
    assert len(captured_datums) == 2
    assert captured_datums[0].model_input.routing_matrices == [
        "",
        "",
        *long_seq.routing_matrices,
    ]
    assert captured_datums[1].model_input.routing_matrices == [
        "",
        "",
        *short_seq.routing_matrices,
    ]


def test_fixed_eval_reuses_carved_rows_without_training(monkeypatch, tmp_path):
    class _Immediate:
        def __init__(self, value):
            self.value = value

        def result(self, timeout=None):
            del timeout
            return self.value

    sequences = [
        SimpleNamespace(tokens=[1], finish_reason="stop"),
        SimpleNamespace(tokens=[2, 2], finish_reason="length"),
    ]

    class _Sampler:
        def sample(self, **kwargs):
            assert kwargs["num_samples"] == 2
            return _Immediate(SimpleNamespace(sequences=sequences))

        def close(self):
            return None

    class _Service:
        def create_sampling_client(self, **kwargs):
            assert kwargs["model_path"] == "tinker://eval-snapshot"
            return _Sampler()

    class _TrainingClient:
        def save_weights_for_sampler(self, name):
            assert name == "cd-eval-0005"
            return _Immediate(SimpleNamespace(path="tinker://eval-snapshot"))

    prompt = countdown_rl.tinker.ModelInput.from_ints([7, 8, 9])

    class _Renderer:
        def build_generation_prompt(self, messages):
            del messages
            return prompt

        def get_stop_sequences(self):
            return []

        def parse_response(self, tokens):
            return ("correct" if tokens == [1] else "wrong", None)

    monkeypatch.setattr(countdown_rl, "get_text_content", lambda content: content)
    monkeypatch.setattr(
        countdown_rl,
        "composite_reward",
        lambda content, ground_truth: 1.0 if content == "correct" else 0.0,
    )

    runner = object.__new__(countdown_rl.ServerlessCountdownRL)
    runner.cfg = Config(
        eval_prompt_groups=2,
        eval_group_size=2,
        prompt_concurrency=2,
        max_sample_tokens=8,
        max_seq_len=64,
    )
    runner.training_client = _TrainingClient()
    runner.service = _Service()
    runner.tokenizer = None
    runner.renderer = _Renderer()
    runner.eval_rows = [
        {"messages": [], "ground_truth": '{"numbers": [1], "target": 1}'},
        {"messages": [], "ground_truth": '{"numbers": [2], "target": 2}'},
    ]
    runner.eval_row_indices = [10, 11]
    runner.eval_metrics_path = tmp_path / "eval_metrics.jsonl"
    runner.eval_completions_dir = tmp_path / "eval_completions"
    runner.eval_completions_dir.mkdir()

    rec = runner._evaluate(5)

    assert rec["eval/raw_reward"] == 0.5
    assert rec["eval/samples"] == 4
    assert rec["eval/truncated_samples"] == 2
    assert rec["eval/truncation_ratio"] == 0.5
    assert (runner.eval_completions_dir / "step-0005.jsonl").exists()
