"""Pure-logic tests for the async_rl_loop runtime helpers.

Covers the deterministic pieces that don't require tinker, the Fireworks
SDK, or a deployment.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import tinker

from training.recipes import async_rl_loop
from training.utils.runner import UserConfigError


class _StopAfterProvisioning(RuntimeError):
    pass


class _StopAfterRolloutSetup(RuntimeError):
    pass


def test_evaluation_rollout_context_is_explicit_and_compatible() -> None:
    seen: list[tuple[int, bool]] = []

    async def rollout(_row, *, sample_index: int, evaluation: bool = False):
        seen.append((sample_index, evaluation))
        return None

    evaluation_rollout = async_rl_loop.make_evaluation_rollout_fn(rollout)
    asyncio.run(evaluation_rollout({}, sample_index=2, cursor_index=7))

    assert seen == [(2, True)]


def test_evaluation_rollout_omits_unsupported_context() -> None:
    seen: list[int] = []

    async def rollout(_row, *, sample_index: int):
        seen.append(sample_index)
        return None

    evaluation_rollout = async_rl_loop.make_evaluation_rollout_fn(rollout)
    asyncio.run(evaluation_rollout({}, sample_index=3, cursor_index=7))

    assert seen == [3]


def test_evaluation_rollout_inherits_training_length_limits() -> None:
    setup = SimpleNamespace(
        sample_kwargs={"max_tokens": 1024, "max_seq_len": 4096},
    )
    seen: list[tuple[int, int, bool]] = []

    async def rollout(_row, *, evaluation: bool = False):
        seen.append(
            (
                setup.sample_kwargs["max_tokens"],
                setup.sample_kwargs["max_seq_len"],
                evaluation,
            )
        )
        return None

    evaluation_rollout = async_rl_loop.make_evaluation_rollout_fn(rollout)
    asyncio.run(evaluation_rollout({}))

    assert seen == [(1024, 4096, True)]


class TestConfigDefaults:
    def test_config_has_no_runner_state(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert not hasattr(cfg, "runner")
        assert not hasattr(async_rl_loop, "RunnerIO")

    def test_config_has_no_conditional_initial_sync(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert not hasattr(cfg, "weight_sync_before_training")

    def test_config_cleanup_defaults_on(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.cleanup_on_exit is True

    def test_config_recovery_defaults_preserve_existing_behavior(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.warm_start_from_adapter is None
        assert cfg.dcp_save_interval == 0
        assert cfg.weight_sync_timeout == 600

    def test_config_pipeline_chunks_default_to_one(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.pipeline_chunks_per_step == 1

    def test_config_exposes_policy_loss_knobs(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.kl_beta == 0.001
        assert cfg.eps_clip == 0.2
        assert cfg.eps_clip_high is None
        assert cfg.anchor_logp == "old_policy"
        assert cfg.server_side_grpo is False
        assert cfg.policy_loss == "grpo"
        assert cfg.gspo.clip_ratio_low == 3e-4
        assert cfg.gspo.clip_ratio_high == 4e-4
        assert cfg.gspo_execution == "builtin"
        assert cfg.dapo.eps_clip_high == 0.28
        assert cfg.dro.beta == 0.05
        assert cfg.cispo.eps_high == 0.28
        assert cfg.dppo.divergence == "binary_tv"
        assert cfg.dppo.threshold == 0.15
        assert cfg.score_centering.top_k == 5
        assert cfg.grad_norm_metrics == "off"
        assert cfg.router_replay is True
        assert cfg.router_replay_completion_only is True
        assert not hasattr(cfg, "loss_path")
        assert not hasattr(cfg, "eval_max_completion_tokens")
        assert not hasattr(cfg, "eval_max_seq_len")


def test_policy_loss_metadata_has_one_algorithm_config() -> None:
    config = async_rl_loop.Config(
        log_path="gs://logs",
        policy_loss="dppo",
        kl_beta=0,
    )

    metadata = async_rl_loop.policy_loss_metadata(config)

    assert metadata["trainer_loss"] == "client_dppo"
    assert metadata["policy_loss"] == "dppo"
    assert metadata["dppo"] == {
        "divergence": "binary_tv",
        "threshold": 0.15,
        "ratio_log_cap": 20.0,
    }
    assert metadata["gspo"] is None
    assert metadata["score_centering"] is None


def test_server_side_grpo_calls_only_builtin_ppo_and_emits_kld() -> None:
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([10, 11, 12]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=[11, 12, 13],
                dtype="int64",
                shape=[3],
            ),
            "weights": tinker.TensorData(
                data=[0.0, 1.0, 1.0],
                dtype="float32",
                shape=[3],
            ),
        },
    )
    calls = []

    class FakePolicy:
        def forward_backward(self, data, loss_fn, loss_fn_config=None):
            calls.append((data, loss_fn, loss_fn_config))
            return SimpleNamespace(
                loss_fn_outputs=[
                    {
                        "logprobs": tinker.TensorData(
                            data=[-0.4, -0.2, -0.1],
                            dtype="float32",
                            shape=[3],
                        )
                    }
                ],
                metrics={"loss:sum": 1.0},
            )

        def forward_backward_custom(self, *_args, **_kwargs):
            raise AssertionError("server-side GRPO must not call a custom loss")

    result = async_rl_loop._run_server_side_policy_loss(
        FakePolicy(),
        data=[datum],
        advantages=[1.0],
        prompt_lens=[2],
        rollout_logprobs=[[-0.4, -0.3, -0.1]],
        raw_inference_logprobs=[[-0.4, -0.4, -0.2]],
        old_policy_logprobs=[[-0.4, -0.3, -0.1]],
        config=async_rl_loop.Config(
            log_path="gs://logs",
            kl_beta=0,
            server_side_grpo=True,
        ),
    )

    assert len(calls) == 1
    server_data, loss_fn, loss_config = calls[0]
    assert loss_fn == "ppo"
    assert loss_config == {
        "clip_low_threshold": 0.8,
        "clip_high_threshold": 1.2,
    }
    assert set(server_data[0].loss_fn_inputs) == {
        "target_tokens",
        "logprobs",
        "advantages",
    }
    assert result.metrics["inference_k3"] >= 0
    assert result.metrics["raw_inference_logprob_coverage"] == 1.0


def test_server_side_gspo_preserves_zero_advantage_response_membership() -> None:
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([10, 11, 12]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=[11, 12, 13],
                dtype="int64",
                shape=[3],
            ),
            "weights": tinker.TensorData(
                data=[0.0, 1.0, 1.0],
                dtype="float32",
                shape=[3],
            ),
        },
    )
    calls = []

    class FakePolicy:
        def forward_backward(self, data, loss_fn, loss_fn_config=None):
            calls.append((data, loss_fn, loss_fn_config))
            return SimpleNamespace(
                loss_fn_outputs=[
                    {
                        "logprobs": tinker.TensorData(
                            data=[-0.4, -0.2, -0.1],
                            dtype="float32",
                            shape=[3],
                        )
                    }
                ],
                metrics={"loss:sum": 1.0},
            )

    config = async_rl_loop.Config(
        log_path="gs://logs",
        kl_beta=0,
        server_side_grpo=True,
        policy_loss="gspo",
    )
    result = async_rl_loop._run_server_side_policy_loss(
        FakePolicy(),
        data=[datum],
        advantages=[0.0],
        prompt_lens=[2],
        rollout_logprobs=[[-0.4, -0.3, -0.1]],
        raw_inference_logprobs=[[-0.4, -0.4, -0.2]],
        old_policy_logprobs=[[-0.4, -0.3, -0.1]],
        config=config,
    )

    server_data, loss_fn, loss_config = calls[0]
    assert loss_fn == "gspo"
    assert loss_config == {
        "clip_low_threshold": 1.0 - 3e-4,
        "clip_high_threshold": 1.0 + 4e-4,
        "seq_ratio_log_cap": 10.0,
    }
    assert server_data[0].loss_fn_inputs["advantages"].data == [0.0, 0.0, 0.0]
    assert server_data[0].loss_fn_inputs["response_mask"].data == [0, 1, 1]
    assert result.metrics["gspo_clip_frac"] == 1.0
    assert result.metrics["gspo_seq_ratio_mean"] > 1.0
    assert (
        async_rl_loop._effective_grad_accumulation_normalization(config)
        == "num_sequences"
    )


def test_two_pass_gspo_uses_local_loss_without_builtin_dispatch(monkeypatch) -> None:
    loss_fn = object()
    result = object()
    captured = {}

    def make_loss(**kwargs):
        captured["loss_kwargs"] = kwargs
        return loss_fn

    class FakePolicy:
        def forward_backward_custom(
            self,
            data,
            actual_loss_fn,
            *,
            precomputed_forward,
        ):
            captured["forward"] = (data, actual_loss_fn, precomputed_forward)
            return result

        def forward_backward(self, *_args, **_kwargs):
            raise AssertionError("two-pass GSPO must not call the built-in kernel")

    monkeypatch.setattr(async_rl_loop, "make_gspo_loss_fn", make_loss)
    config = async_rl_loop.Config(
        log_path="gs://logs",
        kl_beta=0,
        policy_loss="gspo",
        gspo_execution="two_pass",
    )

    actual = async_rl_loop._run_two_pass_gspo(
        FakePolicy(),
        data=["datum"],
        advantages=[1.0],
        ref_logprobs=[[]],
        prompt_lens=[2],
        rollout_logprobs=[[-0.4, -0.3]],
        old_policy_logprobs=[[-0.4, -0.3]],
        precomputed_forward="forward",
        config=config,
    )

    assert actual is result
    assert captured["forward"] == (["datum"], loss_fn, "forward")
    assert captured["loss_kwargs"] == {
        "advantages": [1.0],
        "ref_logprobs": [[]],
        "inf_logprobs": [[-0.4, -0.3]],
        "prompt_len": [2],
        "old_policy_logprobs": [[-0.4, -0.3]],
        "gspo_config": config.gspo,
        "tis_config": config.tis,
    }
    assert (
        async_rl_loop._effective_grad_accumulation_normalization(config)
        == "num_sequences"
    )


def test_optimizer_step_requests_grad_norm_metrics_before_clipping() -> None:
    captured = {}

    class FakePolicy:
        def optim_step(
            self,
            adam_params,
            *,
            grad_accumulation_normalization,
            emit_grad_norm_metrics,
        ):
            captured.update(
                adam_params=adam_params,
                normalization=grad_accumulation_normalization,
                grad_metrics=emit_grad_norm_metrics,
            )
            return "result"

    config = async_rl_loop.Config(
        log_path="gs://logs",
        policy_loss="gspo",
        gspo_execution="two_pass",
        grad_clip_norm=1.5,
        grad_norm_metrics="detailed",
    )

    result = async_rl_loop._run_optimizer_step(
        FakePolicy(),
        config,
        learning_rate=2e-6,
    )

    assert result == "result"
    assert captured["adam_params"].learning_rate == 2e-6
    assert captured["adam_params"].beta1 == 0.9
    assert captured["adam_params"].beta2 == 0.95
    assert captured["adam_params"].eps == 1e-12
    assert captured["adam_params"].weight_decay == 0.01
    assert captured["adam_params"].grad_clip_norm == 1.5
    assert captured["normalization"] == "num_sequences"
    assert captured["grad_metrics"] == "detailed"


def test_gspo_requires_builtin_server_path_and_sequence_normalization() -> None:
    with pytest.raises(ValueError, match="requires server_side_grpo=True"):
        async_rl_loop.main(
            async_rl_loop.Config(
                log_path="gs://logs",
                kl_beta=0,
                policy_loss="gspo",
            ),
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )

    with pytest.raises(
        ValueError,
        match="requires grad_accumulation_normalization='num_sequences'",
    ):
        async_rl_loop.main(
            async_rl_loop.Config(
                log_path="gs://logs",
                kl_beta=0,
                server_side_grpo=True,
                policy_loss="gspo",
                grad_accumulation_normalization="num_loss_tokens",
            ),
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )

    with pytest.raises(ValueError, match="supports only token_reduction='mean'"):
        async_rl_loop.main(
            async_rl_loop.Config(
                log_path="gs://logs",
                kl_beta=0,
                server_side_grpo=True,
                policy_loss="gspo",
                gspo=async_rl_loop.GSPOConfig(token_reduction="sum"),
            ),
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )


def test_main_rejects_server_side_grpo_with_reference_kl() -> None:
    cfg = async_rl_loop.Config(
        log_path="gs://logs",
        server_side_grpo=True,
        kl_beta=0.1,
    )

    with pytest.raises(ValueError, match="server_side_grpo requires kl_beta=0"):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )


@pytest.mark.parametrize(
    "trainer",
    [
        async_rl_loop.TrainerConfig(reference_training_shape_id="ref-shape"),
        async_rl_loop.TrainerConfig(reference_job_id="ref-job"),
    ],
)
def test_main_rejects_unused_reference_trainer_config(trainer) -> None:
    cfg = async_rl_loop.Config(log_path="gs://logs", kl_beta=0, trainer=trainer)

    with pytest.raises(ValueError, match="require kl_beta > 0"):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )


@pytest.mark.parametrize(
    "config_overrides",
    [{"eps_clip": -0.1}, {"eps_clip_high": -0.1}, {"kl_beta": -0.1}],
)
def test_main_rejects_invalid_grpo_config(config_overrides) -> None:
    cfg = async_rl_loop.Config(log_path="gs://logs", **config_overrides)

    with pytest.raises(ValueError, match="must be non-negative"):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )


def test_main_rejects_unknown_anchor_logp() -> None:
    cfg = async_rl_loop.Config(log_path="gs://logs", anchor_logp="unknown")

    with pytest.raises(ValueError, match="anchor_logp must be"):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )


@pytest.mark.parametrize(
    "policy_loss", ["dapo", "dro", "cispo", "dppo", "score_centering"]
)
def test_client_policy_losses_reject_unused_reference_kl(policy_loss) -> None:
    cfg = async_rl_loop.Config(
        log_path="gs://logs",
        policy_loss=policy_loss,
        kl_beta=0.1,
    )

    with pytest.raises(ValueError, match="requires kl_beta=0"):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )


def test_server_side_grpo_rejects_client_only_policy_losses() -> None:
    cfg = async_rl_loop.Config(
        log_path="gs://logs",
        policy_loss="dppo",
        server_side_grpo=True,
        kl_beta=0,
    )

    with pytest.raises(ValueError, match="supports only"):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )


def test_score_centering_uses_loss_token_normalization() -> None:
    config = async_rl_loop.Config(
        log_path="gs://logs",
        policy_loss="score_centering",
        kl_beta=0,
    )

    assert (
        async_rl_loop._effective_grad_accumulation_normalization(config)
        == "num_loss_tokens"
    )


@pytest.mark.parametrize(
    "config_overrides, error",
    [
        (
            {
                "lora_rank": 8,
                "warm_start_from_adapter": "accounts/a/models/adapter",
                "init_from_checkpoint": "step-5",
            },
            "mutually exclusive",
        ),
        (
            {"lora_rank": 0, "warm_start_from_adapter": "accounts/a/models/adapter"},
            "requires lora_rank > 0",
        ),
    ],
)
def test_main_validates_adapter_warm_start(config_overrides, error) -> None:
    cfg = async_rl_loop.Config(log_path="gs://logs", **config_overrides)

    with pytest.raises(UserConfigError, match=error):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )


# ---------------------------------------------------------------------------
# SDK service construction
# ---------------------------------------------------------------------------


def _build_service_kwargs(
    monkeypatch: pytest.MonkeyPatch, cfg: async_rl_loop.Config
) -> dict:
    calls = []

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(async_rl_loop, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(async_rl_loop, "validate_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        async_rl_loop,
        "resolve_router_replay_enabled",
        lambda **kwargs: kwargs["requested"],
    )
    monkeypatch.setattr(
        async_rl_loop, "load_deployment_tokenizer", lambda *args, **kwargs: object()
    )

    def fake_build_service_client(**kwargs):
        calls.append(kwargs)
        raise _StopAfterProvisioning

    monkeypatch.setattr(
        async_rl_loop, "build_service_client", fake_build_service_client
    )

    with pytest.raises(_StopAfterProvisioning):
        async_rl_loop.main(
            cfg,
            rows=[{"prompt": "1+1"}],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )

    assert len(calls) == 1
    return calls[0]


def test_main_requests_cleanup_for_sdk_created_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is True
    assert (
        kwargs["cleanup_deployment_on_close"]
        == async_rl_loop.CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO
    )


def test_main_can_disable_cleanup_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        cleanup_on_exit=False,
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is False
    assert kwargs["cleanup_deployment_on_close"] is None


def test_main_requests_trainer_cleanup_for_empty_job_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        trainer=async_rl_loop.TrainerConfig(job_id=""),
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is True


def test_main_injects_sampler_and_closes_it_before_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    expected_tokenizer = object()

    class FakeSampler:
        model = "accounts/test/deployments/rollout"
        base_url = "https://rollout.example"

        def close(self) -> None:
            events.append("sampler.close")

    sampler = FakeSampler()

    class FakeService:
        trainer_job_id = "trainer"
        max_context_length = 4096

        def close(self) -> None:
            events.append("service.close")

        def create_training_client(self, *_args, **_kwargs):
            return object()

        def create_deployment_sampler(self, *, tokenizer):
            assert tokenizer is expected_tokenizer
            return sampler

        def hotload_sampler_snapshot(self, path: str) -> None:
            assert path == "checkpoint"
            events.append("hotload")

    class FakePolicy:
        def save_weights_for_sampler(self, *_args, **_kwargs):
            events.append("save")
            return SimpleNamespace(path="checkpoint")

    class FakeCheckpoints:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def resume(self, **_kwargs):
            return None

    service = FakeService()
    policy = FakePolicy()
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(async_rl_loop, "setup_wandb", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        async_rl_loop, "validate_config", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        async_rl_loop,
        "resolve_router_replay_enabled",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        async_rl_loop,
        "load_deployment_tokenizer",
        lambda *_args, **_kwargs: expected_tokenizer,
    )
    monkeypatch.setattr(
        async_rl_loop,
        "build_service_client",
        lambda **_kwargs: service,
    )
    monkeypatch.setattr(
        async_rl_loop.ReconnectableClient,
        "from_training_client",
        lambda *_args, **_kwargs: policy,
    )
    monkeypatch.setattr(async_rl_loop, "TrainingCheckpoints", FakeCheckpoints)
    monkeypatch.setattr(async_rl_loop, "log_metrics", lambda *_args, **_kwargs: None)

    def rollout_factory(setup: async_rl_loop.RolloutSetup):
        assert setup.sampler is sampler
        assert setup.inference_base_url == sampler.base_url
        assert setup.model == sampler.model
        assert not hasattr(setup, "max_context_tokens")
        assert setup.sample_kwargs["max_seq_len"] == 4096
        events.append("rollout_factory")
        raise _StopAfterRolloutSetup

    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        kl_beta=0,
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )
    with pytest.raises(_StopAfterRolloutSetup):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=rollout_factory,
        )

    assert events == [
        "save",
        "hotload",
        "rollout_factory",
        "sampler.close",
        "service.close",
    ]
