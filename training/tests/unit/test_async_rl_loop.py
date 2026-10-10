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


class _StopAfterCoordinatorSetup(RuntimeError):
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
        assert cfg.loss_execution == "client"
        assert cfg.policy_loss == "grpo"
        assert cfg.gspo.clip_ratio_low == 3e-4
        assert cfg.gspo.clip_ratio_high == 4e-4
        assert cfg.dapo.eps_clip_high == 0.28
        assert cfg.dro.beta == 0.05
        assert cfg.cispo.eps_high == 0.28
        assert cfg.dppo.divergence == "binary_tv"
        assert cfg.dppo.threshold == 0.15
        assert cfg.score_centering.top_k == 5
        assert cfg.grad_norm_metrics == "off"
        assert cfg.router_replay is True
        assert cfg.router_replay_completion_only is True
        assert not hasattr(cfg, "server_side_grpo")
        assert not hasattr(cfg, "gspo_execution")
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
    assert metadata["loss_execution"] == "client"
    assert metadata["policy_loss"] == "dppo"
    assert metadata["loss_fn_config"] is None
    assert metadata["dppo"] == {
        "divergence": "binary_tv",
        "threshold": 0.15,
        "ratio_log_cap": 20.0,
    }
    assert metadata["gspo"] is None
    assert metadata["score_centering"] is None


@pytest.mark.parametrize(
    "policy_loss, execution, trainer_loss, loss_fn_config",
    [
        ("grpo", "client", "client_grpo", None),
        (
            "grpo",
            "builtin",
            "server_ppo",
            {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
        ),
        (
            "gspo",
            "builtin",
            "server_gspo",
            {
                "clip_low_threshold": 1.0 - 3e-4,
                "clip_high_threshold": 1.0 + 4e-4,
                "seq_ratio_log_cap": 10.0,
            },
        ),
        (
            "cispo",
            "builtin",
            "server_cispo",
            {
                "clip_low_threshold": 0.8,
                "clip_high_threshold": 1.28,
                "ratio_log_cap": 20.0,
            },
        ),
        ("dro", "builtin", "server_dro", {"beta": 0.05}),
        ("importance_sampling", "builtin", "server_importance_sampling", {}),
    ],
)
def test_policy_loss_resolves_trainer_settings_once(
    policy_loss, execution, trainer_loss, loss_fn_config
) -> None:
    loss = async_rl_loop.resolve_recipe_policy_loss(
        async_rl_loop.Config(
            log_path="gs://logs",
            policy_loss=policy_loss,
            loss_execution=execution,
            kl_beta=0,
        )
    )

    assert loss.trainer_loss == trainer_loss
    assert loss.loss_fn_config == loss_fn_config


@pytest.mark.parametrize(
    "policy_loss, normalization",
    [("gspo", "num_sequences"), ("score_centering", "num_loss_tokens")],
)
def test_objective_owned_normalization(policy_loss, normalization) -> None:
    config = async_rl_loop.Config(
        log_path="gs://logs", policy_loss=policy_loss, kl_beta=0
    )

    assert async_rl_loop.resolve_recipe_policy_loss(config).normalization == normalization


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
        grad_clip_norm=1.5,
        grad_norm_metrics="detailed",
    )

    result = async_rl_loop._run_optimizer_step(
        FakePolicy(),
        config,
        learning_rate=2e-6,
        normalization="num_sequences",
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


@pytest.mark.parametrize(
    "overrides, error",
    [
        (
            {"policy_loss": "gspo", "grad_accumulation_normalization": "num_loss_tokens"},
            "requires grad_accumulation_normalization='num_sequences'",
        ),
        (
            {
                "policy_loss": "gspo",
                "loss_execution": "builtin",
                "gspo": async_rl_loop.GSPOConfig(token_reduction="sum"),
            },
            "supports only token_reduction='mean'",
        ),
        ({"loss_execution": "builtin", "kl_beta": 0.1}, "builtin' requires kl_beta=0"),
        (
            {"policy_loss": "score_centering", "loss_execution": "builtin"},
            "has no built-in objective",
        ),
        ({"loss_execution": "fused"}, "loss_execution must be"),
        ({"anchor_logp": "snapshot"}, "anchor_logp must be"),
    ],
)
def test_main_rejects_invalid_policy_loss_settings(overrides, error) -> None:
    cfg = async_rl_loop.Config(log_path="gs://logs", **{"kl_beta": 0, **overrides})

    with pytest.raises(ValueError, match=error):
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


@pytest.mark.parametrize(
    "policy_loss",
    ["dapo", "dro", "cispo", "dppo", "importance_sampling", "score_centering"],
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


@pytest.mark.parametrize("advantage_mode", ["rollout_setup", "default", "mean_only"])
def test_main_injects_sampler_and_advantages_and_closes_before_service(
    monkeypatch: pytest.MonkeyPatch,
    advantage_mode: str,
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
        if advantage_mode == "rollout_setup":
            raise _StopAfterRolloutSetup

        async def rollout(_row):
            return None

        return rollout

    def mean_only(rewards):
        mean = sum(rewards) / len(rewards)
        return [2 * (reward - mean) for reward in rewards]

    def coordinator(**kwargs):
        advantages = kwargs["advantage_fn"]([1.0, 1.0, 0.0, 0.0])
        if advantage_mode == "mean_only":
            assert advantages == [1.0, 1.0, -1.0, -1.0]
        else:
            assert advantages == pytest.approx(
                [0.8660254, 0.8660254, -0.8660254, -0.8660254]
            )
        events.append("coordinator")
        raise _StopAfterCoordinatorSetup

    monkeypatch.setattr(async_rl_loop, "AsyncRLCoordinator", coordinator)

    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        kl_beta=0,
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )
    error = (
        _StopAfterRolloutSetup
        if advantage_mode == "rollout_setup"
        else _StopAfterCoordinatorSetup
    )
    options = {"advantage_fn": mean_only} if advantage_mode == "mean_only" else {}
    with pytest.raises(error):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=rollout_factory,
            **options,
        )

    assert events == [
        "save",
        "hotload",
        "rollout_factory",
        *([] if advantage_mode == "rollout_setup" else ["coordinator"]),
        "sampler.close",
        "service.close",
    ]


class _StopAfterTrainChunk(RuntimeError):
    pass


def _policy_group(*, advantage: float = 1.0):
    from training.utils.rl.losses import PromptGroup

    return PromptGroup(
        data=[
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints([10, 11, 12]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=[11, 12, 13], dtype="int64", shape=[3]
                    ),
                    "weights": tinker.TensorData(
                        data=[0.0, 1.0, 1.0], dtype="float32", shape=[3]
                    ),
                },
            )
        ],
        advantages=[advantage],
        ref_logprobs=None,
        prompt_len=2,
        rewards=[1.0],
        inf_logprobs=[[0.0, -0.3, -0.1]],
        raw_inf_logprobs=[[0.0, -0.4, -0.2]],
        inference_topk_token_ids=[[[1, 2, 3, 4, 5]] * 3],
        inference_topk_logprobs=[[[-3.0] * 5] * 3],
    )


def _logprob_output(rows):
    return SimpleNamespace(
        loss_fn_outputs=[
            {
                "logprobs": tinker.TensorData(
                    data=row, dtype="float32", shape=[len(row)]
                )
            }
            for row in rows
        ]
    )


def _run_train_chunk(monkeypatch, cfg, group, policy):
    """Drive ``main`` until its first ``train_chunk`` returns."""
    outputs = []

    class FakeService:
        trainer_job_id = "trainer"
        max_context_length = 4096

        def close(self):
            pass

        def create_training_client(self, *_args, **_kwargs):
            return object()

        def create_deployment_sampler(self, *, tokenizer):
            return SimpleNamespace(model="m", base_url="u", close=lambda: None)

        def hotload_sampler_snapshot(self, path):
            pass

    class FakeCheckpoints:
        def __init__(self, *_args, **_kwargs):
            pass

        def resume(self, **_kwargs):
            return None

    class FakeBatch:
        async def chunks(self):
            yield SimpleNamespace(groups=[group])

    class FakeCoordinator:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def snapshot(self):
            return None

        async def next_batch(self):
            return FakeBatch()

        def raise_if_failed(self, *_args):
            pass

        async def run_blocking(self, name, fn, chunk, **_kwargs):
            assert name == "train_chunk"
            outputs.append(fn(chunk))
            raise _StopAfterTrainChunk

    class FakeTelemetry:
        def __init__(self, **_kwargs):
            pass

        def start(self, *_args):
            pass

        async def aclose(self):
            pass

    policy.save_weights_for_sampler = lambda *_a, **_k: SimpleNamespace(
        path="checkpoint"
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    for name, value in {
        "setup_wandb": lambda *_a, **_k: None,
        "validate_config": lambda *_a, **_k: None,
        "resolve_router_replay_enabled": lambda **_k: False,
        "load_deployment_tokenizer": lambda *_a, **_k: object(),
        "build_service_client": lambda **_k: FakeService(),
        "TrainingCheckpoints": FakeCheckpoints,
        "log_metrics": lambda *_a, **_k: None,
        "AsyncRLCoordinator": FakeCoordinator,
        "AsyncRLTelemetry": FakeTelemetry,
    }.items():
        monkeypatch.setattr(async_rl_loop, name, value)
    monkeypatch.setattr(
        async_rl_loop.ReconnectableClient,
        "from_training_client",
        lambda *_a, **_k: policy,
    )
    with pytest.raises(_StopAfterTrainChunk):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )
    return outputs[0]["fwd_bwd_result"]


def _config(**overrides):
    return async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        kl_beta=0,
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
        **overrides,
    )


def test_builtin_grpo_submits_rollout_anchor_and_emits_kld(monkeypatch) -> None:
    calls = []

    class FakePolicy:
        def forward_backward(self, data, loss_fn, loss_fn_config=None):
            calls.append((data, loss_fn, loss_fn_config))
            return SimpleNamespace(
                metrics={"loss:sum": 1.0},
                loss_fn_outputs=_logprob_output([[-0.4, -0.2, -0.1]]).loss_fn_outputs,
            )

        def forward(self, *_args, **_kwargs):
            raise AssertionError("a rollout anchor needs no snapshot forward")

        def forward_backward_custom(self, *_args, **_kwargs):
            raise AssertionError("built-in execution must not call a custom loss")

    result = _run_train_chunk(
        monkeypatch,
        _config(loss_execution="builtin", anchor_logp="rollout"),
        _policy_group(),
        FakePolicy(),
    )

    (server_data, loss_fn, loss_config), = calls
    assert loss_fn == "ppo"
    assert loss_config == {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
    inputs = server_data[0].loss_fn_inputs
    assert inputs["logprobs"].data == pytest.approx([0.0, -0.3, -0.1])
    assert inputs["advantages"].data == [0.0, 1.0, 1.0]
    assert inputs["response_mask"].data == [0, 1, 1]
    assert result.metrics["inference_k3"] >= 0
    assert result.metrics["raw_inference_logprob_coverage"] == 1.0


def test_builtin_gspo_uses_snapshot_anchor_and_zero_advantage_membership(
    monkeypatch,
) -> None:
    events = []

    class FakePolicy:
        def forward(self, data, loss_fn):
            events.append("anchor.forward")
            return _logprob_output([[-9.0, -0.5, -0.6]])

        def forward_backward(self, data, loss_fn, loss_fn_config=None):
            events.append(loss_fn)
            assert data[0].loss_fn_inputs["logprobs"].data == pytest.approx(
                [0.0, -0.5, -0.6]
            )
            assert data[0].loss_fn_inputs["advantages"].data == [0.0, 0.0, 0.0]
            assert data[0].loss_fn_inputs["response_mask"].data == [0, 1, 1]
            return SimpleNamespace(
                metrics={},
                loss_fn_outputs=_logprob_output([[-0.4, -0.2, -0.1]]).loss_fn_outputs,
            )

    result = _run_train_chunk(
        monkeypatch,
        _config(loss_execution="builtin", policy_loss="gspo"),
        _policy_group(advantage=0.0),
        FakePolicy(),
    )

    assert events == ["anchor.forward", "gspo"]
    assert result.metrics["gspo_clip_frac"] == 1.0


def test_client_loss_reuses_the_snapshot_forward(monkeypatch) -> None:
    snapshot = _logprob_output([[-9.0, -0.5, -0.6]])
    captured = {}

    class FakePolicy:
        def forward(self, data, loss_fn):
            assert loss_fn == "cross_entropy"
            return snapshot

        def forward_backward_custom(self, data, loss_fn, *, precomputed_forward):
            captured["forward"] = precomputed_forward
            logprobs = [
                output["logprobs"].to_torch().requires_grad_()
                for output in precomputed_forward.loss_fn_outputs
            ]
            loss, metrics = loss_fn(data, logprobs)
            loss.backward()
            return SimpleNamespace(metrics=dict(metrics), loss_fn_outputs=[])

        def forward_backward(self, *_args, **_kwargs):
            raise AssertionError("client execution must not call a built-in loss")

    result = _run_train_chunk(
        monkeypatch, _config(policy_loss="cispo"), _policy_group(), FakePolicy()
    )

    assert captured["forward"] is snapshot
    assert result.metrics["custom_forward_reused"] == 1.0


def test_client_score_centering_runs_its_own_expanded_forward(monkeypatch) -> None:
    events = []

    class FakePolicy:
        def forward(self, data, loss_fn):
            shape = data[0].loss_fn_inputs["target_tokens"].shape
            events.append(("forward", tuple(shape)))
            rows = data[0].loss_fn_inputs["target_tokens"].data
            return SimpleNamespace(
                loss_fn_outputs=[
                    {
                        "logprobs": tinker.TensorData(
                            data=[-2.0] * len(rows), dtype="float32", shape=shape
                        )
                    }
                ]
            )

        def forward_backward_custom(self, data, loss_fn, *, precomputed_forward):
            events.append("custom")
            return SimpleNamespace(metrics={}, loss_fn_outputs=[])

    result = _run_train_chunk(
        monkeypatch,
        _config(policy_loss="score_centering"),
        _policy_group(),
        FakePolicy(),
    )

    assert events == [("forward", (3, 6)), "custom"]
    assert result.metrics["custom_forward_reused"] == 0.0
