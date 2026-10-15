"""Pure-logic tests for the async_rl_loop runtime helpers.

Covers the deterministic pieces that don't require tinker, the Fireworks
SDK, or a deployment.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fireworks.training import sdk
from training.recipes import async_rl_loop
from training.utils.rl.heldout import evaluation_rollout_fn
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

    evaluation_rollout = evaluation_rollout_fn(rollout)
    asyncio.run(evaluation_rollout({}, sample_index=2, cursor_index=7))

    assert seen == [(2, True)]


def test_evaluation_rollout_omits_unsupported_context() -> None:
    seen: list[int] = []

    async def rollout(_row, *, sample_index: int):
        seen.append(sample_index)
        return None

    evaluation_rollout = evaluation_rollout_fn(rollout)
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

    evaluation_rollout = evaluation_rollout_fn(rollout)
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

    def test_config_has_only_grpo_hyperparameters(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.eps_clip == 0.2
        assert cfg.eps_clip_high is None
        for removed in (
            "kl_beta",
            "top_p",
            "top_sampling_logprobs",
            "policy_loss",
            "loss_execution",
            "target_logprob_support",
            "anchor_logp",
            "grad_accumulation_normalization",
            "grad_norm_metrics",
            "router_replay",
            "router_replay_completion_only",
            "gspo",
            "dapo",
            "dro",
            "cispo",
            "dppo",
            "score_centering",
            "eval_max_completion_tokens",
            "eval_max_seq_len",
        ):
            assert not hasattr(cfg, removed), removed


def test_optimizer_step_requests_grad_norm_metrics() -> None:
    captured = {}

    class FakePolicy:
        default_timeout = 3600

        def optim_step(self, adam_params, **kwargs):
            captured.update(adam_params=adam_params, **kwargs)
            return "result"

    config = async_rl_loop.Config(log_path="gs://logs", grad_clip_norm=1.5)

    result = async_rl_loop._run_optimizer_step(FakePolicy(), config, learning_rate=2e-6)

    assert result == "result"
    assert captured["adam_params"].learning_rate == 2e-6
    assert captured["adam_params"].beta1 == 0.9
    assert captured["adam_params"].beta2 == 0.95
    assert captured["adam_params"].eps == 1e-12
    assert captured["adam_params"].weight_decay == 0.01
    assert captured["adam_params"].grad_clip_norm == 1.5
    assert captured["emit_grad_norm_metrics"] is True
    assert captured["grad_accumulation_normalization"] == "num_loss_tokens"


@pytest.mark.parametrize(
    "config_overrides",
    [{"eps_clip": -0.1}, {"eps_clip_high": -0.1}],
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


def _build_service_kwargs(monkeypatch: pytest.MonkeyPatch, cfg: async_rl_loop.Config) -> dict:
    calls = []

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(async_rl_loop, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(async_rl_loop, "validate_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        async_rl_loop,
        "resolve_router_replay_enabled",
        lambda **kwargs: kwargs["requested"],
    )
    monkeypatch.setattr(async_rl_loop, "load_deployment_tokenizer", lambda *args, **kwargs: object())

    def fake_build_service_client(**kwargs):
        calls.append(kwargs)
        raise _StopAfterProvisioning

    monkeypatch.setattr(async_rl_loop, "build_service_client", fake_build_service_client)

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
    assert kwargs["cleanup_deployment_on_close"] == async_rl_loop.CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO


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
        default_timeout = 3600

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
    monkeypatch.setattr(async_rl_loop, "validate_config", lambda *_args, **_kwargs: None)
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
            assert advantages == pytest.approx([0.8660254, 0.8660254, -0.8660254, -0.8660254])
        events.append("coordinator")
        raise _StopAfterCoordinatorSetup

    monkeypatch.setattr(async_rl_loop, "AsyncRLCoordinator", coordinator)

    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )
    error = _StopAfterRolloutSetup if advantage_mode == "rollout_setup" else _StopAfterCoordinatorSetup
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


class _StopAfterTrainGroup(RuntimeError):
    pass


def _policy_group(*, advantage: float = 1.0):
    from training.utils.rl.losses import PromptGroup

    return PromptGroup(
        data=[
            sdk.Datum(
                model_input=sdk.ModelInput.from_ints([10, 11, 12]),
                loss_fn_inputs={
                    "target_tokens": sdk.TensorData(data=[11, 12, 13], dtype="int64", shape=[3]),
                    "weights": sdk.TensorData(data=[0.0, 1.0, 1.0], dtype="float32", shape=[3]),
                },
            )
        ],
        advantages=[advantage],
        prompt_len=2,
        rewards=[1.0],
        inf_logprobs=[[0.0, -0.3, -0.1]],
        raw_inf_logprobs=[[0.0, -0.4, -0.2]],
        inference_topk_token_ids=[[[1, 2, 3, 4, 5]] * 3],
        inference_topk_logprobs=[[[-3.0] * 5] * 3],
    )


def _logprob_output(rows):
    return SimpleNamespace(
        loss_fn_outputs=[{"logprobs": sdk.TensorData(data=row, dtype="float32", shape=[len(row)])} for row in rows]
    )


def _run_train_group(monkeypatch, cfg, group, policy, setups=None, heartbeat=None):
    """Drive ``main`` until its first ``train_group`` returns."""
    outputs = []

    def rollout_fn_factory(setup):
        if setups is not None:
            setups.append(setup)
        return lambda _sample: None

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
        batch_id = 1
        _train_group_wait_time = 0.0

        def start_training(self, **kwargs):
            pass

        async def ready_group_batches(self):
            yield [
                SimpleNamespace(
                    group=group,
                    batch_id=1,
                    source_token=0,
                    submit_version=0,
                    ready_at=0.0,
                    ready_utc=0.0,
                )
            ]

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

        async def run_blocking(self, name, fn, *args, **_kwargs):
            if name == "train_group":
                if heartbeat is not None:
                    asyncio.get_running_loop().call_later(0.01, heartbeat.set)
                return await asyncio.to_thread(fn, *args)
            assert name == "optimizer"
            raise _StopAfterTrainGroup

    class FakeTelemetry:
        def __init__(self, **_kwargs):
            pass

        def start(self, *_args):
            pass

        async def aclose(self):
            pass

    original_diagnostics = async_rl_loop.add_batch_diagnostics

    def record_diagnostics(submissions, results, config, objective):
        original_diagnostics(submissions, results, config, objective)
        outputs.extend(results)

    monkeypatch.setattr(async_rl_loop, "add_batch_diagnostics", record_diagnostics)
    policy.save_weights_for_sampler = lambda *_a, **_k: SimpleNamespace(path="checkpoint")
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
    with pytest.raises(_StopAfterTrainGroup):
        async_rl_loop.main(cfg, rows=[], rollout_fn_factory=rollout_fn_factory)
    return outputs[0]


def _config(**overrides):
    return async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
        **overrides,
    )


def test_train_group_runs_server_grpo_on_the_rollout_anchor(monkeypatch) -> None:
    calls = []

    class FakePolicy:
        default_timeout = 3600

        def submit_forward_backward(self, data, loss_fn, loss_fn_config=None):
            calls.append((data, loss_fn, loss_fn_config))
            result = SimpleNamespace(
                metrics={"loss:sum": 1.0},
                loss_fn_outputs=_logprob_output([[-0.4, -0.2, -0.1]]).loss_fn_outputs,
            )

            async def result_async(timeout=None):
                return result

            return SimpleNamespace(result_async=result_async)

        def forward(self, *_args, **_kwargs):
            raise AssertionError("a rollout anchor needs no snapshot forward")

        def forward_backward_custom(self, *_args, **_kwargs):
            raise AssertionError("built-in execution must not call a custom loss")

    result = _run_train_group(
        monkeypatch,
        _config(),
        _policy_group(),
        FakePolicy(),
    )

    ((server_data, loss_fn, loss_config),) = calls
    assert loss_fn == "ppo"
    assert loss_config == {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
    inputs = server_data[0].loss_fn_inputs
    assert inputs["logprobs"].data == pytest.approx([0.0, -0.3, -0.1])
    assert inputs["advantages"].data == [0.0, 1.0, 1.0]
    assert inputs["response_mask"].data == [0, 1, 1]
    assert result.metrics["inference_k3"] >= 0
    assert result.metrics["raw_inference_logprob_coverage"] == 1.0
    assert result.metrics["ppo_ratio_mean"] > 0
    assert result.metrics["active_tokens"] == 2


def test_group_preparation_and_submission_leave_event_loop_responsive(monkeypatch) -> None:
    heartbeat = threading.Event()
    main_thread = threading.get_ident()
    original_build = async_rl_loop.objective.build_datums

    def slow_build(*args):
        assert threading.get_ident() != main_thread
        assert heartbeat.wait(timeout=2), "event-loop timer stalled during preparation"
        return original_build(*args)

    class Policy:
        default_timeout = 17

        def submit_forward_backward(self, *_args, **_kwargs):
            assert threading.get_ident() != main_thread

            async def result_async(timeout=None):
                assert timeout == self.default_timeout
                return SimpleNamespace(
                    metrics={},
                    loss_fn_outputs=_logprob_output([[-0.4, -0.2, -0.1]]).loss_fn_outputs,
                )

            return SimpleNamespace(result_async=result_async)

    monkeypatch.setattr(async_rl_loop.objective, "build_datums", slow_build)
    _run_train_group(monkeypatch, _config(), _policy_group(), Policy(), heartbeat=heartbeat)
