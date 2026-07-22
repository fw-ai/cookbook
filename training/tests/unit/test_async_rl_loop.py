"""Pure-logic tests for the async_rl_loop runtime helpers.

Covers the deterministic pieces that don't require tinker, the Fireworks
SDK, or a deployment.
"""

from __future__ import annotations

import inspect

import pytest

from training.recipes import async_rl_loop


class _StopAfterProvisioning(RuntimeError):
    pass


class TestConfigDefaults:
    def test_config_has_no_runner_state(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert not hasattr(cfg, "runner")

    def test_config_cleanup_defaults_on(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.cleanup_on_exit is True

    def test_config_pipeline_chunks_default_to_one(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.pipeline_chunks_per_step == 1

    def test_config_exposes_only_grpo_knobs(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.kl_beta == 0.001
        assert cfg.eps_clip == 0.2
        assert cfg.eps_clip_high is None
        assert cfg.anchor_logp == "old_policy"
        assert not hasattr(cfg, "policy_loss")
        assert not hasattr(cfg, "loss_path")


def test_main_has_direct_client_grpo_customization_boundary() -> None:
    source = inspect.getsource(async_rl_loop.main)

    assert "make_grpo_loss_fn(" in source
    assert "policy.forward_backward_custom(" in source
    assert 'cfg.anchor_logp == "old_policy"' in source
    assert "To switch to built-in PPO or another loss" in source
    assert "adding dispatch" in source
    assert "skills/fireworks-training/references/rl-custom-loss.md" in source
    assert "build_loss_fn" not in source
    assert "loss_path" not in source


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
