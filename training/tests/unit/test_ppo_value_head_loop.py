"""Wiring of ``recipes.experiment.ppo_value_head_loop`` (SAO)."""

from __future__ import annotations

import pytest

import training.recipes.experiment.ppo_value_head_loop as module
from training.utils.checkpoints import ResumeInfo


def test_default_config_uses_separate_trainers_for_same_base() -> None:
    config = module.Config(log_path="/tmp/test")

    assert config.actor_base_model == module.QWEN3_4B
    assert config.critic_base_model == module.QWEN3_4B
    assert config.critic_lora_rank > 0
    assert config.critic_updates == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dis_low": 1.0}, "dis_low"),
        ({"critic_updates": 0}, "critic_updates must be a positive integer"),
        ({"critic_lora_rank": 0}, "critic_lora_rank must be a positive integer"),
        ({"gae_alpha": 0.0}, "gae_alpha must be positive"),
        ({"gamma": 1.5}, "gamma must be finite and in"),
        ({"value_clip": 0.0}, "value_clip must be positive"),
    ],
)
def test_validate_settings_rejects_invalid(overrides, message) -> None:
    config = module.Config(
        log_path="/tmp/test",
        critic_trainer=module.TrainerConfig(training_shape_id="critic-shape"),
        **overrides,
    )
    with pytest.raises(ValueError, match=message):
        module.validate_settings(config)


def test_missing_critic_shape_fails_before_provisioning(monkeypatch) -> None:
    def unexpected_provision(**_kwargs):
        raise AssertionError("invalid topology must fail before resource creation")

    monkeypatch.setattr(module, "build_service_client", unexpected_provision)
    with pytest.raises(ValueError, match="critic_trainer.training_shape_id"):
        module.main(module.Config(log_path="/tmp/test"), rows=[{"id": 1}])


def _patch_setup(monkeypatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(module, "setup_wandb", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "validate_config", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module,
        "load_deployment_tokenizer",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("injected rollouts do not need a tokenizer")
        ),
    )


async def _unused_sampler(_row, *, cursor_index: int):
    raise AssertionError(f"sampling should not start during setup: {cursor_index}")


def _injected_config() -> module.Config:
    return module.Config(
        log_path="/tmp/ppo-value-test",
        critic_trainer=module.TrainerConfig(training_shape_id="critic-shape"),
        deployment=module.DeployConfig(tokenizer_model="tokenizer"),
    )


def test_injected_rollouts_do_not_provision_actor_deployment(monkeypatch) -> None:
    class StopAfterActorService(RuntimeError):
        pass

    captured: list[dict] = []

    def build_service_client(**kwargs):
        captured.append(kwargs)
        raise StopAfterActorService

    _patch_setup(monkeypatch)
    monkeypatch.setattr(module, "build_service_client", build_service_client)

    with pytest.raises(StopAfterActorService):
        module.main(_injected_config(), sample_prompt_fn=_unused_sampler, rows=[{"id": 1}])

    assert len(captured) == 1
    assert captured[0]["deployment"] is None


def test_main_creates_mlp_only_projection_critic(monkeypatch) -> None:
    class StopAfterCritic(RuntimeError):
        pass

    services: list[dict] = []
    created: list[dict] = []

    class Service:
        trainer_job_id = "job"
        max_context_length = 4096

        def create_training_client(self, base_model, **kwargs):
            created.append(kwargs)
            if kwargs.get("train_attn") is False:
                raise StopAfterCritic
            return object()

        def close(self):
            pass

    def build_service_client(**kwargs):
        services.append(kwargs)
        return Service()

    _patch_setup(monkeypatch)
    monkeypatch.setattr(module, "build_service_client", build_service_client)
    monkeypatch.setattr(
        module.ReconnectableClient,
        "from_training_client",
        classmethod(lambda cls, *_a, **_k: object()),
    )

    with pytest.raises(StopAfterCritic):
        module.main(_injected_config(), sample_prompt_fn=_unused_sampler, rows=[{"id": 1}])

    assert "projection_head_dim" not in services[0]
    assert services[1]["projection_head_dim"] == 1
    assert "train_attn" not in created[0]
    assert created[1]["train_attn"] is False
    assert created[1]["train_mlp"] is True
    assert created[1]["train_unembed"] is False


def test_resume_requires_an_aligned_actor_critic_pair() -> None:
    assert module.resume_position(None, None) == (0, 0)
    assert module.resume_position(
        ResumeInfo(step=3, data_consumed=24), ResumeInfo(step=3, data_consumed=24)
    ) == (3, 24)
    with pytest.raises(ValueError, match="different \\(step, rows\\) positions"):
        module.resume_position(None, ResumeInfo(step=3, data_consumed=24))
    with pytest.raises(ValueError, match="Resume an aligned pair"):
        module.resume_position(
            ResumeInfo(step=1, data_consumed=8), ResumeInfo(step=1, data_consumed=16)
        )
