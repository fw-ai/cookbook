from __future__ import annotations

from types import SimpleNamespace

import pytest
import tinker
import torch

import training.recipes.experiment.ppo_value_head_loop as module
from training.utils.rl.losses import PromptGroup


def _policy_datum(tokens: list[int], weights: list[float]) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=tokens,
                dtype="int64",
                shape=[len(tokens)],
            ),
            "weights": tinker.TensorData(
                data=weights,
                dtype="float32",
                shape=[len(weights)],
            ),
        },
    )


def test_build_projection_value_batch_reuses_full_sequence() -> None:
    policy = _policy_datum([10, 11, 12, 20, 21], [0, 0, 1, 1, 1])

    batch = module.build_projection_value_batch([policy])

    assert batch.data[0].model_input is policy.model_input
    assert batch.data[0].loss_fn_inputs == {}
    assert torch.equal(
        batch.action_masks[0],
        torch.tensor([False, False, True, True, True]),
    )


def test_projection_value_batch_requires_terminal_contiguous_response() -> None:
    with pytest.raises(ValueError, match="terminal, contiguous"):
        module.build_projection_value_batch(
            [_policy_datum([10, 11, 12, 20], [0, 1, 0, 1])]
        )


@pytest.mark.parametrize("dimension", [0, -1, True])
def test_value_decoder_requires_positive_integer_dimension(dimension) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        module.validate_value_decoder(dimension, None)


def test_scalar_value_decoder_rejects_support() -> None:
    with pytest.raises(ValueError, match="only valid for categorical"):
        module.validate_value_decoder(1, (0.0,))


def test_categorical_value_decoder_requires_matching_finite_support() -> None:
    with pytest.raises(ValueError, match="one value per"):
        module.validate_value_decoder(3, (-1.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        module.validate_value_decoder(2, (-1.0, float("inf")))


def test_projection_values_decode_scalar_identity() -> None:
    projection = torch.tensor([[1.5], [-2.0]])

    values = module.projection_values(
        projection,
        projection_head_dim=1,
        value_support=None,
    )

    assert torch.equal(values, torch.tensor([1.5, -2.0]))


def test_projection_values_decode_categorical_support_expectation() -> None:
    projection = torch.tensor([[0.0, 0.0], [-20.0, 20.0]])

    values = module.projection_values(
        projection,
        projection_head_dim=2,
        value_support=(-1.0, 1.0),
    )

    assert torch.allclose(values, torch.tensor([0.0, 1.0]), atol=1e-6)


def test_projection_values_validate_shape() -> None:
    with pytest.raises(ValueError, match=r"expected \[tokens, 2\]"):
        module.projection_values(
            torch.zeros(3, 1),
            projection_head_dim=2,
            value_support=(-1.0, 1.0),
        )


def test_reference_targets_match_terminal_reward_gae() -> None:
    mask = torch.tensor([False, False, True, True, True])
    values = torch.zeros(5)

    advantages, returns = module.reference_targets(
        terminal_rewards=[1.0],
        old_values=[values],
        action_masks=[mask],
        gamma=1.0,
        gae_lambda=0.95,
        critic_lambda=1.0,
    )

    assert torch.allclose(
        advantages[0],
        torch.tensor([0.0, 0.0, 0.95**2, 0.95, 1.0]),
    )
    assert torch.equal(returns[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))


def test_scalar_projection_value_loss_masks_prompt_and_captures_advantages() -> None:
    capture: dict = {}
    loss_fn = module.make_projection_value_loss_fn(
        terminal_rewards=[1.0],
        action_masks=[torch.tensor([False, True, True])],
        projection_head_dim=1,
        value_support=None,
        gamma=1.0,
        gae_lambda=1.0,
        critic_lambda=1.0,
        normalize_advantages_enabled=False,
        value_clip=0.2,
        value_loss_coef=0.5,
        capture=capture,
    )
    projection = torch.zeros((3, 1), requires_grad=True)

    loss, metrics = loss_fn([], [projection])
    loss.backward()

    assert loss.item() == 0.5
    assert metrics["value/loss"] == 1.0
    assert torch.equal(capture["old_values"][0], torch.zeros(3))
    assert torch.equal(capture["advantages"][0], torch.tensor([0.0, 1.0, 1.0]))
    assert projection.grad is not None
    assert projection.grad[0].item() == 0.0
    assert torch.equal(projection.grad[1:], torch.tensor([[-0.5], [-0.5]]))


def test_value_clip_uses_frozen_values_and_targets_after_critic_update() -> None:
    capture: dict = {}
    loss_fn = module.make_projection_value_loss_fn(
        terminal_rewards=[1.0],
        action_masks=[torch.tensor([False, True, True])],
        reference_values=[torch.zeros(3)],
        projection_head_dim=1,
        value_support=None,
        gamma=1.0,
        gae_lambda=0.95,
        critic_lambda=0.5,
        normalize_advantages_enabled=False,
        value_clip=0.2,
        value_loss_coef=1.0,
        capture=capture,
    )
    projection = torch.tensor([[0.0], [0.5], [0.5]], requires_grad=True)

    loss, _metrics = loss_fn([], [projection])
    loss.backward()

    # The pre-update targets are [0.5, 1.0]. Clipping at 0.2 produces
    # per-token squared errors 0.09 and 0.64, with no gradient past the bound.
    assert loss.item() == pytest.approx(0.365)
    assert torch.equal(capture["returns"][0], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.equal(projection.grad, torch.zeros_like(projection))


def test_categorical_projection_value_loss_backpropagates_output_gradients() -> None:
    capture: dict = {}
    loss_fn = module.make_projection_value_loss_fn(
        terminal_rewards=[1.0],
        action_masks=[torch.tensor([False, True])],
        projection_head_dim=2,
        value_support=(-1.0, 1.0),
        gamma=1.0,
        gae_lambda=1.0,
        critic_lambda=1.0,
        normalize_advantages_enabled=False,
        value_clip=0.2,
        value_loss_coef=1.0,
        capture=capture,
    )
    projection = torch.zeros((2, 2), requires_grad=True)

    loss, _metrics = loss_fn([], [projection])
    loss.backward()

    assert loss.item() == 1.0
    assert projection.grad is not None
    assert torch.equal(projection.grad[0], torch.zeros(2))
    assert torch.allclose(projection.grad[1].sum(), torch.tensor(0.0))
    assert projection.grad[1, 0] > 0
    assert projection.grad[1, 1] < 0


def test_projection_value_loss_uses_per_response_then_batch_normalization() -> None:
    capture: dict = {}
    loss_fn = module.make_projection_value_loss_fn(
        terminal_rewards=[1.0, 2.0],
        action_masks=[torch.tensor([True, True]), torch.tensor([True])],
        projection_head_dim=1,
        value_support=None,
        gamma=1.0,
        gae_lambda=1.0,
        critic_lambda=1.0,
        normalize_advantages_enabled=False,
        value_clip=0.2,
        value_loss_coef=1.0,
        capture=capture,
    )

    loss, _metrics = loss_fn(
        [],
        [
            torch.zeros((2, 1), requires_grad=True),
            torch.zeros((1, 1), requires_grad=True),
        ],
    )

    assert loss.item() == 2.5


def test_ppo_policy_loss_is_mean_per_response_then_batch() -> None:
    loss_fn = module.make_ppo_policy_loss_fn(
        old_logprobs=[[0.0, 0.0, 0.0], [0.0, 0.0]],
        advantages=[torch.tensor([0.0, 1.0, 3.0]), torch.tensor([2.0, 4.0])],
        action_masks=[
            torch.tensor([False, True, True]),
            torch.tensor([True, True]),
        ],
        clip_low=0.2,
        clip_high=0.28,
    )
    current = [torch.zeros(3, requires_grad=True), torch.zeros(2, requires_grad=True)]

    loss, metrics = loss_fn([], current)
    loss.backward()

    assert loss.item() == -2.5
    assert metrics["actor/pg_clipfrac"] == 0.0
    assert all(row.grad is not None for row in current)


def test_normalize_token_advantages_uses_only_active_tokens() -> None:
    normalized = module.normalize_token_advantages(
        [torch.tensor([100.0, 1.0, 2.0, 3.0])],
        [torch.tensor([False, True, True, True])],
        enabled=True,
    )

    active = normalized[0][1:]
    assert torch.isclose(active.mean(), torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(active.std(unbiased=False), torch.tensor(1.0))
    assert normalized[0][0] == 100.0


def test_terminal_rewards_require_one_reward_per_datum() -> None:
    group = PromptGroup(
        data=[
            _policy_datum([1, 2], [0, 1]),
            _policy_datum([3, 4], [0, 1]),
        ],
        advantages=[0.0, 0.0],
        ref_logprobs=None,
        prompt_len=1,
        rewards=[1.0],
    )

    with pytest.raises(ValueError, match="one datum per rollout run"):
        module.aligned_terminal_rewards([group])


def test_default_config_uses_separate_handles_for_same_base() -> None:
    config = module.Config(log_path="/tmp/test")

    assert config.actor_base_model == module.QWEN3_4B
    assert config.critic_base_model == module.QWEN3_4B
    assert config.critic_projection_head_dim == 1
    assert config.critic_lora_rank > 0


def test_injected_rollouts_do_not_provision_actor_deployment(monkeypatch) -> None:
    class StopAfterActorService(RuntimeError):
        pass

    captured: list[dict] = []

    def build_service_client(**kwargs):
        captured.append(kwargs)
        raise StopAfterActorService

    async def sample_prompt(_row, *, cursor_index: int):
        raise AssertionError(f"sampling should not start during setup: {cursor_index}")

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(module, "setup_wandb", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "validate_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "build_service_client", build_service_client)
    monkeypatch.setattr(
        module,
        "load_deployment_tokenizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("injected rollouts do not need a tokenizer")
        ),
    )

    with pytest.raises(StopAfterActorService):
        module.main(
            module.Config(
                log_path="/tmp/ppo-value-test",
                critic_trainer=module.TrainerConfig(training_shape_id="critic-shape"),
                deployment=module.DeployConfig(tokenizer_model="tokenizer"),
            ),
            sample_prompt_fn=sample_prompt,
            rows=[{"id": 1}],
        )

    assert len(captured) == 1
    assert captured[0]["deployment"] is None


def test_injected_rollouts_do_not_read_managed_deployment_metadata() -> None:
    class TrainerOnlyService:
        @property
        def deployment_id(self):
            raise AssertionError("trainer-only services have no deployment id")

    assert module.managed_deployment_id(TrainerOnlyService(), enabled=False) is None


def test_critic_helper_selects_projection_output() -> None:
    calls = []

    class Critic:
        def forward_backward_custom(self, data, loss_fn, **kwargs):
            calls.append((data, loss_fn, kwargs))
            return "critic-result"

    config = module.Config(log_path="/tmp/test")
    data = [
        tinker.Datum(model_input=tinker.ModelInput.from_ints([1]), loss_fn_inputs={})
    ]

    result = module.forward_backward_projection_critic(
        Critic(),
        data=data,
        terminal_rewards=[1.0],
        action_masks=[torch.tensor([True])],
        config=config,
        capture={},
    )

    assert result == "critic-result"
    assert calls[0][0] is data
    assert callable(calls[0][1])
    assert calls[0][2] == {"output": "projection"}


def test_actor_helper_selects_language_model_logprobs() -> None:
    calls = []
    forward_result = SimpleNamespace(
        loss_fn_outputs=[
            {"logprobs": tinker.TensorData(data=[0.0], dtype="float32", shape=[1])}
        ]
    )

    class Actor:
        def forward(self, data, loss_fn):
            calls.append(("forward", data, loss_fn))
            return forward_result

        def forward_backward_custom(self, data, loss_fn, **kwargs):
            calls.append(("backward", data, loss_fn, kwargs))
            return "actor-result"

    data = [_policy_datum([1], [1])]
    result = module.forward_backward_ppo_actor(
        Actor(),
        data=data,
        advantages=[torch.tensor([1.0])],
        action_masks=[torch.tensor([True])],
        clip_low=0.2,
        clip_high=0.28,
    )

    assert result == "actor-result"
    assert calls[0] == ("forward", data, "cross_entropy")
    assert calls[1][0] == "backward"
    assert calls[1][3] == {"precomputed_forward": forward_result}


def test_missing_critic_shape_fails_before_provisioning(monkeypatch) -> None:
    def unexpected_provision(**_kwargs):
        raise AssertionError("invalid topology must fail before resource creation")

    monkeypatch.setattr(module, "build_service_client", unexpected_provision)
    with pytest.raises(ValueError, match="critic_trainer.training_shape_id"):
        module.main(module.Config(log_path="/tmp/test"), rows=[{"id": 1}])
