"""Projection-head critic utilities in ``training.utils.rl.critic``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from fireworks.training import sdk
from training.utils.rl.critic import (
    CriticBatch,
    advantage_metrics,
    critic_batch,
    make_value_loss_fn,
    predict_values,
    projection_values,
    token_gae,
    train_critic,
    value_statistics,
)
from training.utils.rl.losses import PromptGroup, combine_prompt_groups


def _datum(tokens: list[int], weights: list[float]) -> sdk.Datum:
    return sdk.Datum(
        model_input=sdk.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": sdk.TensorData(data=tokens, dtype="int64", shape=[len(tokens)]),
            "weights": sdk.TensorData(data=weights, dtype="float32", shape=[len(weights)]),
        },
    )


def _group(data: list[sdk.Datum], rewards: list[float]) -> PromptGroup:
    return PromptGroup(
        data=data,
        advantages=list(rewards),
        prompt_len=1,
        rewards=rewards,
    )


def test_critic_batch_reuses_full_sequence_without_targets() -> None:
    policy = _datum([10, 11, 12, 20, 21], [0, 0, 1, 1, 1])

    batch = critic_batch(combine_prompt_groups([_group([policy], [1.0])]))

    assert batch.data[0].model_input is policy.model_input
    assert batch.data[0].loss_fn_inputs == {}
    assert batch.terminal_rewards == [1.0]
    assert torch.equal(batch.action_masks[0], torch.tensor([False, False, True, True, True]))


def test_critic_batch_requires_terminal_contiguous_response() -> None:
    group = _group([_datum([10, 11, 12, 20], [0, 1, 0, 1])], [1.0])
    with pytest.raises(ValueError, match="terminal, contiguous"):
        critic_batch(combine_prompt_groups([group]))


def test_critic_batch_requires_one_reward_per_datum() -> None:
    group = _group([_datum([1, 2], [0, 1]), _datum([3, 4], [0, 1])], [1.0])
    with pytest.raises(ValueError, match="one datum per rollout run"):
        critic_batch(combine_prompt_groups([group]))


def test_projection_values_decode_scalar_head() -> None:
    assert torch.equal(projection_values(torch.tensor([[1.5], [-2.0]])), torch.tensor([1.5, -2.0]))
    with pytest.raises(ValueError, match=r"expected \[tokens, 1\]"):
        projection_values(torch.zeros(3, 2))


def test_token_gae_matches_terminal_reward_scan() -> None:
    mask = torch.tensor([False, False, True, True, True])

    advantages, _ = token_gae([1.0], [torch.zeros(5)], [mask], gamma=1.0, lam=0.95)
    _, returns = token_gae([1.0], [torch.zeros(5)], [mask], gamma=1.0, lam=1.0)

    assert torch.allclose(advantages[0], torch.tensor([0.0, 0.0, 0.95**2, 0.95, 1.0]))
    assert torch.equal(returns[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))


def test_token_gae_accepts_one_lambda_per_response() -> None:
    masks = [torch.tensor([True, True]), torch.tensor([True, True])]
    values = [torch.zeros(2), torch.zeros(2)]

    advantages, _ = token_gae([1.0, 1.0], values, masks, gamma=1.0, lam=[0.5, 1.0])

    assert torch.allclose(advantages[0], torch.tensor([0.5, 1.0]))
    assert torch.allclose(advantages[1], torch.tensor([1.0, 1.0]))
    with pytest.raises(ValueError, match="must align by response"):
        token_gae([1.0, 1.0], values, masks, gamma=1.0, lam=[0.5])


def _batch(rewards: list[float], masks: list[torch.Tensor]) -> CriticBatch:
    data = [
        sdk.Datum(
            model_input=sdk.ModelInput.from_ints(list(range(1, len(mask) + 1))),
            loss_fn_inputs={},
        )
        for mask in masks
    ]
    return CriticBatch(data=data, terminal_rewards=rewards, action_masks=masks)


def test_value_loss_masks_prompt_and_captures_old_values() -> None:
    capture: dict = {}
    loss_fn = make_value_loss_fn(
        _batch([1.0], [torch.tensor([False, True, True])]),
        gamma=1.0,
        critic_lambda=1.0,
        value_clip=0.2,
        capture=capture,
    )
    projection = torch.zeros((3, 1), requires_grad=True)

    loss, metrics = loss_fn([], [projection])
    loss.backward()

    assert loss.item() == 1.0
    assert metrics["value/loss"] == 1.0
    assert capture["loss"] == 1.0
    assert torch.equal(capture["old_values"][0], torch.zeros(3))
    assert projection.grad[0].item() == 0.0
    assert torch.equal(projection.grad[1:], torch.tensor([[-1.0], [-1.0]]))


def test_value_clip_uses_frozen_old_values_and_targets() -> None:
    loss_fn = make_value_loss_fn(
        _batch([1.0], [torch.tensor([False, True, True])]),
        gamma=1.0,
        critic_lambda=0.5,
        value_clip=0.2,
        capture={},
        old_values=[torch.zeros(3)],
    )
    projection = torch.tensor([[0.0], [0.5], [0.5]], requires_grad=True)

    loss, _metrics = loss_fn([], [projection])
    loss.backward()

    # Targets from the old values are [0.5, 1.0]. Clipping at 0.2 produces
    # squared errors 0.09 and 0.64, with no gradient past the bound.
    assert loss.item() == pytest.approx(0.365)
    assert torch.equal(projection.grad, torch.zeros_like(projection))


def test_value_loss_averages_per_response_then_batch() -> None:
    loss_fn = make_value_loss_fn(
        _batch([1.0, 2.0], [torch.tensor([True, True]), torch.tensor([True])]),
        gamma=1.0,
        critic_lambda=1.0,
        value_clip=0.2,
        capture={},
    )

    loss, _metrics = loss_fn([], [torch.zeros((2, 1)), torch.zeros((1, 1))])

    assert loss.item() == 2.5


class _Critic:
    """Values are one shared scalar that each optimizer step raises by 0.25."""

    def __init__(self):
        self.value = 0.0
        self.events: list[str] = []
        self.seen_values: list[float] = []

    def _rows(self, data):
        return [torch.full((d.model_input.length, 1), self.value) for d in data]

    def forward_backward_custom(self, data, loss_fn, *, output):
        assert output == "projection"
        self.events.append("fwd_bwd")
        self.seen_values.append(self.value)
        projections = [row.requires_grad_() for row in self._rows(data)]
        loss, metrics = loss_fn(data, projections)
        loss.backward()
        return SimpleNamespace(metrics=metrics)

    def optim_step(self, _params, **_kwargs):
        self.events.append("optim")
        self.value += 0.25
        return SimpleNamespace(metrics={})

    def forward_projection(self, data):
        self.events.append("forward")
        return SimpleNamespace(
            loss_fn_outputs=[
                {
                    "projection": sdk.TensorData(
                        data=row.flatten().tolist(),
                        dtype="float32",
                        shape=list(row.shape),
                    )
                }
                for row in self._rows(data)
            ]
        )


def test_train_critic_rereads_values_with_frozen_targets() -> None:
    critic = _Critic()
    batch = _batch([1.0], [torch.tensor([False, True, True])])

    update = train_critic(
        critic,
        batch,
        updates=2,
        gamma=1.0,
        critic_lambda=1.0,
        value_clip=0.2,
        adam_params=sdk.AdamParams(learning_rate=1e-6),
    )
    values = predict_values(critic, batch)

    assert critic.events == ["fwd_bwd", "optim", "fwd_bwd", "optim", "forward"]
    # The second update's loss sees the parameters left by the first.
    assert critic.seen_values == [0.0, 0.25]
    assert update.value_losses == pytest.approx([1.0, 0.64])
    assert len(update.fwd_bwd_results) == 2
    assert torch.allclose(values[0], torch.full((3,), 0.5))


def test_value_statistics_report_bias_that_explained_variance_ignores() -> None:
    masks = [torch.tensor([True, True]), torch.tensor([True, True])]
    offset = 0.5
    values = [torch.tensor([0.0, 0.0]) + offset, torch.tensor([1.0, 1.0]) + offset]

    stats = value_statistics(values, [0.0, 1.0], masks)

    assert stats["explained_variance"] == pytest.approx(1.0)
    assert stats["bias"] == pytest.approx(offset)
    assert stats["token_mse"] == pytest.approx(offset**2)
    assert stats["constant_baseline_mse"] == pytest.approx(0.25)


def test_value_statistics_target_discounted_returns() -> None:
    # A critic that predicts the discounted return exactly has zero GAE and
    # must also show zero calibration error when gamma < 1.
    gamma = 0.5
    values = [torch.tensor([0.25, 0.5, 1.0])]
    masks = [torch.tensor([True, True, True])]
    advantages, _returns = token_gae([1.0], values, masks, gamma=gamma, lam=1.0)
    assert advantages[0].abs().max().item() == pytest.approx(0.0)

    stats = value_statistics(values, [1.0], masks, gamma=gamma)

    assert stats["token_mse"] == pytest.approx(0.0)
    assert stats["bias"] == pytest.approx(0.0)
    assert value_statistics(values, [1.0], masks)["token_mse"] == pytest.approx(0.270833, abs=1e-6)


def test_advantage_metrics_use_only_action_tokens() -> None:
    metrics = advantage_metrics(
        [torch.tensor([100.0, 1.0, -3.0])],
        [torch.tensor([9.0, 0.5, 1.5])],
        [torch.tensor([False, True, True])],
    )

    assert metrics["train/advantage_mean"] == -1.0
    assert metrics["train/advantage_max_abs"] == 3.0
    assert metrics["train/old_value_mean"] == 1.0
