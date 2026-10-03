"""SAO switches of ``recipes.experiment.ppo_value_head_loop``.

The oracles are small, independent reference implementations of SAO's DIS
loss (arXiv 2607.07508, Eqs. 1-3) and of an action-only GAE scan, so the
recipe's loss and advantages are checked against separately written math.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import tinker
import torch

import training.recipes.experiment.ppo_value_head_loop as module
from training.utils.checkpoints import ResumeInfo


def oracle_dis_loss(current, behavior, advantages, mask, *, eps_low, eps_high):
    lp, old, adv = current[mask], behavior[mask], advantages[mask].detach()
    delta = lp.detach().double() - old.detach().double()
    keep = (delta > math.log1p(-eps_low)) & (delta < math.log1p(eps_high))
    weights = torch.zeros_like(delta)
    weights[keep] = delta[keep].exp()
    return -(weights.to(lp.dtype) * adv * lp).mean()


def oracle_gae(rewards, values, mask, *, gamma, lam):
    active = mask.nonzero().flatten().tolist()
    r = rewards.double().tolist()
    v = values.double().tolist()
    adv = [0.0] * len(r)
    next_value, next_adv = 0.0, 0.0
    for t in reversed(active):
        delta = r[t] + gamma * next_value - v[t]
        next_adv = delta + gamma * lam * next_adv
        adv[t] = next_adv
        next_value = v[t]
    return torch.tensor(adv, dtype=torch.float64)


def _datum(tokens: list[int], weights: list[int]) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=tokens, dtype="int64", shape=[len(tokens)]
            ),
            "weights": tinker.TensorData(
                data=weights, dtype="int64", shape=[len(weights)]
            ),
        },
    )


# -- DIS actor objective ------------------------------------------------------


def test_dis_loss_and_gradient_match_reference() -> None:
    generator = torch.Generator().manual_seed(0)
    masks = [
        torch.tensor([False, True, True, True, True, True]),
        torch.tensor([False, False, True, True]),
    ]
    behaviors = [torch.randn(6, generator=generator) - 2 for _ in range(1)] + [
        torch.randn(4, generator=generator) - 2
    ]
    # Push some tokens outside (0.7, 6) in both directions.
    currents = [
        behaviors[0] + torch.tensor([0.0, 0.1, -0.5, 2.5, -0.01, 0.3]),
        behaviors[1] + torch.tensor([0.0, 0.0, 1.0, -1.0]),
    ]
    advantages = [
        torch.tensor([0.0, 1.0, -2.0, 0.5, 3.0, -1.0]),
        torch.tensor([0.0, 0.0, -0.5, 2.0]),
    ]
    recipe_inputs = [c.clone().requires_grad_() for c in currents]
    loss_fn = module.make_dis_policy_loss_fn(
        behavior_logprobs=behaviors,
        advantages=advantages,
        action_masks=masks,
        dis_low=0.3,
        dis_high=5.0,
    )
    loss, metrics = loss_fn([], recipe_inputs)
    loss.backward()

    oracle_inputs = [c.clone().requires_grad_() for c in currents]
    oracle = torch.stack(
        [
            oracle_dis_loss(c, b, a, m, eps_low=0.3, eps_high=5.0)
            for c, b, a, m in zip(oracle_inputs, behaviors, advantages, masks)
        ]
    ).mean()
    oracle.backward()

    assert torch.allclose(loss, oracle)
    for mine, theirs in zip(recipe_inputs, oracle_inputs):
        assert torch.allclose(mine.grad, theirs.grad)
    assert 0 < metrics["actor/rejected_token_fraction"] < 1


def test_dis_rejects_both_advantage_signs_and_keeps_denominator() -> None:
    mask = torch.tensor([True, True, True, True])
    behavior = torch.zeros(4)
    # ratios: 1, e^-1 (< 0.7, rejected), e^2 (> 6, rejected), 1
    current = torch.tensor([0.0, -1.0, 2.0, 0.0], requires_grad=True)
    advantage = torch.tensor([1.0, 1.0, -1.0, -1.0])
    loss_fn = module.make_dis_policy_loss_fn(
        behavior_logprobs=[behavior],
        advantages=[advantage],
        action_masks=[mask],
        dis_low=0.3,
        dis_high=5.0,
    )

    loss, metrics = loss_fn([], [current])
    loss.backward()

    # Stop-gradient weight: dL/dlogp = -w * A / N with N = 4 (rejected kept).
    assert torch.allclose(current.grad, torch.tensor([-0.25, 0.0, 0.0, 0.25]))
    assert metrics["actor/rejected_token_fraction"] == 0.5


def test_dis_ratio_uses_sampler_logprobs_not_trainer_recompute() -> None:
    """Reference PPO's ratio is 1; DIS measures the sampler/trainer gap."""
    mask = torch.tensor([True, True])
    current = torch.tensor([-1.0, -1.0], requires_grad=True)
    behavior = torch.tensor([-1.2, -0.9])
    loss_fn = module.make_dis_policy_loss_fn(
        behavior_logprobs=[behavior],
        advantages=[torch.tensor([1.0, 1.0])],
        action_masks=[mask],
        dis_low=0.3,
        dis_high=5.0,
    )

    _loss, metrics = loss_fn([], [current])

    assert metrics["actor/behavior_logprob_mean_abs_delta"] == pytest.approx(0.15)
    assert metrics["actor/behavior_logprob_max_abs_delta"] == pytest.approx(0.2)


class _Actor:
    def __init__(self, logprobs: list[list[float]]):
        self.logprobs = logprobs
        self.calls: list[tuple] = []

    def forward(self, data, loss_fn):
        self.calls.append(("forward", len(data)))
        return SimpleNamespace(
            loss_fn_outputs=[
                {
                    "logprobs": tinker.TensorData(
                        data=self.logprobs[i],
                        dtype="float32",
                        shape=[len(self.logprobs[i])],
                    )
                }
                for i in range(len(data))
            ]
        )

    def forward_backward_custom(self, data, loss_fn, **kwargs):
        self.calls.append(("custom", kwargs))
        current = [
            torch.tensor(row, requires_grad=True) for row in self.logprobs[: len(data)]
        ]
        loss, metrics = loss_fn(data, current)
        loss.backward()
        return SimpleNamespace(metrics=metrics, grads=[row.grad for row in current])


def test_dis_actor_reuses_reference_forward_with_sampler_denominator() -> None:
    actor = _Actor([[0.0, -1.0, -1.0]])

    result = module.forward_backward_ppo_actor(
        actor,
        data=[_datum([1, 2, 3], [0, 1, 1])],
        advantages=[torch.tensor([0.0, 1.0, 1.0])],
        action_masks=[torch.tensor([False, True, True])],
        clip_low=0.2,
        clip_high=0.28,
        behavior_logprobs=[torch.tensor([0.0, -1.2, -0.9])],
        dis_bounds=(0.3, 5.0),
    )

    # Same call shape as reference PPO: one forward, reused by the custom loss.
    assert [call[0] for call in actor.calls] == ["forward", "custom"]
    assert "precomputed_forward" in actor.calls[1][1]
    # The ratio is against the sampler, not 1, so equal advantages get
    # different per-token gradients.
    assert not torch.allclose(result.grads[0][1], result.grads[0][2])


def test_reference_ppo_ratio_is_one_without_sampler_denominator() -> None:
    actor = _Actor([[0.0, -1.0, -1.0]])

    result = module.forward_backward_ppo_actor(
        actor,
        data=[_datum([1, 2, 3], [0, 1, 1])],
        advantages=[torch.tensor([0.0, 1.0, 1.0])],
        action_masks=[torch.tensor([False, True, True])],
        clip_low=0.2,
        clip_high=0.28,
    )

    assert result.metrics["actor/pg_clipfrac"] == 0.0
    assert result.metrics["actor/ppo_kl"] == 0.0
    assert torch.allclose(result.grads[0][1], result.grads[0][2])


def test_dis_actor_requires_sampler_logprobs() -> None:
    with pytest.raises(ValueError, match="behavior logprobs"):
        module.forward_backward_ppo_actor(
            _Actor([[0.0, -1.0]]),
            data=[_datum([1, 2], [0, 1])],
            advantages=[torch.tensor([0.0, 1.0])],
            action_masks=[torch.tensor([False, True])],
            clip_low=0.2,
            clip_high=0.28,
            dis_bounds=(0.3, 5.0),
        )


def test_behavior_logprob_rows_validate_alignment_and_finiteness() -> None:
    masks = [torch.tensor([False, True])]
    assert torch.equal(
        module.behavior_logprob_rows([[0.0, -0.5]], masks)[0], torch.tensor([0.0, -0.5])
    )
    with pytest.raises(ValueError, match="sampler logprobs for every datum"):
        module.behavior_logprob_rows([], masks)
    with pytest.raises(ValueError, match="1 sampler logprobs for 2 target"):
        module.behavior_logprob_rows([[0.0]], masks)
    with pytest.raises(ValueError, match="non-finite"):
        module.behavior_logprob_rows([[0.0, float("nan")]], masks)


# -- Adaptive GAE --------------------------------------------------------------


def test_adaptive_lambda_per_response_length() -> None:
    masks = [torch.tensor([False] + [True] * 10), torch.tensor([True] * 16384)]

    lambdas = module.adaptive_gae_lambdas(masks, alpha=1.5)

    assert lambdas[0] == pytest.approx(1 - 1 / 15)
    assert lambdas[1] == pytest.approx(1 - 1 / (1.5 * 16384))
    config = module.Config(log_path="/tmp/test")
    assert module.actor_gae_lambdas(masks, config) == [0.95, 0.95]


def test_long_horizon_gae_matches_float64_oracle() -> None:
    length = 16384
    mask = torch.ones(length, dtype=torch.bool)
    mask[:100] = False
    values = torch.linspace(0.1, 0.9, length, dtype=torch.float32)
    lam = module.adaptive_gae_lambdas([mask], alpha=1.5)[0]
    rewards = torch.zeros(length)
    rewards[-1] = 1.0

    advantages, returns = module.reference_targets(
        terminal_rewards=[1.0],
        old_values=[values],
        action_masks=[mask],
        gamma=1.0,
        gae_lambda=[lam],
        critic_lambda=1.0,
    )

    expected = oracle_gae(rewards, values, mask, gamma=1.0, lam=lam)
    assert torch.allclose(advantages[0].double(), expected, atol=1e-6)
    # gamma = lambda_critic = 1: every action token's target is the reward.
    assert torch.equal(returns[0][mask], torch.ones(int(mask.sum())))


def test_reference_targets_accept_per_trajectory_lambdas() -> None:
    masks = [torch.tensor([True, True]), torch.tensor([True, True])]
    values = [torch.zeros(2), torch.zeros(2)]

    advantages, _ = module.reference_targets(
        terminal_rewards=[1.0, 1.0],
        old_values=values,
        action_masks=masks,
        gamma=1.0,
        gae_lambda=[0.5, 1.0],
        critic_lambda=1.0,
    )

    assert torch.allclose(advantages[0], torch.tensor([0.5, 1.0]))
    assert torch.allclose(advantages[1], torch.tensor([1.0, 1.0]))
    with pytest.raises(ValueError, match="one actor GAE lambda per trajectory"):
        module.reference_targets(
            terminal_rewards=[1.0, 1.0],
            old_values=values,
            action_masks=masks,
            gamma=1.0,
            gae_lambda=[0.5],
            critic_lambda=1.0,
        )


# -- Critic updates with refreshed values ---------------------------------------


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
                    "projection": tinker.TensorData(
                        data=row.flatten().tolist(),
                        dtype="float32",
                        shape=list(row.shape),
                    )
                }
                for row in self._rows(data)
            ]
        )


def _critic_batch() -> module.CriticBatch:
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]), loss_fn_inputs={}
    )
    return module.CriticBatch(
        data=[datum],
        terminal_rewards=[1.0],
        action_masks=[torch.tensor([False, True, True])],
    )


def test_sao_critic_refreshes_values_and_uses_post_update_advantages() -> None:
    critic = _Critic()
    config = module.Config(
        log_path="/tmp/test", critic_updates=2, actor_values="post_update"
    )

    outcome = module.train_critic_then_advantages(
        critic,
        batch=_critic_batch(),
        actor_lambdas=[1.0],
        config=config,
        adam_params=tinker.AdamParams(learning_rate=1e-6),
        normalization="none",
    )

    assert critic.events == ["fwd_bwd", "optim", "fwd_bwd", "optim", "forward"]
    # The second update's loss sees the parameters left by the first.
    assert critic.seen_values == [0.0, 0.25]
    assert torch.allclose(outcome.actor_values[0], torch.full((3,), 0.5))
    # lambda=1: advantage = reward - V(s_t) on action tokens, from the fresh read.
    assert torch.allclose(outcome.advantages[0], torch.tensor([0.0, 0.5, 0.5]))
    assert torch.allclose(outcome.pre_update_values[0], torch.zeros(3))
    assert len(outcome.fwd_bwd_results) == 2
    assert outcome.metrics["train/critic-value_loss/epoch_1"] == pytest.approx(1.0)
    assert outcome.metrics["train/critic-value_loss/epoch_2"] == pytest.approx(0.64)
    assert outcome.metrics["train/critic_post/bias"] == pytest.approx(-0.5)


def test_reference_critic_uses_pre_update_values_without_extra_forward() -> None:
    critic = _Critic()
    config = module.Config(log_path="/tmp/test")

    outcome = module.train_critic_then_advantages(
        critic,
        batch=_critic_batch(),
        actor_lambdas=[0.95],
        config=config,
        adam_params=tinker.AdamParams(learning_rate=1e-6),
        normalization="none",
    )

    assert critic.events == ["fwd_bwd", "optim"]
    assert torch.allclose(outcome.advantages[0], torch.tensor([0.0, 0.95, 1.0]))


# -- Guards and diagnostics -----------------------------------------------------


def test_critic_statistics_report_bias_that_explained_variance_ignores() -> None:
    masks = [torch.tensor([True, True]), torch.tensor([True, True])]
    offset = 0.5
    values = [torch.tensor([0.0, 0.0]) + offset, torch.tensor([1.0, 1.0]) + offset]

    stats = module.critic_value_statistics(values, [0.0, 1.0], masks)

    assert stats["explained_variance"] == pytest.approx(1.0)
    assert stats["bias"] == pytest.approx(offset)
    assert stats["token_mse"] == pytest.approx(offset**2)
    assert stats["constant_baseline_mse"] == pytest.approx(0.25)


# -- Config -----------------------------------------------------------------------


def test_sao_config_applies_sao_settings() -> None:
    config = module.sao_config("/tmp/test", critic_learning_rate=1e-5)

    assert config.policy_objective == "dis"
    assert config.gae_mode == "adaptive"
    assert config.critic_updates == 2
    assert config.actor_values == "post_update"
    assert config.critic_warmup_batches == 10
    assert config.critic_train_attn is False
    assert config.value_pretrain_steps == 64
    assert config.critic_learning_rate == 1e-5
    module.validate_sao_settings(
        config,
        module.ValuePretrainData(
            train=[_trajectory(1.0)], validation=[_trajectory(0.0)]
        ),
    )


def test_default_config_is_reference_ppo() -> None:
    config = module.Config(log_path="/tmp/test")

    assert config.policy_objective == "ppo"
    assert config.gae_mode == "fixed"
    assert config.critic_updates == 1
    assert config.actor_values == "pre_update"
    assert config.critic_warmup_batches == 0
    assert config.critic_train_attn and config.critic_train_mlp
    module.validate_sao_settings(config)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"policy_objective": "grpo"}, "policy_objective must be one of"),
        ({"dis_low": 1.0}, "dis_low"),
        ({"critic_updates": 0}, "critic_updates must be a positive integer"),
        ({"value_pretrain_steps": 4}, "needs value_pretrain_data"),
        ({"critic_train_attn": False, "critic_lora_rank": 0}, "critic_lora_rank > 0"),
        ({"critic_train_attn": False, "critic_train_mlp": False}, "at least one"),
    ],
)
def test_validate_sao_settings_rejects_invalid(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        module.validate_sao_settings(module.Config(log_path="/tmp/test", **overrides))


def test_main_creates_mlp_only_critic_session(monkeypatch) -> None:
    class StopAfterCritic(RuntimeError):
        pass

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

    async def sample_prompt(_row, *, cursor_index: int):
        raise AssertionError(cursor_index)

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(module, "setup_wandb", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "validate_config", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "build_service_client", lambda **_k: Service())
    monkeypatch.setattr(
        module.ReconnectableClient,
        "from_training_client",
        classmethod(lambda cls, *_a, **_k: object()),
    )

    with pytest.raises(StopAfterCritic):
        module.main(
            module.Config(
                log_path="/tmp/ppo-value-test",
                critic_train_attn=False,
                deployment=module.DeployConfig(tokenizer_model="tokenizer"),
            ),
            sample_prompt_fn=sample_prompt,
            rows=[{"id": 1}],
        )

    assert created[0].get("train_attn", True) is True
    assert created[1]["train_attn"] is False
    assert created[1]["train_mlp"] is True
    assert created[1]["train_unembed"] is False
    assert "train_unembed" not in created[0]


# -- Value pretraining ------------------------------------------------------------


def _trajectory(reward: float) -> module.PromptGroup:
    """One saved trajectory: prompt [1, 2], completion [3, 4]."""
    return module.PromptGroup(
        data=[_datum([1, 2, 3], [0, 1, 1])],
        advantages=[reward],
        ref_logprobs=None,
        prompt_len=2,
        rewards=[reward],
    )


class _Checkpoints:
    def __init__(self, critic):
        self.critic = critic
        self.saved: list[float] = []
        self.restored = False

    def save(self, name, *, resumable, promotable, data_consumed):
        assert (name, resumable, promotable, data_consumed) == (
            "value-pretrain-best",
            True,
            False,
            0,
        )
        self.saved.append(self.critic.value)

    def resume(self, *, restore_optimizer):
        assert restore_optimizer
        self.critic.value = self.saved[-1]
        self.restored = True
        return SimpleNamespace(step=0, data_consumed=0)


def _pretrain_config(**overrides) -> module.Config:
    settings = {
        "value_pretrain_steps": 4,
        "value_pretrain_batch_size": 2,
        "value_pretrain_eval_interval": 2,
    }
    return module.Config(log_path="/tmp/test", **{**settings, **overrides})


def test_value_pretraining_restores_best_held_out_checkpoint(monkeypatch) -> None:
    metrics = []
    monkeypatch.setattr(module, "log_metrics", lambda values, **_k: metrics.append(values))
    critic = _Critic()
    checkpoints = _Checkpoints(critic)
    # Validation targets are 0.5: MSE is best at value 0.5 (after 2 steps),
    # then worsens at 1.0 (after 4 steps).
    data = module.ValuePretrainData(
        train=[_trajectory(1.0), _trajectory(0.0)],
        validation=[_trajectory(0.5)],
    )

    best = module.run_value_pretraining(
        critic,
        checkpoints,
        pretrain_data=data,
        config=_pretrain_config(),
        adam_params=tinker.AdamParams(learning_rate=1e-6),
        normalization="none",
    )

    assert best == 2
    assert checkpoints.saved == [0.5]
    assert checkpoints.restored
    assert critic.value == 0.5
    training_metrics = [m for m in metrics if "pretrain/train/step_wall_time_s" in m]
    assert len(training_metrics) == 4
    assert all(m["pretrain/train/token_positions"] == 6 for m in training_metrics)
    assert all(
        m["pretrain/train/step_wall_time_s"] >= m["pretrain/train/fwd_bwd_time_s"] > 0
        for m in training_metrics
    )


def test_value_pretraining_refuses_actor_training_without_improvement(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "log_metrics", lambda *_a, **_k: None)
    critic = _Critic()
    data = module.ValuePretrainData(
        train=[_trajectory(1.0)],
        validation=[_trajectory(0.0)],
    )

    with pytest.raises(RuntimeError, match="refusing actor training"):
        module.run_value_pretraining(
            critic,
            _Checkpoints(critic),
            pretrain_data=data,
            config=_pretrain_config(),
            adam_params=tinker.AdamParams(learning_rate=1e-6),
            normalization="none",
        )


# -- Warmup checkpoints ------------------------------------------------------------


class _SavedCheckpoints:
    def __init__(self):
        self.saved: list[tuple[str, int]] = []

    def save(self, name, *, resumable, promotable, data_consumed):
        assert resumable and not promotable
        self.saved.append((name, data_consumed))


def test_warmup_periodic_save_checkpoints_only_critic() -> None:
    actor, critic = _SavedCheckpoints(), _SavedCheckpoints()

    module.save_periodic_checkpoints(
        actor, critic, actor_updated=False, actor_step=0, rollout_batch=2, data_consumed=16
    )
    module.save_periodic_checkpoints(
        actor, critic, actor_updated=True, actor_step=1, rollout_batch=3, data_consumed=24
    )

    assert actor.saved == [("step-1", 24)]
    assert critic.saved == [("step-2", 16), ("step-3", 24)]


def test_resume_accepts_untouched_actor_with_warmup_critic() -> None:
    critic = ResumeInfo(step=2, data_consumed=16)

    assert (
        module.resume_rows_consumed(None, critic, critic_warmup_batches=2) == 16
    )
    # A weights-only actor init also reports step 0 and no consumed rows.
    assert (
        module.resume_rows_consumed(
            ResumeInfo(step=0, data_consumed=0),
            critic,
            critic_warmup_batches=2,
        )
        == 16
    )
    # A prior final save may record an older cursor for the still-untouched
    # actor before a later critic-only warmup checkpoint advances the cursor.
    assert (
        module.resume_rows_consumed(
            ResumeInfo(step=0, data_consumed=16),
            ResumeInfo(step=3, data_consumed=24),
            critic_warmup_batches=10,
        )
        == 24
    )


@pytest.mark.parametrize(
    ("actor", "critic"),
    [
        # Past warmup, the actor must have been checkpointed with the critic.
        (None, ResumeInfo(step=3, data_consumed=24)),
        (
            ResumeInfo(step=1, data_consumed=24),
            ResumeInfo(step=4, data_consumed=32),
        ),
        # Never move an untouched actor's dataset cursor backward.
        (
            ResumeInfo(step=0, data_consumed=24),
            ResumeInfo(step=2, data_consumed=16),
        ),
    ],
)
def test_resume_rejects_misaligned_pair(actor, critic) -> None:
    with pytest.raises(ValueError, match="different dataset cursors"):
        module.resume_rows_consumed(actor, critic, critic_warmup_batches=2)
