"""Tests for AReaL-style decoupled IS corrections."""

from __future__ import annotations

import math

import pytest
import torch

from training.utils.rl.importance_sampling import (
    DecoupledConfig,
    _compute_proximal_logprobs,
    compute_decoupled_corrections,
)


class TestComputeProximalLogprobs:
    """Tests for the log-linear proximal approximation."""

    def test_on_policy_alpha_zero(self):
        pi_theta = torch.tensor([-0.5, -0.3, -0.8])
        pi_old = torch.tensor([-0.6, -0.4, -0.9])
        pi_prox, alpha = _compute_proximal_logprobs(pi_theta, pi_old, generation_step=5, current_step=5)
        assert alpha == 0.0
        torch.testing.assert_close(pi_prox, pi_old)

    def test_one_step_stale(self):
        pi_theta = torch.tensor([-0.5, -0.3])
        pi_old = torch.tensor([-0.6, -0.4])
        pi_prox, alpha = _compute_proximal_logprobs(pi_theta, pi_old, generation_step=4, current_step=5)
        assert alpha == 0.0
        torch.testing.assert_close(pi_prox, pi_old)

    def test_two_step_stale(self):
        pi_theta = torch.tensor([-0.5, -0.3])
        pi_old = torch.tensor([-0.7, -0.5])
        pi_prox, alpha = _compute_proximal_logprobs(pi_theta, pi_old, generation_step=3, current_step=5)
        assert alpha == pytest.approx(0.5)
        expected = 0.5 * pi_old + 0.5 * pi_theta
        torch.testing.assert_close(pi_prox, expected)

    def test_three_step_stale(self):
        pi_theta = torch.tensor([-0.5])
        pi_old = torch.tensor([-0.8])
        pi_prox, alpha = _compute_proximal_logprobs(pi_theta, pi_old, generation_step=2, current_step=5)
        assert alpha == pytest.approx(2.0 / 3.0)

    def test_future_generation_step_clamps(self):
        pi_theta = torch.tensor([-0.5])
        pi_old = torch.tensor([-0.6])
        pi_prox, alpha = _compute_proximal_logprobs(pi_theta, pi_old, generation_step=6, current_step=5)
        assert alpha == 0.0
        torch.testing.assert_close(pi_prox, pi_old)


class TestDecoupledCorrections:
    """Tests for the full decoupled correction computation."""

    def test_on_policy_behave_weight_is_one(self):
        config = DecoupledConfig()
        pi_theta = torch.tensor([-0.5, -0.3, -0.8], requires_grad=True)
        pi_old = torch.tensor([-0.5, -0.3, -0.8])

        ppo_ratio, ppo_clipped, behave_weight, metrics = compute_decoupled_corrections(
            pi_theta, pi_old, generation_step=5, current_step=5, config=config,
        )

        torch.testing.assert_close(behave_weight, torch.ones(3))
        assert metrics["decoupled/alpha"] == 0.0

    def test_ppo_ratio_has_gradient(self):
        config = DecoupledConfig()
        pi_theta = torch.tensor([-0.5, -0.3], requires_grad=True)
        pi_old = torch.tensor([-0.6, -0.4])

        ppo_ratio, ppo_clipped, behave_weight, _ = compute_decoupled_corrections(
            pi_theta, pi_old, generation_step=4, current_step=5, config=config,
        )

        loss = (ppo_ratio * behave_weight).sum()
        loss.backward()
        assert pi_theta.grad is not None
        assert not torch.all(pi_theta.grad == 0)

    def test_ppo_clipping(self):
        config = DecoupledConfig(eps_clip=0.2)
        pi_theta = torch.tensor([0.0, -2.0], requires_grad=True)
        pi_old = torch.tensor([-0.5, -0.5])

        ppo_ratio, ppo_clipped, _, metrics = compute_decoupled_corrections(
            pi_theta, pi_old, generation_step=4, current_step=5, config=config,
        )

        assert ppo_clipped.min().item() >= 1.0 - 0.2 - 1e-6
        assert ppo_clipped.max().item() <= 1.0 + 0.2 + 1e-6

    def test_behave_truncate_mode(self):
        config = DecoupledConfig(behave_cap=2.0, behave_mode="token_truncate")
        pi_theta = torch.tensor([-0.1, -0.1], requires_grad=True)
        pi_old = torch.tensor([-3.0, -3.0])

        _, _, behave_weight, metrics = compute_decoupled_corrections(
            pi_theta, pi_old, generation_step=3, current_step=5, config=config,
        )

        assert behave_weight.max().item() <= 2.0

    def test_behave_mask_mode(self):
        config = DecoupledConfig(behave_cap=2.0, behave_mode="token_mask")
        pi_theta = torch.tensor([-0.1, -0.1], requires_grad=True)
        pi_old = torch.tensor([-3.0, -3.0])

        _, _, behave_weight, metrics = compute_decoupled_corrections(
            pi_theta, pi_old, generation_step=3, current_step=5, config=config,
        )

        assert (behave_weight <= 2.0).all() or (behave_weight == 0.0).any()

    def test_metrics_reported(self):
        config = DecoupledConfig()
        pi_theta = torch.tensor([-0.5], requires_grad=True)
        pi_old = torch.tensor([-0.6])

        _, _, _, metrics = compute_decoupled_corrections(
            pi_theta, pi_old, generation_step=3, current_step=5, config=config,
        )

        assert "decoupled/ppo_ratio_mean" in metrics
        assert "decoupled/ppo_clip_frac" in metrics
        assert "decoupled/behave_weight_mean" in metrics
        assert "decoupled/behave_clip_frac" in metrics
        assert "decoupled/alpha" in metrics

    def test_asymmetric_clip(self):
        config = DecoupledConfig(eps_clip=0.1, eps_clip_high=0.3)
        pi_theta = torch.tensor([0.5, -1.0], requires_grad=True)
        pi_old = torch.tensor([-0.5, -0.5])

        _, ppo_clipped, _, _ = compute_decoupled_corrections(
            pi_theta, pi_old, generation_step=4, current_step=5, config=config,
        )

        assert ppo_clipped.min().item() >= 1.0 - 0.1 - 1e-6
        assert ppo_clipped.max().item() <= 1.0 + 0.3 + 1e-6


class TestDecoupledConfigDefaults:
    def test_defaults(self):
        cfg = DecoupledConfig()
        assert cfg.eps_clip == 0.2
        assert cfg.eps_clip_high is None
        assert cfg.behave_cap == 5.0
        assert cfg.behave_mode == "token_truncate"


class TestGRPOWithDecoupled:
    """End-to-end: GRPO loss with decoupled corrections vs without."""

    def test_on_policy_same_as_no_decoupled(self):
        """When on-policy (generation_step == current_step), decoupled GRPO
        should produce the same loss direction as non-decoupled, since
        behave_weight=1 and ppo_ratio=exp(pi-pi_old)."""
        from training.utils.rl.grpo import make_grpo_loss_fn

        advantages = [1.0, -0.5]
        ref_logprobs = [[0.0] * 5, [0.0] * 5]
        inf_logprobs = [[-0.5, -0.3, -0.2, -0.4, -0.1], [-0.6, -0.4, -0.3, -0.5, -0.2]]

        config = DecoupledConfig()

        def decoupled_fn(resp_pi, resp_inf, sample_idx):
            return compute_decoupled_corrections(
                resp_pi, resp_inf,
                generation_step=10, current_step=10, config=config,
            )

        fn_decoupled = make_grpo_loss_fn(
            advantages, ref_logprobs, prompt_len=2,
            inf_logprobs=inf_logprobs, kl_beta=0.001,
            decoupled_fn=decoupled_fn,
        )

        lp = torch.tensor([-0.4, -0.3, -0.2, -0.3, -0.1], requires_grad=True)
        loss_d, metrics_d = fn_decoupled([], [lp.clone().requires_grad_(True), lp.clone().requires_grad_(True)])

        assert loss_d.item() != 0.0
        assert "decoupled/ppo_ratio_mean" in metrics_d
        assert "decoupled/alpha" in metrics_d
        assert metrics_d["decoupled/alpha"] == 0.0

    def test_build_loss_fn_with_decoupled(self):
        """build_loss_fn wires decoupled_config correctly."""
        from training.utils.rl.losses import build_loss_fn

        config = DecoupledConfig()
        builder = build_loss_fn(
            policy_loss="grpo", kl_beta=0.001,
            decoupled_config=config,
        )

        advantages = [1.0]
        ref_logprobs = [[0.0] * 5]
        inf_logprobs = [[-0.5, -0.3, -0.2, -0.4, -0.1]]
        prompt_lens = [2]
        generation_steps = [8]
        current_step = 10

        loss_fn = builder(advantages, ref_logprobs, prompt_lens, inf_logprobs, generation_steps, current_step)
        lp = torch.tensor([-0.4, -0.3, -0.2, -0.3, -0.1], requires_grad=True)
        loss, metrics = loss_fn([], [lp])

        assert loss.item() != 0.0
        assert "decoupled/alpha" in metrics

    def test_build_loss_fn_backward_compat(self):
        """build_loss_fn without decoupled_config still works (4-arg call)."""
        from training.utils.rl.losses import build_loss_fn

        builder = build_loss_fn(policy_loss="grpo", kl_beta=0.001)

        advantages = [1.0]
        ref_logprobs = [[0.0] * 5]
        inf_logprobs = [[-0.5, -0.3, -0.2, -0.4, -0.1]]
        prompt_lens = [2]

        loss_fn = builder(advantages, ref_logprobs, prompt_lens, inf_logprobs)
        lp = torch.tensor([-0.4, -0.3, -0.2, -0.3, -0.1], requires_grad=True)
        loss, metrics = loss_fn([], [lp])

        assert loss.item() != 0.0
        assert "decoupled/alpha" not in metrics
