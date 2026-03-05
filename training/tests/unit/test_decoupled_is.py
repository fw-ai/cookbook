"""Tests for the clean decoupled IS corrections (PPO ratio + behavioral weight)."""

from __future__ import annotations

import pytest
import torch

from training.utils.rl.importance_sampling import (
    DecoupledConfig,
    compute_behave_weight,
)


class TestDecoupledConfigDefaults:
    def test_defaults(self):
        cfg = DecoupledConfig()
        assert cfg.eps_clip == 0.2
        assert cfg.eps_clip_high is None
        assert cfg.behave_cap == 5.0
        assert cfg.behave_mode == "token_truncate"


class TestComputeBehaveWeight:
    def test_same_logprobs_gives_weight_one(self):
        prox = torch.tensor([-0.5, -0.3, -0.8])
        inf = torch.tensor([-0.5, -0.3, -0.8])
        config = DecoupledConfig()

        weight, metrics = compute_behave_weight(prox, inf, config)

        torch.testing.assert_close(weight, torch.ones(3))
        assert metrics["behave/clip_frac"] == 0.0

    def test_truncate_mode_caps(self):
        prox = torch.tensor([-0.1, -0.1])
        inf = torch.tensor([-3.0, -3.0])
        config = DecoupledConfig(behave_cap=2.0, behave_mode="token_truncate")

        weight, metrics = compute_behave_weight(prox, inf, config)

        assert weight.max().item() <= 2.0 + 1e-6
        assert metrics["behave/clip_frac"] > 0

    def test_mask_mode_zeros(self):
        prox = torch.tensor([-0.1, -0.1])
        inf = torch.tensor([-3.0, -3.0])
        config = DecoupledConfig(behave_cap=2.0, behave_mode="token_mask")

        weight, metrics = compute_behave_weight(prox, inf, config)

        assert (weight == 0.0).any()

    def test_metrics_reported(self):
        prox = torch.tensor([-0.5])
        inf = torch.tensor([-0.6])
        config = DecoupledConfig()

        _, metrics = compute_behave_weight(prox, inf, config)

        assert "behave/weight_mean" in metrics
        assert "behave/clip_frac" in metrics

    def test_detached(self):
        prox = torch.tensor([-0.5], requires_grad=False)
        inf = torch.tensor([-0.6])
        config = DecoupledConfig()

        weight, _ = compute_behave_weight(prox, inf, config)
        assert not weight.requires_grad


class TestGRPOWithProxLogprobs:
    """End-to-end: GRPO loss with pre-computed prox_logprobs."""

    def test_same_prox_and_pi_gives_ratio_one(self):
        from training.utils.rl.grpo import make_grpo_loss_fn

        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[-0.5, -0.3, -0.2, -0.4, -0.1]]
        lp_vals = [-0.4, -0.3, -0.2, -0.3, -0.1]
        prox = [lp_vals[:]]

        fn = make_grpo_loss_fn(adv, ref, prompt_len=2, inf_logprobs=inf, prox_logprobs=prox, kl_beta=0.001)
        lp = torch.tensor(lp_vals, requires_grad=True)
        loss, metrics = fn([], [lp])

        assert loss.item() != 0.0
        assert metrics["ppo_clip_frac"] == 0.0

    def test_gradient_flows(self):
        from training.utils.rl.grpo import make_grpo_loss_fn

        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[-0.5, -0.3, -0.2, -0.4, -0.1]]
        prox = [[-0.5, -0.3, -0.2, -0.4, -0.1]]

        fn = make_grpo_loss_fn(adv, ref, prompt_len=2, inf_logprobs=inf, prox_logprobs=prox, kl_beta=0.001)
        lp = torch.tensor([-0.4, -0.3, -0.2, -0.3, -0.1], requires_grad=True)
        loss, _ = fn([], [lp])
        loss.backward()

        assert lp.grad is not None
        assert not torch.all(lp.grad == 0)

    def test_build_loss_fn_passes_prox(self):
        from training.utils.rl.losses import build_loss_fn

        builder = build_loss_fn(policy_loss="grpo", kl_beta=0.001)
        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[-0.5, -0.3, -0.2, -0.4, -0.1]]
        prox = [[-0.5, -0.3, -0.2, -0.4, -0.1]]

        loss_fn = builder(adv, ref, [2], inf, prox)
        lp = torch.tensor([-0.4, -0.3, -0.2, -0.3, -0.1], requires_grad=True)
        loss, metrics = loss_fn([], [lp])

        assert loss.item() != 0.0
        assert "behave/weight_mean" in metrics

    def test_dapo_with_prox(self):
        from training.utils.rl.dapo import DAPOConfig, make_dapo_loss_fn

        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[-0.5, -0.3, -0.2, -0.4, -0.1]]
        prox = [[-0.5, -0.3, -0.2, -0.4, -0.1]]

        fn = make_dapo_loss_fn(adv, ref, inf, prompt_len=2, prox_logprobs=prox, dapo_config=DAPOConfig())
        lp = torch.tensor([-0.4, -0.3, -0.2, -0.3, -0.1], requires_grad=True)
        loss, metrics = fn([], [lp])

        assert loss.item() != 0.0
        assert "dapo_clip_frac" in metrics
        assert "behave/weight_mean" in metrics

    def test_cispo_with_prox(self):
        from training.utils.rl.cispo import CISPOConfig, make_cispo_loss_fn

        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[-0.5, -0.3, -0.2, -0.4, -0.1]]
        prox = [[-0.5, -0.3, -0.2, -0.4, -0.1]]

        fn = make_cispo_loss_fn(adv, ref, inf, prompt_len=2, prox_logprobs=prox, cispo_config=CISPOConfig())
        lp = torch.tensor([-0.4, -0.3, -0.2, -0.3, -0.1], requires_grad=True)
        loss, metrics = fn([], [lp])

        assert loss.item() != 0.0
        assert "cispo_mask_frac" in metrics
        assert "behave/weight_mean" in metrics

    def test_gspo_with_prox(self):
        from training.utils.rl.gspo import GSPOConfig, make_gspo_loss_fn

        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[-0.5, -0.3, -0.2, -0.4, -0.1]]
        prox = [[-0.5, -0.3, -0.2, -0.4, -0.1]]

        fn = make_gspo_loss_fn(adv, ref, inf, prompt_len=2, prox_logprobs=prox, gspo_config=GSPOConfig())
        lp = torch.tensor([-0.4, -0.3, -0.2, -0.3, -0.1], requires_grad=True)
        loss, metrics = fn([], [lp])

        assert loss.item() != 0.0
        assert "gspo_clip_frac" in metrics
        assert "behave/weight_mean" in metrics
