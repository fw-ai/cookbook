"""Unit tests for GRPO loss metrics."""

from __future__ import annotations

import pytest
import torch

from training.utils.rl.grpo import make_grpo_loss_fn


class TestGRPOMetrics:
    def test_reports_reference_kl_separately_from_ppo_kl(self):
        pi_vals = [-2.0, -1.0, -0.5]
        ref_vals = [-2.0, -1.3, -0.1]
        pi = torch.tensor(pi_vals, requires_grad=True)

        fn = make_grpo_loss_fn(
            advantages=[1.0],
            ref_logprobs=[ref_vals],
            prompt_len=1,
            inf_logprobs=[pi_vals],
            old_policy_logprobs=[pi_vals],
            kl_beta=0.0,
        )

        _, metrics = fn([], [pi])

        ref_log_diff = torch.tensor(ref_vals) - torch.tensor(pi_vals)
        expected_ref_kl = (torch.exp(ref_log_diff) - ref_log_diff - 1.0).mean().item()
        assert metrics["ref_kl"] == pytest.approx(expected_ref_kl)
        assert metrics["ppo_kl"] == pytest.approx(0.0)

    def test_reference_kl_contributes_k3_gradient(self):
        pi_vals = [-1.0, -1.0]
        ref_vals = [-2.0, -0.5]
        beta = 0.2
        pi = torch.tensor(pi_vals, requires_grad=True)

        fn = make_grpo_loss_fn(
            advantages=[0.0],
            ref_logprobs=[ref_vals],
            prompt_len=1,
            inf_logprobs=[pi_vals],
            old_policy_logprobs=[pi_vals],
            kl_beta=beta,
        )

        loss, metrics = fn([], [pi])
        loss.backward()

        ref_log_ratio = torch.tensor(ref_vals) - torch.tensor(pi_vals)
        expected_per_token_kl = torch.exp(ref_log_ratio) - ref_log_ratio - 1.0
        expected_grad = beta * (1.0 - torch.exp(ref_log_ratio))
        assert pi.grad is not None
        assert pi.grad.tolist() == pytest.approx(expected_grad.tolist())
        assert metrics["mean_kl_penalty"] == pytest.approx(
            beta * expected_per_token_kl.mean().item()
        )

    def test_reports_raw_inference_observability_metrics(self):
        pi_vals = [-1.0, -1.2]
        raw_inf_vals = [-1.1, -1.0]
        pi = torch.tensor(pi_vals, requires_grad=True)

        fn = make_grpo_loss_fn(
            advantages=[1.0],
            ref_logprobs=[],
            prompt_len=1,
            inf_logprobs=[pi_vals],
            old_policy_logprobs=[pi_vals],
            kl_beta=0.0,
            raw_inf_logprobs=[raw_inf_vals],
        )

        _, metrics = fn([], [pi])

        diff = torch.tensor(pi_vals) - torch.tensor(raw_inf_vals)
        expected_kld = (torch.exp(diff) - diff - 1.0).mean().item()
        assert metrics["raw_inference_logprob_coverage"] == 1.0
        assert metrics["inference_diff"] == pytest.approx(diff.abs().mean().item())
        assert metrics["inference_kld"] == pytest.approx(expected_kld)

    def test_raw_inference_logprobs_do_not_change_loss_or_gradient(self):
        pi_vals = [-1.0, -1.2]

        def run(raw_inf_vals):
            pi = torch.tensor(pi_vals, requires_grad=True)
            fn = make_grpo_loss_fn(
                advantages=[1.0],
                ref_logprobs=[],
                prompt_len=1,
                inf_logprobs=[pi_vals],
                old_policy_logprobs=[pi_vals],
                kl_beta=0.0,
                raw_inf_logprobs=[raw_inf_vals],
            )
            loss, metrics = fn([], [pi])
            loss.backward()
            return loss.detach(), pi.grad, metrics

        loss_a, grad_a, metrics_a = run([-1.1, -1.0])
        loss_b, grad_b, metrics_b = run([-4.0, -4.0])

        assert torch.equal(loss_a, loss_b)
        assert torch.equal(grad_a, grad_b)
        assert metrics_a["inference_diff"] != metrics_b["inference_diff"]
