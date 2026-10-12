"""Unit tests for GRPO loss metrics."""

from __future__ import annotations

import pytest
import torch

from training.utils.rl.algorithm.grpo import GRPOConfig, make_loss_fn
from training.utils.rl.losses import PolicyBatch


def make_grpo_loss_fn(
    *, advantages, ref_logprobs, prompt_len, inf_logprobs, kl_beta, raw_inf_logprobs=()
):
    """Client GRPO loss over a bare :class:`PolicyBatch` (no datum masks)."""
    n = len(advantages)
    batch = PolicyBatch(
        data=[],
        advantages=advantages,
        prompt_lens=prompt_len if isinstance(prompt_len, list) else [prompt_len] * n,
        rollout_logprobs=inf_logprobs,
        ref_logprobs=ref_logprobs,
        raw_inference_logprobs=list(raw_inf_logprobs),
    )
    return make_loss_fn(batch, GRPOConfig(kl_beta=kl_beta))


class TestGRPOMetrics:
    def test_reports_policy_gradient_estimator_variance_proxy(self):
        pi_rows = [
            torch.tensor([-1.0, -1.0], requires_grad=True),
            torch.tensor([-1.0, -1.0], requires_grad=True),
        ]
        fn = make_grpo_loss_fn(
            advantages=[1.0, -1.0],
            ref_logprobs=[],
            prompt_len=[1, 1],
            inf_logprobs=[[-1.0, -1.0], [-1.0, -1.0]],
            kl_beta=0.0,
        )

        _, metrics = fn([], pi_rows)

        # At ratio=1, the two datum coefficients are -1 and +1.
        assert metrics["policy_gradient/coefficient_variance_proxy"] == pytest.approx(
            2.0
        )
        assert metrics["policy_gradient/estimator_variance_proxy"] == pytest.approx(1.0)
        assert metrics["policy_gradient/estimator_std_error_proxy"] == pytest.approx(
            1.0
        )
        assert metrics["policy_gradient/sample_count"] == 2

    def test_reports_reference_kl_separately_from_ppo_kl(self):
        pi_vals = [-2.0, -1.0, -0.5]
        ref_vals = [-2.0, -1.3, -0.1]
        pi = torch.tensor(pi_vals, requires_grad=True)

        fn = make_grpo_loss_fn(
            advantages=[1.0],
            ref_logprobs=[ref_vals],
            prompt_len=1,
            inf_logprobs=[pi_vals],
            kl_beta=0.0,
        )

        _, metrics = fn([], [pi])

        ref_log_diff = torch.tensor(ref_vals) - torch.tensor(pi_vals)
        expected_ref_kl = (torch.exp(ref_log_diff) - ref_log_diff - 1.0).mean().item()
        assert metrics["mean_kl"] == pytest.approx(
            (torch.tensor(pi_vals) - torch.tensor(ref_vals)).mean().item()
        )
        assert metrics["ref_kl"] == pytest.approx(expected_ref_kl)
        assert metrics["ppo_kl"] == pytest.approx(0.0)

    def test_omits_reference_kl_metrics_without_reference(self):
        pi_vals = [-1.0, -1.2]
        pi = torch.tensor(pi_vals, requires_grad=True)
        fn = make_grpo_loss_fn(
            advantages=[1.0],
            ref_logprobs=[],
            prompt_len=1,
            inf_logprobs=[pi_vals],
            kl_beta=0.0,
        )

        _, metrics = fn([], [pi])

        assert "mean_kl" not in metrics
        assert "ref_kl" not in metrics
        assert "mean_kl_penalty" not in metrics

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

    def test_raw_inference_logprobs_do_not_change_loss_or_gradient(self):
        pi_vals = [-1.0, -1.2]

        def run(raw_inf_vals):
            pi = torch.tensor(pi_vals, requires_grad=True)
            fn = make_grpo_loss_fn(
                advantages=[1.0],
                ref_logprobs=[],
                prompt_len=1,
                inf_logprobs=[pi_vals],
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
        # Client losses no longer compute inference drift; see inference_drift_metrics.
        assert not {"inference_k1", "inference_k3", "raw_inference_logprob_coverage"} & metrics_a.keys()

    def test_token_advantages_weight_each_response_token(self):
        pi_vals = [-1.0, -1.0, -1.0]
        pi = torch.tensor(pi_vals, requires_grad=True)
        batch = PolicyBatch(
            data=[],
            advantages=[100.0],
            prompt_lens=[1],
            rollout_logprobs=[pi_vals],
            token_advantages=[[2.0, -1.0, 0.5]],
        )

        loss, _ = make_loss_fn(batch, GRPOConfig())([], [pi])
        loss.backward()

        # At ratio 1 the per-token gradient is -A_t; the scalar advantage is unused.
        assert pi.grad.tolist() == pytest.approx([-2.0, 1.0, -0.5])
        assert loss.item() == pytest.approx(-1.5)
