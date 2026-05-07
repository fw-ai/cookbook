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
            prox_logprobs=[pi_vals],
            kl_beta=0.0,
        )

        _, metrics = fn([], [pi])

        ref_log_diff = torch.tensor(ref_vals) - torch.tensor(pi_vals)
        expected_ref_kl = (torch.exp(ref_log_diff) - ref_log_diff - 1.0).mean().item()
        assert metrics["ref_kl"] == pytest.approx(expected_ref_kl)
        assert metrics["ppo_kl"] == pytest.approx(0.0)
