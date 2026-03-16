"""Unit tests for the CISPO loss function.

Validates:
- Clipping behavior (positive/negative/zero advantage)
- Equivalence between int and list prompt_len
- Gradient flow through logprobs
- Metric reporting (cispo_clip_frac, mean_kl)
- Config defaults
"""

from __future__ import annotations

import torch
import pytest

from training.utils.rl.cispo import CISPOConfig, make_cispo_loss_fn


def _make_logprobs(seq_len: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(seq_len, requires_grad=True)


class TestCISPOConfigDefaults:
    def test_defaults(self):
        cfg = CISPOConfig()
        assert cfg.eps_low == 0.2
        assert cfg.eps_high == 0.28
        assert cfg.ratio_log_cap == 20.0

    def test_custom(self):
        cfg = CISPOConfig(eps_low=0.1, eps_high=0.5)
        assert cfg.eps_low == 0.1
        assert cfg.eps_high == 0.5


class TestCISPOClipping:
    """Test that CISPO clips the ratio to [1-eps_low, 1+eps_high]."""

    def _build_and_call(
        self, adv: float, pi_offset: float, eps_low: float = 0.2, eps_high: float = 0.28,
    ):
        """Build a single-response-token CISPO loss where pi - pi_old = pi_offset.

        Uses prompt_len=2, seq_len=2 so response_start = max(0, 2-1) = 1,
        meaning only token at index 1 is the response token.
        """
        prompt_len = 2
        seq_len = 2
        cfg = CISPOConfig(eps_low=eps_low, eps_high=eps_high)

        inf_lp = [0.0] * seq_len
        ref_lp = [0.0] * seq_len
        pi_vals = [0.0, inf_lp[1] + pi_offset]
        pi_logprobs = torch.tensor(pi_vals, requires_grad=True)

        fn = make_cispo_loss_fn(
            advantages=[adv],
            ref_logprobs=[ref_lp],
            inf_logprobs=[inf_lp],
            prompt_len=prompt_len,
            prox_logprobs=[inf_lp],
            cispo_config=cfg,
        )
        loss, metrics = fn([], [pi_logprobs])
        return loss, metrics["cispo_clip_frac"]

    def test_positive_adv_ratio_within_bounds(self):
        """A > 0, ratio < 1 + eps_high -> not clipped, clip_frac=0."""
        loss, clip_frac = self._build_and_call(adv=1.0, pi_offset=0.1)
        assert clip_frac == pytest.approx(0.0)
        assert loss.item() != 0.0

    def test_positive_adv_ratio_exceeds_eps_high(self):
        """A > 0, ratio > 1 + eps_high -> clipped, loss still non-zero."""
        loss, clip_frac = self._build_and_call(adv=1.0, pi_offset=1.0)
        assert clip_frac == pytest.approx(1.0)
        assert loss.item() != 0.0

    def test_negative_adv_ratio_within_bounds(self):
        """A < 0, ratio > 1 - eps_low -> not clipped, clip_frac=0."""
        loss, clip_frac = self._build_and_call(adv=-1.0, pi_offset=-0.05)
        assert clip_frac == pytest.approx(0.0)
        assert loss.item() != 0.0

    def test_negative_adv_ratio_below_eps_low(self):
        """A < 0, ratio < 1 - eps_low -> clipped, loss still non-zero."""
        loss, clip_frac = self._build_and_call(adv=-1.0, pi_offset=-1.0)
        assert clip_frac == pytest.approx(1.0)
        assert loss.item() != 0.0

    def test_zero_advantage_loss_is_zero(self):
        """A == 0 -> loss is zero regardless of clipping (adv factor is 0)."""
        loss, clip_frac = self._build_and_call(adv=0.0, pi_offset=5.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestCISPOBatched:
    def test_int_vs_list_prompt_len(self):
        """Loss with prompt_len=10 should equal prompt_lens=[10, 10]."""
        adv = [1.0, -0.5]
        ref = [[0.1] * 20, [0.2] * 20]
        inf = [[0.05] * 20, [0.15] * 20]
        lp0 = _make_logprobs(20, seed=42)
        lp1 = _make_logprobs(20, seed=43)

        prox = [[0.05] * 20, [0.15] * 20]
        fn_int = make_cispo_loss_fn(adv, ref, inf, prompt_len=10, prox_logprobs=prox)
        fn_list = make_cispo_loss_fn(adv, ref, inf, prompt_len=[10, 10], prox_logprobs=prox)

        loss_int, met_int = fn_int([], [lp0.detach().requires_grad_(True), lp1.detach().requires_grad_(True)])
        loss_list, met_list = fn_list([], [lp0.detach().requires_grad_(True), lp1.detach().requires_grad_(True)])

        assert torch.allclose(loss_int, loss_list, atol=1e-6)
        assert abs(met_int["mean_kl"] - met_list["mean_kl"]) < 1e-6

    def test_mixed_prompt_lens(self):
        """Different prompt_lens should produce different losses."""
        adv = [1.0]
        ref = [[0.0] * 10]
        inf = [[0.0] * 10]
        lp = _make_logprobs(10, seed=0)

        prox = [[0.0] * 10]
        fn_short = make_cispo_loss_fn(adv, ref, inf, prompt_len=2, prox_logprobs=prox)
        fn_long = make_cispo_loss_fn(adv, ref, inf, prompt_len=8, prox_logprobs=prox)

        loss_short, _ = fn_short([], [lp.detach().requires_grad_(True)])
        loss_long, _ = fn_long([], [lp.detach().requires_grad_(True)])

        assert loss_short.item() != pytest.approx(loss_long.item(), abs=1e-6)

    def test_gradient_flows(self):
        """Gradients should flow through logprobs for response tokens."""
        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[0.0] * 5]
        lp = _make_logprobs(5, seed=0)
        lp_grad = lp.detach().requires_grad_(True)

        prox = [[0.0] * 5]
        fn = make_cispo_loss_fn(adv, ref, inf, prompt_len=1, prox_logprobs=prox)
        loss, _ = fn([], [lp_grad])
        loss.backward()

        assert lp_grad.grad is not None
        assert lp_grad.grad[1:].abs().sum() > 0, "Should have non-zero gradients for response tokens"


class TestCISPOMetrics:
    def test_reports_mean_kl(self):
        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[0.0] * 5]
        prox = [[0.0] * 5]
        lp = _make_logprobs(5, seed=0).detach().requires_grad_(True)

        fn = make_cispo_loss_fn(adv, ref, inf, prompt_len=1, prox_logprobs=prox)
        _, metrics = fn([], [lp])

        assert "mean_kl" in metrics
        assert "cispo_clip_frac" in metrics

    def test_reports_inference_metrics(self):
        """CISPO should report inference_diff and inference_kld."""
        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[0.0] * 5]
        prox = [[0.0] * 5]
        lp = _make_logprobs(5, seed=0).detach().requires_grad_(True)

        fn = make_cispo_loss_fn(adv, ref, inf, prompt_len=1, prox_logprobs=prox)
        _, metrics = fn([], [lp])

        assert "inference_diff" in metrics
        assert "inference_kld" in metrics

    def test_requires_inf_logprobs(self):
        """CISPO should raise if inf_logprobs is empty."""
        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[]]
        prox = [[0.0] * 5]
        lp = _make_logprobs(5, seed=0).detach().requires_grad_(True)

        fn = make_cispo_loss_fn(adv, ref, inf, prompt_len=1, prox_logprobs=prox)
        with pytest.raises(ValueError, match="CISPO requires inference logprobs"):
            fn([], [lp])

    def test_requires_sufficient_inf_logprobs(self):
        """CISPO should raise if inf_logprobs is too short."""
        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[0.0] * 2]
        prox = [[0.0] * 5]
        lp = _make_logprobs(5, seed=0).detach().requires_grad_(True)

        fn = make_cispo_loss_fn(adv, ref, inf, prompt_len=1, prox_logprobs=prox)
        with pytest.raises(ValueError, match="CISPO requires at least"):
            fn([], [lp])
