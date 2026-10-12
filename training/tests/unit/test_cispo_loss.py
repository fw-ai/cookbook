"""Unit tests for the CISPO loss function.

Validates:
- Clipping behavior (positive/negative/zero advantage)
- Per-sample prompt boundaries
- Gradient flow through logprobs
- Metric reporting (cispo_clip_frac, mean_kl)
- Config defaults
"""

from __future__ import annotations

import torch
import pytest

from training.utils.rl.algorithm.cispo import CISPOConfig, make_loss_fn
from training.utils.rl.losses import PolicyBatch


def make_cispo_loss_fn(advantages, ref_logprobs, inf_logprobs, prompt_len, cispo_config=None):
    """Client CISPO loss over a :class:`PolicyBatch` anchored on ``inf_logprobs``."""
    batch = PolicyBatch(
        data=[],
        advantages=advantages,
        prompt_lens=[prompt_len] * len(advantages),
        rollout_logprobs=inf_logprobs,
        ref_logprobs=ref_logprobs,
    )
    return make_loss_fn(batch, cispo_config or CISPOConfig())


def _make_logprobs(seq_len: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(seq_len, requires_grad=True)


class TestCISPOClipping:
    """Test that CISPO clips the ratio to [1-eps_low, 1+eps_high]."""

    def _build_and_call(
        self,
        adv: float,
        pi_offset: float,
        eps_low: float = 0.2,
        eps_high: float = 0.28,
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


    def test_gradient_uses_detached_clipped_weight(self):
        """d(loss)/d(logprob) is -clip(ratio) * A: the ratio itself is not differentiated."""
        pi = torch.tensor([0.0, 1.0], requires_grad=True)
        fn = make_cispo_loss_fn([2.0], [], [[0.0, 0.0]], prompt_len=2)
        loss, _ = fn([], [pi])
        loss.backward()
        assert pi.grad.tolist() == pytest.approx([0.0, -1.28 * 2.0])


class TestCISPOBatched:
    def test_mixed_prompt_lens(self):
        """Different prompt_lens should produce different losses."""
        adv = [1.0]
        ref = [[0.0] * 10]
        inf = [[0.0] * 10]
        lp = _make_logprobs(10, seed=0)

        fn_short = make_cispo_loss_fn(adv, ref, inf, prompt_len=2)
        fn_long = make_cispo_loss_fn(adv, ref, inf, prompt_len=8)

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

        fn = make_cispo_loss_fn(
            adv,
            ref,
            inf,
            prompt_len=1,
        )
        loss, _ = fn([], [lp_grad])
        loss.backward()

        assert lp_grad.grad is not None
        assert lp_grad.grad[1:].abs().sum() > 0, "Should have non-zero gradients for response tokens"


class TestCISPOMetrics:
    def test_reports_mean_kl(self):
        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[0.0] * 5]
        lp = _make_logprobs(5, seed=0).detach().requires_grad_(True)

        fn = make_cispo_loss_fn(
            adv,
            ref,
            inf,
            prompt_len=1,
        )
        _, metrics = fn([], [lp])

        assert "mean_kl" in metrics
        assert "cispo_clip_frac" in metrics

    def test_requires_inf_logprobs(self):
        """CISPO should raise if inf_logprobs is empty."""
        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[]]
        lp = _make_logprobs(5, seed=0).detach().requires_grad_(True)

        fn = make_cispo_loss_fn(adv, ref, inf, prompt_len=1)
        with pytest.raises(ValueError, match="CISPO requires rollout_logprobs"):
            fn([], [lp])

    def test_requires_sufficient_inf_logprobs(self):
        """CISPO should raise if inf_logprobs is too short."""
        adv = [1.0]
        ref = [[0.0] * 5]
        inf = [[0.0] * 2]
        lp = _make_logprobs(5, seed=0).detach().requires_grad_(True)

        fn = make_cispo_loss_fn(adv, ref, inf, prompt_len=1)
        with pytest.raises(ValueError, match="CISPO requires at least"):
            fn([], [lp])
