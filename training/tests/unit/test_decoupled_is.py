"""Tests for TIS (train-inference IS) weight computation."""

from __future__ import annotations

import torch

from training.utils.rl.tis import (
    TISConfig,
    compute_tis_weight,
)


class TestComputeTISWeight:
    def test_same_logprobs_gives_weight_one(self):
        prox = torch.tensor([-0.5, -0.3, -0.8])
        inf = torch.tensor([-0.5, -0.3, -0.8])

        weight, metrics = compute_tis_weight(prox, inf, TISConfig())

        torch.testing.assert_close(weight, torch.ones(3))
        assert metrics["tis/clip_frac"] == 0.0

    def test_clamps_at_cap(self):
        prox = torch.tensor([-0.1, -0.1])
        inf = torch.tensor([-3.0, -3.0])

        weight, _ = compute_tis_weight(prox, inf, TISConfig(cap=2.0))

        assert weight.max().item() <= 2.0 + 1e-6


class TestSequenceLevelTIS:
    def test_all_tokens_get_same_weight(self):
        prox = torch.tensor([-0.5, -0.3, -0.8])
        inf = torch.tensor([-0.6, -0.4, -0.9])

        weight, metrics = compute_tis_weight(prox, inf, TISConfig(level="sequence"))

        assert weight[0] == weight[1] == weight[2]
        assert "tis/seq_ratio" in metrics

    def test_geometric_mean_formula(self):
        prox = torch.tensor([-0.2, -0.4])
        inf = torch.tensor([-0.5, -0.3])

        weight, _ = compute_tis_weight(prox, inf, TISConfig(level="sequence", cap=100.0))

        expected = torch.exp((prox - inf).mean())
        torch.testing.assert_close(weight[0], expected)

    def test_token_level_gives_different_weights(self):
        prox = torch.tensor([-0.2, -0.8])
        inf = torch.tensor([-0.5, -0.3])

        weight, _ = compute_tis_weight(prox, inf, TISConfig(cap=100.0))

        assert weight[0] != weight[1]


class TestActiveFilterRegression:
    """Regression: TIS/drift averaged over the full response slice
    (including masked bridge / user-feedback positions in multi-turn
    rollouts) silently contaminated the per-sample weight at active
    positions when ``level="sequence"``.  Both ``utils/rl/common.py`` and
    ``utils/rl/losses.py`` now pre-filter to ``loss_mask > 0`` positions
    before calling :func:`compute_tis_weight` and expand the result back
    to the full response shape with 1.0 at masked positions.
    """

    def test_sequence_weight_ignores_masked_positions(self):
        # 4 response tokens: indices 0 and 3 are active assistant tokens;
        # indices 1 and 2 are a masked bridge / user-feedback span with
        # an extreme log-ratio that, if averaged in, would dominate.
        resp_prox = torch.tensor([-0.5, 5.0, 5.0, -0.5])
        resp_inf = torch.tensor([-0.5, 0.0, 0.0, -0.5])
        resp_mask = torch.tensor([1.0, 0.0, 0.0, 1.0])
        active = resp_mask > 0.5

        tis_weight_active, _ = compute_tis_weight(
            resp_prox[active], resp_inf[active], TISConfig(level="sequence", cap=100.0),
        )
        tis_weight = torch.ones(resp_prox.shape[0])
        tis_weight[active] = tis_weight_active.to(tis_weight.dtype)

        # Active-only mean of (prox - inf) is 0 -> weight 1.0 at active
        # positions.  Contaminated full-slice mean would be (5+5)/4 = 2.5
        # -> exp(2.5) ~= 12.18; that's the wrong answer the fix prevents.
        torch.testing.assert_close(tis_weight[0], torch.tensor(1.0))
        torch.testing.assert_close(tis_weight[3], torch.tensor(1.0))
        assert tis_weight[1].item() == 1.0  # masked: identity
        assert tis_weight[2].item() == 1.0

    def test_token_weight_isolates_active_positions(self):
        # With per-token IS, the active-only filter shouldn't change
        # active values (a no-op there), but it lets us assign 1.0 to
        # masked positions instead of whatever the per-token formula
        # would have produced from masked log-ratios.
        resp_prox = torch.tensor([-0.5, 5.0, -0.5])
        resp_inf = torch.tensor([-0.5, 0.0, -0.5])
        resp_mask = torch.tensor([1.0, 0.0, 1.0])
        active = resp_mask > 0.5

        tis_weight_active, _ = compute_tis_weight(
            resp_prox[active], resp_inf[active], TISConfig(level="token", cap=100.0),
        )
        tis_weight = torch.ones(resp_prox.shape[0])
        tis_weight[active] = tis_weight_active.to(tis_weight.dtype)

        torch.testing.assert_close(tis_weight[0], torch.tensor(1.0))
        torch.testing.assert_close(tis_weight[2], torch.tensor(1.0))
        # Masked position is the identity weight, not exp(5.0) ≈ 148.4.
        assert tis_weight[1].item() == 1.0
