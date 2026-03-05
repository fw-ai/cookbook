"""Tests for TIS (train-inference IS) weight computation."""

from __future__ import annotations

import torch

from training.utils.rl.importance_sampling import (
    ISConfig,
    compute_tis_weight,
)


class TestComputeTISWeight:
    def test_same_logprobs_gives_weight_one(self):
        prox = torch.tensor([-0.5, -0.3, -0.8])
        inf = torch.tensor([-0.5, -0.3, -0.8])

        weight, metrics = compute_tis_weight(prox, inf, ISConfig())

        torch.testing.assert_close(weight, torch.ones(3))
        assert metrics["tis/clip_frac"] == 0.0

    def test_clamps_at_cap(self):
        prox = torch.tensor([-0.1, -0.1])
        inf = torch.tensor([-3.0, -3.0])

        weight, _ = compute_tis_weight(prox, inf, ISConfig(tis_cap=2.0))

        assert weight.max().item() <= 2.0 + 1e-6


class TestSequenceLevelTIS:
    def test_all_tokens_get_same_weight(self):
        prox = torch.tensor([-0.5, -0.3, -0.8])
        inf = torch.tensor([-0.6, -0.4, -0.9])

        weight, metrics = compute_tis_weight(prox, inf, ISConfig(tis_level="sequence"))

        assert weight[0] == weight[1] == weight[2]
        assert "tis/seq_ratio" in metrics

    def test_geometric_mean_formula(self):
        prox = torch.tensor([-0.2, -0.4])
        inf = torch.tensor([-0.5, -0.3])

        weight, _ = compute_tis_weight(prox, inf, ISConfig(tis_level="sequence", tis_cap=100.0))

        expected = torch.exp((prox - inf).mean())
        torch.testing.assert_close(weight[0], expected)

    def test_token_level_gives_different_weights(self):
        prox = torch.tensor([-0.2, -0.8])
        inf = torch.tensor([-0.5, -0.3])

        weight, _ = compute_tis_weight(prox, inf, ISConfig(tis_cap=100.0))

        assert weight[0] != weight[1]
