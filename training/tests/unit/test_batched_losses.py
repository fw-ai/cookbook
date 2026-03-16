"""Tests for loss function utilities and DPO batching."""

from __future__ import annotations

import pytest
import torch

from training.utils.losses import (
    make_batch_dpo_loss_fn,
    make_batch_orpo_loss_fn,
    make_batch_weighted_sft_loss_fn,
    make_orpo_loss_fn,
)
from training.utils.rl.common import _normalize_prompt_lens
from training.utils.rl.losses import build_loss_fn
from training.utils.supervised import build_datum_from_token_mask


def _make_dummy_logprobs(seq_len: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(seq_len, requires_grad=True)


def _zeros(n: int) -> list[float]:
    return [0.0] * n


class TestNormalizePromptLens:
    def test_int_broadcasts(self):
        assert _normalize_prompt_lens(5, 3) == [5, 5, 5]

    def test_list_passthrough(self):
        assert _normalize_prompt_lens([1, 2, 3], 3) == [1, 2, 3]

    def test_list_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Expected 3 prompt lengths, got 2"):
            _normalize_prompt_lens([1, 2], 3)


class TestLossBuilder:
    def test_rejects_unknown_policy_loss(self):
        builder = build_loss_fn(policy_loss="unknown", kl_beta=0.01)
        with pytest.raises(ValueError, match="Unsupported policy_loss"):
            builder([1.0], [_zeros(4)], [2], [_zeros(4)], [_zeros(4)])


class TestBatchDPOLoss:

    def _make_ref_logprobs(self, seq_len: int, seed: int = 0) -> list[float]:
        torch.manual_seed(seed)
        return torch.randn(seq_len).tolist()

    def test_multi_pair_averages_correctly(self):
        """Batched loss with 2 pairs == average of two single-pair calls."""
        ref_c0 = self._make_ref_logprobs(8, seed=0)
        ref_r0 = self._make_ref_logprobs(8, seed=1)
        ref_c1 = self._make_ref_logprobs(12, seed=2)
        ref_r1 = self._make_ref_logprobs(12, seed=3)
        rs0, rs1 = 2, 5
        beta = 0.1

        lp_c0 = _make_dummy_logprobs(8, seed=10)
        lp_r0 = _make_dummy_logprobs(8, seed=11)
        lp_c1 = _make_dummy_logprobs(12, seed=12)
        lp_r1 = _make_dummy_logprobs(12, seed=13)

        fn0 = make_batch_dpo_loss_fn([ref_c0], [ref_r0], [rs0], beta)
        fn1 = make_batch_dpo_loss_fn([ref_c1], [ref_r1], [rs1], beta)
        loss0, _ = fn0([], [lp_c0.clone(), lp_r0.clone()])
        loss1, _ = fn1([], [lp_c1.clone(), lp_r1.clone()])
        expected_avg = (loss0 + loss1) / 2

        fn_batch = make_batch_dpo_loss_fn(
            [ref_c0, ref_c1], [ref_r0, ref_r1], [rs0, rs1], beta,
        )
        loss_b, met_b = fn_batch(
            [], [lp_c0.clone(), lp_r0.clone(), lp_c1.clone(), lp_r1.clone()],
        )

        assert torch.allclose(expected_avg, loss_b, atol=1e-5)
        assert met_b["batch_pairs"] == 2

    def test_wrong_logprobs_count_raises(self):
        fn = make_batch_dpo_loss_fn([[0.0]], [[0.0]], [0], 0.1)
        lp = _make_dummy_logprobs(1, seed=0)
        with pytest.raises(AssertionError, match="Expected 2 logprobs"):
            fn([], [lp])

    def test_microbatch_sizes_preserve_accumulated_loss(self):
        ref_c0 = self._make_ref_logprobs(8, seed=0)
        ref_r0 = self._make_ref_logprobs(8, seed=1)
        ref_c1 = self._make_ref_logprobs(12, seed=2)
        ref_r1 = self._make_ref_logprobs(12, seed=3)
        rs0, rs1 = 2, 5
        beta = 0.1

        lp_c0 = _make_dummy_logprobs(8, seed=10)
        lp_r0 = _make_dummy_logprobs(8, seed=11)
        lp_c1 = _make_dummy_logprobs(12, seed=12)
        lp_r1 = _make_dummy_logprobs(12, seed=13)

        fn0 = make_batch_dpo_loss_fn([ref_c0], [ref_r0], [rs0], beta)
        fn1 = make_batch_dpo_loss_fn([ref_c1], [ref_r1], [rs1], beta)
        loss0, _ = fn0([], [lp_c0.clone(), lp_r0.clone()])
        loss1, _ = fn1([], [lp_c1.clone(), lp_r1.clone()])

        fn_accum = make_batch_dpo_loss_fn(
            [ref_c0, ref_c1],
            [ref_r0, ref_r1],
            [rs0, rs1],
            beta,
            microbatch_sizes=[1, 1],
        )
        loss_accum, metrics = fn_accum(
            [],
            [lp_c0.clone(), lp_r0.clone(), lp_c1.clone(), lp_r1.clone()],
        )

        assert torch.allclose(loss_accum, loss0 + loss1, atol=1e-5)
        assert metrics["microbatch_count"] == 2


class TestBatchORPOLoss:

    def test_multi_pair_preserves_pairwise_accumulation(self):
        rs0, rs1 = 2, 4
        orpo_lambda = 0.5

        lp_c0 = _make_dummy_logprobs(8, seed=20)
        lp_r0 = _make_dummy_logprobs(8, seed=21)
        lp_c1 = _make_dummy_logprobs(10, seed=22)
        lp_r1 = _make_dummy_logprobs(10, seed=23)

        fn0 = make_orpo_loss_fn(rs0, orpo_lambda)
        fn1 = make_orpo_loss_fn(rs1, orpo_lambda)
        loss0, metrics0 = fn0([], [lp_c0.clone(), lp_r0.clone()])
        loss1, metrics1 = fn1([], [lp_c1.clone(), lp_r1.clone()])

        fn_batch = make_batch_orpo_loss_fn([rs0, rs1], orpo_lambda)
        loss_batch, metrics_batch = fn_batch(
            [],
            [lp_c0.clone(), lp_r0.clone(), lp_c1.clone(), lp_r1.clone()],
        )

        assert torch.allclose(loss_batch, loss0 + loss1, atol=1e-5)
        assert metrics_batch["batch_pairs"] == 2
        assert metrics_batch["orpo_loss"] == pytest.approx(
            (metrics0["orpo_loss"] + metrics1["orpo_loss"]) / 2
        )


class TestWeightedSFTLoss:

    def test_microbatch_sizes_preserve_accumulated_loss(self):
        datum_a = build_datum_from_token_mask(
            token_ids=[10, 11, 12, 13],
            token_mask=[0, 0, 1, 1],
        ).datum
        datum_b = build_datum_from_token_mask(
            token_ids=[20, 21, 22],
            token_mask=[0, 1, 1],
        ).datum

        lp_a = torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float32, requires_grad=True)
        lp_b = torch.tensor([-0.4, -0.5], dtype=torch.float32, requires_grad=True)

        fn_single = make_batch_weighted_sft_loss_fn()
        loss_a, _ = fn_single([datum_a], [lp_a.clone()])
        loss_b, _ = fn_single([datum_b], [lp_b.clone()])

        fn_batch = make_batch_weighted_sft_loss_fn(microbatch_sizes=[1, 1])
        loss_batch, metrics = fn_batch([datum_a, datum_b], [lp_a.clone(), lp_b.clone()])

        assert torch.allclose(loss_batch, loss_a + loss_b, atol=1e-5)
        assert metrics["microbatch_count"] == 2
