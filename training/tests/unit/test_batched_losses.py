"""Tests for batched (multi-prompt) loss functions.

Verifies that passing per-datum ``prompt_lens: List[int]`` produces the
same result as the single ``prompt_len: int`` path when all prompt lengths
are identical, and handles mixed prompt lengths correctly.
"""

from __future__ import annotations

import torch
import pytest

from training.utils.losses import make_batch_dpo_loss_fn
from training.utils.rl.dapo import DAPOConfig, make_dapo_loss_fn
from training.utils.rl.grpo import make_grpo_loss_fn
from training.utils.rl.gspo import GSPOConfig, make_gspo_loss_fn
from training.utils.rl.common import _normalize_prompt_lens
from training.utils.rl.losses import build_loss_fn


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


class TestGRPOLossBatched:
    def test_single_int_equals_list(self):
        adv = [1.0, -0.5]
        ref = [[0.1] * 20, [0.2] * 20]
        inf = [_zeros(20), _zeros(20)]
        prox = [_zeros(20), _zeros(20)]
        lp0 = _make_dummy_logprobs(20, seed=42)
        lp1 = _make_dummy_logprobs(20, seed=43)

        fn_int = make_grpo_loss_fn(adv, ref, prompt_len=10, inf_logprobs=inf, prox_logprobs=prox, kl_beta=0.01)
        fn_list = make_grpo_loss_fn(adv, ref, prompt_len=[10, 10], inf_logprobs=inf, prox_logprobs=prox, kl_beta=0.01)

        loss_int, met_int = fn_int([], [lp0.detach().requires_grad_(True), lp1.detach().requires_grad_(True)])
        loss_list, met_list = fn_list([], [lp0.detach().requires_grad_(True), lp1.detach().requires_grad_(True)])

        assert torch.allclose(loss_int, loss_list, atol=1e-6)
        assert abs(met_int["mean_kl"] - met_list["mean_kl"]) < 1e-6

    def test_mixed_prompt_lens(self):
        adv = [1.0]
        ref = [_zeros(10)]
        inf = [_zeros(10)]
        prox = [_zeros(10)]
        lp = _make_dummy_logprobs(10, seed=0)

        fn_short = make_grpo_loss_fn(adv, ref, prompt_len=2, inf_logprobs=inf, prox_logprobs=prox, kl_beta=0.0)
        fn_long = make_grpo_loss_fn(adv, ref, prompt_len=8, inf_logprobs=inf, prox_logprobs=prox, kl_beta=0.0)

        loss_short, _ = fn_short([], [lp.detach().requires_grad_(True)])
        loss_long, _ = fn_long([], [lp.detach().requires_grad_(True)])

        assert loss_short.item() != pytest.approx(loss_long.item(), abs=1e-6)

    def test_gradient_equivalence(self):
        adv_a = [1.0, -0.5]
        adv_b = [0.3, 0.7]
        ref_a = [[0.1] * 15, [0.2] * 15]
        ref_b = [[0.3] * 15, [0.15] * 15]
        inf_a = [_zeros(15), _zeros(15)]
        inf_b = [_zeros(15), _zeros(15)]
        prox_a = [_zeros(15), _zeros(15)]
        prox_b = [_zeros(15), _zeros(15)]
        prompt_len_a, prompt_len_b = 5, 8

        lp = [_make_dummy_logprobs(15, seed=i) for i in range(4)]

        fn_a = make_grpo_loss_fn(adv_a, ref_a, prompt_len=prompt_len_a, inf_logprobs=inf_a, prox_logprobs=prox_a, kl_beta=0.01)
        fn_b = make_grpo_loss_fn(adv_b, ref_b, prompt_len=prompt_len_b, inf_logprobs=inf_b, prox_logprobs=prox_b, kl_beta=0.01)
        loss_a, _ = fn_a([], [lp[0], lp[1]])
        loss_b, _ = fn_b([], [lp[2], lp[3]])
        separate_loss = loss_a + loss_b

        combined_adv = adv_a + adv_b
        combined_ref = ref_a + ref_b
        combined_inf = inf_a + inf_b
        combined_prox = prox_a + prox_b
        combined_lens = [prompt_len_a] * 2 + [prompt_len_b] * 2
        fn_combined = make_grpo_loss_fn(combined_adv, combined_ref, prompt_len=combined_lens, inf_logprobs=combined_inf, prox_logprobs=combined_prox, kl_beta=0.01)

        lp_fresh = [_make_dummy_logprobs(15, seed=i) for i in range(4)]
        combined_loss, _ = fn_combined([], lp_fresh)

        assert torch.allclose(separate_loss, combined_loss, atol=1e-5)

    def test_reports_train_inference_metrics(self):
        adv = [1.0]
        ref = [_zeros(6)]
        inf_lp = [[0.0, -0.6, -0.3, -0.1, -0.2, -0.5]]
        prox_lp = [_zeros(6)]
        lp = torch.tensor([0.0, -0.5, -0.2, -0.1, -0.3, -0.4], requires_grad=True)

        fn = make_grpo_loss_fn(adv, ref, prompt_len=2, kl_beta=0.0, inf_logprobs=inf_lp, prox_logprobs=prox_lp)
        _, metrics = fn([], [lp])

        assert "inference_diff" in metrics
        assert "inference_kld" in metrics

    def test_reports_behave_metrics(self):
        adv = [1.0]
        ref = [_zeros(6)]
        inf_lp = [[0.0, -0.6, -0.3, -0.1, -0.2, -0.5]]
        prox_lp = [[0.0, -0.5, -0.2, -0.1, -0.3, -0.4]]
        lp = torch.tensor([0.0, -0.5, -0.2, -0.1, -0.3, -0.4], requires_grad=True)

        fn = make_grpo_loss_fn(adv, ref, prompt_len=2, kl_beta=0.0, inf_logprobs=inf_lp, prox_logprobs=prox_lp)
        _, metrics = fn([], [lp])

        assert "behave/weight_mean" in metrics

    def test_dapo_reports_train_inference_metrics(self):
        adv = [1.0]
        ref = [_zeros(6)]
        inf_lp = [[0.0, -0.6, -0.3, -0.1, -0.2, -0.5]]
        prox_lp = [_zeros(6)]
        lp = torch.tensor([0.0, -0.5, -0.2, -0.1, -0.3, -0.4], requires_grad=True)

        fn = make_dapo_loss_fn(adv, ref, inf_lp, prompt_len=2, prox_logprobs=prox_lp, dapo_config=DAPOConfig())
        _, metrics = fn([], [lp])

        assert "inference_diff" in metrics
        assert "inference_kld" in metrics

    def test_gspo_reports_train_inference_metrics(self):
        adv = [1.0]
        ref = [_zeros(6)]
        inf_lp = [[0.0, -0.6, -0.3, -0.1, -0.2, -0.5]]
        prox_lp = [_zeros(6)]
        lp = torch.tensor([0.0, -0.5, -0.2, -0.1, -0.3, -0.4], requires_grad=True)

        fn = make_gspo_loss_fn(adv, ref, inf_lp, prompt_len=2, prox_logprobs=prox_lp, gspo_config=GSPOConfig())
        _, metrics = fn([], [lp])

        assert "inference_diff" in metrics
        assert "inference_kld" in metrics


class TestBatchDPOLoss:
    """Tests for ``make_batch_dpo_loss_fn``."""

    def _make_ref_logprobs(self, seq_len: int, seed: int = 0) -> list[float]:
        torch.manual_seed(seed)
        return torch.randn(seq_len).tolist()

    def test_single_pair_produces_valid_loss(self):
        ref_c = self._make_ref_logprobs(10, seed=0)
        ref_r = self._make_ref_logprobs(10, seed=1)
        rs = 3
        beta = 0.1

        lp_c = _make_dummy_logprobs(10, seed=10)
        lp_r = _make_dummy_logprobs(10, seed=11)

        fn = make_batch_dpo_loss_fn([ref_c], [ref_r], [rs], beta)
        loss, met = fn([], [lp_c.clone(), lp_r.clone()])

        assert loss.dim() == 0
        assert met["batch_pairs"] == 1
        assert "dpo_loss" in met
        assert "margin" in met
        assert met["accuracy"] in (0.0, 1.0)

    def test_multi_pair_averages_correctly(self):
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

    def test_gradient_flows(self):
        ref_c = self._make_ref_logprobs(6, seed=0)
        ref_r = self._make_ref_logprobs(6, seed=1)
        fn = make_batch_dpo_loss_fn([ref_c], [ref_r], [2], 0.1)

        lp_c = torch.randn(6, requires_grad=True)
        lp_r = torch.randn(6, requires_grad=True)
        loss, _ = fn([], [lp_c, lp_r])
        loss.backward()

        assert lp_c.grad is not None
        assert lp_r.grad is not None
