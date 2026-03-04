"""Tests for batched (multi-prompt) loss functions.

Verifies that passing per-datum ``prompt_lens: List[int]`` produces the
same result as the single ``prompt_len: int`` path when all prompt lengths
are identical, and handles mixed prompt lengths correctly.
"""

from __future__ import annotations

import torch
import pytest

from training.utils.rl.dapo import DAPOConfig, make_dapo_loss_fn
from training.utils.rl.grpo import make_grpo_loss_fn
from training.utils.rl.gspo import GSPOConfig, make_gspo_loss_fn
from training.utils.rl.common import _normalize_prompt_lens
from training.utils.rl.losses import build_loss_fn
from training.utils.rl.importance_sampling import ISConfig, make_tis_weights_fn


def _make_dummy_logprobs(seq_len: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(seq_len, requires_grad=True)


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
            builder([1.0], [[0.0] * 4], [2], [[0.0] * 4])


class TestGRPOLossBatched:
    def test_single_int_equals_list(self):
        """Loss with prompt_len=10 should equal prompt_lens=[10, 10]."""
        adv = [1.0, -0.5]
        ref = [[0.1] * 20, [0.2] * 20]
        inf = [[0.0] * 20, [0.0] * 20]
        lp0 = _make_dummy_logprobs(20, seed=42)
        lp1 = _make_dummy_logprobs(20, seed=43)

        fn_int = make_grpo_loss_fn(adv, ref, prompt_len=10, inf_logprobs=inf, kl_beta=0.01)
        fn_list = make_grpo_loss_fn(adv, ref, prompt_len=[10, 10], inf_logprobs=inf, kl_beta=0.01)

        loss_int, met_int = fn_int([], [lp0.detach().requires_grad_(True), lp1.detach().requires_grad_(True)])
        loss_list, met_list = fn_list([], [lp0.detach().requires_grad_(True), lp1.detach().requires_grad_(True)])

        assert torch.allclose(loss_int, loss_list, atol=1e-6)
        assert abs(met_int["mean_kl"] - met_list["mean_kl"]) < 1e-6

    def test_mixed_prompt_lens(self):
        """Different prompt_lens per datum should use different response_start."""
        adv = [1.0, 1.0]
        ref = [[0.0] * 10, [0.0] * 10]
        inf = [[0.0] * 10]
        lp = _make_dummy_logprobs(10, seed=0)

        fn_short = make_grpo_loss_fn(adv[:1], ref[:1], prompt_len=2, inf_logprobs=inf, kl_beta=0.0)
        fn_long = make_grpo_loss_fn(adv[:1], ref[:1], prompt_len=8, inf_logprobs=inf, kl_beta=0.0)

        loss_short, _ = fn_short([], [lp.detach().requires_grad_(True)])
        loss_long, _ = fn_long([], [lp.detach().requires_grad_(True)])

        assert loss_short.item() != pytest.approx(loss_long.item(), abs=1e-6), (
            "Different prompt_lens should produce different losses"
        )

    def test_gradient_equivalence(self):
        """Gradients through batched call match sum of per-prompt calls."""
        adv_a = [1.0, -0.5]
        adv_b = [0.3, 0.7]
        ref_a = [[0.1] * 15, [0.2] * 15]
        ref_b = [[0.3] * 15, [0.15] * 15]
        inf_a = [[0.0] * 15, [0.0] * 15]
        inf_b = [[0.0] * 15, [0.0] * 15]
        prompt_len_a, prompt_len_b = 5, 8

        lp = [_make_dummy_logprobs(15, seed=i) for i in range(4)]

        fn_a = make_grpo_loss_fn(adv_a, ref_a, prompt_len=prompt_len_a, inf_logprobs=inf_a, kl_beta=0.01)
        fn_b = make_grpo_loss_fn(adv_b, ref_b, prompt_len=prompt_len_b, inf_logprobs=inf_b, kl_beta=0.01)
        loss_a, _ = fn_a([], [lp[0], lp[1]])
        loss_b, _ = fn_b([], [lp[2], lp[3]])
        separate_loss = loss_a + loss_b

        combined_adv = adv_a + adv_b
        combined_ref = ref_a + ref_b
        combined_inf = inf_a + inf_b
        combined_lens = [prompt_len_a] * 2 + [prompt_len_b] * 2
        fn_combined = make_grpo_loss_fn(
            combined_adv,
            combined_ref,
            prompt_len=combined_lens,
            inf_logprobs=combined_inf,
            kl_beta=0.01,
        )

        lp_fresh = [_make_dummy_logprobs(15, seed=i) for i in range(4)]
        combined_loss, _ = fn_combined([], lp_fresh)

        assert torch.allclose(separate_loss, combined_loss, atol=1e-5), (
            f"Combined loss {combined_loss.item()} != separate sum {separate_loss.item()}"
        )

    def test_reports_train_inference_metrics_when_inf_logprobs_present(self):
        adv = [1.0]
        ref = [[0.0] * 6]
        inf_lp = [[0.0, -0.6, -0.3, -0.1, -0.2, -0.5]]
        lp = torch.tensor([0.0, -0.5, -0.2, -0.1, -0.3, -0.4], requires_grad=True)

        fn = make_grpo_loss_fn(
            adv,
            ref,
            prompt_len=2,
            kl_beta=0.0,
            inf_logprobs=inf_lp,
        )
        _, metrics = fn([], [lp])

        diff = lp.detach()[1:] - torch.tensor(inf_lp[0][1:], dtype=lp.dtype)
        expected_diff = diff.abs().mean().item()
        expected_kld = (torch.exp(diff) - 1.0 - diff).mean().item()

        assert metrics["inference_diff"] == pytest.approx(expected_diff)
        assert metrics["inference_kld"] == pytest.approx(expected_kld)

    def test_dapo_reports_train_inference_metrics(self):
        adv = [1.0]
        ref = [[0.0] * 6]
        inf_lp = [[0.0, -0.6, -0.3, -0.1, -0.2, -0.5]]
        lp = torch.tensor([0.0, -0.5, -0.2, -0.1, -0.3, -0.4], requires_grad=True)

        fn = make_dapo_loss_fn(
            advantages=adv,
            ref_logprobs=ref,
            inf_logprobs=inf_lp,
            prompt_len=2,
            dapo_config=DAPOConfig(),
        )
        _, metrics = fn([], [lp])

        assert "inference_diff" in metrics
        assert "inference_kld" in metrics

    def test_gspo_reports_train_inference_metrics(self):
        adv = [1.0]
        ref = [[0.0] * 6]
        inf_lp = [[0.0, -0.6, -0.3, -0.1, -0.2, -0.5]]
        lp = torch.tensor([0.0, -0.5, -0.2, -0.1, -0.3, -0.4], requires_grad=True)

        fn = make_gspo_loss_fn(
            advantages=adv,
            ref_logprobs=ref,
            inf_logprobs=inf_lp,
            prompt_len=2,
            gspo_config=GSPOConfig(),
        )
        _, metrics = fn([], [lp])

        assert "inference_diff" in metrics
        assert "inference_kld" in metrics


class TestTISWeights:
    def test_rejects_short_inference_logprobs(self):
        weights_fn = make_tis_weights_fn(
            inf_logprobs=[[0.0, -0.1]],
            prompt_len=2,
            tis_config=ISConfig(),
        )
        with pytest.raises(ValueError, match="requires at least"):
            weights_fn(torch.tensor([-0.5, -0.3, -0.2]), 0)
