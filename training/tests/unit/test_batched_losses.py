"""Tests for loss function utilities and DPO batching."""

from __future__ import annotations

import math
import struct
import sys
from types import SimpleNamespace

import pytest
import torch
from fireworks.training import sdk
from training.utils.losses import (
    make_batch_dpo_loss_fn,
    make_batch_orpo_loss_fn,
    make_batch_weighted_sft_loss_fn,
    make_orpo_loss_fn,
    make_sft_loss_fn,
    perplexity_from_nll,
)
from training.utils.rl.algorithm import grpo
from training.utils.rl.common import (
    align_sample_logprobs_to_target_tokens,
    inference_drift_metrics,
)
from training.utils.rl.losses import PolicyBatch, build_policy_datums
from training.utils.supervised import build_datum_from_token_mask


def _make_dummy_logprobs(seq_len: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(seq_len, requires_grad=True)


def _zeros(n: int) -> list[float]:
    return [0.0] * n


def test_perplexity_from_nll_uses_float64_range() -> None:
    ppl = perplexity_from_nll(torch.tensor(100.0, dtype=torch.float32))

    assert ppl == pytest.approx(math.exp(100.0))
    assert math.isfinite(ppl)


def test_perplexity_from_nll_handles_float64_boundary() -> None:
    max_finite_nll = math.log(sys.float_info.max)

    assert perplexity_from_nll(max_finite_nll) == pytest.approx(sys.float_info.max)
    assert perplexity_from_nll(math.nextafter(max_finite_nll, math.inf)) == math.inf


@pytest.mark.parametrize(
    ("nll", "expected"),
    [
        (math.inf, math.inf),
        (-math.inf, 0.0),
    ],
)
def test_perplexity_from_nll_handles_infinities(
    nll: float,
    expected: float,
) -> None:
    assert perplexity_from_nll(nll) == expected


def test_perplexity_from_nll_preserves_nan() -> None:
    assert math.isnan(perplexity_from_nll(math.nan))


class TestAlignSampleLogprobs:
    def test_pads_completion_logprobs_to_target_tokens(self):
        sampled = SimpleNamespace(
            full_tokens=[1, 2, 3, 10, 11],
            prompt_len=3,
            sampling_logprobs=[-0.5, -0.6],
        )

        assert align_sample_logprobs_to_target_tokens(
            sampled,
            attr="sampling_logprobs",
            source="rollout_logprobs",
            sample_idx=0,
            required=True,
        ) == [0.0, 0.0, -0.5, -0.6]

    def test_accepts_echoed_logprobs_already_target_aligned(self):
        sampled = SimpleNamespace(
            full_tokens=[1, 2, 3, 10, 11],
            prompt_len=3,
            sampling_logprobs=[-1.0, -1.1, -0.5, -0.6],
            logprobs_echoed=True,
        )

        assert align_sample_logprobs_to_target_tokens(
            sampled,
            attr="sampling_logprobs",
            source="rollout_logprobs",
            sample_idx=0,
            required=True,
        ) == [-1.0, -1.1, -0.5, -0.6]

    def test_pads_null_echoed_prompt_sampling_logprobs(self):
        sampled = SimpleNamespace(
            full_tokens=[1, 2, 3, 10, 11],
            prompt_len=3,
            # Serving emits behavior-policy logprobs only for generated tokens.
            sampling_logprobs=[None, None, -0.5, -0.6],
            logprobs_echoed=True,
        )

        assert align_sample_logprobs_to_target_tokens(
            sampled,
            attr="sampling_logprobs",
            source="rollout_logprobs",
            sample_idx=0,
            required=True,
        ) == [0.0, 0.0, -0.5, -0.6]

    def test_rejects_null_echoed_completion_sampling_logprob(self):
        sampled = SimpleNamespace(
            full_tokens=[1, 2, 3, 10, 11],
            prompt_len=3,
            sampling_logprobs=[None, None, -0.5, None],
            logprobs_echoed=True,
        )

        with pytest.raises(RuntimeError, match="null completion logprob"):
            align_sample_logprobs_to_target_tokens(
                sampled,
                attr="sampling_logprobs",
                source="rollout_logprobs",
                sample_idx=0,
                required=True,
            )

    def test_rejects_null_echoed_completion_with_single_token_prompt(self):
        sampled = SimpleNamespace(
            full_tokens=[1, 10, 11],
            prompt_len=1,
            sampling_logprobs=[None, -0.6],
            logprobs_echoed=True,
        )

        with pytest.raises(RuntimeError, match="null completion logprob"):
            align_sample_logprobs_to_target_tokens(
                sampled,
                attr="sampling_logprobs",
                source="rollout_logprobs",
                sample_idx=0,
                required=True,
            )

    @pytest.mark.parametrize("prompt_len", [-1, 6])
    def test_rejects_invalid_prompt_len(self, prompt_len):
        sampled = SimpleNamespace(
            full_tokens=[1, 2, 3, 10, 11],
            prompt_len=prompt_len,
            sampling_logprobs=[None, None, -0.5, -0.6],
            logprobs_echoed=True,
        )

        with pytest.raises(RuntimeError, match="invalid prompt_len"):
            align_sample_logprobs_to_target_tokens(
                sampled,
                attr="sampling_logprobs",
                source="rollout_logprobs",
                sample_idx=0,
                required=True,
            )

    def test_rejects_null_echoed_raw_inference_logprob(self):
        sampled = SimpleNamespace(
            full_tokens=[1, 2, 3, 10, 11],
            prompt_len=3,
            inference_logprobs=[None, -1.1, -0.5, -0.6],
            logprobs_echoed=True,
        )

        with pytest.raises(RuntimeError, match="null target-aligned logprob"):
            align_sample_logprobs_to_target_tokens(
                sampled,
                attr="inference_logprobs",
                source="raw inference logprobs",
                sample_idx=0,
                required=True,
            )

    def test_rejects_non_completion_only_logprobs(self):
        sampled = SimpleNamespace(
            full_tokens=[1, 2, 3, 10, 11],
            prompt_len=3,
            sampling_logprobs=[-9.0, -9.0, -9.0, -0.5, -0.6],
        )

        with pytest.raises(RuntimeError, match="expected 2 completion logprobs"):
            align_sample_logprobs_to_target_tokens(
                sampled,
                attr="sampling_logprobs",
                source="rollout_logprobs",
                sample_idx=0,
                required=True,
            )

    def test_rejects_misaligned_echoed_logprobs(self):
        sampled = SimpleNamespace(
            full_tokens=[1, 2, 3, 10, 11],
            prompt_len=3,
            sampling_logprobs=[-0.5, -0.6],
            logprobs_echoed=True,
        )

        with pytest.raises(RuntimeError, match="expected 4 target-aligned logprobs"):
            align_sample_logprobs_to_target_tokens(
                sampled,
                attr="sampling_logprobs",
                source="rollout_logprobs",
                sample_idx=0,
                required=True,
            )


def _policy_batch(data, advantages, rollout_logprobs, prompt_lens, **kwargs) -> PolicyBatch:
    return PolicyBatch(
        data=data,
        advantages=advantages,
        prompt_lens=prompt_lens,
        rollout_logprobs=rollout_logprobs,
        **kwargs,
    )


class TestBuildPolicyDatums:
    @staticmethod
    def _datum():
        return build_datum_from_token_mask(
            token_ids=[10, 11, 12, 13],
            token_mask=[0, 1, 1, 1],
        ).datum

    def test_builds_exact_builtin_contract(self):
        datum = build_policy_datums(_policy_batch([self._datum()], [1.0], [[-0.1, -0.2, -0.3]], [2]))[0]

        assert set(datum.loss_fn_inputs) == {
            "target_tokens",
            "logprobs",
            "advantages",
            "response_mask",
        }
        assert datum.loss_fn_inputs["logprobs"].data == pytest.approx([0.0, -0.2, -0.3])
        assert datum.loss_fn_inputs["response_mask"].data == [0, 1, 1]

    @pytest.mark.parametrize(
        ("field", "kwargs"),
        [
            ("advantages", {"advantages": []}),
            ("rollout_logprobs", {"rollout_logprobs": []}),
            ("prompt_lens", {"prompt_lens": []}),
            ("anchor_logprobs", {"anchor_logprobs": []}),
            ("token_advantages", {"token_advantages": []}),
        ],
    )
    def test_rejects_misaligned_rows(self, field, kwargs):
        inputs = {
            "data": [self._datum()],
            "advantages": [1.0],
            "rollout_logprobs": [[-0.1, -0.2, -0.3]],
            "prompt_lens": [2],
        }
        inputs.update(kwargs)

        with pytest.raises(ValueError, match=field):
            build_policy_datums(PolicyBatch(**inputs))

    def test_rejects_short_rollout_row_instead_of_padding(self):
        with pytest.raises(ValueError, match="expected 3, got 2"):
            build_policy_datums(_policy_batch([self._datum()], [1.0], [[-0.1, -0.2]], [2]))

    @pytest.mark.parametrize("prompt_len", [-1, 5])
    def test_rejects_prompt_boundary_outside_sequence(self, prompt_len):
        with pytest.raises(ValueError, match="prompt_len"):
            build_policy_datums(_policy_batch([self._datum()], [1.0], [[-0.1, -0.2, -0.3]], [prompt_len]))

    def test_keeps_response_advantages_unweighted(self):
        datum = build_policy_datums(_policy_batch([self._datum()], [2.0], [[-1.0, -1.0, -1.0]], [2]))[0]

        assert datum.loss_fn_inputs["advantages"].data == pytest.approx([0.0, 2.0, 2.0])

    def test_token_advantages_replace_scalar_advantage_on_active_tokens(self):
        datum = sdk.Datum(
            model_input=sdk.ModelInput.from_ints([10, 11, 12, 13]),
            loss_fn_inputs={
                "target_tokens": sdk.TensorData(data=[11, 12, 13, 14], dtype="int64", shape=[4]),
                "weights": sdk.TensorData(data=[0, 1, 0, 1], dtype="float32", shape=[4]),
            },
        )
        batch = _policy_batch(
            [datum],
            [5.0],
            [[-0.1, -0.2, None, -0.4]],
            [2],
            token_advantages=[[9.0, 1.0, -7.0, -2.0]],
        )

        inputs = build_policy_datums(batch)[0].loss_fn_inputs

        # Prompt and masked targets get no advantage; the scalar is ignored.
        assert inputs["advantages"].data == [0.0, 1.0, 0.0, -2.0]
        assert inputs["response_mask"].data == [0, 1, 0, 1]
        assert inputs["logprobs"].data == pytest.approx([0.0, -0.2, 0.0, -0.4])

    def test_token_advantages_row_must_match_target_tokens(self):
        batch = _policy_batch([self._datum()], [1.0], [[-0.1, -0.2, -0.3]], [2], token_advantages=[[1.0, 1.0]])
        with pytest.raises(ValueError, match="2 values for 3 target tokens"):
            build_policy_datums(batch)

    def test_anchor_logprobs_replace_rollout_as_datum_logprobs(self):
        batch = _policy_batch(
            [self._datum()],
            [1.0],
            [[-0.1, -0.2, -0.3]],
            [2],
            anchor_logprobs=[[-9.0, -0.5, -0.6]],
        )

        inputs = build_policy_datums(batch)[0].loss_fn_inputs

        assert inputs["logprobs"].data == pytest.approx([0.0, -0.5, -0.6])
        assert inputs["advantages"].data == [0.0, 1.0, 1.0]

    def test_anchor_defaults_to_rollout_logprobs(self):
        rollout = [[-0.1, -0.2, -0.3]]
        assert _policy_batch([self._datum()], [1.0], rollout, [2]).anchor_logprobs is rollout

    @pytest.mark.parametrize("advantage", [0.0, -0.0, 1.0 / 3, -1.23456789])
    @pytest.mark.parametrize(
        ("prefix", "response_len", "all_masked"),
        [(0, 0, False), (4, 0, False), (0, 17, False), (3, 17, True), (5, 4097, False)],
    )
    def test_preserves_scalar_advantage_arithmetic_exactly(
        self,
        advantage: float,
        prefix: int,
        response_len: int,
        all_masked: bool,
    ) -> None:
        # Masked spans represent intervening tool/user turns; fractional masks
        # also exercise the existing active-token threshold and final weighting.
        masks = (
            [0.0] * response_len if all_masked else [0.0, 1.0, 1.0, 0.25, 0.0, 1.0, 0.75] * ((response_len + 6) // 7)
        )
        masks = masks[:response_len]
        count = prefix + response_len
        source = sdk.Datum(
            model_input=sdk.ModelInput.from_ints(list(range(count))),
            loss_fn_inputs={
                "target_tokens": sdk.TensorData(data=list(range(count)), dtype="int64", shape=[count]),
                "weights": sdk.TensorData(data=[0.0] * prefix + masks, dtype="float32", shape=[count]),
            },
        )
        old_lp = [0.0] * prefix + [-0.13 * (i % 7 + 1) for i in range(response_len)]
        rollout_lp = [0.0] * prefix + [
            None if mask <= 0.5 else old_lp[prefix + i] - [-3.0, -0.3, 0.7, 3.0][i % 4] for i, mask in enumerate(masks)
        ]
        tensor_mask = torch.tensor(masks, dtype=torch.float32)
        expected = [0.0] * prefix + [float(advantage * tensor_mask[i].item()) for i in range(response_len)]

        result = build_policy_datums(_policy_batch([source], [advantage], [rollout_lp], [prefix + 1]))[0]

        actual = result.loss_fn_inputs["advantages"]
        assert actual.shape == [count] and actual.dtype == "float32"
        # TensorData stores float32; compare its exact bits, including signed zero.
        assert struct.pack(f"{count}f", *actual.data) == struct.pack(f"{count}f", *expected)
        assert result.model_input is source.model_input
        assert result.loss_fn_inputs["target_tokens"].data == list(range(count))
        expected_logprobs = [0.0 if value is None else value for value in rollout_lp]
        assert result.loss_fn_inputs["logprobs"].data == pytest.approx(expected_logprobs)


def _logprob_result(*rows):
    return SimpleNamespace(
        loss_fn_outputs=[{"logprobs": sdk.TensorData(data=row, dtype="float32", shape=[len(row)])} for row in rows]
    )


class TestServerGRPOMetrics:
    @staticmethod
    def _batch(rollout_logprobs, raw_logprobs, **kwargs):
        datum = build_datum_from_token_mask(
            token_ids=[10, 11, 12, 13],
            token_mask=[0, 1, 1, 1],
        ).datum
        return _policy_batch(
            [datum],
            [1.0],
            rollout_logprobs,
            [2],
            raw_inference_logprobs=raw_logprobs,
            **kwargs,
        )

    def test_uses_returned_policy_logprobs_without_another_forward(self):
        batch = self._batch([[-0.4, -0.3, -0.1]], [[-0.4, -0.4, -0.2]])
        result = _logprob_result([-0.4, -0.2, -0.1])

        metrics = grpo.metrics(batch, result, grpo.GRPOConfig())

        expected_ratios = torch.exp(torch.tensor([0.1, 0.0]))
        assert metrics["ppo_ratio_mean"] == pytest.approx(expected_ratios.mean().item())
        assert metrics["ppo_clip_frac"] == 0.0
        log_ratio = torch.tensor([0.1, 0.0])
        assert metrics["ppo_kl"] == pytest.approx((torch.exp(log_ratio) - log_ratio - 1.0).mean().item())
        assert metrics["active_tokens"] == 2
        assert metrics["total_resp_tokens"] == 2
        # Sampler/trainer drift is a separate call, not part of the objective metrics.
        assert "inference_k1" not in metrics

        drift = inference_drift_metrics(batch, result)
        assert drift["inference_k1"] == pytest.approx(0.15)
        expected_diff = torch.tensor([0.2, 0.1])
        assert drift["inference_k3"] == pytest.approx((torch.exp(expected_diff) - expected_diff - 1.0).mean().item())
        assert drift["raw_inference_logprob_coverage"] == 1.0

    def test_anchor_override_and_clip_fraction(self):
        batch = self._batch([[-0.4, -0.3, -0.1]], [[]], anchor_logprobs=[[-0.4, -0.5, -0.1]])
        result = _logprob_result([-0.4, 0.0, -0.1])

        metrics = grpo.metrics(batch, result, grpo.GRPOConfig(eps_clip=0.1))

        # Ratios exp(0.5) and 1.0: one of two active tokens is outside [0.9, 1.1].
        assert metrics["ppo_clip_frac"] == pytest.approx(0.5)
        assert inference_drift_metrics(batch, result) == {"raw_inference_logprob_coverage": 0.0}

    def test_rejects_misaligned_trainer_outputs(self):
        batch = self._batch([[-0.4, -0.3, -0.1]], [[-0.4, -0.4, -0.2]])
        with pytest.raises(ValueError, match="policy_logprobs=0"):
            grpo.metrics(batch, _logprob_result(), grpo.GRPOConfig())


class TestInferenceDriftMetrics:
    def test_preserves_per_sequence_macro_average(self):
        batch = _policy_batch(
            [],
            [0.0, 0.0],
            [[0.0], [0.0, 0.0, 0.0]],
            [1, 1],
            raw_inference_logprobs=[[0.0], [0.0, 0.0, 0.0]],
        )

        metrics = inference_drift_metrics(batch, _logprob_result([1.0], [0.0, 0.0, 0.0]))

        # The one-token and three-token sequences have equal weight. A
        # token-weighted reduction would return 0.25 instead of 0.5.
        assert metrics["inference_k1"] == pytest.approx(0.5)
        assert metrics["inference_k3"] == pytest.approx((torch.exp(torch.tensor(1.0)).item() - 2.0) / 2.0)

    @pytest.mark.parametrize("raw", [[], [[], []]])
    def test_coverage_is_zero_without_raw_rows(self, raw):
        batch = _policy_batch([], [1.0, 1.0], [[-0.1, -0.2], [-0.1, -0.2]], [1, 1], raw_inference_logprobs=raw)

        metrics = inference_drift_metrics(batch, _logprob_result([-0.1, -0.2], [-0.1, -0.2]))

        assert metrics == {"raw_inference_logprob_coverage": 0.0}

    def test_partial_coverage_reports_share_of_active_tokens(self):
        batch = _policy_batch(
            [],
            [1.0, 1.0],
            [[-0.1, -0.2], [-0.1, -0.2, -0.3, -0.4]],
            [1, 1],
            raw_inference_logprobs=[[-0.1, -0.3], []],
        )

        metrics = inference_drift_metrics(batch, _logprob_result([-0.1, -0.2], [-0.1, -0.2, -0.3, -0.4]))

        assert metrics["raw_inference_logprob_coverage"] == pytest.approx(2 / 6)
        assert metrics["inference_k1"] == pytest.approx(0.05)

    def test_no_active_tokens_reports_nothing(self):
        batch = _policy_batch([], [], [], [])
        assert inference_drift_metrics(batch, _logprob_result()) == {}


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
            [ref_c0, ref_c1],
            [ref_r0, ref_r1],
            [rs0, rs1],
            beta,
        )
        loss_b, met_b = fn_batch(
            [],
            [lp_c0.clone(), lp_r0.clone(), lp_c1.clone(), lp_r1.clone()],
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
        assert metrics_batch["orpo_loss"] == pytest.approx((metrics0["orpo_loss"] + metrics1["orpo_loss"]) / 2)


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


class TestDPOResponseStartOffByOne:
    """DPO loss must include the first response token's logprob.

    `logprobs[i]` predicts `token[i+1]`, so a response that starts at
    index ``response_start`` has its first-token prediction at
    ``logprobs[response_start - 1]``. A naive slice ``lp[response_start:]``
    drops the most discriminating token between chosen and rejected.
    ORPO uses ``lp_start = max(0, response_start - 1)``; DPO must agree.
    """

    def test_dpo_loss_includes_first_response_token(self) -> None:
        """chosen=[A, B, C] vs rejected=[A, B, D] with response_start=2.

        The only differing token is the last one. With the buggy slice
        ``lp[2:]`` the differing logprob is dropped and the margin collapses
        to 0.
        """
        chosen_logprobs = torch.tensor([-0.1, -0.1], dtype=torch.float32)
        rejected_logprobs = torch.tensor([-0.1, -2.0], dtype=torch.float32)
        ref_chosen = [-0.5, -1.0]
        ref_rejected = [-0.5, -1.0]
        response_start = 2

        loss_fn = make_batch_dpo_loss_fn(
            [ref_chosen],
            [ref_rejected],
            [response_start],
            beta=0.1,
        )
        _, metrics = loss_fn([None, None], [chosen_logprobs, rejected_logprobs])

        assert metrics["margin"] != pytest.approx(0.0, abs=1e-6), (
            "DPO margin is 0 because the first response token's logprob was sliced away. "
            "Fix: use `lp_start = max(0, response_start - 1)` (mirroring make_batch_orpo_loss_fn)."
        )
        assert metrics["margin"] > 0.0, (
            f"Policy strongly favors chosen over rejected on the only differing token, "
            f"so DPO margin should be positive. Got margin={metrics['margin']:.4f}."
        )

    def test_dpo_and_orpo_agree_on_preferred_direction(self) -> None:
        """Given identical inputs, DPO and ORPO should both identify chosen as preferred."""
        chosen_logprobs = torch.tensor([-0.1, -0.1], dtype=torch.float32)
        rejected_logprobs = torch.tensor([-0.1, -2.0], dtype=torch.float32)
        ref_chosen = [-0.5, -1.0]
        ref_rejected = [-0.5, -1.0]
        response_start = 2

        dpo_loss = make_batch_dpo_loss_fn(
            [ref_chosen],
            [ref_rejected],
            [response_start],
            beta=0.1,
        )
        _, dpo_metrics = dpo_loss(
            [None, None],
            [chosen_logprobs.clone(), rejected_logprobs.clone()],
        )

        orpo_loss = make_batch_orpo_loss_fn([response_start], orpo_lambda=1.0)
        _, orpo_metrics = orpo_loss(
            [None, None],
            [chosen_logprobs.clone(), rejected_logprobs.clone()],
        )

        assert (
            orpo_metrics["log_odds_ratio"] > 0.0
        ), "Sanity check: ORPO should prefer chosen; if this fails, the test setup is wrong."
        assert dpo_metrics["accuracy"] == orpo_metrics["accuracy"], (
            f"DPO and ORPO disagree on preferred direction "
            f"(dpo_acc={dpo_metrics['accuracy']}, orpo_acc={orpo_metrics['accuracy']}). "
            "Root cause: DPO's logprob slice is off by one relative to ORPO's."
        )


class TestSFTLossResponseStartOffByOne:
    """`make_sft_loss_fn` had the same off-by-one as DPO — its response-start
    slice must start at ``max(0, response_start - 1)`` so the first response
    token's prediction is counted."""

    def test_single_sample_sft_loss_includes_first_response_token(self) -> None:
        """Full sequence [A, B, C] with response_start=2 (response is just "C").

        target_tokens=[B, C], logprobs=[logp(B|A), logp(C|A,B)].

        Expected loss: -logp(C|A,B) = 1.0.
        Buggy code slices ``lp[2:]`` which is empty and returns 0.0.
        """
        target_tokens = [10, 20]  # B, C
        response_start = 2
        logprobs = torch.tensor([0.0, -1.0], dtype=torch.float32)

        loss_fn = make_sft_loss_fn(response_start, target_tokens)
        loss, metrics = loss_fn([None], [logprobs])

        assert metrics["response_tokens"] == 1, (
            f"response_start=2 covers exactly 1 response token in a length-3 sequence; "
            f"got {metrics['response_tokens']}"
        )
        assert loss.item() == pytest.approx(1.0, abs=1e-5), (
            f"SFT single-sample loss dropped the first response token's logprob. "
            f"Expected CE=1.0 (from logp(C|A,B)=-1.0), got {loss.item():.4f}. "
            "Fix: use `lp_start = max(0, response_start - 1)` (same fix as DPO)."
        )


def test_build_policy_datums_rejects_null_rollout_logprobs_after_anchor_recompute():
    datum = sdk.Datum(
        model_input=sdk.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": sdk.TensorData(data=[2, 3], dtype="int64", shape=[2])},
    )
    batch = PolicyBatch(
        data=[datum],
        advantages=[1.0],
        prompt_lens=[1],
        rollout_logprobs=[[-0.1, None]],
        anchor_logprobs=[[-0.1, -0.2]],
    )
    with pytest.raises(ValueError, match="rollout_logprobs"):
        build_policy_datums(batch)
