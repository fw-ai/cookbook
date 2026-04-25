"""Tests for loss function utilities and DPO batching."""

from __future__ import annotations

import pytest
import torch

from training.utils.losses import (
    make_batch_dpo_loss_fn,
    make_batch_orpo_loss_fn,
    make_batch_weighted_sft_loss_fn,
    make_orpo_loss_fn,
    make_sft_loss_fn,
)
from training.utils.rl.common import _normalize_prompt_lens
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.losses import (
    LOSS_REGISTRY,
    build_builtin_loss_datums,
    build_loss_fn,
    get_builtin_loss_config,
    resolve_builtin_loss,
)
from training.utils.rl.spec import LossSpec
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

    def test_grpo_requires_inference_logprobs(self):
        builder = build_loss_fn(policy_loss="grpo", kl_beta=0.01)
        loss_fn = builder([1.0], [_zeros(4)], [2], [[]], [_zeros(4)])

        with pytest.raises(ValueError, match="GRPO requires inference logprobs"):
            loss_fn([], [_make_dummy_logprobs(4, seed=0)])

    def test_client_only_loss_registration_uses_custom_path(self, monkeypatch):
        events: dict[str, object] = {}

        def fake_client_loss_factory(**kwargs):
            events["kwargs"] = kwargs
            return "client-only-loss"

        monkeypatch.setitem(
            LOSS_REGISTRY,
            "client_only_test",
            LossSpec(
                name="client_only_test",
                client_loss_factory=fake_client_loss_factory,
            ),
        )

        assert resolve_builtin_loss("client_only_test") is None

        builder = build_loss_fn(policy_loss="client_only_test", kl_beta=0.01)
        loss_fn = builder([1.0], [_zeros(4)], [2], [_zeros(4)], [_zeros(4)])

        assert loss_fn == "client-only-loss"
        assert events["kwargs"]["advantages"] == [1.0]
        assert events["kwargs"]["prompt_lens"] == [2]
        assert events["kwargs"]["kl_beta"] == pytest.approx(0.01)


class TestBuiltinLossConfig:
    def test_grpo_preserves_zero_upper_clip_bound(self):
        kernel, config = get_builtin_loss_config(
            "grpo", eps_clip=0.2, eps_clip_high=0.0,
        )

        assert kernel == "ppo"
        assert config["clip_low_threshold"] == pytest.approx(0.8)
        assert config["clip_high_threshold"] == pytest.approx(1.0)

    def test_gspo_preserves_zero_clip_bounds(self):
        kernel, config = get_builtin_loss_config(
            "gspo",
            gspo_config=GSPOConfig(
                clip_ratio_low=0.0,
                clip_ratio_high=0.0,
            ),
        )

        assert kernel == "gspo"
        assert config["clip_low_threshold"] == pytest.approx(1.0)
        assert config["clip_high_threshold"] == pytest.approx(1.0)

    def test_importance_sampling_uses_ratio_log_cap(self):
        kernel, config = get_builtin_loss_config(
            "importance_sampling",
            ratio_log_cap=7.5,
        )

        assert kernel == "importance_sampling"
        assert config["ratio_log_cap"] == pytest.approx(7.5)

    def test_builtin_datums_require_inference_logprobs(self):
        datum = build_datum_from_token_mask(
            token_ids=[10, 11, 12, 13],
            token_mask=[0, 1, 1, 1],
        ).datum

        with pytest.raises(ValueError, match="GRPO requires inference logprobs"):
            build_builtin_loss_datums(
                [datum],
                [1.0],
                [[-0.1, -0.2, -0.3]],
                [[]],
                [2],
                policy_loss="grpo",
            )


class TestKLBetaRoutesToClientSide:
    """Documents the silent-KL-drop bug and the fix contract.

    Bug: the server-side builtin kernels receive datums produced by
    :func:`training.utils.rl.losses.build_builtin_loss_datums`, whose
    ``loss_fn_inputs`` are exactly ``{target_tokens, logprobs, advantages}``
    -- no ``ref_logprobs`` field is ever sent to the trainer. So when a caller
    sets ``kl_beta > 0`` expecting a KL penalty, the builtin PPO kernel
    silently drops the ``kl_beta * (pi - pi_ref)`` term.

    Fix: ``resolve_builtin_loss`` must gate on ``kl_beta`` and return ``None``
    when positive, forcing the recipe onto ``forward_backward_custom(...)``,
    which is the only path that actually applies the KL penalty.
    """

    @staticmethod
    def _builtin_datum():
        """Run the builtin datum packer the way rl_loop does."""
        datum = build_datum_from_token_mask(
            token_ids=[10, 11, 12, 13],
            token_mask=[0, 1, 1, 1],
        ).datum
        rl_datums = build_builtin_loss_datums(
            data=[datum],
            advantages=[1.0],
            prox_logprobs=[[-0.1, -0.2, -0.3, -0.4]],
            inf_logprobs=[[-0.1, -0.2, -0.3, -0.4]],
            prompt_lens=[2],
            policy_loss="grpo",
        )
        return rl_datums[0]

    def test_builtin_datum_has_no_ref_logprobs_field(self):
        """Structural invariant: the server-side path carries no ref logprobs.

        This is the root cause of the silent KL drop. Encoded as a test so the
        motivation for the ``kl_beta`` gate remains explicit: if this invariant
        ever changes (e.g. the server kernel learns to consume ref logprobs),
        the gate below can be relaxed.
        """
        fields = set(self._builtin_datum().loss_fn_inputs.keys())

        assert fields == {"target_tokens", "logprobs", "advantages"}, (
            "build_builtin_loss_datums emitted unexpected fields "
            f"{fields!r}; the server-side builtin kernel contract is "
            "(target_tokens, logprobs, advantages) only."
        )
        assert "ref_logprobs" not in fields, (
            "ref_logprobs leaked into the builtin datum -- if the server "
            "kernel now consumes it, the kl_beta gate can be lifted."
        )

    def test_resolve_builtin_loss_gates_on_kl_beta_for_grpo(self):
        """With ``kl_beta > 0`` the dispatch must return ``None`` (client-side)."""
        try:
            result = resolve_builtin_loss("grpo", profile=None, kl_beta=0.01)
        except TypeError as exc:
            pytest.fail(
                "resolve_builtin_loss currently ignores kl_beta "
                f"({exc}). The server-side builtin datum has no "
                "ref_logprobs field (see "
                "test_builtin_datum_has_no_ref_logprobs_field), so "
                "kl_beta > 0 routed to the builtin kernel is silently "
                "dropped. Add a `kl_beta: float = 0.0` parameter to "
                "resolve_builtin_loss and return None when it is positive."
            )

        assert result is None, (
            "kl_beta>0 requested a KL penalty, but resolve_builtin_loss "
            f"returned the server-side builtin kernel {result!r}. That "
            "kernel silently drops the KL term (no ref_logprobs in the "
            "datum). Must return None to force forward_backward_custom."
        )

    def test_resolve_builtin_loss_kl_beta_zero_keeps_server_builtin(self):
        """Regression guard: ``kl_beta == 0`` must still take the fast path."""
        result = resolve_builtin_loss("grpo", profile=None)

        assert result is not None, (
            "GRPO with kl_beta=0 must keep the fast server-side PPO path. "
            "The kl_beta gate must not over-broadly block the builtin path."
        )
        kernel, _config = result
        assert kernel == "ppo"


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
            [ref_chosen], [ref_rejected], [response_start], beta=0.1,
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
            [ref_chosen], [ref_rejected], [response_start], beta=0.1,
        )
        _, dpo_metrics = dpo_loss(
            [None, None], [chosen_logprobs.clone(), rejected_logprobs.clone()],
        )

        orpo_loss = make_batch_orpo_loss_fn([response_start], orpo_lambda=1.0)
        _, orpo_metrics = orpo_loss(
            [None, None], [chosen_logprobs.clone(), rejected_logprobs.clone()],
        )

        assert orpo_metrics["log_odds_ratio"] > 0.0, (
            "Sanity check: ORPO should prefer chosen; if this fails, the test setup is wrong."
        )
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
