"""Unit tests for the GSPO sequence-level importance ratio.

``baseline_gspo_loss`` mirrors the built-in GSPO loss before optimizer-step
sequence normalization. Logits, hidden states and the LM-head projection are
dropped: the cookbook loss consumes logprobs directly, so the projection only
adds noise to the comparison.
"""

from __future__ import annotations

import math

import pytest
import tinker
import torch

from training.utils.rl.gspo import GSPOConfig, make_gspo_loss_fn
from training.utils.rl.tis import TISConfig, compute_tis_weight

SEQ_RATIO_LOG_CAP = 10.0

PI = [-1.0, -1.2, -0.5, -0.7]
OLD_POLICY = [-1.1, -1.0, -0.9, -0.4]
MASK = [1.0, 1.0, 0.0, 0.0]
# Wide clip bounds keep the ratio unclipped so the assertions read the raw value.
CONFIG = GSPOConfig(clip_ratio_low=0.5, clip_ratio_high=0.5)
ACTIVE_SEQ_LOG_RATIO = ((PI[0] - OLD_POLICY[0]) + (PI[1] - OLD_POLICY[1])) / 2


def _datum(mask: list[float]) -> tinker.Datum:
    count = len(mask)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(count))),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=list(range(count)), dtype="int64", shape=[count]
            ),
            "weights": tinker.TensorData(
                data=[int(value) for value in mask], dtype="int64", shape=[count]
            ),
        },
    )


def _run(old_policy: list[float], config: GSPOConfig = CONFIG):
    pi = torch.tensor(PI, requires_grad=True)
    fn = make_gspo_loss_fn(
        advantages=[1.0],
        ref_logprobs=[],
        inf_logprobs=[OLD_POLICY],
        prompt_len=1,
        old_policy_logprobs=[old_policy],
        gspo_config=config,
    )
    loss, metrics = fn([_datum(MASK)], [pi])
    loss.backward()
    return loss.detach(), pi.grad, metrics


def test_sequence_ratio_averages_only_active_positions():
    loss, grad, metrics = _run(OLD_POLICY)
    ratio = math.exp(ACTIVE_SEQ_LOG_RATIO)

    assert metrics["ppo_ratio_mean"] == pytest.approx(ratio)
    assert metrics["gspo_clip_frac"] == pytest.approx(0.0)
    # The two active token losses are averaged within the sequence.
    assert loss.item() == pytest.approx(-ratio)
    assert grad.tolist() == pytest.approx(
        [-ratio / 2.0, -ratio / 2.0, 0.0, 0.0]
    )


def test_token_sum_scales_loss_and_gradient_by_active_response_length():
    mean_loss, mean_grad, _ = _run(OLD_POLICY)
    sum_loss, sum_grad, _ = _run(
        OLD_POLICY,
        GSPOConfig(
            clip_ratio_low=0.5,
            clip_ratio_high=0.5,
            token_reduction="sum",
        ),
    )

    assert sum_loss.item() == pytest.approx(mean_loss.item() * 2)
    assert sum_grad.tolist() == pytest.approx(
        [value * 2 for value in mean_grad.tolist()]
    )


def test_masked_positions_cannot_move_the_sequence_ratio():
    baseline_loss, baseline_grad, _ = _run(OLD_POLICY)
    perturbed_loss, perturbed_grad, _ = _run(OLD_POLICY[:2] + [-40.0, 12.0])

    assert perturbed_loss.item() == pytest.approx(baseline_loss.item())
    assert perturbed_grad.tolist() == pytest.approx(baseline_grad.tolist())


def baseline_gspo_loss(
    policy_logps: torch.Tensor,  # (batch, resp_len)
    sampling_logps: torch.Tensor,  # (batch, resp_len)
    mask: torch.Tensor,  # (batch, resp_len)
    advantages: torch.Tensor,  # (batch, 1)
    epsilon: float,
    token_weights: torch.Tensor | None = None,  # (batch, resp_len)
) -> torch.Tensor:
    """Logprob-level built-in GSPO loss with additive sequence means."""
    policy_logps = policy_logps * mask
    sampling_logps = sampling_logps * mask

    negative_approx_kl = policy_logps - sampling_logps.detach()
    seq_lengths = mask.sum(dim=-1).clamp(min=1)
    negative_approx_kl_seq = (negative_approx_kl * mask).sum(dim=-1) / seq_lengths

    log_seq_importance_ratio = (
        policy_logps
        - policy_logps.detach()
        + negative_approx_kl_seq.detach().unsqueeze(-1)
    )
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=SEQ_RATIO_LOG_CAP)
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(
        seq_importance_ratio, 1 - epsilon, 1 + epsilon
    )
    per_token_loss = torch.maximum(pg_losses1, pg_losses2)
    if token_weights is not None:
        per_token_loss = per_token_loss * token_weights

    per_sequence_loss = (per_token_loss * mask).sum(dim=-1) / seq_lengths
    return per_sequence_loss.sum()


def _expected_tis_weights(
    old_policy: torch.Tensor,  # (batch, resp_len)
    inference: torch.Tensor,  # (batch, resp_len)
    mask: torch.Tensor,  # (batch, resp_len)
    tis_config: TISConfig,
) -> torch.Tensor:
    """Rebuild run_loss_loop's per-token TIS weight: identity where inactive."""
    weights = torch.ones_like(old_policy)
    for row in range(old_policy.shape[0]):
        active = mask[row] > 0.5
        active_weights, _ = compute_tis_weight(
            old_policy[row][active], inference[row][active], tis_config
        )
        weights[row][active] = active_weights
    return weights


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("resp_len", [16, 128])
@pytest.mark.parametrize("prompt_len", [1, 5])
@pytest.mark.parametrize("epsilon", [0.02, 0.05])
@pytest.mark.parametrize("use_weights", [True, False])
def test_matches_train_py_baseline(
    batch_size: int,
    resp_len: int,
    prompt_len: int,
    epsilon: float,
    use_weights: bool,
) -> None:
    """Cookbook GSPO matches built-in GSPO before optimizer normalization."""
    generator = torch.Generator().manual_seed(20260912)
    response_start = prompt_len - 1
    target_len = response_start + resp_len
    shape = (batch_size, target_len)

    pi_full = torch.randn(shape, generator=generator) - 1.0
    old_full = pi_full + 0.05 * torch.randn(shape, generator=generator)
    inference_full = (
        old_full + 0.3 * torch.randn(shape, generator=generator)
        if use_weights
        else old_full.clone()
    )
    mask_full = torch.zeros(shape)
    mask_full[:, response_start:] = (
        torch.rand((batch_size, resp_len), generator=generator) > 0.4
    ).float()
    mask_full[:, response_start] = 1.0  # run_loss_loop skips fully masked samples
    advantages = torch.randn(batch_size, generator=generator).tolist()

    tis_config = TISConfig()
    gspo_config = GSPOConfig(clip_ratio_low=epsilon, clip_ratio_high=epsilon)
    data = [_datum(mask_full[i].tolist()) for i in range(batch_size)]

    def build_loss_fn():
        return make_gspo_loss_fn(
            advantages=advantages,
            ref_logprobs=[],
            inf_logprobs=inference_full.tolist(),
            prompt_len=prompt_len,
            old_policy_logprobs=old_full.tolist(),
            gspo_config=gspo_config,
            tis_config=tis_config,
        )

    sequence_means = []
    for i in range(batch_size):
        pi_row = pi_full[i].clone().requires_grad_()
        loss_fn = make_gspo_loss_fn(
            advantages=[advantages[i]],
            ref_logprobs=[],
            inf_logprobs=[inference_full[i].tolist()],
            prompt_len=prompt_len,
            old_policy_logprobs=[old_full[i].tolist()],
            gspo_config=gspo_config,
            tis_config=tis_config,
        )
        sequence_mean, _ = loss_fn([data[i]], [pi_row])
        sequence_means.append(sequence_mean.detach())

    batched_rows = [pi_full[i].clone().requires_grad_() for i in range(batch_size)]
    batched_loss, metrics = build_loss_fn()(data, batched_rows)
    batched_loss.backward()
    assert batched_loss.item() == pytest.approx(
        sum(value.item() for value in sequence_means), rel=1e-6
    )
    for key in ("ppo_kl", "gspo_clip_frac", "ppo_ratio_mean", "tis/weight_mean"):
        assert key in metrics

    mask = mask_full[:, response_start:]
    policy_logps = pi_full[:, response_start:].clone().requires_grad_()
    token_weights = (
        _expected_tis_weights(
            old_full[:, response_start:],
            inference_full[:, response_start:],
            mask,
            tis_config,
        )
        if use_weights
        else None
    )
    expected = baseline_gspo_loss(
        policy_logps,
        old_full[:, response_start:],
        mask,
        torch.tensor(advantages).unsqueeze(-1),
        epsilon,
        token_weights=token_weights,
    )
    expected.backward()

    torch.testing.assert_close(
        batched_loss.detach(), expected.detach(), rtol=1e-5, atol=1e-6
    )

    for i in range(batch_size):
        torch.testing.assert_close(
            batched_rows[i].grad[response_start:],
            policy_logps.grad[i],
            rtol=1e-5,
            atol=1e-6,
        )
        # Prompt positions are never part of the response slice.
        assert not batched_rows[i].grad[:response_start].any()


def test_reference_logprobs_add_kl_metrics_without_changing_loss() -> None:
    """GSPO's surrogate ignores the reference policy; it only feeds KL metrics."""
    generator = torch.Generator().manual_seed(7)
    pi_values = (torch.randn(12, generator=generator) - 1.0).tolist()
    old_values = (torch.randn(12, generator=generator) - 1.0).tolist()
    ref_values = (torch.randn(12, generator=generator) - 1.0).tolist()
    mask = [0.0, 0.0] + [1.0] * 10

    def run(ref_logprobs: list[list[float]]):
        pi = torch.tensor(pi_values, requires_grad=True)
        loss_fn = make_gspo_loss_fn(
            advantages=[1.5],
            ref_logprobs=ref_logprobs,
            inf_logprobs=[old_values],
            prompt_len=3,
            old_policy_logprobs=[old_values],
        )
        loss, metrics = loss_fn([_datum(mask)], [pi])
        loss.backward()
        return loss.detach(), pi.grad, metrics

    without_loss, without_grad, without_metrics = run([])
    with_loss, with_grad, with_metrics = run([ref_values])

    assert with_loss.item() == pytest.approx(without_loss.item())
    torch.testing.assert_close(with_grad, without_grad)
    assert not {"mean_kl", "ref_kl"}.intersection(without_metrics)
    assert {"mean_kl", "ref_kl"}.issubset(with_metrics)
