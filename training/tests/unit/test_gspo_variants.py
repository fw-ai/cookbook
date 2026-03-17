from __future__ import annotations

import pytest
import torch

from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.losses import build_loss_fn
from training.utils.rl.tis import TISConfig, compute_tis_weight


def _example_inputs() -> dict:
    return {
        "pi": torch.tensor(
            [-0.2, -0.7, -0.4, -1.1, -0.6],
            dtype=torch.float32,
        ),
        "prox": [-0.2, -0.9, -0.5, -0.8, -0.3],
        "inf": [-0.2, -0.9, -0.5, -0.8, -0.3],
        "ref": [-0.2, -0.95, -0.45, -0.85, -0.25],
        "prompt_len": 2,
    }


def _reference_gspo_sequence_loss(
    pi_logprobs: torch.Tensor,
    prox_logprobs: list[float],
    inf_logprobs: list[float],
    advantage: float,
    prompt_len: int,
    config: GSPOConfig,
    tis_config: TISConfig,
) -> torch.Tensor:
    response_start = prompt_len - 1
    resp_pi = pi_logprobs[response_start:]
    device = resp_pi.device
    dtype = resp_pi.dtype
    resp_prox = torch.tensor(prox_logprobs[response_start:], dtype=dtype, device=device)
    resp_inf = torch.tensor(inf_logprobs[response_start:], dtype=dtype, device=device)

    clip_low = (
        config.clip_ratio if config.clip_ratio_low is None else config.clip_ratio_low
    )
    clip_high = (
        config.clip_ratio if config.clip_ratio_high is None else config.clip_ratio_high
    )
    seq_log_ratio = torch.clamp(
        (resp_pi - resp_prox).mean(),
        max=config.seq_ratio_log_cap,
    )
    seq_ratio = torch.exp(seq_log_ratio)
    clipped_ratio = torch.clamp(seq_ratio, min=1.0 - clip_low, max=1.0 + clip_high)
    tis_weight, _ = compute_tis_weight(
        resp_prox=resp_prox,
        resp_inf=resp_inf,
        config=tis_config,
    )
    loss = torch.maximum(-seq_ratio * advantage, -clipped_ratio * advantage)
    return (loss * tis_weight).sum()


def _reference_gspo_token_loss(
    pi_logprobs: torch.Tensor,
    prox_logprobs: list[float],
    inf_logprobs: list[float],
    advantages: float | list[float],
    prompt_len: int,
    config: GSPOConfig,
    tis_config: TISConfig,
) -> torch.Tensor:
    response_start = prompt_len - 1
    resp_pi = pi_logprobs[response_start:]
    device = resp_pi.device
    dtype = resp_pi.dtype
    resp_prox = torch.tensor(prox_logprobs[response_start:], dtype=dtype, device=device)
    resp_inf = torch.tensor(inf_logprobs[response_start:], dtype=dtype, device=device)

    if isinstance(advantages, list):
        adv_t = torch.tensor(advantages, dtype=dtype, device=device)
    else:
        adv_t = torch.full(
            (len(resp_pi),),
            float(advantages),
            dtype=dtype,
            device=device,
        )

    clip_low = (
        config.clip_ratio if config.clip_ratio_low is None else config.clip_ratio_low
    )
    clip_high = (
        config.clip_ratio if config.clip_ratio_high is None else config.clip_ratio_high
    )
    seq_log_ratio = (resp_pi - resp_prox).mean()
    token_log_ratio = resp_pi - resp_pi.detach() + seq_log_ratio.detach()
    token_log_ratio = torch.clamp(token_log_ratio, max=config.seq_ratio_log_cap)
    token_ratio = torch.exp(token_log_ratio)
    clipped_ratio = torch.clamp(token_ratio, min=1.0 - clip_low, max=1.0 + clip_high)
    tis_weight, _ = compute_tis_weight(
        resp_prox=resp_prox,
        resp_inf=resp_inf,
        config=tis_config,
    )
    per_token_loss = (
        torch.maximum(-token_ratio * adv_t, -clipped_ratio * adv_t) * tis_weight
    )
    return per_token_loss.sum()


def test_gspo_sequence_matches_reference_loss_and_gradient() -> None:
    inputs = _example_inputs()
    config = GSPOConfig(clip_ratio=0.15, clip_ratio_low=0.15, clip_ratio_high=0.2)
    tis_config = TISConfig(cap=100.0)

    pi = inputs["pi"].clone().requires_grad_(True)
    builder = build_loss_fn(
        policy_loss="gspo",
        kl_beta=0.0,
        gspo_config=config,
        tis_config=tis_config,
    )
    loss_fn = builder(
        [1.25],
        [inputs["ref"]],
        [inputs["prompt_len"]],
        [inputs["inf"]],
        [inputs["prox"]],
    )
    loss, _ = loss_fn([], [pi])

    ref_pi = inputs["pi"].clone().requires_grad_(True)
    ref_loss = _reference_gspo_sequence_loss(
        pi_logprobs=ref_pi,
        prox_logprobs=inputs["prox"],
        inf_logprobs=inputs["inf"],
        advantage=1.25,
        prompt_len=inputs["prompt_len"],
        config=config,
        tis_config=tis_config,
    )

    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-5)
    loss.backward()
    ref_loss.backward()
    torch.testing.assert_close(pi.grad, ref_pi.grad, rtol=1e-5, atol=1e-5)


def test_gspo_token_matches_reference_loss_and_gradient() -> None:
    inputs = _example_inputs()
    config = GSPOConfig(clip_ratio=0.15, clip_ratio_low=0.15, clip_ratio_high=0.2)
    tis_config = TISConfig(cap=100.0)

    pi = inputs["pi"].clone().requires_grad_(True)
    builder = build_loss_fn(
        policy_loss="gspo-token",
        kl_beta=0.0,
        gspo_config=config,
        tis_config=tis_config,
    )
    loss_fn = builder(
        [1.25],
        [inputs["ref"]],
        [inputs["prompt_len"]],
        [inputs["inf"]],
        [inputs["prox"]],
    )
    loss, _ = loss_fn([], [pi])

    ref_pi = inputs["pi"].clone().requires_grad_(True)
    ref_loss = _reference_gspo_token_loss(
        pi_logprobs=ref_pi,
        prox_logprobs=inputs["prox"],
        inf_logprobs=inputs["inf"],
        advantages=1.25,
        prompt_len=inputs["prompt_len"],
        config=config,
        tis_config=tis_config,
    )

    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-5)
    loss.backward()
    ref_loss.backward()
    torch.testing.assert_close(pi.grad, ref_pi.grad, rtol=1e-5, atol=1e-5)


def test_gspo_and_gspo_token_match_for_scalar_advantages_with_uniform_tis() -> None:
    inputs = _example_inputs()
    config = GSPOConfig(clip_ratio=0.15, clip_ratio_low=0.15, clip_ratio_high=0.2)
    tis_config = TISConfig(cap=100.0)

    seq_pi = inputs["pi"].clone().requires_grad_(True)
    tok_pi = inputs["pi"].clone().requires_grad_(True)

    seq_builder = build_loss_fn(
        policy_loss="gspo",
        kl_beta=0.0,
        gspo_config=config,
        tis_config=tis_config,
    )
    tok_builder = build_loss_fn(
        policy_loss="gspo-token",
        kl_beta=0.0,
        gspo_config=config,
        tis_config=tis_config,
    )

    seq_loss_fn = seq_builder(
        [0.5],
        [inputs["ref"]],
        [inputs["prompt_len"]],
        [inputs["inf"]],
        [inputs["prox"]],
    )
    tok_loss_fn = tok_builder(
        [0.5],
        [inputs["ref"]],
        [inputs["prompt_len"]],
        [inputs["inf"]],
        [inputs["prox"]],
    )

    seq_loss, _ = seq_loss_fn([], [seq_pi])
    tok_loss, _ = tok_loss_fn([], [tok_pi])

    torch.testing.assert_close(seq_loss, tok_loss, rtol=1e-5, atol=1e-5)
    seq_loss.backward()
    tok_loss.backward()
    torch.testing.assert_close(seq_pi.grad, tok_pi.grad, rtol=1e-5, atol=1e-5)


def test_gspo_and_gspo_token_can_have_different_gradients_with_token_level_tis() -> (
    None
):
    inputs = _example_inputs()
    inputs["inf"] = [-0.2, -1.05, -0.25, -1.0, -0.55]
    config = GSPOConfig(clip_ratio=0.15, clip_ratio_low=0.15, clip_ratio_high=0.2)
    tis_config = TISConfig(cap=100.0, level="token")

    seq_pi = inputs["pi"].clone().requires_grad_(True)
    tok_pi = inputs["pi"].clone().requires_grad_(True)

    seq_builder = build_loss_fn(
        policy_loss="gspo",
        kl_beta=0.0,
        gspo_config=config,
        tis_config=tis_config,
    )
    tok_builder = build_loss_fn(
        policy_loss="gspo-token",
        kl_beta=0.0,
        gspo_config=config,
        tis_config=tis_config,
    )

    seq_loss_fn = seq_builder(
        [0.5],
        [inputs["ref"]],
        [inputs["prompt_len"]],
        [inputs["inf"]],
        [inputs["prox"]],
    )
    tok_loss_fn = tok_builder(
        [0.5],
        [inputs["ref"]],
        [inputs["prompt_len"]],
        [inputs["inf"]],
        [inputs["prox"]],
    )

    seq_loss, _ = seq_loss_fn([], [seq_pi])
    tok_loss, _ = tok_loss_fn([], [tok_pi])

    torch.testing.assert_close(seq_loss, tok_loss, rtol=1e-5, atol=1e-5)
    seq_loss.backward()
    tok_loss.backward()
    assert not torch.allclose(seq_pi.grad, tok_pi.grad)


def test_gspo_token_supports_per_token_advantages() -> None:
    inputs = _example_inputs()
    config = GSPOConfig(clip_ratio=0.15, clip_ratio_low=0.15, clip_ratio_high=0.2)
    tis_config = TISConfig(cap=100.0)
    token_advantages = [1.0, 0.5, -0.25, 0.75]

    pi = inputs["pi"].clone().requires_grad_(True)
    builder = build_loss_fn(
        policy_loss="gspo-token",
        kl_beta=0.0,
        gspo_config=config,
        tis_config=tis_config,
    )
    loss_fn = builder(
        [token_advantages],
        [inputs["ref"]],
        [inputs["prompt_len"]],
        [inputs["inf"]],
        [inputs["prox"]],
    )
    loss, _ = loss_fn([], [pi])

    ref_pi = inputs["pi"].clone().requires_grad_(True)
    ref_loss = _reference_gspo_token_loss(
        pi_logprobs=ref_pi,
        prox_logprobs=inputs["prox"],
        inf_logprobs=inputs["inf"],
        advantages=token_advantages,
        prompt_len=inputs["prompt_len"],
        config=config,
        tis_config=tis_config,
    )

    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-5)
    loss.backward()
    ref_loss.backward()
    torch.testing.assert_close(pi.grad, ref_pi.grad, rtol=1e-5, atol=1e-5)


def test_gspo_rejects_per_token_advantages() -> None:
    inputs = _example_inputs()
    builder = build_loss_fn(policy_loss="gspo", kl_beta=0.0, gspo_config=GSPOConfig())
    loss_fn = builder(
        [[1.0, 0.5, -0.25, 0.75]],
        [inputs["ref"]],
        [inputs["prompt_len"]],
        [inputs["inf"]],
        [inputs["prox"]],
    )

    with pytest.raises(
        ValueError,
        match="Sequence-level GSPO expects one scalar advantage",
    ):
        loss_fn([], [inputs["pi"].clone().requires_grad_(True)])


@pytest.mark.parametrize("policy_loss", ["gspo", "gspo-token"])
def test_gspo_variants_omit_mean_kl_without_reference(policy_loss: str) -> None:
    inputs = _example_inputs()
    builder = build_loss_fn(
        policy_loss=policy_loss,
        kl_beta=0.0,
        gspo_config=GSPOConfig(),
        tis_config=TISConfig(cap=100.0),
    )
    loss_fn = builder(
        [0.5],
        [],
        [inputs["prompt_len"]],
        [inputs["inf"]],
        [inputs["prox"]],
    )

    _, metrics = loss_fn([], [inputs["pi"].clone().requires_grad_(True)])

    assert "mean_kl" not in metrics
