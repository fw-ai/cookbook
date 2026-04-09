"""GSPO and GSPO-token losses for RL training.

`make_gspo_loss_fn` implements the sequence-level GSPO objective from Eq. 5/10
of https://arxiv.org/abs/2507.18071.

`make_gspo_token_loss_fn` implements the GSPO-token objective from Eq. 13/17,
including the stop-gradient tokenized ratio construction. As in the paper,
GSPO-token reduces to sequence-level GSPO when every token in a response shares
the same advantage and there is no additional token-wise reweighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens, run_loss_loop
from training.utils.rl.spec import LossSpec
from training.utils.rl.tis import TISConfig


@dataclass
class GSPOConfig:
    """GSPO clipping configuration.

    ``clip_ratio`` is the symmetric fallback clip epsilon.
    Set ``clip_ratio_low`` / ``clip_ratio_high`` for asymmetric clipping.
    """

    clip_ratio: float = 0.2
    clip_ratio_low: float | None = None
    clip_ratio_high: float | None = None
    seq_ratio_log_cap: float = 10.0
    kl_beta: float = 0.001


Advantage = float | List[float]


def _clip_bounds(config: GSPOConfig) -> tuple[float, float]:
    clip_low = (
        config.clip_ratio if config.clip_ratio_low is None else config.clip_ratio_low
    )
    clip_high = (
        config.clip_ratio if config.clip_ratio_high is None else config.clip_ratio_high
    )
    return clip_low, clip_high


def _sequence_ratio(
    resp_pi: torch.Tensor, resp_prox: torch.Tensor, config: GSPOConfig
) -> torch.Tensor:
    seq_log_ratio = (resp_pi - resp_prox).mean()
    seq_log_ratio = torch.clamp(seq_log_ratio, max=config.seq_ratio_log_cap)
    return torch.exp(seq_log_ratio)


def _token_ratio(
    resp_pi: torch.Tensor, resp_prox: torch.Tensor, config: GSPOConfig
) -> torch.Tensor:
    seq_log_ratio = (resp_pi - resp_prox).mean()
    log_seq_ratio = resp_pi - resp_pi.detach() + seq_log_ratio.detach()
    log_seq_ratio = torch.clamp(log_seq_ratio, max=config.seq_ratio_log_cap)
    return torch.exp(log_seq_ratio)


def _sequence_advantage(advantage: torch.Tensor) -> torch.Tensor:
    if advantage.ndim == 0:
        return advantage
    if advantage.numel() != 1:
        raise ValueError(
            "Sequence-level GSPO expects one scalar advantage per response."
        )
    return advantage.reshape(())


def _token_advantages(advantage: torch.Tensor, resp_len: int) -> torch.Tensor:
    if advantage.ndim == 0:
        return advantage.expand(resp_len)
    if advantage.numel() != resp_len:
        raise ValueError(
            f"GSPO-token expected {resp_len} token advantages, got {advantage.numel()}."
        )
    return advantage.reshape(resp_len)


def make_gspo_loss_fn(
    advantages: List[Advantage],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    prox_logprobs: List[List[float]],
    gspo_config: GSPOConfig | None = None,
    tis_config: TISConfig | None = None,
) -> ...:
    """Build the sequence-level GSPO loss closure."""
    if gspo_config is None:
        gspo_config = GSPOConfig()
    if tis_config is None:
        tis_config = TISConfig()
    clip_low, clip_high = _clip_bounds(gspo_config)
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))
    has_reference = bool(ref_logprobs)

    def policy_fn(ctx):
        seq_ratio = _sequence_ratio(
            resp_pi=ctx.resp_pi,
            resp_prox=ctx.resp_prox,
            config=gspo_config,
        )
        clipped_seq_ratio = torch.clamp(
            seq_ratio,
            min=1.0 - clip_low,
            max=1.0 + clip_high,
        )
        clip_frac = float((clipped_seq_ratio.detach() != seq_ratio.detach()).item())
        ratio_mean = seq_ratio.detach().item()

        adv_t = _sequence_advantage(ctx.adv)
        surr1 = -seq_ratio * adv_t
        surr2 = -clipped_seq_ratio * adv_t
        per_token_loss = torch.maximum(surr1, surr2) * ctx.tis_weight * ctx.resp_mask
        return per_token_loss, {"clip_frac": clip_frac, "ratio_mean": ratio_mean}

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(
            advantages,
            ref_logprobs,
            inf_logprobs,
            prompt_lens,
            prox_logprobs,
            tis_config,
            data,
            logprobs_list,
            "gspo",
            policy_fn,
        )
        metrics = dict(result.base_metrics)
        if not has_reference:
            metrics.pop("mean_kl", None)
        ns = result.n_samples
        metrics["gspo_clip_frac"] = result.extra_sums.get("clip_frac", 0.0) / ns
        metrics["ppo_ratio_mean"] = result.extra_sums.get("ratio_mean", 0.0) / ns
        return result.total_loss, metrics

    return loss_fn


def make_gspo_token_loss_fn(
    advantages: List[Advantage],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    prox_logprobs: List[List[float]],
    gspo_config: GSPOConfig | None = None,
    tis_config: TISConfig | None = None,
) -> ...:
    """Build the GSPO-token loss closure."""
    if gspo_config is None:
        gspo_config = GSPOConfig()
    if tis_config is None:
        tis_config = TISConfig()
    clip_low, clip_high = _clip_bounds(gspo_config)
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))
    has_reference = bool(ref_logprobs)

    def policy_fn(ctx):
        token_ratio = _token_ratio(
            resp_pi=ctx.resp_pi,
            resp_prox=ctx.resp_prox,
            config=gspo_config,
        )
        clipped_token_ratio = torch.clamp(
            token_ratio,
            min=1.0 - clip_low,
            max=1.0 + clip_high,
        )
        clip_frac = (
            (clipped_token_ratio.detach() != token_ratio.detach())
            .float()
            .mean()
            .item()
        )
        ratio_mean = token_ratio.detach().mean().item()

        adv_t = _token_advantages(ctx.adv, resp_len=len(ctx.resp_pi))
        surr1 = -token_ratio * adv_t
        surr2 = -clipped_token_ratio * adv_t
        per_token_loss = (
            torch.maximum(surr1, surr2) * ctx.tis_weight * ctx.resp_mask
        )
        return per_token_loss, {"clip_frac": clip_frac, "ratio_mean": ratio_mean}

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(
            advantages,
            ref_logprobs,
            inf_logprobs,
            prompt_lens,
            prox_logprobs,
            tis_config,
            data,
            logprobs_list,
            "gspo-token",
            policy_fn,
        )
        metrics = dict(result.base_metrics)
        if not has_reference:
            metrics.pop("mean_kl", None)
        ns = result.n_samples
        metrics["gspo_clip_frac"] = result.extra_sums.get("clip_frac", 0.0) / ns
        metrics["ppo_ratio_mean"] = result.extra_sums.get("ratio_mean", 0.0) / ns
        return result.total_loss, metrics

    return loss_fn


def _builtin_config(
    *, gspo_config: GSPOConfig | None = None, **_kw: Any,
) -> tuple[str, dict[str, Any]]:
    cfg = gspo_config or GSPOConfig()
    clip_low, clip_high = _clip_bounds(cfg)
    return "gspo", {
        "clip_low_threshold": 1.0 - clip_low,
        "clip_high_threshold": 1.0 + clip_high,
        "seq_ratio_log_cap": cfg.seq_ratio_log_cap,
    }


def _client_loss_factory(
    *,
    advantages: List[Advantage],
    ref_logprobs: List[List[float]],
    prompt_lens: List[int],
    inf_logprobs: List[List[float]],
    prox_logprobs: List[List[float]],
    gspo_config: GSPOConfig | None,
    tis_config: TISConfig,
    **_kw: Any,
) -> Any:
    return make_gspo_loss_fn(
        advantages,
        ref_logprobs,
        inf_logprobs,
        prompt_lens,
        prox_logprobs,
        gspo_config,
        tis_config=tis_config,
    )


def _client_token_loss_factory(
    *,
    advantages: List[Advantage],
    ref_logprobs: List[List[float]],
    prompt_lens: List[int],
    inf_logprobs: List[List[float]],
    prox_logprobs: List[List[float]],
    gspo_config: GSPOConfig | None,
    tis_config: TISConfig,
    **_kw: Any,
) -> Any:
    return make_gspo_token_loss_fn(
        advantages,
        ref_logprobs,
        inf_logprobs,
        prompt_lens,
        prox_logprobs,
        gspo_config,
        tis_config=tis_config,
    )


LOSS_SPEC = LossSpec(
    name="gspo",
    client_loss_factory=_client_loss_factory,
    builtin_config_builder=_builtin_config,
)

GSPO_TOKEN_LOSS_SPEC = LossSpec(
    name="gspo-token",
    client_loss_factory=_client_token_loss_factory,
)
