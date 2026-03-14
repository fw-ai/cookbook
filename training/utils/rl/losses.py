"""RL loss dispatch and rollout data types."""

from __future__ import annotations

from typing import Any, List, Tuple, Callable
from dataclasses import field, dataclass

import tinker

from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.dro import DROConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.importance_sampling import ISConfig


def _cispo_kernel_config(
    *, cispo_config: CISPOConfig | None = None, **_kw: Any,
) -> tuple[str, dict[str, Any]]:
    cfg = cispo_config or CISPOConfig()
    return "cispo", {
        "clip_low_threshold": 1.0 - cfg.eps_low,
        "clip_high_threshold": 1.0 + cfg.eps_high,
        "ratio_log_cap": cfg.ratio_log_cap,
    }


def _gspo_kernel_config(
    *, gspo_config: GSPOConfig | None = None, **_kw: Any,
) -> tuple[str, dict[str, Any]]:
    cfg = gspo_config or GSPOConfig()
    clip_low = cfg.clip_ratio_low or cfg.clip_ratio
    clip_high = cfg.clip_ratio_high or cfg.clip_ratio
    return "gspo", {
        "clip_low_threshold": 1.0 - clip_low,
        "clip_high_threshold": 1.0 + clip_high,
        "seq_ratio_log_cap": cfg.seq_ratio_log_cap,
    }


def _dapo_kernel_config(
    *, dapo_config: DAPOConfig | None = None, **_kw: Any,
) -> tuple[str, dict[str, Any]]:
    cfg = dapo_config or DAPOConfig()
    config: dict[str, Any] = {
        "clip_low_threshold": 1.0 - cfg.eps_clip,
        "clip_high_threshold": 1.0 + cfg.eps_clip_high,
        "ratio_log_cap": cfg.ratio_log_cap,
    }
    if cfg.eps_clip_c is not None:
        config["eps_clip_c"] = cfg.eps_clip_c
        return "dapo", config
    return "ppo", config


def _grpo_kernel_config(
    *, is_config: ISConfig | None = None, **_kw: Any,
) -> tuple[str, dict[str, Any]]:
    cfg = is_config or ISConfig()
    eps_high = cfg.eps_clip_high or cfg.eps_clip
    return "ppo", {
        "clip_low_threshold": 1.0 - cfg.eps_clip,
        "clip_high_threshold": 1.0 + eps_high,
    }


def _is_kernel_config(
    *, is_config: ISConfig | None = None, **_kw: Any,
) -> tuple[str, dict[str, Any]]:
    cfg = is_config or ISConfig()
    return "importance_sampling", {
        "ratio_log_cap": 20.0,
    }


def _dro_kernel_config(
    *, dro_config: DROConfig | None = None, **_kw: Any,
) -> tuple[str, dict[str, Any]]:
    cfg = dro_config or DROConfig()
    return "dro", {
        "beta": cfg.beta,
    }


BUILTIN_LOSS_REGISTRY: dict[str, Callable[..., tuple[str, dict[str, Any]]]] = {
    "grpo": _grpo_kernel_config,
    "importance_sampling": _is_kernel_config,
    "cispo": _cispo_kernel_config,
    "gspo": _gspo_kernel_config,
    "dapo": _dapo_kernel_config,
    "dro": _dro_kernel_config,
}
"""Mapping from ``policy_loss`` name to a builder that returns
``(kernel_loss_name, kernel_config_dict)`` for the server-side built-in loss
kernel.  Any ``policy_loss`` not in this registry falls back to the two-pass
``forward_backward_custom`` path."""


def get_builtin_loss_config(
    policy_loss: str,
    *,
    dapo_config: DAPOConfig | None = None,
    dro_config: DROConfig | None = None,
    gspo_config: GSPOConfig | None = None,
    cispo_config: CISPOConfig | None = None,
    is_config: ISConfig | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Return ``(kernel_loss_name, kernel_config)`` for a supported built-in loss.

    Returns ``None`` if *policy_loss* is not in :data:`BUILTIN_LOSS_REGISTRY`,
    signalling the caller to fall back to ``forward_backward_custom``.
    """
    builder = BUILTIN_LOSS_REGISTRY.get(policy_loss)
    if builder is None:
        return None
    return builder(
        dapo_config=dapo_config,
        dro_config=dro_config,
        gspo_config=gspo_config,
        cispo_config=cispo_config,
        is_config=is_config,
    )


@dataclass
class PromptGroup:
    """Processed data from one prompt's rollout, ready for training."""

    data: List[tinker.Datum]
    advantages: List[float]
    ref_logprobs: List[List[float]] | None
    prompt_len: int
    rewards: List[float]
    ref_data: List[tinker.Datum] = field(default_factory=list)
    """Reference-only datums (no routing matrices)."""
    inf_logprobs: List[List[float]] = field(default_factory=list)
    completion_lens: List[int] = field(default_factory=list)
    """Per-sample completion lengths in tokens."""
    truncated: List[bool] = field(default_factory=list)
    """Per-sample flag: True if completion hit max_completion_tokens."""
    prompt: list[dict] | None = None
    """Original prompt messages (for trajectory logging)."""
    completions: list[str] | None = None
    """Raw completion texts (for trajectory logging)."""
    row_meta: dict | None = None
    """Dataset row metadata, e.g. ground_truth (for trajectory logging)."""


def combine_prompt_groups(
    groups: List[PromptGroup],
) -> Tuple[List[tinker.Datum], List[float], List[List[float]], List[int], List[List[float]]]:
    """Flatten a list of PromptGroups into combined arrays for a fwd_bwd call.

    Returns (data, advantages, ref_logprobs, prompt_lens, inf_logprobs).
    """
    data: List[tinker.Datum] = []
    advantages: List[float] = []
    ref_logprobs: List[List[float]] = []
    prompt_lens: List[int] = []
    inf_logprobs: List[List[float]] = []

    for pg in groups:
        data.extend(pg.data)
        advantages.extend(pg.advantages)
        if pg.ref_logprobs is not None:
            ref_logprobs.extend(pg.ref_logprobs)
        prompt_lens.extend([pg.prompt_len] * len(pg.data))
        inf_logprobs.extend(pg.inf_logprobs)

    return data, advantages, ref_logprobs, prompt_lens, inf_logprobs


def build_builtin_loss_datums(
    data: List[tinker.Datum],
    advantages: List[float],
    prox_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_lens: List[int],
    is_config: ISConfig | None = None,
) -> List[tinker.Datum]:
    """Build datums with per-token sampling_logprobs and advantages for server-side built-in loss.

    Folds the TIS weight ``exp(prox - inf)`` into per-token advantages so the
    server only sees ``sampling_logprobs`` (= prox_lp) and ``advantages``
    (= advantage * tis_weight * loss_mask).

    Uses ``compute_tis_weight`` for behavioral IS correction and
    ``_get_loss_mask`` for multi-turn tool-call masking (matching the
    pattern in grpo/dapo/cispo).
    """
    import torch
    from training.utils.rl.common import _get_loss_mask
    from training.utils.rl.importance_sampling import compute_tis_weight

    if is_config is None:
        is_config = ISConfig()
    result: List[tinker.Datum] = []
    adv_idx = 0

    for i, datum in enumerate(data):
        target_data = datum.loss_fn_inputs["target_tokens"]
        target_tokens = list(target_data.data)
        n_tokens = len(target_tokens)
        response_start = max(0, prompt_lens[i] - 1)
        prox_lp = list(prox_logprobs[i])
        inf_lp = list(inf_logprobs[i]) if inf_logprobs else [0.0] * len(prox_lp)

        resp_len = max(0, n_tokens - response_start)
        if resp_len > 0 and inf_lp:
            resp_prox = torch.tensor(prox_lp[response_start:response_start + resp_len], dtype=torch.float32)
            resp_inf = torch.tensor(inf_lp[response_start:response_start + resp_len], dtype=torch.float32)
            tis_weight, _ = compute_tis_weight(resp_prox, resp_inf, is_config)
        else:
            tis_weight = torch.ones(resp_len, dtype=torch.float32)

        loss_mask = _get_loss_mask(
            datum, response_start, resp_len, dtype=torch.float32, device=torch.device("cpu"),
        )

        per_token_adv = [0.0] * response_start
        adv_val = advantages[adv_idx] if adv_idx < len(advantages) else 0.0
        for r in range(resp_len):
            per_token_adv.append(float(adv_val * tis_weight[r].item() * loss_mask[r].item()))

        slp_padded = prox_lp[:n_tokens] if len(prox_lp) >= n_tokens else prox_lp + [0.0] * (n_tokens - len(prox_lp))
        new_datum = tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=target_tokens, dtype="int64", shape=[n_tokens],
                ),
                "logprobs": tinker.TensorData(
                    data=slp_padded, dtype="float32", shape=[n_tokens],
                ),
                "advantages": tinker.TensorData(
                    data=per_token_adv, dtype="float32", shape=[n_tokens],
                ),
            },
        )
        result.append(new_datum)
        adv_idx += 1

    return result


LossFnBuilder = Callable[..., Any]
"""Signature for the loss builder returned by ``build_loss_fn``."""


def build_loss_fn(
    policy_loss: str,
    kl_beta: float,
    dapo_config: Any = None,
    dro_config: Any = None,
    gspo_config: Any = None,
    cispo_config: Any = None,
    is_config: ISConfig | None = None,
) -> LossFnBuilder:
    """Create a loss builder that dispatches to grpo/dapo/gspo/cispo.

    Returns a callable:
      (advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs) -> loss_fn
    """
    if is_config is None:
        is_config = ISConfig()

    from training.utils.rl.dapo import make_dapo_loss_fn
    from training.utils.rl.dro import make_dro_loss_fn
    from training.utils.rl.grpo import make_grpo_loss_fn
    from training.utils.rl.gspo import make_gspo_loss_fn
    from training.utils.rl.cispo import make_cispo_loss_fn
    from training.utils.rl.is_loss import make_is_loss_fn

    def build(
        advantages: List[float],
        ref_logprobs: List[List[float]],
        prompt_lens: List[int],
        inf_logprobs: List[List[float]],
        prox_logprobs: List[List[float]],
    ) -> Any:
        if policy_loss == "importance_sampling":
            return make_is_loss_fn(
                advantages, ref_logprobs, inf_logprobs,
                prompt_lens, prox_logprobs,
                is_config=is_config,
            )
        if policy_loss == "dapo":
            return make_dapo_loss_fn(
                advantages, ref_logprobs, inf_logprobs,
                prompt_lens, prox_logprobs,
                dapo_config, is_config=is_config,
            )
        if policy_loss == "dro":
            return make_dro_loss_fn(
                advantages, ref_logprobs, inf_logprobs,
                prompt_lens, prox_logprobs,
                dro_config, is_config=is_config,
            )
        if policy_loss == "gspo":
            return make_gspo_loss_fn(
                advantages, ref_logprobs, inf_logprobs,
                prompt_lens, prox_logprobs,
                gspo_config, is_config=is_config,
            )
        if policy_loss == "cispo":
            return make_cispo_loss_fn(
                advantages, ref_logprobs, inf_logprobs,
                prompt_lens, prox_logprobs,
                cispo_config, is_config=is_config,
            )
        if policy_loss == "grpo":
            return make_grpo_loss_fn(
                advantages, ref_logprobs,
                prompt_lens, inf_logprobs=inf_logprobs,
                prox_logprobs=prox_logprobs,
                kl_beta=kl_beta, is_config=is_config,
            )
        raise ValueError(
            f"Unsupported policy_loss '{policy_loss}'. "
            f"Expected one of: grpo, importance_sampling, dapo, dro, gspo, cispo."
        )

    return build
