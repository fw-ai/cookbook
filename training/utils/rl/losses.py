"""RL loss dispatch, builtin resolution, and rollout data types."""

from __future__ import annotations

from typing import Any, List, Tuple, Callable
from dataclasses import field, dataclass

import tinker

from training.utils.rl.cispo import CISPOConfig, LOSS_SPEC as CISPO_LOSS_SPEC
from training.utils.rl.dapo import DAPOConfig, LOSS_SPEC as DAPO_LOSS_SPEC
from training.utils.rl.dro import DROConfig, LOSS_SPEC as DRO_LOSS_SPEC
from training.utils.rl.gspo import GSPOConfig, LOSS_SPEC as GSPO_LOSS_SPEC
from training.utils.rl.grpo import LOSS_SPEC as GRPO_LOSS_SPEC
from training.utils.rl.is_loss import LOSS_SPEC as IS_LOSS_SPEC
from training.utils.rl.reinforce import LOSS_SPEC as REINFORCE_LOSS_SPEC
from training.utils.rl.spec import LossSpec
from training.utils.rl.tis import TISConfig


LOSS_REGISTRY: dict[str, LossSpec] = {
    spec.name: spec
    for spec in (
        GRPO_LOSS_SPEC,
        IS_LOSS_SPEC,
        DAPO_LOSS_SPEC,
        DRO_LOSS_SPEC,
        GSPO_LOSS_SPEC,
        CISPO_LOSS_SPEC,
        REINFORCE_LOSS_SPEC,
    )
}
"""Single source of truth for both RL execution paths.

Each :class:`~training.utils.rl.spec.LossSpec` can provide:
- ``builtin_config_builder`` for the server-side ``forward_backward(...)`` path
- ``client_loss_factory`` for the client-side ``forward_backward_custom(...)`` path

Losses may support both paths, or may be intentionally registered as
client-side-only by leaving ``builtin_config_builder=None``.
"""

SUPPORTED_POLICY_LOSSES: tuple[str, ...] = tuple(LOSS_REGISTRY)


def _supported_policy_losses_text() -> str:
    return ", ".join(SUPPORTED_POLICY_LOSSES)


def resolve_builtin_loss(
    policy_loss: str,
    profile: Any | None = None,
    *,
    dapo_config: DAPOConfig | None = None,
    dro_config: DROConfig | None = None,
    gspo_config: GSPOConfig | None = None,
    cispo_config: CISPOConfig | None = None,
    ratio_log_cap: float = 20.0,
    eps_clip: float = 0.2,
    eps_clip_high: float | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Resolve the builtin server-side loss kernel for a policy loss.

    Returns ``None`` when *policy_loss* is registered as client-side-only, or
    when it otherwise has no builtin kernel config. In both cases the caller
    should fall back to ``forward_backward_custom(...)``.

    Raises ``ValueError`` when the current *profile* cannot use builtin losses
    (for example, PP > 1), instead of silently falling back.
    """
    spec = LOSS_REGISTRY.get(policy_loss)
    if spec is None or spec.builtin_config_builder is None:
        return None

    if profile is not None:
        pp = getattr(profile, "pipeline_parallelism", 1)
        if pp > 1:
            raise ValueError(
                f"Pipeline parallelism (PP={pp}) is not supported with server-side "
                f"built-in loss '{policy_loss}'. Use a training shape with PP=1, "
                f"or use a custom policy_loss (which falls back to two-pass)."
            )

    return spec.builtin_config_builder(
        dapo_config=dapo_config,
        dro_config=dro_config,
        gspo_config=gspo_config,
        cispo_config=cispo_config,
        ratio_log_cap=ratio_log_cap,
        eps_clip=eps_clip,
        eps_clip_high=eps_clip_high,
    )


def check_builtin_loss_eligibility(
    policy_loss: str,
    profile: Any | None,
) -> None:
    """Reject unsupported parallelism configurations early.

    Must be called before rollout generation to avoid wasted work.

    Raises ``ValueError`` when:
    - Any builtin loss is used with PP > 1 (server kernels don't support PP)
    - GSPO + TP/CP is caught server-side only (profile doesn't expose TP/CP)
    """
    resolve_builtin_loss(policy_loss, profile)


def get_builtin_loss_config(
    policy_loss: str,
    *,
    dapo_config: DAPOConfig | None = None,
    dro_config: DROConfig | None = None,
    gspo_config: GSPOConfig | None = None,
    cispo_config: CISPOConfig | None = None,
    ratio_log_cap: float = 20.0,
    eps_clip: float = 0.2,
    eps_clip_high: float | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Return ``(kernel_loss_name, kernel_config)`` for a supported built-in loss.

    Returns ``None`` for losses that are registered as client-side-only, or
    that otherwise have no builtin kernel config.
    """
    return resolve_builtin_loss(
        policy_loss,
        None,
        dapo_config=dapo_config,
        dro_config=dro_config,
        gspo_config=gspo_config,
        cispo_config=cispo_config,
        ratio_log_cap=ratio_log_cap,
        eps_clip=eps_clip,
        eps_clip_high=eps_clip_high,
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
    generation_step: int | None = None
    """Training step when this group's rollout was submitted (async mode)."""
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
    tis_config: TISConfig | None = None,
    policy_loss: str = "rl_loss",
) -> List[tinker.Datum]:
    """Build datums with per-token sampling_logprobs and advantages for server-side built-in loss.

    Folds the TIS weight ``exp(prox - inf)`` into per-token advantages so the
    server only sees ``sampling_logprobs`` (= prox_lp) and ``advantages``
    (= advantage * tis_weight * loss_mask).

    Uses ``compute_tis_weight`` for behavioral TIS correction and
    ``_get_loss_mask`` for multi-turn tool-call masking.
    """
    import torch
    from training.utils.rl.common import _get_loss_mask, validate_inference_logprobs_for_sample
    from training.utils.rl.tis import compute_tis_weight

    if tis_config is None:
        tis_config = TISConfig()
    result: List[tinker.Datum] = []
    adv_idx = 0

    for i, datum in enumerate(data):
        target_data = datum.loss_fn_inputs["target_tokens"]
        target_tokens = list(target_data.data)
        n_tokens = len(target_tokens)
        response_start = max(0, prompt_lens[i] - 1)
        prox_lp = list(prox_logprobs[i])
        inf_lp = list(inf_logprobs[i]) if i < len(inf_logprobs) else []

        resp_len = max(0, n_tokens - response_start)
        loss_mask = _get_loss_mask(
            datum, response_start, resp_len, dtype=torch.float32, device=torch.device("cpu"),
        )
        active_count = int((loss_mask > 0.5).sum().item())

        if resp_len > 0 and active_count > 0:
            validate_inference_logprobs_for_sample(
                policy_loss, i, inf_lp, response_start + resp_len,
            )
            resp_prox = torch.tensor(prox_lp[response_start:response_start + resp_len], dtype=torch.float32)
            resp_inf = torch.tensor(inf_lp[response_start:response_start + resp_len], dtype=torch.float32)
            tis_weight, _ = compute_tis_weight(resp_prox, resp_inf, tis_config)
        else:
            tis_weight = torch.ones(resp_len, dtype=torch.float32)

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


ClientLossBuilder = Callable[..., Any]
"""Signature for the client-side loss builder used by ``forward_backward_custom``."""


def build_loss_fn(
    policy_loss: str,
    kl_beta: float,
    dapo_config: Any = None,
    dro_config: Any = None,
    gspo_config: Any = None,
    cispo_config: Any = None,
    tis_config: TISConfig | None = None,
    ratio_log_cap: float = 20.0,
    eps_clip: float = 0.2,
    eps_clip_high: float | None = None,
) -> ClientLossBuilder:
    """Create the client-side loss builder for one registered RL policy loss.

    The returned callable is only used on the
    ``forward_backward_custom(...)`` path. Builtin server-side kernels are
    resolved separately by :func:`resolve_builtin_loss`. Losses registered with
    ``builtin_config_builder=None`` always take this client-side path.

    Returns a callable:
    ``(advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs) -> loss_fn``
    """
    if tis_config is None:
        tis_config = TISConfig()
    spec = LOSS_REGISTRY.get(policy_loss)

    def build(
        advantages: List[float],
        ref_logprobs: List[List[float]],
        prompt_lens: List[int],
        inf_logprobs: List[List[float]],
        prox_logprobs: List[List[float]],
    ) -> Any:
        if spec is None:
            supported = _supported_policy_losses_text()
            raise ValueError(
                f"Unsupported policy_loss '{policy_loss}'. "
                f"Expected one of: {supported}."
            )
        return spec.client_loss_factory(
            advantages=advantages,
            ref_logprobs=ref_logprobs,
            prompt_lens=prompt_lens,
            inf_logprobs=inf_logprobs,
            prox_logprobs=prox_logprobs,
            kl_beta=kl_beta,
            dapo_config=dapo_config,
            dro_config=dro_config,
            gspo_config=gspo_config,
            cispo_config=cispo_config,
            tis_config=tis_config,
            ratio_log_cap=ratio_log_cap,
            eps_clip=eps_clip,
            eps_clip_high=eps_clip_high,
        )

    return build
