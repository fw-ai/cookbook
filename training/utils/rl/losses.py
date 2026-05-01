"""RL loss dispatch, builtin resolution, and rollout data types."""

from __future__ import annotations

from typing import Any, List, Literal, Protocol, Tuple, Callable
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


PolicyLoss = Literal[
    "grpo",
    "importance_sampling",
    "dapo",
    "dro",
    "gspo",
    "cispo",
    "reinforce",
]
"""Names of registered RL policy losses.

Kept in lockstep with ``LOSS_REGISTRY`` keys; a startup assertion below
catches drift if a new ``LossSpec`` is registered without updating this
``Literal``.
"""


LossPath = Literal["builtin", "client"]
"""Which forward/backward path the recipe uses.

- ``"builtin"`` -- server-side ``forward_backward(...)`` with a fused kernel
  (PPO, GSPO, etc.). Faster, but the kernel sees only
  ``(target_tokens, logprobs, advantages)`` so KL penalties (``kl_beta>0``)
  cannot be applied; pipeline parallelism > 1 is also unsupported.
- ``"client"`` -- client-side ``forward_backward_custom(...)`` with the
  Python loss closure. Always works, slower because of an extra forward pass
  for old-policy logprobs.

The choice is **explicit**: ``validate_loss_path`` raises with an
actionable message if the user picked ``"builtin"`` in a configuration that
forbids it, instead of silently falling back to client-side. This avoids
the historical footgun where setting ``kl_beta > 0`` would magically route
to the client path -- a behavior that broke once anyone reordered the gate
or introduced a new builtin kernel that consumed ref logprobs.
"""


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

# Drift guard: PolicyLoss Literal must list every registered loss.
import typing as _typing  # noqa: E402
assert set(_typing.get_args(PolicyLoss)) == set(LOSS_REGISTRY), (
    "PolicyLoss Literal is out of sync with LOSS_REGISTRY. "
    f"Literal={set(_typing.get_args(PolicyLoss))!r}, "
    f"registry={set(LOSS_REGISTRY)!r}."
)


class LossArgs(Protocol):
    """Loss-related fields ``build_loss_fn`` reads off the recipe Config.

    Recipe ``Config`` dataclasses naturally implement this Protocol when
    they expose the listed fields at the top level. Tests and external
    callers can use :class:`LossConfig` (concrete dataclass below) when
    they don't already have a Config to pass.
    """

    policy_loss: PolicyLoss
    loss_path: LossPath
    kl_beta: float
    eps_clip: float
    eps_clip_high: float | None
    ratio_log_cap: float
    dapo: DAPOConfig
    dro: DROConfig
    gspo: GSPOConfig
    cispo: CISPOConfig
    tis: TISConfig


@dataclass
class LossConfig:
    """Concrete :class:`LossArgs` implementation with sensible defaults.

    Useful when a caller (e.g. a unit test or an ad-hoc script) needs to
    invoke :func:`build_loss_fn` without a full recipe ``Config``.
    """

    policy_loss: PolicyLoss = "grpo"
    loss_path: LossPath = "client"
    """Default ``"client"`` because it works for every loss/PP/kl_beta combo;
    opt into ``"builtin"`` explicitly when you want the server-side fast path."""
    kl_beta: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    ratio_log_cap: float = 20.0
    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    dro: DROConfig = field(default_factory=DROConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    tis: TISConfig = field(default_factory=TISConfig)


def _supported_policy_losses_text() -> str:
    return ", ".join(SUPPORTED_POLICY_LOSSES)


def validate_loss_path(args: LossArgs, profile: Any | None = None) -> None:
    """Reject loss-path / config combinations that would silently misbehave.

    Call this **once at recipe startup** so a misconfigured run fails before
    rollouts begin instead of after. Replaces the historical ``resolve_builtin_loss``
    fallback (which returned ``None`` when builtin was ineligible and let the
    recipe quietly switch to client-side).

    No-op when ``args.loss_path == "client"`` (client path always works).
    Raises ``ValueError`` when ``args.loss_path == "builtin"`` and any of:

    - the loss is registered as client-side-only (no builtin kernel),
    - ``args.kl_beta > 0`` (builtin datums have no ``ref_logprobs`` field, so
      the KL term ``kl_beta * (pi - pi_ref)`` would be silently dropped --
      see :func:`build_builtin_loss_datums`),
    - ``profile.pipeline_parallelism > 1`` (server kernels don't support PP).
    """
    if args.loss_path == "client":
        return

    spec = LOSS_REGISTRY.get(args.policy_loss)
    if spec is None:
        supported = _supported_policy_losses_text()
        raise ValueError(
            f"Unsupported policy_loss '{args.policy_loss}'. "
            f"Expected one of: {supported}."
        )
    if spec.builtin_config_builder is None:
        raise ValueError(
            f"loss_path='builtin' requested but '{args.policy_loss}' is "
            f"registered as client-side-only (no builtin kernel). "
            f"Set loss_path='client' or pick a different policy_loss."
        )
    if args.kl_beta > 0.0:
        raise ValueError(
            f"loss_path='builtin' requested with kl_beta={args.kl_beta} > 0. "
            f"Server-side builtin kernels receive datums without ref_logprobs "
            f"(see build_builtin_loss_datums) so the KL term would be "
            f"silently dropped. Set loss_path='client' or kl_beta=0."
        )
    if profile is not None:
        pp = getattr(profile, "pipeline_parallelism", 1)
        if pp > 1:
            raise ValueError(
                f"loss_path='builtin' requested with pipeline_parallelism={pp} > 1. "
                f"Server-side builtin kernels do not support PP. "
                f"Set loss_path='client' or use a PP=1 training shape."
            )


def get_builtin_kernel_config(args: LossArgs) -> tuple[str, dict[str, Any]]:
    """Return ``(kernel_loss_name, kernel_config)`` for the builtin path.

    Caller must have already passed :func:`validate_loss_path` -- this helper
    will raise ``KeyError`` / ``AttributeError`` on a config that wasn't
    validated, by design (loud failure beats silent misconfiguration).
    """
    if args.loss_path != "builtin":
        raise ValueError(
            f"get_builtin_kernel_config called with loss_path={args.loss_path!r}; "
            f"only valid when loss_path='builtin'."
        )
    spec = LOSS_REGISTRY[args.policy_loss]
    return spec.builtin_config_builder(
        dapo_config=args.dapo,
        dro_config=args.dro,
        gspo_config=args.gspo,
        cispo_config=args.cispo,
        ratio_log_cap=args.ratio_log_cap,
        eps_clip=args.eps_clip,
        eps_clip_high=args.eps_clip_high,
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


def build_loss_fn(args: LossArgs) -> ClientLossBuilder:
    """Create the client-side loss builder for one registered RL policy loss.

    The returned callable is only used on the
    ``forward_backward_custom(...)`` path. Builtin server-side kernels are
    resolved separately by :func:`get_builtin_kernel_config`. Losses
    registered with ``builtin_config_builder=None`` always take this
    client-side path.

    *args* implements the :class:`LossArgs` protocol -- typically the recipe
    ``Config`` dataclass itself, which exposes ``policy_loss``, ``kl_beta``,
    and per-loss config fields at the top level.

    Returns a callable:
    ``(advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs) -> loss_fn``
    """
    spec = LOSS_REGISTRY.get(args.policy_loss)

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
                f"Unsupported policy_loss '{args.policy_loss}'. "
                f"Expected one of: {supported}."
            )
        return spec.client_loss_factory(
            advantages=advantages,
            ref_logprobs=ref_logprobs,
            prompt_lens=prompt_lens,
            inf_logprobs=inf_logprobs,
            prox_logprobs=prox_logprobs,
            kl_beta=args.kl_beta,
            dapo_config=args.dapo,
            dro_config=args.dro,
            gspo_config=args.gspo,
            cispo_config=args.cispo,
            tis_config=args.tis,
            ratio_log_cap=args.ratio_log_cap,
            eps_clip=args.eps_clip,
            eps_clip_high=args.eps_clip_high,
        )

    return build
