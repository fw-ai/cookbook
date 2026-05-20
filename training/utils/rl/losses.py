"""RL loss dispatch, path selection, and rollout data types.

Two parallel registries live in sibling files and are wired in here:

- :mod:`training.utils.rl.client_losses` -- :data:`CLIENT_LOSSES`, the
  Python loss closures that run inside ``forward_backward_custom(...)``.
- :mod:`training.utils.rl.builtin_losses` -- :data:`BUILTIN_LOSSES`, the
  server-side fused losses dispatched via ``forward_backward(...)``.

Per-loss math files (``grpo.py``, ``dapo.py``, ...) hold only the
``XConfig`` dataclass and ``make_x_loss_fn`` builder; they are intentionally
unaware of the registries and the path-selection logic.
"""

from __future__ import annotations

from typing import Any, List, Literal, Protocol, Tuple, Callable
from dataclasses import field, dataclass

import tinker

from training.utils.rl.builtin_losses import BUILTIN_LOSSES
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.client_losses import CLIENT_LOSSES
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.dro import DROConfig
from training.utils.rl.gspo import GSPOConfig
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

Kept in lockstep with :data:`CLIENT_LOSSES` keys (and a strict subset of
:data:`BUILTIN_LOSSES` keys); a startup assertion below catches drift if a
new client loss is added without updating this ``Literal``.
"""


LossPath = Literal["builtin", "client"]
"""Which forward/backward path the recipe uses.

- ``"builtin"`` -- server-side ``forward_backward(...)`` with a fused
  loss. Faster, but the loss sees only
  ``(target_tokens, logprobs, advantages)`` so KL penalties (``kl_beta>0``)
  cannot be applied; pipeline parallelism > 1 is also unsupported.
- ``"client"`` -- client-side ``forward_backward_custom(...)`` with the
  Python loss closure. Always works, slower because of an extra forward
  pass for old-policy logprobs.

The choice is **explicit**: ``validate_loss_path`` raises with an
actionable message if the user picked ``"builtin"`` in a configuration that
forbids it, instead of silently falling back to client-side. This avoids
the historical footgun where setting ``kl_beta > 0`` would magically route
to the client path -- a behavior that broke once anyone reordered the gate
or introduced a new builtin loss that consumed ref logprobs.
"""


SUPPORTED_POLICY_LOSSES: tuple[str, ...] = tuple(CLIENT_LOSSES)

# Drift guard: PolicyLoss Literal must list every client-registered loss,
# and every builtin must also be in CLIENT_LOSSES (a builtin without a
# client fallback is a footgun -- there's no way to apply KL or run on PP>1).
import typing as _typing  # noqa: E402
assert set(_typing.get_args(PolicyLoss)) == set(CLIENT_LOSSES), (
    "PolicyLoss Literal is out of sync with CLIENT_LOSSES. "
    f"Literal={set(_typing.get_args(PolicyLoss))!r}, "
    f"client={set(CLIENT_LOSSES)!r}."
)
assert set(BUILTIN_LOSSES).issubset(CLIENT_LOSSES), (
    "Every builtin loss must also have a client-side fallback; "
    f"orphaned builtins: {set(BUILTIN_LOSSES) - set(CLIENT_LOSSES)!r}."
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

    if args.policy_loss not in CLIENT_LOSSES:
        supported = _supported_policy_losses_text()
        raise ValueError(
            f"Unsupported policy_loss '{args.policy_loss}'. "
            f"Expected one of: {supported}."
        )
    if args.policy_loss not in BUILTIN_LOSSES:
        raise ValueError(
            f"loss_path='builtin' requested but '{args.policy_loss}' is "
            f"registered as client-side-only (no builtin loss). "
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


def get_builtin_loss_config(args: LossArgs) -> tuple[str, dict[str, Any]]:
    """Return ``(loss_name, loss_config_dict)`` for the builtin path.

    Caller must have already passed :func:`validate_loss_path` -- this helper
    will raise ``KeyError`` on a config that wasn't validated, by design
    (loud failure beats silent misconfiguration).
    """
    if args.loss_path != "builtin":
        raise ValueError(
            f"get_builtin_loss_config called with loss_path={args.loss_path!r}; "
            f"only valid when loss_path='builtin'."
        )
    builder = BUILTIN_LOSSES[args.policy_loss]
    return builder(args)


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
    prompt_lens: List[int] | None = None
    """Per-sample prompt boundaries.  Heterogeneous rollouts (multi-turn,
    tool branches) have different prefix lengths per sample, so the scalar
    ``prompt_len`` cannot represent them faithfully -- ``prompt_lens[i]``
    is the boundary for ``data[i]``.  Left ``None`` for legacy single-turn
    rollouts where every sample shares the same prefix; ``combine_prompt_groups``
    then falls back to ``[prompt_len] * len(data)``."""


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
        if pg.prompt_lens is not None:
            prompt_lens.extend(pg.prompt_lens)
        else:
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
            # Active-only filter mirrors common.py: keep masked bridge tokens
            # out of the sequence-level TIS weight.
            active = loss_mask > 0.5
            tis_weight_active, _ = compute_tis_weight(
                resp_prox[active], resp_inf[active], tis_config,
            )
            tis_weight = torch.ones(resp_len, dtype=torch.float32)
            tis_weight[active] = tis_weight_active.to(torch.float32)
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
    factory = CLIENT_LOSSES.get(args.policy_loss)

    def build(
        advantages: List[float],
        ref_logprobs: List[List[float]],
        prompt_lens: List[int],
        inf_logprobs: List[List[float]],
        prox_logprobs: List[List[float]],
    ) -> Any:
        if factory is None:
            supported = _supported_policy_losses_text()
            raise ValueError(
                f"Unsupported policy_loss '{args.policy_loss}'. "
                f"Expected one of: {supported}."
            )
        return factory(
            args, advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs,
        )

    return build
