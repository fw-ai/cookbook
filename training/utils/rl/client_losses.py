"""Client-side loss adapters -- single registry.

The client path runs each loss as a Python closure inside
``forward_backward_custom(...)``. Each loss exposes a single public
constructor in its own file (``make_X_loss_fn`` in ``X.py``); this file
adapts the unified :class:`~training.utils.rl.losses.LossArgs` bundle
to that loss's specific signature, and exposes the result as
:data:`CLIENT_LOSSES`.

Mirrors :mod:`training.utils.rl.builtin_losses` (server-side path). Keeping
the adapters here -- not in the per-loss files -- means:

* per-loss files (``grpo.py``, ``dapo.py``, ...) only contain the math
  (``make_X_loss_fn`` and any ``XConfig`` dataclass),
* the full client-side surface is reviewable in one place,
* registering a new loss does not require importing :class:`LossArgs` (or
  any registry-specific machinery) into the math file.

When adding a new loss with a client path:

1. Define ``XConfig`` and ``make_x_loss_fn`` in ``x.py`` (no registry import).
2. Add a ``_x_client_factory`` here that pulls the fields it needs out of
   the :class:`LossArgs` bundle and calls ``make_x_loss_fn``.
3. Add the ``"x": _x_client_factory`` entry to :data:`CLIENT_LOSSES`.
4. Add ``"x"`` to :data:`~training.utils.rl.losses.PolicyLoss`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List

from training.utils.rl.cispo import make_cispo_loss_fn
from training.utils.rl.dapo import make_dapo_loss_fn
from training.utils.rl.dro import make_dro_loss_fn
from training.utils.rl.grpo import make_grpo_loss_fn
from training.utils.rl.gspo import make_gspo_loss_fn
from training.utils.rl.is_loss import make_is_loss_fn
from training.utils.rl.reinforce import make_reinforce_loss_fn

if TYPE_CHECKING:
    from training.utils.rl.losses import LossArgs


ClientLossFactory = Callable[
    [
        "LossArgs",
        List[float],          # advantages
        List[List[float]],    # ref_logprobs
        List[int],            # prompt_lens
        List[List[float]],    # inf_logprobs
        List[List[float]],    # prox_logprobs
    ],
    Any,
]
"""Builds a client-side loss closure from a :class:`LossArgs` bundle plus
the per-step rollout tensors. The returned closure is what
``forward_backward_custom(..., loss_fn=...)`` invokes.
"""


def _grpo_client_factory(
    args: "LossArgs",
    advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs,
):
    return make_grpo_loss_fn(
        advantages,
        ref_logprobs,
        prompt_lens,
        inf_logprobs=inf_logprobs,
        prox_logprobs=prox_logprobs,
        kl_beta=args.kl_beta,
        eps_clip=args.eps_clip,
        eps_clip_high=args.eps_clip_high,
        tis_config=args.tis,
    )


def _is_client_factory(
    args: "LossArgs",
    advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs,
):
    return make_is_loss_fn(
        advantages,
        ref_logprobs,
        inf_logprobs,
        prompt_lens,
        prox_logprobs,
        ratio_log_cap=args.ratio_log_cap,
        tis_config=args.tis,
    )


def _dapo_client_factory(
    args: "LossArgs",
    advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs,
):
    return make_dapo_loss_fn(
        advantages,
        ref_logprobs,
        inf_logprobs,
        prompt_lens,
        prox_logprobs,
        args.dapo,
        tis_config=args.tis,
    )


def _dro_client_factory(
    args: "LossArgs",
    advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs,
):
    return make_dro_loss_fn(
        advantages,
        ref_logprobs,
        inf_logprobs,
        prompt_lens,
        prox_logprobs,
        args.dro,
        tis_config=args.tis,
    )


def _gspo_client_factory(
    args: "LossArgs",
    advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs,
):
    return make_gspo_loss_fn(
        advantages,
        ref_logprobs,
        inf_logprobs,
        prompt_lens,
        prox_logprobs,
        args.gspo,
        tis_config=args.tis,
    )


def _cispo_client_factory(
    args: "LossArgs",
    advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs,
):
    return make_cispo_loss_fn(
        advantages,
        ref_logprobs,
        inf_logprobs,
        prompt_lens,
        prox_logprobs,
        args.cispo,
        tis_config=args.tis,
    )


def _reinforce_client_factory(
    args: "LossArgs",
    advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs,
):
    return make_reinforce_loss_fn(
        advantages,
        ref_logprobs,
        prompt_lens,
        inf_logprobs=inf_logprobs,
        prox_logprobs=prox_logprobs,
        kl_beta=args.kl_beta,
        tis_config=args.tis,
    )


CLIENT_LOSSES: dict[str, ClientLossFactory] = {
    "grpo": _grpo_client_factory,
    "importance_sampling": _is_client_factory,
    "dapo": _dapo_client_factory,
    "dro": _dro_client_factory,
    "gspo": _gspo_client_factory,
    "cispo": _cispo_client_factory,
    "reinforce": _reinforce_client_factory,
}
"""Single source of truth for client-side loss adapters.

Membership defines which :data:`~training.utils.rl.losses.PolicyLoss` names
are supported on the ``forward_backward_custom`` path. The drift-guard
assertion in :mod:`training.utils.rl.losses` checks this against
:data:`PolicyLoss` and :data:`BUILTIN_LOSSES` to keep all three in sync.
"""
