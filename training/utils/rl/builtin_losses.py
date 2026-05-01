"""Server-side builtin loss configs -- single registry.

The builtin path packs each rollout into a ``Datum`` whose
``loss_fn_inputs`` are exactly ``(target_tokens, logprobs, advantages)``
(see :func:`training.utils.rl.losses.build_builtin_loss_datums`) and
dispatches to a fused server-side loss via
``forward_backward(..., loss_fn=<name>, loss_fn_config=<config>)``.

This file maps each :data:`~training.utils.rl.losses.PolicyLoss` name to a
callable that produces ``(loss_name, loss_config_dict)`` for the trainer.
The mapping is intentionally separate from the per-loss client files
(``grpo.py``, ``dapo.py``, ...) so that:

* per-loss files only contain client-side loss closures and config
  dataclasses -- no server-side knowledge,
* losses with no builtin path simply omit themselves from this registry,
* the full builtin surface is reviewable in one place.

When adding a new loss with a server-side path:

1. Define its ``XConfig`` dataclass and ``make_x_loss_fn`` in ``x.py``
   (client-side only).
2. Add a ``_x_builtin_config`` here that returns ``(loss_name, config_dict)``.
3. Add the ``"x": _x_builtin_config`` entry to :data:`BUILTIN_LOSSES`.
4. Add ``"x"`` to :data:`~training.utils.rl.losses.PolicyLoss`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from training.utils.rl.losses import LossArgs


BuiltinLossConfigBuilder = Callable[["LossArgs"], tuple[str, dict[str, Any]]]
"""Returns ``(loss_name, loss_config_dict)`` for the trainer.

Receives the full :class:`~training.utils.rl.losses.LossArgs` bundle and
reads only the fields it cares about.
"""


def _grpo_builtin_config(args: "LossArgs") -> tuple[str, dict[str, Any]]:
    high = args.eps_clip if args.eps_clip_high is None else args.eps_clip_high
    return "ppo", {
        "clip_low_threshold": 1.0 - args.eps_clip,
        "clip_high_threshold": 1.0 + high,
    }


def _is_builtin_config(args: "LossArgs") -> tuple[str, dict[str, Any]]:
    return "importance_sampling", {
        "ratio_log_cap": args.ratio_log_cap,
    }


def _dapo_builtin_config(args: "LossArgs") -> tuple[str, dict[str, Any]]:
    cfg = args.dapo
    config: dict[str, Any] = {
        "clip_low_threshold": 1.0 - cfg.eps_clip,
        "clip_high_threshold": 1.0 + cfg.eps_clip_high,
        "ratio_log_cap": cfg.ratio_log_cap,
    }
    if cfg.eps_clip_c is not None:
        config["eps_clip_c"] = cfg.eps_clip_c
        return "dapo", config
    return "ppo", config


def _dro_builtin_config(args: "LossArgs") -> tuple[str, dict[str, Any]]:
    return "dro", {
        "beta": args.dro.beta,
    }


def _gspo_builtin_config(args: "LossArgs") -> tuple[str, dict[str, Any]]:
    cfg = args.gspo
    return "gspo", {
        "clip_low_threshold": 1.0 - cfg.clip_ratio_low,
        "clip_high_threshold": 1.0 + cfg.clip_ratio_high,
        "seq_ratio_log_cap": cfg.seq_ratio_log_cap,
    }


def _cispo_builtin_config(args: "LossArgs") -> tuple[str, dict[str, Any]]:
    cfg = args.cispo
    return "cispo", {
        "clip_low_threshold": 1.0 - cfg.eps_low,
        "clip_high_threshold": 1.0 + cfg.eps_high,
        "ratio_log_cap": cfg.ratio_log_cap,
    }


BUILTIN_LOSSES: dict[str, BuiltinLossConfigBuilder] = {
    "grpo": _grpo_builtin_config,
    "importance_sampling": _is_builtin_config,
    "dapo": _dapo_builtin_config,
    "dro": _dro_builtin_config,
    "gspo": _gspo_builtin_config,
    "cispo": _cispo_builtin_config,
    # ``reinforce`` intentionally omitted: client-side only.
}
"""Single source of truth for losses that have a server-side builtin path.

Membership in this registry is what determines whether ``loss_path='builtin'``
is allowed for a given ``policy_loss``; absence means client-side-only and
:func:`training.utils.rl.losses.validate_loss_path` will raise.
"""
