"""Truncated importance sampling (TIS) for a recomputed anchor.

With the anchor recomputed on the trainer
(:func:`training.utils.rl.anchor.recompute_anchor`), rollouts still come
from the sampler ``mu`` while the loss clips against the trainer snapshot
``pi_old``. TIS weights each trained token, or each response, by
``min(pi_old / mu, cap)``. The weight is positive and not differentiated, so
scaling the datum advantages applies it to any server-side policy loss. With
the rollout logprobs as the anchor every weight is 1.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Tuple

import torch
from fireworks.training import sdk

TISLevel = Literal["token", "sequence"]


def apply_tis(
    datums: List[sdk.Datum],
    batch: Any,
    *,
    cap: float,
    level: TISLevel = "token",
) -> Tuple[List[sdk.Datum], Dict[str, float]]:
    """Scale datum advantages by truncated anchor/rollout weights.

    ``datums`` come from ``build_policy_datums(batch)`` after
    :func:`~training.utils.rl.anchor.recompute_anchor`, so their ``logprobs``
    input is the anchor and ``batch.rollout_logprobs`` is the sampler side.
    ``level="sequence"`` uses one weight per
    response from the summed log-ratio. Returns the new datums and
    ``tis_weight_mean`` (mean applied weight) and ``tis_clip_frac`` (share of
    weights truncated at ``cap``), both averaged per response.
    """
    if not cap >= 1.0:
        raise ValueError(f"TIS cap must be >= 1; got {cap!r}")
    if level not in ("token", "sequence"):
        raise ValueError(f"TIS level must be 'token' or 'sequence'; got {level!r}")
    rollout_logprobs = batch.rollout_logprobs
    if len(datums) != len(rollout_logprobs):
        raise ValueError("apply_tis needs one rollout_logprobs row per datum")
    log_cap = math.log(cap)
    weighted: List[sdk.Datum] = []
    weight_sum = clip_sum = 0.0
    responses = 0
    for datum, rollout in zip(datums, rollout_logprobs, strict=True):
        inputs = datum.loss_fn_inputs
        anchor = list(inputs["logprobs"].data)
        advantages = list(inputs["advantages"].data)
        active = [t for t, m in enumerate(inputs["response_mask"].data) if m]
        if not active:
            weighted.append(datum)
            continue
        # Both rows in the datum's float32, so an unchanged anchor weighs exactly 1.
        log_ratios = torch.tensor([anchor[t] for t in active], dtype=torch.float32) - torch.tensor(
            [float(rollout[t]) for t in active], dtype=torch.float32
        )
        if level == "sequence":
            log_ratios = log_ratios.sum().expand(len(active))
        weights = torch.exp(torch.clamp(log_ratios, max=log_cap))
        for t, w in zip(active, weights.tolist()):
            advantages[t] *= w
        weight_sum += weights.mean().item()
        clip_sum += (log_ratios > log_cap).float().mean().item()
        responses += 1
        weighted.append(
            sdk.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={
                    **inputs,
                    "advantages": sdk.TensorData(data=advantages, dtype="float32", shape=[len(advantages)]),
                },
            )
        )
    if not responses:
        return weighted, {}
    return weighted, {
        "tis_weight_mean": weight_sum / responses,
        "tis_clip_frac": clip_sum / responses,
    }
