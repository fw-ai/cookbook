"""Recompute the clipping anchor on the trainer (one extra forward per batch).

The reference loops clip against the rollout logprobs. A fork that clips
against a pre-update trainer snapshot instead calls :func:`recompute_anchor`
before building datums (timed as ``perf/old_policy_forward_time``) and logs
:func:`anchor_drift_metrics`. No importance weights are applied here; see
:mod:`training.utils.rl.tis` for a truncated correction.
"""

from dataclasses import replace
from typing import Any, Dict, Tuple

import torch

from training.utils.rl.common import iter_response_logprobs
from training.utils.rl.losses import PolicyBatch
from training.utils.timer import elapsed_timer


def recompute_anchor(policy: Any, batch: PolicyBatch) -> Tuple[PolicyBatch, Any]:
    """Return ``batch`` anchored on a trainer forward, and that forward's result.

    The forward result can be reused (for example by IGPO scoring). Trainer
    forward returns full-vocabulary logprobs; a support-normalized snapshot is
    not available here.
    """
    with elapsed_timer("old_policy_forward"):
        forward = policy.forward(batch.data, "cross_entropy")
        if callable(getattr(forward, "result", None)):
            forward = forward.result()
    rows = [list(output["logprobs"].data) for output in forward.loss_fn_outputs]
    if len(rows) != len(batch.data) or any(
        len(row) != len(datum.loss_fn_inputs["target_tokens"].data)
        for row, datum in zip(rows, batch.data, strict=True)
    ):
        raise ValueError("Trainer anchor logprobs must align with target tokens")
    return replace(batch, anchor_logprobs=rows), forward


def anchor_drift_metrics(batch: PolicyBatch) -> Dict[str, float]:
    """``anchor_k1`` / ``anchor_k3``: rollout-to-anchor drift, averaged per response.

    Measures the sampler/trainer mismatch on trained tokens at the snapshot,
    the quantity a truncated importance-sampling correction compensates.
    """
    k1 = k3 = 0.0
    responses = 0
    for pair in iter_response_logprobs(
        batch.data,
        [torch.tensor(row, dtype=torch.float32) for row in batch.anchor_logprobs],
        batch.rollout_logprobs,
        batch.prompt_lens,
        loss_name="anchor",
        source="rollout_logprobs",
    ):
        diff = pair.policy - pair.reference
        k1 += diff.mean().item()
        k3 += (torch.exp(diff) - diff - 1.0).mean().item()
        responses += 1
    if not responses:
        return {}
    return {"anchor_k1": k1 / responses, "anchor_k3": k3 / responses}
