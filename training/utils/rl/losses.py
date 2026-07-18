"""Rollout batch types and explicit built-in GRPO datum preparation.

The generic sync and async recipes call ``make_grpo_loss_fn`` directly. Recipe
forks that intentionally switch to the trainer's built-in PPO kernel can use
``build_grpo_datums``. There is no registry or runtime loss dispatch here.
"""

from __future__ import annotations

from typing import List
from dataclasses import field, dataclass

import tinker
import torch

from training.utils.rl.common import (
    _coerce_response_logprobs,
    _get_loss_mask,
    validate_inference_logprobs_for_sample,
)
from training.utils.rl.tis import TISConfig, compute_tis_weight


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
    """``rollout_logprobs`` aligned to ``target_tokens``.

    These are sampled-token logprobs after rollout temperature and sampling masks.
    They feed direct rollout-logprob reuse and TIS denominators.
    """
    raw_inf_logprobs: List[List[float]] = field(default_factory=list)
    """Raw model logprobs aligned to ``target_tokens`` for observability only.

    The direct client GRPO builder uses them only for drift metrics. They must
    never replace behavior logprobs in TIS.
    """
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
    *,
    include_raw: bool = False,
):
    """Flatten a list of PromptGroups into combined arrays for a fwd_bwd call.

    Returns ``(data, advantages, ref_logprobs, prompt_lens, inf_logprobs)``.
    With ``include_raw=True``, appends observability-only ``raw_inf_logprobs``
    for ``train/inference_*`` drift metrics.
    """
    data: List[tinker.Datum] = []
    advantages: List[float] = []
    ref_logprobs: List[List[float]] = []
    prompt_lens: List[int] = []
    inf_logprobs: List[List[float]] = []
    raw_inf_logprobs: List[List[float]] = []

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
        if include_raw:
            if pg.raw_inf_logprobs:
                raw_inf_logprobs.extend(pg.raw_inf_logprobs)
            else:
                raw_inf_logprobs.extend([[] for _ in pg.data])

    if include_raw:
        return (
            data,
            advantages,
            ref_logprobs,
            prompt_lens,
            inf_logprobs,
            raw_inf_logprobs,
        )
    return data, advantages, ref_logprobs, prompt_lens, inf_logprobs


def build_grpo_datums(
    data: List[tinker.Datum],
    advantages: List[float],
    old_policy_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_lens: List[int],
    tis_config: TISConfig | None = None,
) -> List[tinker.Datum]:
    """Build strictly aligned datums for an explicit server-side GRPO fork.

    Folds the TIS weight ``exp(old_policy - behavior)`` into per-token
    advantages so the server only sees ``logprobs`` (= old_policy_lp) and
    ``advantages`` (= advantage * tis_weight * loss_mask).

    Uses ``compute_tis_weight`` for behavioral TIS correction and
    ``_get_loss_mask`` for multi-turn tool-call masking.
    """
    if tis_config is None:
        tis_config = TISConfig()

    n = len(data)
    aligned = {
        "advantages": len(advantages),
        "old_policy_logprobs": len(old_policy_logprobs),
        "rollout_logprobs": len(inf_logprobs),
        "prompt_lens": len(prompt_lens),
    }
    mismatched = {name: size for name, size in aligned.items() if size != n}
    if mismatched:
        details = ", ".join(f"{name}={size}" for name, size in mismatched.items())
        raise ValueError(
            f"GRPO requires {n} aligned rows; mismatched inputs: {details}."
        )

    result: List[tinker.Datum] = []
    for i, (datum, advantage, old_policy_row, rollout_row, prompt_len) in enumerate(
        zip(
            data,
            advantages,
            old_policy_logprobs,
            inf_logprobs,
            prompt_lens,
            strict=True,
        )
    ):
        target_data = datum.loss_fn_inputs["target_tokens"]
        target_tokens = list(target_data.data)
        n_tokens = len(target_tokens)
        if prompt_len < 0:
            raise ValueError(
                f"GRPO prompt_len must be non-negative for sample {i}, got {prompt_len}."
            )
        response_start = max(0, prompt_len - 1)
        if response_start > n_tokens:
            raise ValueError(
                "GRPO prompt_len exceeds the datum sequence for sample "
                f"{i}: prompt_len={prompt_len}, target_tokens={n_tokens}."
            )
        old_policy_lp = list(old_policy_row)
        inf_lp = list(rollout_row)

        if len(old_policy_lp) != n_tokens:
            raise ValueError(
                "GRPO old_policy_logprobs must align exactly with target_tokens "
                f"for sample {i}: expected {n_tokens}, got {len(old_policy_lp)}."
            )

        resp_len = max(0, n_tokens - response_start)
        loss_mask = _get_loss_mask(
            datum,
            response_start,
            resp_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        active_count = int((loss_mask > 0.5).sum().item())

        if resp_len > 0 and active_count > 0:
            validate_inference_logprobs_for_sample(
                "grpo",
                i,
                inf_lp,
                response_start + resp_len,
                source="rollout_logprobs",
            )
            resp_old_policy = torch.tensor(
                old_policy_lp[response_start : response_start + resp_len],
                dtype=torch.float32,
            )
            # Active-only filter mirrors common.py: keep masked bridge tokens
            # out of the sequence-level TIS weight.
            active = loss_mask > 0.5
            resp_inf_values = _coerce_response_logprobs(
                inf_lp[response_start : response_start + resp_len],
                active,
                policy_loss="grpo",
                sample_idx=i,
                source="rollout_logprobs",
            )
            resp_inf = torch.tensor(resp_inf_values, dtype=torch.float32)
            tis_weight_active, _ = compute_tis_weight(
                resp_old_policy[active],
                resp_inf[active],
                tis_config,
            )
            tis_weight = torch.ones(resp_len, dtype=torch.float32)
            tis_weight[active] = tis_weight_active.to(torch.float32)
        else:
            tis_weight = torch.ones(resp_len, dtype=torch.float32)

        per_token_adv = [0.0] * response_start
        for r in range(resp_len):
            per_token_adv.append(
                float(advantage * tis_weight[r].item() * loss_mask[r].item())
            )

        new_datum = tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=target_tokens,
                    dtype="int64",
                    shape=[n_tokens],
                ),
                "logprobs": tinker.TensorData(
                    data=old_policy_lp,
                    dtype="float32",
                    shape=[n_tokens],
                ),
                "advantages": tinker.TensorData(
                    data=per_token_adv,
                    dtype="float32",
                    shape=[n_tokens],
                ),
            },
        )
        result.append(new_datum)

    return result
