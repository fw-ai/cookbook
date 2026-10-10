"""Rollout batches and built-in policy datum preparation.

Recipes select their objective in ``algorithm``. This module combines rollout
groups and prepares token-aligned advantages, denominators, and response masks.
"""

from __future__ import annotations

from typing import List
from dataclasses import field, dataclass

import tinker
import torch
from fireworks.training.sdk.routing import RoutingReferences

from training.utils.rl.common import (
    _coerce_response_logprobs,
    _get_loss_mask,
    validate_inference_logprobs_for_sample,
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
    """``rollout_logprobs`` aligned to ``target_tokens``.

    These are sampled-token logprobs after rollout temperature and sampling masks.
    They feed direct rollout-logprob reuse denominators.
    """
    raw_inf_logprobs: List[List[float]] = field(default_factory=list)
    """Raw model logprobs aligned to ``target_tokens`` for observability only.

    The direct client GRPO builder uses them only for drift metrics. They must
    never replace sampling logprobs in the policy ratio.
    """
    inference_topk_token_ids: List[List[List[int]]] = field(default_factory=list)
    """Per-datum sampler top-k token ids in shifted target coordinates."""
    inference_topk_logprobs: List[List[List[float]]] = field(default_factory=list)
    """Per-datum sampler top-k logprobs aligned with ``inference_topk_token_ids``."""
    top_sampling_references: List[RoutingReferences | None] = field(default_factory=list)
    """Per-datum Parquet top-K sampling references over model-input positions."""
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
    run_metadata: List[dict] = field(default_factory=list)
    """One metadata mapping per logical rollout run.

    Multi-turn callers may split one run into several trainer datums. Keeping
    run metadata beside the group preserves the logical-trajectory boundary for
    metrics without treating history segments as extra completions.
    """
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
    include_topk: bool = False,
    include_top_sampling_references: bool = False,
):
    """Flatten a list of PromptGroups into combined arrays for a fwd_bwd call.

    Returns ``(data, advantages, ref_logprobs, prompt_lens, inf_logprobs)``.
    ``include_raw`` appends observability-only ``raw_inf_logprobs``.
    ``include_topk`` appends sampler top-k token IDs and logprobs.
    ``include_top_sampling_references`` appends Parquet top-K references.
    """
    data: List[tinker.Datum] = []
    advantages: List[float] = []
    ref_logprobs: List[List[float]] = []
    prompt_lens: List[int] = []
    inf_logprobs: List[List[float]] = []
    raw_inf_logprobs: List[List[float]] = []
    inference_topk_token_ids: List[List[List[int]]] = []
    inference_topk_logprobs: List[List[List[float]]] = []
    top_sampling_references: List[RoutingReferences | None] = []

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
        if include_topk:
            inference_topk_token_ids.extend(
                pg.inference_topk_token_ids or [[] for _ in pg.data]
            )
            inference_topk_logprobs.extend(
                pg.inference_topk_logprobs or [[] for _ in pg.data]
            )
        if include_top_sampling_references:
            top_sampling_references.extend(
                pg.top_sampling_references or [None for _ in pg.data]
            )

    result = (data, advantages, ref_logprobs, prompt_lens, inf_logprobs)
    if include_raw:
        result += (raw_inf_logprobs,)
    if include_topk:
        result += (inference_topk_token_ids, inference_topk_logprobs)
    if include_top_sampling_references:
        result += (top_sampling_references,)
    return result


def build_grpo_datums(
    data: List[tinker.Datum],
    advantages: List[float],
    inf_logprobs: List[List[float]],
    prompt_lens: List[int],
    *,
    include_response_mask: bool = False,
    old_policy_logprobs: List[List[float]] | None = None,
) -> List[tinker.Datum]:
    """Build masked advantages and a fixed denominator for built-in losses.

    The denominator defaults to rollout probabilities; clipped objectives can
    supply a separate snapshot anchor. Advantages are never preweighted.
    """
    n = len(data)
    denominator_logprobs = (
        inf_logprobs if old_policy_logprobs is None else old_policy_logprobs
    )
    aligned = {
        "anchor_logprobs": len(denominator_logprobs),
        "advantages": len(advantages),
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
    for i, (datum, advantage, rollout_row, prompt_len) in enumerate(
        zip(
            data,
            advantages,
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
        inf_lp = list(rollout_row)

        if len(inf_lp) != n_tokens:
            raise ValueError(
                "GRPO rollout_logprobs must align exactly with target_tokens "
                f"for sample {i}: expected {n_tokens}, got {len(inf_lp)}."
            )

        resp_len = max(0, n_tokens - response_start)
        loss_mask = _get_loss_mask(
            datum,
            response_start,
            resp_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        if resp_len > 0:
            validate_inference_logprobs_for_sample(
                "grpo",
                i,
                inf_lp,
                response_start + resp_len,
                source="rollout_logprobs",
            )
            active = loss_mask > 0.5
            resp_inf_values = _coerce_response_logprobs(
                inf_lp[response_start : response_start + resp_len],
                active,
                policy_loss="grpo",
                sample_idx=i,
                source="rollout_logprobs",
            )
            inf_lp[response_start:] = resp_inf_values
        inf_lp[:response_start] = [
            0.0 if value is None else float(value) for value in inf_lp[:response_start]
        ]
        anchor_lp = inf_lp
        if old_policy_logprobs is not None:
            anchor_lp = list(denominator_logprobs[i])
            if len(anchor_lp) != n_tokens:
                raise ValueError(
                    "old_policy_logprobs must align exactly with target_tokens"
                )
            anchor_lp[:response_start] = [0.0] * response_start
            if resp_len:
                anchor_lp[response_start:] = _coerce_response_logprobs(
                    anchor_lp[response_start:],
                    loss_mask > 0.5,
                    policy_loss="grpo",
                    sample_idx=i,
                    source="old_policy_logprobs",
                )
        per_token_adv = [0.0] * response_start
        # Bulk extraction avoids per-token tensor calls while retaining Python-float arithmetic.
        per_token_adv.extend(float(advantage * mask) for mask in loss_mask.tolist())

        loss_fn_inputs = {
            "target_tokens": tinker.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[n_tokens],
            ),
            "logprobs": tinker.TensorData(
                data=anchor_lp,
                dtype="float32",
                shape=[n_tokens],
            ),
            "advantages": tinker.TensorData(
                data=per_token_adv,
                dtype="float32",
                shape=[n_tokens],
            ),
        }
        if include_response_mask:
            # GSPO's sequence ratio and num_sequences denominator must include
            # sampled responses whose group-relative advantage is exactly zero.
            # Inferring membership from per_token_adv would silently drop them.
            response_mask = [0] * response_start
            response_mask.extend(int(value > 0.5) for value in loss_mask.tolist())
            loss_fn_inputs["response_mask"] = tinker.TensorData(
                data=response_mask,
                dtype="int64",
                shape=[n_tokens],
            )

        new_datum = tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs=loss_fn_inputs,
        )
        result.append(new_datum)

    return result
