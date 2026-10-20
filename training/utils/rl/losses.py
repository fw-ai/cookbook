"""Rollout batches and policy datum preparation.

:func:`combine_prompt_groups` flattens rollout groups into a
:class:`PolicyBatch`; :func:`build_policy_datums` turns one into the datums of a
server-side policy loss (advantages, the batch anchor and a response mask).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from fireworks.training import sdk
from fireworks.training.sdk.routing import RoutingReferences

from training.utils.rl.common import (
    _coerce_response_logprobs,
    _get_loss_mask,
    validate_inference_logprobs_for_sample,
)


@dataclass
class PromptGroup:
    """Processed data from one prompt's rollout, ready for training."""

    data: List[sdk.Datum]
    advantages: List[float]
    prompt_len: int
    rewards: List[float]
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
    top_sampling_references: List[RoutingReferences | None] = field(
        default_factory=list
    )
    sampling_mask_references: List[RoutingReferences | None] = field(
        default_factory=list
    )
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


def has_reward_variance(group: PromptGroup) -> bool:
    """Dynamic filter for GRPO: a group whose rewards are all equal has zero advantage."""
    return len(set(group.rewards)) > 1


@dataclass
class PolicyBatch:
    """Prompt groups flattened into aligned per-datum rows for one trainer call.

    ``anchor_logprobs`` is what the policy ratio is clipped against; it defaults
    to ``rollout_logprobs``. :func:`training.utils.rl.anchor.recompute_anchor`
    replaces it with a trainer snapshot, and every datum builder, loss and
    metric reads this one field.
    """

    data: List[sdk.Datum]
    advantages: List[float]
    prompt_lens: List[int]
    rollout_logprobs: List[List[float]]
    """Sampling logprobs aligned to ``target_tokens``."""
    anchor_logprobs: List[List[float]] | None = None
    """Clipping anchor aligned to ``target_tokens``; ``None`` means ``rollout_logprobs``."""
    groups: List[PromptGroup] = field(default_factory=list)
    raw_inference_logprobs: List[List[float]] = field(default_factory=list)
    """Raw model logprobs for drift metrics; an empty row where serving sent none."""
    token_advantages: List[List[float]] | None = None
    """Optional per-target-token advantages (e.g. IGPO); replaces ``advantages``."""
    ref_logprobs: List[List[float]] = field(default_factory=list)
    """Optional reference-model logprobs for a client-side KL term (opt-in fork)."""
    inference_topk_token_ids: List[List[List[int]]] = field(default_factory=list)
    inference_topk_logprobs: List[List[List[float]]] = field(default_factory=list)
    top_sampling_references: List[RoutingReferences | None] = field(
        default_factory=list
    )
    sampling_mask_references: List[RoutingReferences | None] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if self.anchor_logprobs is None:
            self.anchor_logprobs = self.rollout_logprobs


def combine_prompt_groups(groups: List[PromptGroup]) -> PolicyBatch:
    """Flatten prompt groups into one :class:`PolicyBatch`."""
    data: List[sdk.Datum] = []
    advantages: List[float] = []
    prompt_lens: List[int] = []
    rollout: List[List[float]] = []
    raw: List[List[float]] = []
    topk_ids: List[List[List[int]]] = []
    topk_logprobs: List[List[List[float]]] = []
    references: List[RoutingReferences | None] = []
    mask_references: List[RoutingReferences | None] = []
    for pg in groups:
        rows = len(pg.data)
        data.extend(pg.data)
        advantages.extend(pg.advantages)
        prompt_lens.extend(
            pg.prompt_lens if pg.prompt_lens is not None else [pg.prompt_len] * rows
        )
        rollout.extend(pg.inf_logprobs)
        raw.extend(pg.raw_inf_logprobs or [[] for _ in range(rows)])
        topk_ids.extend(pg.inference_topk_token_ids or [[] for _ in range(rows)])
        topk_logprobs.extend(pg.inference_topk_logprobs or [[] for _ in range(rows)])
        references.extend(pg.top_sampling_references or [None for _ in range(rows)])
        mask_references.extend(
            pg.sampling_mask_references or [None for _ in range(rows)]
        )
    return PolicyBatch(
        data=data,
        advantages=advantages,
        prompt_lens=prompt_lens,
        rollout_logprobs=rollout,
        groups=list(groups),
        raw_inference_logprobs=raw,
        inference_topk_token_ids=topk_ids,
        inference_topk_logprobs=topk_logprobs,
        top_sampling_references=references,
        sampling_mask_references=mask_references,
    )


def build_policy_datums(batch: PolicyBatch) -> List[sdk.Datum]:
    """Datums for a server-side policy loss over ``batch``.

    Each datum carries ``target_tokens``, the batch anchor as ``logprobs``,
    masked ``advantages`` (``batch.token_advantages`` when set, else one
    advantage per datum) and a ``response_mask`` that keeps zero-advantage
    responses in sequence-level objectives. Advantages are never preweighted.
    """
    n = len(batch.data)
    aligned = {
        "advantages": len(batch.advantages),
        "rollout_logprobs": len(batch.rollout_logprobs),
        "anchor_logprobs": len(batch.anchor_logprobs),
        "prompt_lens": len(batch.prompt_lens),
    }
    if batch.token_advantages is not None:
        aligned["token_advantages"] = len(batch.token_advantages)
    mismatched = {name: size for name, size in aligned.items() if size != n}
    if mismatched:
        details = ", ".join(f"{name}={size}" for name, size in mismatched.items())
        raise ValueError(
            f"Policy datums require {n} aligned rows; mismatched inputs: {details}."
        )

    result: List[sdk.Datum] = []
    for i, datum in enumerate(batch.data):
        prompt_len = batch.prompt_lens[i]
        target_tokens = list(datum.loss_fn_inputs["target_tokens"].data)
        n_tokens = len(target_tokens)
        if prompt_len < 0:
            raise ValueError(
                f"prompt_len must be non-negative for sample {i}, got {prompt_len}."
            )
        response_start = max(0, prompt_len - 1)
        if response_start > n_tokens:
            raise ValueError(
                "prompt_len exceeds the datum sequence for sample "
                f"{i}: prompt_len={prompt_len}, target_tokens={n_tokens}."
            )
        resp_len = max(0, n_tokens - response_start)
        loss_mask = _get_loss_mask(
            datum,
            response_start,
            resp_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        active = loss_mask > 0.5

        rollout_row = list(batch.rollout_logprobs[i])
        if len(rollout_row) != n_tokens:
            raise ValueError(
                "rollout_logprobs must align exactly with target_tokens "
                f"for sample {i}: expected {n_tokens}, got {len(rollout_row)}."
            )
        if resp_len > 0:
            # Rollout rows feed TIS and drift even after the anchor is recomputed.
            validate_inference_logprobs_for_sample(
                "policy",
                i,
                rollout_row,
                response_start + resp_len,
                source="rollout_logprobs",
            )
            _coerce_response_logprobs(
                rollout_row[response_start:],
                active,
                policy_loss="policy",
                sample_idx=i,
                source="rollout_logprobs",
            )
        anchor = list(batch.anchor_logprobs[i])
        if len(anchor) != n_tokens:
            raise ValueError(
                f"anchor_logprobs must align exactly with target_tokens for sample {i}: "
                f"expected {n_tokens}, got {len(anchor)}."
            )
        anchor[:response_start] = [0.0] * response_start
        if resp_len:
            anchor[response_start:] = _coerce_response_logprobs(
                anchor[response_start:],
                active,
                policy_loss="policy",
                sample_idx=i,
                source="anchor_logprobs",
            )

        mask = loss_mask.tolist()
        if batch.token_advantages is None:
            per_token_adv = [0.0] * response_start
            # Bulk extraction avoids per-token tensor calls while retaining Python-float arithmetic.
            per_token_adv.extend(float(batch.advantages[i] * m) for m in mask)
        else:
            row = batch.token_advantages[i]
            if len(row) != n_tokens:
                raise ValueError(
                    f"token_advantages row {i} has {len(row)} values for {n_tokens} target tokens."
                )
            per_token_adv = [0.0] * response_start
            per_token_adv.extend(
                float(value) if m > 0.5 else 0.0
                for value, m in zip(row[response_start:], mask, strict=True)
            )
        response_mask = [0] * response_start
        response_mask.extend(int(m > 0.5) for m in mask)

        result.append(
            sdk.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={
                    "target_tokens": sdk.TensorData(
                        data=target_tokens, dtype="int64", shape=[n_tokens]
                    ),
                    "logprobs": sdk.TensorData(
                        data=anchor, dtype="float32", shape=[n_tokens]
                    ),
                    "advantages": sdk.TensorData(
                        data=per_token_adv, dtype="float32", shape=[n_tokens]
                    ),
                    "response_mask": sdk.TensorData(
                        data=response_mask, dtype="int64", shape=[n_tokens]
                    ),
                },
            )
        )
    return result
