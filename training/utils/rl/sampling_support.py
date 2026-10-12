"""Sampling-support replay for top-p rollouts.

With ``top_p < 1`` rollouts sample from a truncated distribution while the
trainer normalizes over the full vocabulary. Replay trains over the recorded
support instead, in three steps a forked loop calls:

1. :func:`sampling_support_kwargs` -- sampler kwargs that truncate to the
   nucleus and publish each completion token's top-K support to trainer-shared
   storage (Parquet); rollout assembly carries the returned references into
   ``PolicyBatch.top_sampling_references``.
2. :func:`attach_top_sampling_references` -- hand them to the built-in datums.
3. :func:`sampling_support_loss_fn_config` -- ask the trainer to normalize
   target logprobs over that support.

Nothing here applies importance weights or changes advantages.
"""

from __future__ import annotations

from typing import Any, List

import tinker
from fireworks.training.sdk.routing import RoutingReferences


def attach_top_sampling_references(
    data: List[tinker.Datum],
    top_sampling_references: List[RoutingReferences | None],
) -> List[tinker.Datum]:
    """Return datums whose model inputs carry ``top_sampling_references``.

    A datum without references gets an all-gap reference; the trainer rejects
    it if a trained response position lacks recorded support.
    """
    if len(data) != len(top_sampling_references):
        raise ValueError("top_sampling_references must provide one entry per datum")
    # Replay is opt-in; importing the recipe must also work with older SDKs.
    try:
        from fireworks.training.sdk.routing import top_sampling_model_input_kwargs
    except ImportError as exc:
        raise ValueError(
            "Sampling support requires an SDK with Parquet sampling-reference "
            "transport; upgrade fireworks-ai before enabling sampling_support"
        ) from exc

    result: List[tinker.Datum] = []
    for datum, references in zip(data, top_sampling_references, strict=True):
        length = datum.model_input.length
        if references is None:
            references = RoutingReferences(
                length,
                (),
                ({"input_token_start": 0, "count": length},) if length else (),
            )
        result.append(
            tinker.Datum(
                model_input=datum.model_input.model_copy(
                    update=top_sampling_model_input_kwargs(references, length)
                ),
                loss_fn_inputs=datum.loss_fn_inputs,
            )
        )
    return result


def sampling_support_kwargs(*, top_p: float, support_size: int) -> dict[str, Any]:
    """Sampler kwargs: sample the ``top_p`` nucleus and record its top-K support.

    ``support_size`` is a storage width (K candidates per token), not a top-k
    filter; it must cover the support of every trained token.
    """
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1]; got {top_p!r}")
    if type(support_size) is not int or support_size < 1:
        raise ValueError("support_size must be a positive integer")
    return {
        "top_p": top_p,
        "top_sampling_logprobs": support_size,
        "top_sampling_format": "parquet_v1",
    }


def sampling_support_loss_fn_config(loss_fn_config: dict[str, Any]) -> dict[str, Any]:
    """A built-in loss config that normalizes target logprobs over the recorded support."""
    return {**loss_fn_config, "target_logprob_support": "sampling_support"}
