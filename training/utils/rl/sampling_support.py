"""Attach recorded sampler support to built-in policy datums.

The sampler publishes each completion token's top-K support to trainer-shared
storage (Parquet) and returns references. Rollout assembly aligns them with
model-input positions; this module hands them to the trainer unchanged. It
never applies importance weights or changes advantages.
"""

from __future__ import annotations

from typing import List

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
