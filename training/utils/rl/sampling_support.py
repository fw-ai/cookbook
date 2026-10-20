"""Independent replay-mask and score-centering probability-head capture.

Compose replay_mask_kwargs and sc_head_kwargs as needed, then attach each
channel's references to the datums. Loss selection remains with the caller.
"""

from __future__ import annotations

from typing import Any, List

from fireworks.training import sdk
from fireworks.training.sdk.routing import RoutingReferences


def attach_top_sampling_references(
    data: List[sdk.Datum],
    top_sampling_references: List[RoutingReferences | None],
) -> List[sdk.Datum]:
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

    result: List[sdk.Datum] = []
    for datum, references in zip(data, top_sampling_references, strict=True):
        length = datum.model_input.length
        if references is None:
            references = RoutingReferences(
                length,
                (),
                ({"input_token_start": 0, "count": length},) if length else (),
            )
        result.append(
            sdk.Datum(
                model_input=datum.model_input.model_copy(
                    update=top_sampling_model_input_kwargs(references, length)
                ),
                loss_fn_inputs=datum.loss_fn_inputs,
            )
        )
    return result


def attach_sampling_mask_references(
    data: List[sdk.Datum],
    sampling_mask_references: List[RoutingReferences | None],
) -> List[sdk.Datum]:
    """Return datums whose model inputs carry ``sampling_mask_references``.

    A datum without references gets an all-gap reference; the trainer rejects
    it if a trained response position lacks recorded support.
    """
    if len(data) != len(sampling_mask_references):
        raise ValueError("sampling_mask_references must provide one entry per datum")
    # Replay is opt-in; importing the recipe must also work with older SDKs.
    try:
        from fireworks.training.sdk.routing import sampling_mask_model_input_kwargs
    except ImportError as exc:
        raise ValueError(
            "Sampling support requires an SDK with Parquet sampling-reference "
            "transport; upgrade fireworks-ai before enabling sampling_support"
        ) from exc

    result: List[sdk.Datum] = []
    for datum, references in zip(data, sampling_mask_references, strict=True):
        length = datum.model_input.length
        if references is None:
            references = RoutingReferences(
                length,
                (),
                ({"input_token_start": 0, "count": length},) if length else (),
            )
        result.append(
            sdk.Datum(
                model_input=datum.model_input.model_copy(
                    update=sampling_mask_model_input_kwargs(references, length)
                ),
                loss_fn_inputs=datum.loss_fn_inputs,
            )
        )
    return result


def replay_mask_kwargs(*, top_p: float) -> dict[str, Any]:
    """Capture the complete ID-only replay support, without requesting an SC head."""
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1]; got {top_p!r}")
    return {"top_p": top_p, "sampling_mask": "parquet_v1"}


def sc_head_kwargs(k: int) -> dict[str, Any]:
    """Capture only the probability head used by score centering."""
    if type(k) is not int or not 1 <= k <= 20000:
        raise ValueError("k must be an integer in [1, 20000]")
    return {"top_sampling_logprobs": k, "top_sampling_format": "parquet_v1"}


def sampling_support_loss_fn_config(loss_fn_config: dict[str, Any]) -> dict[str, Any]:
    """A built-in loss config that normalizes target logprobs over the recorded support."""
    return {**loss_fn_config, "target_logprob_support": "sampling_support"}
