from __future__ import annotations

import pytest


@pytest.mark.parametrize("head,mask", [(True, False), (False, True), (True, True)])
def test_sampling_support_request_keeps_independent_references_on_wire(
    head: bool, mask: bool
) -> None:
    from fireworks.training import sdk
    from fireworks.training.sdk._core.lib._pydantic_conv import to_pydantic_input
    from fireworks.training.sdk.routing import RoutingReferences
    from fireworks.training.sdk.types.forward_backward_input import ForwardBackwardInput
    from training.utils.rl.losses import PolicyBatch, build_policy_datums
    from training.utils.rl.sampling_support import (
        attach_sampling_mask_references,
        attach_top_sampling_references,
    )

    sampling = sdk.FiretitanSamplingParams(
        max_tokens=2,
        top_sampling_logprobs=128 if head else None,
        sampling_mask="parquet_v1" if mask else None,
    )
    assert sampling.sampling_mask == ("parquet_v1" if mask else None)

    datum = sdk.Datum(
        model_input=sdk.ModelInput.from_ints([10, 11]),
        loss_fn_inputs={
            "target_tokens": sdk.TensorData(data=[11, 12], dtype="int64", shape=[2]),
            "weights": sdk.TensorData(data=[0, 1], dtype="int64", shape=[2]),
        },
    )
    references = {}
    for channel in ("head", "mask"):
        references[channel] = RoutingReferences(
            2,
            ({"file_id": f"{channel}-file", "format": "parquet_v1", "row_count": 2},),
            (
                {
                    "input_token_start": 0,
                    "count": 2,
                    "file_index": 0,
                    "file_row_start": 0,
                },
            ),
        )
    data = build_policy_datums(
        PolicyBatch(
            data=[datum],
            advantages=[3.0],
            prompt_lens=[1],
            rollout_logprobs=[[0.0, -0.2]],
        )
    )
    if head:
        data = attach_top_sampling_references(data, [references["head"]])
    if mask:
        data = attach_sampling_mask_references(data, [references["mask"]])
    config = {}
    if head:
        config["score_centering"] = True
    if mask:
        config["target_logprob_support"] = "sampling_support"
    request = ForwardBackwardInput(
        data=data,
        loss_fn="reinforce" if head else "ppo",
        loss_fn_config=config,
    )

    body = to_pydantic_input(request).model_dump(mode="json")

    assert body["loss_fn_config"] == config
    assert body["data"][0]["loss_fn_inputs"]["advantages"]["data"] == [0.0, 3.0]
    model_input = body["data"][0]["model_input"]
    assert model_input.get("top_sampling_references") == (
        references["head"].to_dict() if head else None
    )
    assert model_input.get("sampling_mask_references") == (
        references["mask"].to_dict() if mask else None
    )
