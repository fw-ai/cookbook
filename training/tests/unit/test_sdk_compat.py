from __future__ import annotations





def test_sampling_support_request_keeps_references_and_choice_on_wire():
    import tinker
    from tinker.lib._pydantic_conv import to_pydantic_input
    from tinker.types.forward_backward_input import ForwardBackwardInput

    from fireworks.training.sdk.routing import RoutingReferences
    from training.utils.rl.losses import PolicyBatch, build_policy_datums
    from training.utils.rl.sampling_support import attach_top_sampling_references

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([10, 11]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[11, 12], dtype="int64", shape=[2]),
            "weights": tinker.TensorData(data=[0, 1], dtype="int64", shape=[2]),
        },
    )
    references = RoutingReferences(2, (), ({"input_token_start": 0, "count": 2},))
    data = attach_top_sampling_references(
        build_policy_datums(
            PolicyBatch(
                data=[datum], advantages=[3.0], prompt_lens=[1], rollout_logprobs=[[0.0, -0.2]]
            )
        ),
        [references],
    )
    request = ForwardBackwardInput(
        data=data,
        loss_fn="ppo",
        loss_fn_config={"target_logprob_support": "sampling_support"},
    )

    body = to_pydantic_input(request).model_dump(mode="json")

    assert body["loss_fn_config"] == {"target_logprob_support": "sampling_support"}
    assert body["data"][0]["loss_fn_inputs"]["advantages"]["data"] == [0.0, 3.0]
    assert (
        body["data"][0]["model_input"]["top_sampling_references"]
        == references.to_dict()
    )
