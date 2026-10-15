from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from fireworks.training import sdk

from training.utils.rl.losses import PolicyBatch
from training.utils.rl.reference import LoraReference, SamplerReference, with_reference_logprobs


def datum(tokens: tuple[int, ...] = (1, 2, 3)) -> sdk.Datum:
    return sdk.Datum(
        model_input=sdk.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": sdk.TensorData(data=list(tokens[1:]), dtype="int64")},
    )


def future(value: Any) -> Future:
    result = Future()
    result.set_result(value)
    return result


def test_sources_agree_on_target_layout_without_changing_anchor() -> None:
    data = [datum(), datum((4, 5))]
    expected = [[-2.0, -3.0], [-5.0]]
    base = Mock()
    base.forward.return_value = future(
        SimpleNamespace(loss_fn_outputs=[{"logprobs": SimpleNamespace(data=row)} for row in expected])
    )
    sampler = Mock()
    sampler.compute_logprobs.side_effect = [future([None, *row]) for row in expected]
    batch = PolicyBatch(data=data, advantages=[1, -1], prompt_lens=[1, 1], rollout_logprobs=[[-7, -8], [-9]])
    for reference in [LoraReference(base), SamplerReference(sampler)]:
        enriched = with_reference_logprobs(batch, reference)
        assert enriched.ref_logprobs == expected
        assert enriched.anchor_logprobs is batch.anchor_logprobs
        assert enriched.rollout_logprobs is batch.rollout_logprobs
        assert enriched.data is batch.data
    assert batch.ref_logprobs == []
    assert [call.args[0].to_ints() for call in sampler.compute_logprobs.call_args_list] == [[1, 2, 3], [4, 5]]
    base.forward.assert_called_once_with(data, "cross_entropy")
    base.forward_backward.assert_not_called()


@pytest.mark.parametrize("scores", [[None, -1], [None, -1, None], [None, float("nan"), -1], [None, -1, float("inf")]])
def test_sampler_rejects_incomplete_or_invalid_target_scores(scores: list[float | None]) -> None:
    sampler = Mock()
    sampler.compute_logprobs.return_value = future(scores)
    with pytest.raises(ValueError, match="logprob length|unscored target"):
        SamplerReference(sampler).forward([datum()])


def test_sampler_rejects_unshifted_targets_before_request() -> None:
    data = datum()
    data.loss_fn_inputs["target_tokens"] = sdk.TensorData(data=[7, 8], dtype="int64")
    sampler = Mock()
    with pytest.raises(ValueError, match="shifted text"):
        SamplerReference(sampler).forward([data])
    sampler.compute_logprobs.assert_not_called()


@pytest.mark.parametrize("rows", [[], [[-1]], [[-1, -2], [-1, -2]], [[-1, float("nan")]]])
def test_batch_rejects_bad_trainer_output(rows: list[list[float]]) -> None:
    reference = Mock()
    reference.forward.return_value = SimpleNamespace(
        loss_fn_outputs=[{"logprobs": SimpleNamespace(data=row)} for row in rows]
    )
    batch = PolicyBatch(data=[datum()], advantages=[1], prompt_lens=[1], rollout_logprobs=[[-1, -2]])
    with pytest.raises(ValueError, match="align with target tokens"):
        with_reference_logprobs(batch, reference)


def test_forward_failure_propagates() -> None:
    failed = Future()
    failed.set_exception(RuntimeError("reference unavailable"))
    base = Mock()
    base.forward.return_value = failed
    with pytest.raises(RuntimeError, match="reference unavailable"):
        LoraReference(base).forward([datum()])


def test_lora_reference_uses_own_routing_and_preserves_policy_metadata() -> None:
    original = datum()
    original = sdk.Datum(
        model_input=sdk.ModelInput(
            chunks=original.model_input.chunks,
            routing_matrices=["policy-route-0", "policy-route-1"],
            top_sampling_references={"policy": "sampling-support"},
        ),
        loss_fn_inputs=original.loss_fn_inputs,
    )
    base = Mock()
    base.forward.return_value = future(SimpleNamespace(loss_fn_outputs=[]))
    LoraReference(base).forward([original])
    forwarded = base.forward.call_args.args[0][0]
    assert forwarded.model_input.chunks == original.model_input.chunks
    assert forwarded.loss_fn_inputs == original.loss_fn_inputs
    assert forwarded.model_input.routing_matrices is None
    assert forwarded.model_input.routing_references is None
    assert forwarded.model_input.top_sampling_references is None
    assert original.model_input.routing_matrices == ["policy-route-0", "policy-route-1"]
    assert original.model_input.top_sampling_references == {"policy": "sampling-support"}
