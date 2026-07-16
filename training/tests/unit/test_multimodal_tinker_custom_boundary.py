"""Pinned Tinker client coverage for the multimodal custom-loss wire boundary."""

from __future__ import annotations

import asyncio
from importlib.metadata import version
from typing import Any

import pytest
import tinker
import torch
from tinker._compat import model_dump
from tinker.lib._pydantic_conv import to_pydantic_request
from tinker.lib.public_interfaces.api_future import APIFuture
from tinker.lib.public_interfaces.training_client import TrainingClient
from tinker.types import ImageChunk

from training.utils.rl.grpo import make_grpo_loss_fn
from training.utils.supervised import build_multimodal_policy_datum


class _ImmediateFuture(APIFuture[Any]):
    """Minimal completed future used in place of the HTTP transport."""

    def __init__(self, value: Any) -> None:
        self.value = value

    async def result_async(self, timeout: float | None = None) -> Any:
        del timeout
        return self.value

    def result(self, timeout: float | None = None) -> Any:
        del timeout
        return self.value


class _CapturingTrainingClient(TrainingClient):
    """Use Tinker's real custom-loss method with immediate forward transports."""

    def __init__(self, forward_result: tinker.ForwardBackwardOutput) -> None:
        # ``_CombinedAPIFuture.result_async`` only stores this holder. The fake
        # transports below replace the methods that otherwise need its HTTP and
        # request-sequencing state.
        self.holder = object()
        self.forward_result = forward_result
        self.forward_calls: list[tuple[list[tinker.Datum], str, dict | None]] = []
        self.forward_backward_calls: list[
            tuple[list[tinker.Datum], str, dict | None]
        ] = []

    async def forward_async(
        self,
        data: list[tinker.Datum],
        loss_fn: str,
        loss_fn_config: dict | None = None,
    ) -> APIFuture[tinker.ForwardBackwardOutput]:
        self.forward_calls.append((data, loss_fn, loss_fn_config))
        return _ImmediateFuture(self.forward_result)

    async def forward_backward_async(
        self,
        data: list[tinker.Datum],
        loss_fn: str,
        loss_fn_config: dict | None = None,
    ) -> APIFuture[tinker.ForwardBackwardOutput]:
        self.forward_backward_calls.append((data, loss_fn, loss_fn_config))
        return _ImmediateFuture(
            tinker.ForwardBackwardOutput(
                loss_fn_output_type="cross_entropy",
                loss_fn_outputs=[{}],
                metrics={"surrogate_loss": 0.0},
            )
        )


def test_tinker_custom_grpo_keeps_expanded_multimodal_wire_coordinates() -> None:
    """Exercise Tinker 0.23.0's real two-pass custom-loss implementation."""

    assert version("tinker") == "0.23.0"

    prompt = tinker.ModelInput(
        chunks=[
            tinker.EncodedTextChunk(tokens=[10, 11]),
            ImageChunk(
                data=b"test-image-bytes",
                format="png",
                expected_tokens=2,
            ),
            tinker.EncodedTextChunk(tokens=[12]),
        ]
    )
    datum = build_multimodal_policy_datum(prompt, [30, 31])
    target_tokens = list(datum.loss_fn_inputs["target_tokens"].data)
    target_weights = [float(x) for x in datum.loss_fn_inputs["weights"].data]

    assert datum.model_input.length == 6
    assert target_tokens == [11, 0, 0, 12, 30, 31]
    assert target_weights == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

    # FireTitan's one-dimensional forward response omits ``shape``. Tinker's
    # real custom path must therefore retain the expanded flat six-token tensor
    # rather than trying to reshape it to a compressed text-only target shape.
    expanded_forward_logprobs = [-3.0, -4.0, -5.0, -6.0, -0.25, -0.5]
    forward_result = tinker.ForwardBackwardOutput(
        loss_fn_output_type="cross_entropy",
        loss_fn_outputs=[
            {
                "logprobs": tinker.TensorData(
                    data=expanded_forward_logprobs,
                    dtype="float32",
                    shape=None,
                )
            }
        ],
        metrics={},
    )
    client = _CapturingTrainingClient(forward_result)

    grpo_loss = make_grpo_loss_fn(
        [1.0],
        [[0.0] * 6],
        [5],
        inf_logprobs=[list(expanded_forward_logprobs)],
        old_policy_logprobs=[list(expanded_forward_logprobs)],
        kl_beta=0.0,
    )
    observed_logprobs: list[torch.Tensor] = []

    def capture_loss(
        data: list[tinker.Datum], logprobs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        observed_logprobs.extend(logprobs)
        return grpo_loss(data, logprobs)

    async def run_custom_path() -> tinker.ForwardBackwardOutput:
        future = await client.forward_backward_custom_async([datum], capture_loss)
        return await future.result_async()

    result = asyncio.run(run_custom_path())

    assert len(client.forward_calls) == 1
    forward_data, forward_loss_fn, forward_config = client.forward_calls[0]
    assert forward_data == [datum]
    assert forward_loss_fn == "cross_entropy"
    assert forward_config is None

    assert len(observed_logprobs) == 1
    assert observed_logprobs[0].shape == (6,)
    assert observed_logprobs[0].grad is not None

    assert len(client.forward_backward_calls) == 1
    backward_data, backward_loss_fn, backward_config = client.forward_backward_calls[0]
    assert backward_loss_fn == "cross_entropy"
    assert backward_config is None
    assert len(backward_data) == 1

    wire_datum = backward_data[0]
    wire_targets = wire_datum.loss_fn_inputs["target_tokens"]
    wire_weights = wire_datum.loss_fn_inputs["weights"]
    expected_weights = (-observed_logprobs[0].grad).tolist()

    assert wire_datum.model_input == datum.model_input
    assert wire_targets.data == target_tokens
    assert wire_targets.shape == [6]
    assert wire_weights.shape == [6]
    assert wire_weights.data == pytest.approx(expected_weights)

    # Exercise the exact Pydantic conversion + model dump used by Tinker's
    # JSON HTTP transport as well as the in-memory Datum passed above.
    wire_request = tinker.types.ForwardBackwardRequest(
        forward_backward_input=tinker.types.ForwardBackwardInput(
            data=[wire_datum],
            loss_fn="cross_entropy",
            loss_fn_config=None,
        ),
        model_id="test-model",
        seq_id=1,
    )
    wire_payload = model_dump(
        to_pydantic_request(wire_request),
        exclude_unset=False,
        exclude_none=True,
        mode="json",
    )
    serialized_inputs = wire_payload["forward_backward_input"]["data"][0][
        "loss_fn_inputs"
    ]
    assert serialized_inputs["target_tokens"] == {
        "data": target_tokens,
        "dtype": "int64",
        "shape": [6],
    }
    assert serialized_inputs["weights"]["shape"] == [6]
    assert serialized_inputs["weights"]["data"] == pytest.approx(expected_weights)

    # Image target positions are 1 and 2. The actual GRPO loss masks every
    # prompt/image position, while both completion positions carry gradients.
    assert wire_weights.data[1] == pytest.approx(0.0)
    assert wire_weights.data[2] == pytest.approx(0.0)
    assert all(w == pytest.approx(0.0) for w in wire_weights.data[:4])
    assert all(abs(w) > 0.0 for w in wire_weights.data[4:])

    # ``_CombinedAPIFuture`` from the pinned client also carries custom metrics
    # from the real GRPO closure onto the backward result.
    assert result.metrics["active_tokens"] == 2
    assert result.metrics["surrogate_loss"] == pytest.approx(0.0)
