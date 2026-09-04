"""Differential gates for supervised datum ownership.

The installed ``tinker-cookbook==0.4.3`` package is a temporary oracle. These
tests compare the Fireworks-owned surface against it without making the legacy
namespace a static runtime dependency.
"""

from __future__ import annotations

import importlib
import inspect
import math
from collections.abc import Callable
from typing import Any

import pytest
import tinker
import torch

import training._vendor.tinker_cookbook_0_4_3.supervised.common as vendored_supervised
import training.renderer.supervised as fireworks_supervised

legacy_supervised = importlib.import_module("tinker_cookbook.supervised.common")


def _value_snapshot(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, dict):
        return {key: _value_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value_snapshot(item) for item in value]
    if hasattr(value, "tolist"):
        return _value_snapshot(value.tolist())
    if hasattr(value, "__dict__"):
        return {
            "type": type(value).__name__,
            "fields": _value_snapshot(vars(value)),
        }
    return value


def _datum_snapshot(datum: tinker.Datum) -> dict[str, Any]:
    return {
        "model_input": _value_snapshot(datum.model_input),
        "loss_fn_inputs": _value_snapshot(datum.loss_fn_inputs),
    }


def _text_input() -> tinker.ModelInput:
    return tinker.ModelInput.from_ints([10, 11, 12, 13, 14])


def _multimodal_input() -> tinker.ModelInput:
    return tinker.ModelInput(
        chunks=[
            tinker.EncodedTextChunk(tokens=[10, 11]),
            tinker.types.ImageAssetPointerChunk(
                location="https://example.com/image.png",
                format="png",
                expected_tokens=3,
            ),
            tinker.EncodedTextChunk(tokens=[12, 13, 14]),
        ]
    )


def _trailing_image_input() -> tinker.ModelInput:
    return tinker.ModelInput(
        chunks=[
            tinker.EncodedTextChunk(tokens=[10, 11, 12]),
            tinker.types.ImageAssetPointerChunk(
                location="https://example.com/image.png",
                format="png",
                expected_tokens=2,
            ),
        ]
    )


def test_vendored_source_matches_pinned_package_except_namespace() -> None:
    vendored_source = inspect.getsource(vendored_supervised).replace(
        "from ..exceptions import DataValidationError",
        "from tinker_cookbook.exceptions import DataValidationError",
    )
    assert vendored_source == inspect.getsource(legacy_supervised)


def test_public_surface_exports_vendored_implementation() -> None:
    assert fireworks_supervised.compute_mean_nll is vendored_supervised.compute_mean_nll
    assert (
        fireworks_supervised.create_rightshifted_model_input_and_leftshifted_targets
        is vendored_supervised.create_rightshifted_model_input_and_leftshifted_targets
    )
    assert fireworks_supervised.datum_from_model_input_weights is vendored_supervised.datum_from_model_input_weights


@pytest.mark.parametrize(
    ("input_factory", "weights", "max_length", "reduction"),
    [
        (_text_input, [0, 1, 2, 3, 4], None, "none"),
        (_text_input, [0, 1, 2, 3, 4], 3, "none"),
        (_text_input, [0, 1, 2, 3, 4], None, "mean"),
        (_text_input, [0, 0, 0, 0, 0], None, "mean"),
        (_multimodal_input, [0, 0, 0, 0, 0, 1, 1, 1], None, "none"),
        (_multimodal_input, [0, 0, 0, 0, 0, 1, 1, 1], 7, "none"),
        (_multimodal_input, [0, 0, 0, 0, 0, 1, 1, 1], 4, "none"),
        (_trailing_image_input, [0, 1, 1, 0, 0], None, "none"),
    ],
    ids=[
        "text",
        "partial-text-truncation",
        "mean-reduction",
        "zero-weight-mean",
        "multimodal",
        "partial-multimodal-text-truncation",
        "whole-image-truncation",
        "trailing-image-removal",
    ],
)
def test_serialized_datum_matches_pinned_package(
    input_factory: Callable[[], tinker.ModelInput],
    weights: list[float],
    max_length: int | None,
    reduction: str,
) -> None:
    legacy = legacy_supervised.datum_from_model_input_weights(
        input_factory(),
        torch.tensor(weights, dtype=torch.float32),
        max_length=max_length,
        reduction=reduction,
    )
    fireworks = fireworks_supervised.datum_from_model_input_weights(
        input_factory(),
        torch.tensor(weights, dtype=torch.float32),
        max_length=max_length,
        reduction=reduction,
    )
    assert _datum_snapshot(fireworks) == _datum_snapshot(legacy)


@pytest.mark.parametrize("input_factory", [_text_input, _multimodal_input])
def test_input_target_shift_matches_pinned_package(
    input_factory: Callable[[], tinker.ModelInput],
) -> None:
    legacy_input = input_factory()
    fireworks_input = input_factory()
    legacy_result = legacy_supervised.create_rightshifted_model_input_and_leftshifted_targets(
        list(legacy_input.chunks)
    )
    fireworks_result = fireworks_supervised.create_rightshifted_model_input_and_leftshifted_targets(
        list(fireworks_input.chunks)
    )
    assert _value_snapshot(fireworks_result) == _value_snapshot(legacy_result)


@pytest.mark.parametrize(
    "chunks",
    [
        [],
        [tinker.EncodedTextChunk(tokens=[1])],
        [
            tinker.types.ImageAssetPointerChunk(
                location="https://example.com/image.png",
                format="png",
                expected_tokens=2,
            )
        ],
    ],
    ids=["empty", "one-token", "trailing-image"],
)
def test_input_target_shift_error_matches_pinned_package(
    chunks: list[tinker.ModelInputChunk],
) -> None:
    with pytest.raises(BaseException) as legacy_error:
        legacy_supervised.create_rightshifted_model_input_and_leftshifted_targets(chunks)
    with pytest.raises(BaseException) as fireworks_error:
        fireworks_supervised.create_rightshifted_model_input_and_leftshifted_targets(chunks)

    assert type(fireworks_error.value).__name__ == type(legacy_error.value).__name__
    assert str(fireworks_error.value) == str(legacy_error.value)


def test_unknown_reduction_error_matches_pinned_package() -> None:
    with pytest.raises(ValueError) as legacy_error:
        legacy_supervised.datum_from_model_input_weights(_text_input(), torch.ones(5), reduction="sum")
    with pytest.raises(ValueError) as fireworks_error:
        fireworks_supervised.datum_from_model_input_weights(_text_input(), torch.ones(5), reduction="sum")
    assert str(fireworks_error.value) == str(legacy_error.value)


def test_compute_mean_nll_matches_pinned_package() -> None:
    logprobs = [
        tinker.TensorData(data=[-1.0, -2.0], dtype="float32", shape=[2]),
        tinker.TensorData(data=[-3.0], dtype="float32", shape=[1]),
    ]
    weights = [
        tinker.TensorData(data=[0.0, 1.0], dtype="float32", shape=[2]),
        tinker.TensorData(data=[2.0], dtype="float32", shape=[1]),
    ]
    assert fireworks_supervised.compute_mean_nll(logprobs, weights) == pytest.approx(
        legacy_supervised.compute_mean_nll(logprobs, weights)
    )

    zero_weights = [
        tinker.TensorData(data=[0.0, 0.0], dtype="float32", shape=[2]),
        tinker.TensorData(data=[0.0], dtype="float32", shape=[1]),
    ]
    assert math.isnan(fireworks_supervised.compute_mean_nll(logprobs, zero_weights))
    assert math.isnan(legacy_supervised.compute_mean_nll(logprobs, zero_weights))
