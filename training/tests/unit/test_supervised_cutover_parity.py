"""Frozen behavior gates for Fireworks-owned supervised datum construction."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest
import tinker
import torch

from training._vendor.tinker_cookbook_0_4_3.exceptions import DataValidationError
import training._vendor.tinker_cookbook_0_4_3.supervised.common as vendored_supervised
import training.renderer.supervised as fireworks_supervised


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


def test_public_surface_exports_vendored_implementation() -> None:
    assert fireworks_supervised.compute_mean_nll is vendored_supervised.compute_mean_nll
    assert (
        fireworks_supervised.create_rightshifted_model_input_and_leftshifted_targets
        is vendored_supervised.create_rightshifted_model_input_and_leftshifted_targets
    )
    assert (
        fireworks_supervised.datum_from_model_input_weights
        is vendored_supervised.datum_from_model_input_weights
    )


def _chunk_layout(model_input: tinker.ModelInput) -> list[tuple[Any, ...]]:
    layout: list[tuple[Any, ...]] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            layout.append(("text", *chunk.tokens))
        else:
            layout.append(
                ("image", chunk.location, chunk.format, chunk.expected_tokens)
            )
    return layout


@pytest.mark.parametrize(
    (
        "input_factory",
        "weights",
        "max_length",
        "reduction",
        "expected_layout",
        "expected_targets",
        "expected_weights",
    ),
    [
        (
            _text_input,
            [0, 1, 2, 3, 4],
            None,
            "none",
            [("text", 10, 11, 12, 13)],
            [11, 12, 13, 14],
            [1, 2, 3, 4],
        ),
        (_text_input, [0, 1, 2, 3, 4], 3, "none", [("text", 10, 11)], [11, 12], [1, 2]),
        (
            _text_input,
            [0, 1, 2, 3, 4],
            None,
            "mean",
            [("text", 10, 11, 12, 13)],
            [11, 12, 13, 14],
            [0.1, 0.2, 0.3, 0.4],
        ),
        (
            _text_input,
            [0, 0, 0, 0, 0],
            None,
            "mean",
            [("text", 10, 11, 12, 13)],
            [11, 12, 13, 14],
            [0, 0, 0, 0],
        ),
        (
            _multimodal_input,
            [0, 0, 0, 0, 0, 1, 1, 1],
            None,
            "none",
            [
                ("text", 10, 11),
                ("image", "https://example.com/image.png", "png", 3),
                ("text", 12, 13),
            ],
            [11, 0, 0, 0, 12, 13, 14],
            [0, 0, 0, 0, 1, 1, 1],
        ),
        (
            _multimodal_input,
            [0, 0, 0, 0, 0, 1, 1, 1],
            7,
            "none",
            [
                ("text", 10, 11),
                ("image", "https://example.com/image.png", "png", 3),
                ("text", 12),
            ],
            [11, 0, 0, 0, 12, 13],
            [0, 0, 0, 0, 1, 1],
        ),
        (
            _multimodal_input,
            [0, 0, 0, 0, 0, 1, 1, 1],
            4,
            "none",
            [("text", 10)],
            [11],
            [0],
        ),
        (
            _trailing_image_input,
            [0, 1, 1, 0, 0],
            None,
            "none",
            [("text", 10, 11)],
            [11, 12],
            [1, 1],
        ),
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
def test_serialized_datum_matches_frozen_snapshot(
    input_factory: Callable[[], tinker.ModelInput],
    weights: list[float],
    max_length: int | None,
    reduction: str,
    expected_layout: list[tuple[Any, ...]],
    expected_targets: list[int],
    expected_weights: list[float],
) -> None:
    fireworks = fireworks_supervised.datum_from_model_input_weights(
        input_factory(),
        torch.tensor(weights, dtype=torch.float32),
        max_length=max_length,
        reduction=reduction,
    )
    assert _chunk_layout(fireworks.model_input) == expected_layout
    assert fireworks.loss_fn_inputs["target_tokens"].data == expected_targets
    assert fireworks.loss_fn_inputs["weights"].data == pytest.approx(expected_weights)


@pytest.mark.parametrize("input_factory", [_text_input, _multimodal_input])
def test_input_target_shift_is_stable(
    input_factory: Callable[[], tinker.ModelInput],
) -> None:
    fireworks_input = input_factory()
    input_model, targets = (
        fireworks_supervised.create_rightshifted_model_input_and_leftshifted_targets(
            list(fireworks_input.chunks)
        )
    )
    if input_factory is _text_input:
        assert _chunk_layout(input_model) == [("text", 10, 11, 12, 13)]
        assert targets == [11, 12, 13, 14]
    else:
        assert _chunk_layout(input_model) == [
            ("text", 10, 11),
            ("image", "https://example.com/image.png", "png", 3),
            ("text", 12, 13),
        ]
        assert targets == [11, 0, 0, 0, 12, 13, 14]


@pytest.mark.parametrize(
    ("chunks", "expected_type", "expected_message"),
    [
        ([], AssertionError, "must have at least one chunk"),
        (
            [tinker.EncodedTextChunk(tokens=[1])],
            DataValidationError,
            "need at least 2 tokens",
        ),
        (
            [
                tinker.types.ImageAssetPointerChunk(
                    location="https://example.com/image.png",
                    format="png",
                    expected_tokens=2,
                )
            ],
            DataValidationError,
            "The last chunk must be a text chunk",
        ),
    ],
    ids=["empty", "one-token", "trailing-image"],
)
def test_input_target_shift_error_is_stable(
    chunks: list[tinker.ModelInputChunk],
    expected_type: type[BaseException],
    expected_message: str,
) -> None:
    with pytest.raises(expected_type, match=expected_message):
        fireworks_supervised.create_rightshifted_model_input_and_leftshifted_targets(
            chunks
        )


def test_unknown_reduction_error_is_stable() -> None:
    with pytest.raises(ValueError, match="Unknown reduction mode: 'sum'"):
        fireworks_supervised.datum_from_model_input_weights(
            _text_input(), torch.ones(5), reduction="sum"
        )


def test_compute_mean_nll_is_stable() -> None:
    logprobs = [
        tinker.TensorData(data=[-1.0, -2.0], dtype="float32", shape=[2]),
        tinker.TensorData(data=[-3.0], dtype="float32", shape=[1]),
    ]
    weights = [
        tinker.TensorData(data=[0.0, 1.0], dtype="float32", shape=[2]),
        tinker.TensorData(data=[2.0], dtype="float32", shape=[1]),
    ]
    assert fireworks_supervised.compute_mean_nll(logprobs, weights) == pytest.approx(
        8 / 3
    )

    zero_weights = [
        tinker.TensorData(data=[0.0, 0.0], dtype="float32", shape=[2]),
        tinker.TensorData(data=[0.0], dtype="float32", shape=[1]),
    ]
    assert math.isnan(fireworks_supervised.compute_mean_nll(logprobs, zero_weights))
