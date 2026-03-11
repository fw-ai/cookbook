"""Tests that multimodal datum building excludes image placeholders from target_tokens.

The RLOR server uses ``target_tokens`` to gather logprobs at each position.
If ``target_tokens`` contains zeros at image-chunk positions, the server
interprets them as real token IDs and gathers logprobs for token-0, which
corrupts the loss computation (producing CE loss ~20 instead of ~1).

The fix: ``_build_multimodal_datum`` strips image positions from
``target_tokens`` so only real text token IDs are sent.  These tests verify
that invariant for several multimodal input shapes.
"""

from __future__ import annotations

import pytest
import torch
import tinker

from training.utils.supervised import (
    build_datum_from_model_input_and_weights,
    render_messages_to_datum,
)
from tinker_cookbook.renderers import TrainOnWhat


def _make_multimodal_model_input(
    prefix_tokens: list[int],
    image_expected_tokens: int,
    suffix_tokens: list[int],
) -> tinker.ModelInput:
    """Build a ModelInput: [text] [image] [text]."""
    return tinker.ModelInput(
        chunks=[
            tinker.EncodedTextChunk(tokens=prefix_tokens),
            tinker.types.ImageAssetPointerChunk(
                location="https://example.com/img.png",
                format="png",
                expected_tokens=image_expected_tokens,
            ),
            tinker.EncodedTextChunk(tokens=suffix_tokens),
        ]
    )


class TestMultimodalTargetTokensExcludeImageZeros:
    """Core invariant: target_tokens must contain zero placeholder zeros."""

    def test_simple_image_between_text(self):
        model_input = _make_multimodal_model_input(
            prefix_tokens=[10, 11],
            image_expected_tokens=3,
            suffix_tokens=[12, 13],
        )
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

        rendered = build_datum_from_model_input_and_weights(model_input, weights)
        targets = list(rendered.datum.loss_fn_inputs["target_tokens"].data)

        assert 0 not in targets, (
            f"target_tokens must not contain image placeholder zeros, got {targets}"
        )
        assert targets == [11, 12, 13]

    def test_large_image_chunk(self):
        """192 image tokens (typical Qwen3-VL) must not leak into target_tokens."""
        model_input = _make_multimodal_model_input(
            prefix_tokens=[1, 2, 3],
            image_expected_tokens=192,
            suffix_tokens=[4, 5, 6, 7],
        )
        n = 3 + 192 + 4
        weights = [0.0] * (3 + 192) + [1.0] * 4

        rendered = build_datum_from_model_input_and_weights(model_input, weights)
        targets = list(rendered.datum.loss_fn_inputs["target_tokens"].data)

        assert 0 not in targets
        assert targets == [2, 3, 4, 5, 6, 7]

    def test_multiple_images(self):
        """Two image chunks — neither should produce zeros in target_tokens."""
        model_input = tinker.ModelInput(
            chunks=[
                tinker.EncodedTextChunk(tokens=[10, 11]),
                tinker.types.ImageAssetPointerChunk(
                    location="img1", format="png", expected_tokens=2,
                ),
                tinker.EncodedTextChunk(tokens=[12]),
                tinker.types.ImageAssetPointerChunk(
                    location="img2", format="png", expected_tokens=2,
                ),
                tinker.EncodedTextChunk(tokens=[13, 14]),
            ]
        )
        total = 2 + 2 + 1 + 2 + 2  # 9
        weights = [0.0] * 7 + [1.0] * 2

        rendered = build_datum_from_model_input_and_weights(model_input, weights)
        targets = list(rendered.datum.loss_fn_inputs["target_tokens"].data)

        assert 0 not in targets
        assert targets == [11, 12, 13, 14]

    def test_all_text_tokens_preserved_in_order(self):
        """Text tokens from all chunks appear in target_tokens in sequence order."""
        model_input = _make_multimodal_model_input(
            prefix_tokens=[100, 200],
            image_expected_tokens=5,
            suffix_tokens=[300, 400, 500],
        )
        weights = [0.0] * 7 + [1.0] * 3

        rendered = build_datum_from_model_input_and_weights(model_input, weights)
        targets = list(rendered.datum.loss_fn_inputs["target_tokens"].data)

        assert targets == [200, 300, 400, 500]


class TestMultimodalWeightsAlignment:
    """Weights must still span the full model_input (including image positions)."""

    def test_weights_length_matches_shifted_input(self):
        model_input = _make_multimodal_model_input(
            prefix_tokens=[10, 11],
            image_expected_tokens=3,
            suffix_tokens=[12, 13],
        )
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

        rendered = build_datum_from_model_input_and_weights(model_input, weights)
        datum_weights = list(rendered.datum.loss_fn_inputs["weights"].data)

        assert len(datum_weights) == len(weights) - 1
        assert datum_weights == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

    def test_image_positions_have_zero_weight(self):
        """Weights at image positions should be 0 so images don't contribute to loss."""
        model_input = _make_multimodal_model_input(
            prefix_tokens=[10, 11],
            image_expected_tokens=3,
            suffix_tokens=[12, 13],
        )
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

        rendered = build_datum_from_model_input_and_weights(model_input, weights)
        datum_weights = list(rendered.datum.loss_fn_inputs["weights"].data)

        image_start = 1  # after shift, image occupies positions 1..3
        image_end = 4
        for i in range(image_start, image_end):
            assert datum_weights[i] == 0.0, (
                f"Weight at image position {i} should be 0.0, got {datum_weights[i]}"
            )


class TestMultimodalModelInputPreservesChunks:
    """The datum's model_input must still contain image chunks (not flattened to ints)."""

    def test_image_chunk_preserved_in_datum(self):
        model_input = _make_multimodal_model_input(
            prefix_tokens=[10, 11],
            image_expected_tokens=3,
            suffix_tokens=[12, 13],
        )
        weights = [0.0] * 5 + [1.0] * 2

        rendered = build_datum_from_model_input_and_weights(model_input, weights)
        chunks = list(rendered.datum.model_input.chunks)

        non_text = [c for c in chunks if not isinstance(c, tinker.types.EncodedTextChunk)]
        assert len(non_text) == 1, "Image chunk must be preserved in the datum's model_input"

    def test_truncation_removes_last_text_token_only(self):
        model_input = _make_multimodal_model_input(
            prefix_tokens=[10, 11],
            image_expected_tokens=3,
            suffix_tokens=[12, 13],
        )
        weights = [0.0] * 5 + [1.0] * 2

        rendered = build_datum_from_model_input_and_weights(model_input, weights)
        assert rendered.datum.model_input.length == model_input.length - 1


class TestRendererIntegrationWithMultimodalDatum:
    """Verify the full render_messages_to_datum path excludes image zeros."""

    def test_via_stub_renderer(self):
        """A renderer returning ModelInput with images routes through _build_multimodal_datum."""

        class _VLRenderer:
            def build_supervised_example(self, messages, train_on_what=None):
                mi = _make_multimodal_model_input(
                    prefix_tokens=[10, 11],
                    image_expected_tokens=4,
                    suffix_tokens=[12, 13, 14],
                )
                w = torch.tensor(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    dtype=torch.float32,
                )
                return mi, w

        rendered = render_messages_to_datum(
            [
                {"role": "user", "content": "describe the image"},
                {"role": "assistant", "content": "a cat"},
            ],
            renderer=_VLRenderer(),
        )

        targets = list(rendered.datum.loss_fn_inputs["target_tokens"].data)
        assert 0 not in targets
        assert targets == [11, 12, 13, 14]


class TestTextOnlyPathUnchanged:
    """Text-only inputs should still go through the standard datum builder."""

    def test_text_only_model_input_uses_upstream_builder(self):
        model_input = tinker.ModelInput(
            chunks=[tinker.EncodedTextChunk(tokens=[10, 11, 12, 13])]
        )
        weights = [0.0, 0.0, 1.0, 1.0]

        rendered = build_datum_from_model_input_and_weights(model_input, weights)
        targets = list(rendered.datum.loss_fn_inputs["target_tokens"].data)

        assert targets == [11, 12, 13]
