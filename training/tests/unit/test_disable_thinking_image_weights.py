"""Disable-thinking loss masking must accept multimodal supervised examples."""

from __future__ import annotations

import torch
from fireworks.training import sdk

from training.renderer._think_prefill import DisableThinkingWeightsMixin

_THINK_OPEN = 12
_THINK_CLOSE = 13


class _MarkerTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return {"<think>": [_THINK_OPEN], "</think>": [_THINK_CLOSE]}[text]


class _ImageExampleRenderer:
    def build_supervised_example(self, messages: list[dict], **kwargs: object) -> tuple[sdk.ModelInput, torch.Tensor]:
        image = sdk.types.ImageChunk(data=b"png", format="png", expected_tokens=3)
        model_input = sdk.ModelInput(
            chunks=[
                sdk.EncodedTextChunk(tokens=[1, 2]),
                image,
                sdk.EncodedTextChunk(tokens=[_THINK_OPEN, _THINK_CLOSE, 7, 8]),
            ]
        )
        return model_input, torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)


class _Renderer(DisableThinkingWeightsMixin, _ImageExampleRenderer):
    tokenizer = _MarkerTokenizer()


def test_markers_after_an_image_are_masked_at_expanded_positions() -> None:
    model_input, weights = _Renderer().build_supervised_example(messages=[])
    assert model_input.length == len(weights)
    assert weights.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1]
