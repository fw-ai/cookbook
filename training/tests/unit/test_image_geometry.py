"""Qwen-VL aspect-ratio failures are dataset errors, not internal crashes."""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from training.image_geometry import ImageGeometryError, qwen_vl_aspect_ratio_message
from training.utils.runner import DatasetError
from training.utils.streaming import (
    JSONL_ROW_INDEX_KEY,
    dataset_error_for_rendered_row,
    raise_rendered_dataset_errors,
)
from training.utils.supervised import render_messages_to_datums
from training._vendor.tinker_cookbook_0_4_3.renderers.base import image_to_chunk


class _Qwen2VLImageProcessor:
    merge_size = 2

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
        images_kwargs: dict[str, Any] | None = None,
    ) -> int:
        del height, width, images_kwargs
        return self.merge_size**2


class _OtherImageProcessor:
    merge_size = 2

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
        images_kwargs: dict[str, Any] | None = None,
    ) -> int:
        del height, width, images_kwargs
        return self.merge_size**2


def test_qwen_processor_rejects_extreme_aspect_ratio() -> None:
    image = Image.new("RGB", (1, 630))
    with pytest.raises(ImageGeometryError, match=r"630\.0"):
        image_to_chunk(image, _Qwen2VLImageProcessor())


def test_qwen_processor_accepts_aspect_ratio_at_the_cap() -> None:
    image = Image.new("RGB", (1, 200))
    chunk = image_to_chunk(image, _Qwen2VLImageProcessor())
    assert chunk.expected_tokens == 1


def test_non_qwen_processor_keeps_extreme_aspect_ratio() -> None:
    image = Image.new("RGB", (1, 630))
    chunk = image_to_chunk(image, _OtherImageProcessor())
    assert chunk.expected_tokens == 1


def test_aspect_ratio_message_is_absent_at_the_cap() -> None:
    assert qwen_vl_aspect_ratio_message(1, 200) is None
    assert qwen_vl_aspect_ratio_message(1, 201) is not None


def test_render_messages_to_datums_converts_image_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise ImageGeometryError("Image aspect ratio 630.0 exceeds the maximum of 200")

    monkeypatch.setattr(
        "training.utils.supervised._build_renderer_supervised_examples",
        _boom,
    )
    with pytest.raises(DatasetError, match=r"630\.0") as raised:
        render_messages_to_datums(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
            renderer=object(),
        )
    assert isinstance(raised.value, DatasetError)
    assert not isinstance(raised.value, ImageGeometryError)


def test_rendered_row_error_keeps_public_message() -> None:
    row = {"messages": [{"role": "user", "content": "hi"}], JSONL_ROW_INDEX_KEY: 175}
    error = dataset_error_for_rendered_row(
        row,
        ImageGeometryError("Image aspect ratio 630.0 exceeds the maximum of 200"),
    )
    assert isinstance(error, DatasetError)
    assert str(error).startswith("row 175: Image aspect ratio 630.0")
    with pytest.raises(DatasetError, match="row 175"):
        raise_rendered_dataset_errors([error])
