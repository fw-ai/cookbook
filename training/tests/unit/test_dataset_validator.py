"""Renderer-owned dataset rules stay small and family-scoped."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from training.dataset_utils.validation import image_issue, register_image_rule


def test_dataset_checks_import_without_training_dependencies() -> None:
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "from training.dataset_utils.validation import image_issue; "
            "assert image_issue('qwen3_vl', width=1, height=201) is not None",
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
    )


def test_legacy_imports_share_checks_without_training_dependencies() -> None:
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "from training import dataset_validator, image_geometry; "
            "from training.dataset_utils import validation, image_geometry as geometry; "
            "assert dataset_validator.image_issue is validation.image_issue; "
            "assert dataset_validator.register_image_rule is validation.register_image_rule; "
            "assert image_geometry.ImageGeometryError is geometry.ImageGeometryError; "
            "assert image_geometry.reject_qwen_vl_aspect_ratio is geometry.reject_qwen_vl_aspect_ratio; "
            "assert dataset_validator.image_issue('qwen3_vl', width=1, height=201) is not None",
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
    )


def test_dataset_helpers_do_not_shadow_hugging_face_datasets() -> None:
    # CI and editable installs also run with the training directory on sys.path.
    subprocess.run(
        [sys.executable, "-c", "from datasets import load_dataset; assert callable(load_dataset)"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )


def test_qwen_family_rejects_extreme_aspect_ratio() -> None:
    message = image_issue("qwen3_5_interleaved", width=1, height=630)
    assert message is not None
    assert "630.0" in message
    assert "200" in message


def test_qwen_family_accepts_the_aspect_ratio_cap() -> None:
    assert image_issue("qwen3_vl_instruct", width=1, height=200) is None


def test_other_vision_families_are_not_subject_to_the_qwen_cap() -> None:
    assert image_issue("muse_glimmer", width=1, height=630) is None
    assert image_issue("", width=1, height=630) is None


def test_registered_rule_applies_only_to_its_renderer_family() -> None:
    def _too_wide(width: int, height: int) -> str | None:
        if width > height * 3:
            return "custom family rejects this image"
        return None

    register_image_rule(("custom_vl",), _too_wide)
    assert image_issue("custom_vl_preview", width=10, height=1) == (
        "custom family rejects this image"
    )
    assert image_issue("qwen3_5_interleaved", width=10, height=1) is None
