from __future__ import annotations

import pytest

from training.renderer.muse_spark import RENDERER_NAMES
from training.renderer.thinking_trace import resolve_thinking_trace_renderer_plan
from training.utils.supervised import (
    renderer_supports_images,
    renderer_supports_tool_images,
    resolve_renderer_name,
)


@pytest.fixture(autouse=True)
def _disable_private_resolvers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "training.utils.supervised.resolve_renderer_name_from_plugins",
        lambda _tokenizer_model: None,
    )


def test_open_checkpoint_hf_identity_selects_public_renderer() -> None:
    assert resolve_renderer_name("meta-models/Muse-Spark-1.2-816B-A42B-NVFP4-open") == "muse_spark"
    assert (
        resolve_renderer_name("https://huggingface.co/meta-models/Muse-Spark-1.2-816B-A42B-NVFP4-open") == "muse_spark"
    )


def test_canonical_model_selects_public_renderer() -> None:
    assert resolve_renderer_name("accounts/fireworks/models/muse-spark-open-nvfp4") == "muse_spark"


def test_explicit_public_effort_override_is_preserved() -> None:
    assert (
        resolve_renderer_name("accounts/fireworks/models/muse-spark-open-nvfp4", "muse_spark_low") == "muse_spark_low"
    )


def test_public_renderer_retains_image_capabilities() -> None:
    assert "muse_spark" in RENDERER_NAMES
    assert renderer_supports_images("muse_spark")
    assert renderer_supports_tool_images("muse_spark")


def test_managed_default_snapshot_selects_public_renderer() -> None:
    renderer_id = "meta-models/Muse-Spark-1.2-816B-A42B-NVFP4-open"
    plan = resolve_thinking_trace_renderer_plan(
        renderer_id,
        requested_mode="unspecified",
        default_renderer_name=resolve_renderer_name(renderer_id),
    )
    assert plan.renderer_name == "muse_spark"


def test_unrecognized_internal_selector_is_not_inferred_as_public() -> None:
    with pytest.raises(ValueError):
        resolve_renderer_name("fireworks-internal/not-a-public-renderer")


def test_old_private_checkpoint_is_not_inferred_as_open_checkpoint() -> None:
    with pytest.raises(ValueError):
        resolve_renderer_name("accounts/pyroworks/models/muse-spark-1p2-820b-a42b-nvfp4-obsidian")


def test_unreviewed_model_suffix_is_not_inferred_as_open_checkpoint() -> None:
    with pytest.raises(ValueError):
        resolve_renderer_name("accounts/fireworks/models/muse-spark-open-nvfp4-unreviewed")
