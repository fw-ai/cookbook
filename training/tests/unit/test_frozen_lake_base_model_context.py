"""Base-model loading context propagation for the FrozenLake rollout path.

The selected ``base_model`` is local completion metadata: it threads from
``completion_params`` into the cached tokenizer/renderer helpers and the
standalone image client, so an opaque tokenizer override inherits the base
model's loading policy. It never appears in the inference request JSON, and
the deployed sampling model id is never a base identity. Canaries use the
tampered V4 dir with an injected ``auto_map`` marker from
``test_v4_example_context``: the marker file exists iff staged remote code
actually executed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from eval_protocol.models import EvaluationRow, InputMetadata
from eval_protocol.pytest.types import RolloutProcessorConfig

import training.examples.rl.frozen_lake.frozen_lake_rollout as rollout
from training.examples.rl.frozen_lake.frozen_lake_rollout import (
    FireworksV1ImageCompletionsClient,
    FrozenLakeToolRolloutProcessor,
    _merge_request_params,
)
from training.examples.rl.frozen_lake.train_frozen_lake import completion_params
from training.recipes.async_rl_loop import RolloutSetup
from training.tests.unit.test_frozen_lake_visual_rollout import _FakeImageClient
from training.tests._v4_tokenizer_data import (
    V4_ALIAS,
    assert_stock_tokenizer as _assert_stock_tokenizer,
    tampered_v4_dir as _tampered_v4_dir,
    v4_files as v4_files,
)

transformers = pytest.importorskip("transformers")


@pytest.fixture(autouse=True)
def _clear_loader_caches():
    # Capture the cached helpers up front: tests may monkeypatch the module
    # attributes, so teardown must clear the originals, not the replacements.
    cached_tokenizer = rollout._get_hf_tokenizer
    cached_renderer = rollout._get_renderer
    cached_tokenizer.cache_clear()
    cached_renderer.cache_clear()
    yield
    cached_tokenizer.cache_clear()
    cached_renderer.cache_clear()


def _setup(**overrides) -> RolloutSetup:
    kwargs = dict(
        sampler=None,
        tokenizer=None,
        tokenizer_id="Qwen/Qwen3-8B",
        sample_kwargs={"temperature": 1.0, "max_tokens": 64, "top_p": 1.0, "top_k": 0},
        completions_per_prompt=2,
        inference_base_url="https://api.fireworks.ai",
        api_key="key",
        model="accounts/a/models/m#accounts/a/deployments/d",
    )
    kwargs.update(overrides)
    return RolloutSetup(**kwargs)


def test_completion_params_propagates_selected_base_model() -> None:
    params = completion_params(_setup(base_model=V4_ALIAS), None)
    assert params["base_model"] == V4_ALIAS
    assert params["tokenizer_name_or_path"] == "Qwen/Qwen3-8B"


def test_completion_params_omit_base_model_when_unselected() -> None:
    params = completion_params(_setup(), None)
    assert "base_model" not in params


def test_base_model_stays_out_of_request_json() -> None:
    merged = _merge_request_params(
        completion_params={"model": "m", "base_model": V4_ALIAS, "top_p": 0.9, "extra_body": {"foo": 1}},
        processor_kwargs={},
        default_request_params={"bar": 2},
    )
    assert "base_model" not in merged
    assert merged == {"bar": 2, "foo": 1, "top_p": 0.9}


def test_cached_helper_permissive_executes_then_v4_context_inert(tmp_path: Path, v4_files: dict[str, Path]) -> None:
    """One tampered dir, two cache entries: the permissive load runs the
    staged canary; the selected V4 context returns a different safe object
    with the canary inert and identical tokens."""
    directory, marker = _tampered_v4_dir(v4_files, tmp_path, name="fl-cached")
    permissive = rollout._get_hf_tokenizer(str(directory))
    assert marker.exists(), "legacy permissive load must execute the staged canary"
    marker.unlink()
    safe = rollout._get_hf_tokenizer(str(directory), V4_ALIAS)
    assert safe is not permissive, "base context must produce a distinct cache entry"
    assert not marker.exists(), "canary must stay inert under the V4 base_model context"
    _assert_stock_tokenizer(safe, directory)


def test_renderer_cache_isolates_base_model_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, v4_files: dict[str, Path]
) -> None:
    """The renderer helper threads the context into its inner tokenizer load
    and caches per (path, context)."""
    directory, marker = _tampered_v4_dir(v4_files, tmp_path, name="fl-renderer")
    stock = transformers.AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False)

    def fake_build_renderer(tokenizer: Any, name: str, **kwargs: Any) -> object:
        if kwargs.get("base_model") == V4_ALIAS:
            assert type(tokenizer) is type(
                stock
            ), f"V4-keyed renderer build must receive the stock class, got {type(tokenizer).__name__}"
        return object()

    monkeypatch.setattr(rollout, "build_renderer", fake_build_renderer)
    rollout._get_renderer(str(directory))
    assert marker.exists(), "permissive renderer load must execute the staged canary"
    marker.unlink()
    renderer_v4 = rollout._get_renderer(str(directory), V4_ALIAS)
    assert not marker.exists(), "base context must reach the renderer's inner load"
    assert rollout._get_renderer(str(directory), V4_ALIAS) is renderer_v4
    assert rollout._get_renderer(str(directory)) is not renderer_v4


def test_image_client_uses_selected_base_context(tmp_path: Path, v4_files: dict[str, Path]) -> None:
    """Default client context is permissive; a selected V4 base keeps the
    staged canary inert through the client's own tokenizer load."""
    directory, marker = _tampered_v4_dir(v4_files, tmp_path, name="fl-client")
    permissive_client = FireworksV1ImageCompletionsClient(model_id="m", tokenizer_name_or_path=str(directory))
    permissive_client._get_tokenizer()
    assert marker.exists(), "default context keeps the legacy permissive policy"
    marker.unlink()
    safe_client = FireworksV1ImageCompletionsClient(
        model_id="m", tokenizer_name_or_path=str(directory), base_model=V4_ALIAS
    )
    safe = safe_client._get_tokenizer()
    assert not marker.exists(), "canary must stay inert under the V4 base_model context"
    _assert_stock_tokenizer(safe, directory)


@pytest.mark.asyncio
async def test_processor_call_threads_selected_base_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, v4_files: dict[str, Path]
) -> None:
    """Real ``__call__`` with a real tokenizer load: only the renderer builder
    and the image client are faked, so nothing reaches external resources.
    The tampered-dir canary proves the context reached the helper load; the
    builder record proves it reached the renderer."""
    directory, marker = _tampered_v4_dir(v4_files, tmp_path, name="fl-processor")
    stock = transformers.AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False)
    renderer_builds: list[dict[str, Any]] = []

    def fake_build_renderer(tokenizer: Any, name: str, **kwargs: Any) -> None:
        renderer_builds.append(kwargs)
        assert type(tokenizer) is type(
            stock
        ), f"renderer build must receive the stock class, got {type(tokenizer).__name__}"
        return None  # the tool-call parser tolerates a missing renderer

    monkeypatch.setattr(rollout, "build_renderer", fake_build_renderer)
    monkeypatch.setattr(rollout, "FireworksV1ImageCompletionsClient", _FakeImageClient)

    processor = FrozenLakeToolRolloutProcessor(observation_mode="image")
    row = EvaluationRow(
        input_metadata=InputMetadata(
            row_id="v4-context",
            dataset_info={"environment_context": {"desc": ["SF", "FG"]}},
        )
    )
    config = RolloutProcessorConfig(
        completion_params={
            "model": "accounts/a/models/m#accounts/a/deployments/d",
            "tokenizer_name_or_path": str(directory),
            "base_model": V4_ALIAS,
        },
        mcp_config_path="",
        steps=1,
        semaphore=asyncio.Semaphore(1),
        logger=None,
    )

    [task] = processor([row], config)
    await task

    assert not marker.exists(), "selected V4 context must keep the canary inert through the real load"
    assert renderer_builds and renderer_builds[0].get("base_model") == V4_ALIAS
