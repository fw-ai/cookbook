from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest


def _load_router_replay_module():
    path = Path(__file__).parents[1] / "utils" / "rl" / "router_replay.py"
    spec = importlib.util.spec_from_file_location("_router_replay_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_r3_routing_matrices_pads_completion_only_response() -> None:
    router_replay = _load_router_replay_module()

    assert router_replay.build_r3_routing_matrices(
        ["rm_completion_0", "rm_completion_1"],
        prompt_len=3,
        model_input_len=4,
    ) == ["", "", "rm_completion_0", "rm_completion_1"]


@pytest.mark.parametrize("routing_matrices", [None, []])
def test_build_r3_routing_matrices_preserves_missing_capture(routing_matrices) -> None:
    router_replay = _load_router_replay_module()

    assert (
        router_replay.build_r3_routing_matrices(
            routing_matrices,
            prompt_len=3,
            model_input_len=6,
        )
        == []
    )


def test_build_r3_routing_matrices_preserves_invalid_count(caplog) -> None:
    router_replay = _load_router_replay_module()

    with caplog.at_level(logging.WARNING):
        result = router_replay.build_r3_routing_matrices(
            ["rm_0", "rm_1", "rm_2"],
            prompt_len=3,
            model_input_len=6,
        )
    assert result == ["rm_0", "rm_1", "rm_2"]
    assert "routing_matrices length (3) != expected (4)" in caplog.text


@pytest.mark.parametrize("routing_matrices", [None, []])
def test_validate_r3_routing_matrices_rejects_missing_capture(
    routing_matrices,
) -> None:
    router_replay = _load_router_replay_module()

    with pytest.raises(ValueError, match="missing or empty"):
        router_replay.validate_r3_routing_matrices(
            routing_matrices,
            prompt_len=3,
            model_input_len=6,
        )


def test_validate_r3_routing_matrices_rejects_invalid_count() -> None:
    router_replay = _load_router_replay_module()

    with pytest.raises(ValueError, match="got 3, expected 6 full-sequence or 4"):
        router_replay.validate_r3_routing_matrices(
            ["rm_0", "rm_1", "rm_2"],
            prompt_len=3,
            model_input_len=6,
        )


def test_warn_if_full_sequence_router_replay(caplog) -> None:
    router_replay = _load_router_replay_module()

    with caplog.at_level(logging.WARNING):
        router_replay.warn_if_full_sequence_router_replay(completion_only=True)
    assert "router_replay_completion_only=False" not in caplog.text

    with caplog.at_level(logging.WARNING):
        router_replay.warn_if_full_sequence_router_replay(completion_only=False)
    assert "router_replay_completion_only=False" in caplog.text
    assert (
        "disables prompt KV-cache reuse unless serving enables --cache-logprobs"
        in caplog.text
    )
