from __future__ import annotations

import importlib.util
import logging
from pathlib import Path


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


def test_build_r3_routing_matrices_marks_missing_capture_as_enabled() -> None:
    router_replay = _load_router_replay_module()

    assert router_replay.build_r3_routing_matrices(
        None,
        prompt_len=3,
        model_input_len=6,
    ) == []


def test_build_r3_routing_matrices_warns_and_preserves_invalid_count_for_sdk_check(
    caplog,
) -> None:
    router_replay = _load_router_replay_module()

    with caplog.at_level(logging.WARNING):
        result = router_replay.build_r3_routing_matrices(
            ["rm_0", "rm_1", "rm_2"],
            prompt_len=3,
            model_input_len=6,
        )

    assert result == ["rm_0", "rm_1", "rm_2"]
    assert "R3: routing_matrices length (3) != expected (4)" in caplog.text
    assert "prompt_len=3, model_input_len=6" in caplog.text
