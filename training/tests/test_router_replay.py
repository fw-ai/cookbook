from __future__ import annotations

import importlib.util
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
