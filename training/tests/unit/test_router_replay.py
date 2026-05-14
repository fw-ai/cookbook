from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys

_ROUTER_REPLAY_PATH = Path(__file__).parents[2] / "utils" / "rl" / "router_replay.py"
_SPEC = importlib.util.spec_from_file_location("router_replay_under_test", _ROUTER_REPLAY_PATH)
assert _SPEC is not None
router_replay = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["router_replay_under_test"] = router_replay
_SPEC.loader.exec_module(router_replay)

build_r3_routing_matrices = router_replay.build_r3_routing_matrices
describe_r3_routing_alignment = router_replay.describe_r3_routing_alignment


def _routing(n: int) -> list[str]:
    return [f"rm-{i}" for i in range(n)]


def test_describes_full_echo_routing() -> None:
    alignment = describe_r3_routing_alignment(
        _routing(5),
        prompt_len=3,
        model_input_len=5,
    )

    assert alignment.kind == "full"
    assert alignment.source_len == 5
    assert alignment.completion_only_expected == 3
    assert alignment.aligned_len == 5
    assert alignment.aligned_non_empty == 5
    assert build_r3_routing_matrices(
        _routing(5),
        prompt_len=3,
        model_input_len=5,
    ) == _routing(5)


def test_describes_completion_only_routing() -> None:
    alignment = describe_r3_routing_alignment(
        _routing(3),
        prompt_len=3,
        model_input_len=5,
    )

    assert alignment.kind == "completion_only"
    assert alignment.source_len == 3
    assert alignment.completion_only_expected == 3
    assert alignment.aligned_len == 5
    assert alignment.source_non_empty == 3
    assert alignment.aligned_non_empty == 3
    assert build_r3_routing_matrices(_routing(3), prompt_len=3, model_input_len=5) == [
        "",
        "",
        "rm-0",
        "rm-1",
        "rm-2",
    ]


def test_describes_invalid_short_routing_without_padding(caplog) -> None:
    caplog.set_level(logging.WARNING)
    alignment = describe_r3_routing_alignment(
        _routing(2),
        prompt_len=3,
        model_input_len=5,
    )
    result = build_r3_routing_matrices(_routing(2), prompt_len=3, model_input_len=5)

    assert alignment.kind == "invalid_short"
    assert alignment.source_len == 2
    assert alignment.completion_only_expected == 3
    assert alignment.aligned_len == 4
    assert result == ["", "", "rm-0", "rm-1"]
    assert "kind=invalid_short" in caplog.text
    assert "aligned_len=4" in caplog.text


def test_describes_overlong_routing() -> None:
    alignment = describe_r3_routing_alignment(
        _routing(7),
        prompt_len=3,
        model_input_len=5,
    )

    assert alignment.kind == "overlong"
    assert alignment.source_len == 7
    assert alignment.aligned_len == 5


def test_completion_only_masks_prompt_positions() -> None:
    result = build_r3_routing_matrices(
        _routing(5),
        prompt_len=3,
        model_input_len=5,
        completion_only=True,
    )

    assert result == ["", "", "rm-2", "rm-3", "rm-4"]
