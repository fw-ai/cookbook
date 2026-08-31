from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from fireworks.training.sdk import TITOChatRequest, TITOParsedAssistant
from training.examples.rl.harbor.opencode.artifacts import (
    tool_timeout_count as opencode_tool_timeout_count,
)
from training.examples.rl.harbor.pi.artifacts import (
    tool_timeout_count as pi_tool_timeout_count,
)
from training.examples.rl.harbor.tito.sidecar import (
    build_call_classifier,
)
from training.tito.renderer import (
    _ensure_tool_call_ids,
)


def _request(*, tools=(), adapter_metadata=None):
    return TITOChatRequest(
        messages=({"role": "user", "content": "same prompt"},),
        tools=tuple(tools),
        adapter_metadata=dict(adapter_metadata or {}),
    )


def test_tito_renderer_synthesizes_stable_missing_tool_call_ids() -> None:
    source = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": None,
                "type": "function",
                "function": {"name": "echo", "arguments": {"message": "green"}},
            }
        ],
    }
    first = _ensure_tool_call_ids(source, [10, 20, 30])
    second = _ensure_tool_call_ids(source, [10, 20, 30])
    different = _ensure_tool_call_ids(source, [10, 20, 31])
    assert first["tool_calls"][0]["id"].startswith("call_")
    assert first["tool_calls"][0]["id"] == second["tool_calls"][0]["id"]
    assert first["tool_calls"][0]["id"] != different["tool_calls"][0]["id"]
    assert source["tool_calls"][0]["id"] is None


def test_sdk_canonicalizes_protocol_fields_after_renderer_id_synthesis() -> None:
    source = {
        "role": "assistant",
        "content": "\n",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": '{"z": 1, "a": {"y": 2, "x": 1}}',
                },
            }
        ],
    }
    identified = _ensure_tool_call_ids(source, [10, 20, 30])
    assert identified["content"] == "\n"
    assert identified["tool_calls"][0]["function"]["arguments"] == (
        '{"z": 1, "a": {"y": 2, "x": 1}}'
    )

    normalized = TITOParsedAssistant(message=identified)
    assert normalized.message["content"] == ""
    assert normalized.message["tool_calls"][0]["function"]["arguments"] == (
        '{"a":{"x":1,"y":2},"z":1}'
    )
    assert source["content"] == "\n"


def test_generic_call_classifiers_use_protocol_inputs_only() -> None:
    tools_present = build_call_classifier("tools_present")
    adapter_metadata = build_call_classifier("adapter_metadata")
    assert tools_present(_request(tools=({"type": "function"},))) == (
        "policy",
        "tools_present",
    )
    assert tools_present(_request()) == ("auxiliary", "tools_absent")
    assert adapter_metadata(
        _request(
            adapter_metadata={
                "call_kind": "auxiliary",
                "classifier_source": "pi_compaction_hook",
            }
        )
    ) == ("auxiliary", "pi_compaction_hook")
    with pytest.raises(ValueError, match="missing a valid"):
        adapter_metadata(_request())


def test_all_policy_classifier_does_not_inspect_harness_messages() -> None:
    classifier = build_call_classifier("all_policy")
    assert classifier(_request()) == ("policy", "all_policy")


def test_mini_swe_keeps_task_image_working_directory() -> None:
    pytest.importorskip("harbor")
    from training.examples.rl.harbor.mini_swe.agent import ConfigurableMiniSweAgent

    agent = ConfigurableMiniSweAgent(
        logs_dir="/tmp/logs",
        model_name="ignored",
        sidecar_bundle_path="/tmp/bundle.zip",
        sidecar_launch_spec="{}",
        context_limit=4096,
        output_limit=1024,
        tool_timeout_seconds=123,
    )
    assert agent._config_yaml == "environment:\n  timeout: 123\n"  # noqa: SLF001


def test_unknown_call_classifier_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported TITO call classifier"):
        build_call_classifier("unknown")


def test_tito_layers_do_not_import_or_name_concrete_harnesses() -> None:
    training_root = Path(__file__).parents[2]
    layer_roots = (
        training_root / "tito",
        training_root / "examples" / "rl" / "harbor" / "tito",
    )
    concrete_harnesses = ("opencode", "mini_swe", "mini-swe", "pi")
    concrete_harness_pattern = re.compile(r"opencode|mini[_-]swe|\bpi\b", re.I)

    for layer_root in layer_roots:
        for path in layer_root.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            imported_modules = [
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            ]
            imported_modules.extend(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            assert not any(
                f"harbor.{harness}" in module
                for module in imported_modules
                for harness in concrete_harnesses
            ), f"{path} imports a concrete harness"

            assert concrete_harness_pattern.search(source) is None, (
                f"{path} contains concrete harness knowledge"
            )


def test_harness_packages_do_not_import_each_other() -> None:
    harbor_root = Path(__file__).parents[2] / "examples" / "rl" / "harbor"
    harnesses = ("opencode", "pi", "mini_swe")
    for harness in harnesses:
        for path in (harbor_root / harness).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported_modules = [
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            ]
            imported_modules.extend(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            siblings = set(harnesses) - {harness}
            assert not any(
                f"harbor.{sibling}" in module
                for module in imported_modules
                for sibling in siblings
            ), f"{path} imports a sibling harness"


def test_opencode_tool_timeout_events_are_adapter_owned(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    events = [
        {
            "type": "tool_use",
            "part": {
                "tool": "bash",
                "callID": "call-1",
                "state": {"error": "command timed out"},
            },
        },
        {"type": "message", "content": "timed out is ordinary model text"},
    ]
    (agent / "events.txt").write_text(
        "\n".join(json.dumps(event) for event in events), encoding="utf-8"
    )
    assert opencode_tool_timeout_count(tmp_path) == 1


def test_pi_tool_timeout_events_are_adapter_owned(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    events = [
        {
            "type": "tool_execution_start",
            "toolName": "bash",
            "toolCallId": "call-1",
        },
        {
            "type": "tool_execution_end",
            "toolName": "bash",
            "toolCallId": "call-1",
            "result": {"error": "timed out after 10 seconds"},
        },
    ]
    (agent / "events.txt").write_text(
        "\n".join(json.dumps(event) for event in events), encoding="utf-8"
    )
    assert pi_tool_timeout_count(tmp_path) == 1
