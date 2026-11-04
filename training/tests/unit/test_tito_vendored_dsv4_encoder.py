"""Proofs for the vendored DeepSeek V4 encoder.

The vendored file is the exact prior encoder (byte parity with the
certification hash), incorporated as reviewed package source: stdlib-only
imports, no I/O or dynamic-execution constructs, and identical output to the
repo's older test oracle on the certified rendering contract.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from training.renderer.tito._vendored.deepseek_v4 import encoding_dsv4
from training.renderer.tito.deepseek_v4 import (
    DEEPSEEK_V4_ENCODER_SHA256,
    DeepSeekV4TITORenderer,
)
from training.renderer.tito.deepseek_v41 import (
    DEEPSEEK_V41_ENCODER_FILE,
    DeepSeekV41TITORenderer,
)

VENDORED_PATH = (
    Path(__file__).resolve().parents[2]
    / "renderer"
    / "tito"
    / "_vendored"
    / "deepseek_v4"
    / "encoding_dsv4.py"
)
OLDER_ORACLE_PATH = Path(__file__).resolve().parents[1] / "_encoding_dsv4_oracle.py"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def older_oracle() -> Any:
    """The repo's pre-existing oracle: the initial V4 release encoder."""
    return _load_module(OLDER_ORACLE_PATH, "dsv4_older_oracle")


def test_vendored_encoder_matches_certification_sha256() -> None:
    digest = hashlib.sha256(VENDORED_PATH.read_bytes()).hexdigest()
    assert digest == DEEPSEEK_V4_ENCODER_SHA256


def test_vendored_encoder_imports_only_stdlib() -> None:
    tree = ast.parse(VENDORED_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"typing", "copy", "json", "re"}


def test_vendored_encoder_has_no_io_or_dynamic_execution() -> None:
    tree = ast.parse(VENDORED_PATH.read_text(encoding="utf-8"))
    banned_calls = {"exec", "eval", "compile", "open", "input", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in banned_calls, f"line {node.lineno}: {node.func.id}"


def test_renderer_load_encoder_returns_the_vendored_module() -> None:
    renderer = DeepSeekV4TITORenderer.__new__(DeepSeekV4TITORenderer)
    assert renderer._load_encoder(None, None) is encoding_dsv4  # type: ignore[arg-type]


def test_v41_load_encoder_stays_on_the_artifact_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """V4.1 keeps loading its own different encoder via renderer_artifacts."""
    calls: list[tuple[Any, Any]] = []
    fake_artifact = tmp_path / "encoding.py"
    fake_artifact.write_text('eos_token = "<eos-test>"\n', encoding="utf-8")

    def fake_renderer_artifacts(tokenizer: Any, certification: Any) -> dict[str, Path]:
        calls.append((tokenizer, certification))
        return {DEEPSEEK_V41_ENCODER_FILE: fake_artifact}

    monkeypatch.setattr(
        "training.renderer.tito.deepseek_v41.renderer_artifacts",
        fake_renderer_artifacts,
    )
    renderer = DeepSeekV41TITORenderer.__new__(DeepSeekV41TITORenderer)
    encoder = renderer._load_encoder("tokenizer-sentinel", "certification-sentinel")
    assert calls == [("tokenizer-sentinel", "certification-sentinel")]
    assert encoder.eos_token == "<eos-test>"


def _certified_contract_battery() -> list[list[dict[str, Any]]]:
    tool = {
        "type": "function",
        "function": {
            "name": "run",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {"z": {"type": "string"}, "a": {"type": "integer"}},
            },
        },
    }
    return [
        [{"role": "user", "content": "Hello"}],
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        [
            {"role": "system", "content": "", "tools": [tool]},
            {"role": "user", "content": "Run"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "ordered reasoning",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "run", "arguments": '{"z":"hello","a":2}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "user", "content": "Continue"},
        ],
        [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "One", "reasoning_content": "old thinking"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Two", "reasoning_content": "kept thinking"},
            {"role": "user", "content": "Third"},
        ],
        [
            {"role": "system", "content": "", "tools": [tool]},
            {"role": "user", "content": "Run twice"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "reason",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "run", "arguments": '{"z":"one","a":1}'},
                    },
                    {
                        "type": "function",
                        "function": {"name": "run", "arguments": '{"z":"two","a":2}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "two"},
            {"role": "tool", "tool_call_id": "c1", "content": "one"},
            {"role": "user", "content": "Finish"},
        ],
    ]


@pytest.mark.parametrize("messages", _certified_contract_battery())
def test_certified_contract_matches_older_oracle(
    older_oracle: Any, messages: list[dict[str, Any]]
) -> None:
    """Byte-identical prompt text on the certified V4 rendering contract.

    The pinned V4-Flash-0731 revision is newer than the repo's oracle (the
    initial V4 release); on the certified contract -- thinking mode,
    low/default effort, preserved thinking -- both revisions agree.
    """
    new_text = encoding_dsv4.encode_messages(
        messages,
        thinking_mode="thinking",
        reasoning_effort="low",
        drop_thinking=False,
        add_default_bos_token=True,
    )
    old_text = older_oracle.encode_messages(
        messages,
        thinking_mode="thinking",
        reasoning_effort=None,
        drop_thinking=False,
        add_default_bos_token=True,
    )
    assert new_text == old_text


def test_completion_parsing_matches_older_oracle(older_oracle: Any) -> None:
    completion = "some reasoning</think>final answer" + encoding_dsv4.eos_token
    assert encoding_dsv4.parse_message_from_completion_text(
        completion, thinking_mode="thinking"
    ) == older_oracle.parse_message_from_completion_text(completion, thinking_mode="thinking")


def test_high_effort_prefix_documents_newer_revision(older_oracle: Any) -> None:
    """The pinned revision prepends the high-effort prompt; the older oracle
    predates that behavior. The certified contract always uses low effort."""
    messages = [{"role": "user", "content": "Hello"}]
    new_text = encoding_dsv4.encode_messages(
        messages,
        thinking_mode="thinking",
        reasoning_effort="high",
        drop_thinking=False,
    )
    old_text = older_oracle.encode_messages(
        messages,
        thinking_mode="thinking",
        reasoning_effort="high",
        drop_thinking=False,
    )
    high_prompt = encoding_dsv4.REASONING_EFFORT_PROMPTS["high"]
    assert high_prompt in new_text
    assert high_prompt not in old_text
    assert new_text.replace(high_prompt, "") == old_text
