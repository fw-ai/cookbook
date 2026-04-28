"""Smoke tests for ``training/examples/tools/check_glm5_sft_dataset.py``.

Skips when the GLM-5.1 tokenizer isn't reachable (offline CI), since the
script renders rows through the live tokenizer to inspect special-token
ids ``<think>`` / ``</think>`` / ``<|endoftext|>``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import transformers

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "tools"
    / "check_glm5_sft_dataset.py"
)


@pytest.fixture(scope="module")
def _tokenizer_available() -> bool:
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "zai-org/GLM-5.1", trust_remote_code=True
        )
    except Exception:  # noqa: BLE001 — network / auth / missing template
        return False
    return getattr(tok, "chat_template", None) is not None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run(dataset: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--dataset", str(dataset), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


def test_script_imports() -> None:
    """The script must be importable as a module under any cookbook revision.

    We only check importability (no execution) here so the test stays fast
    on CI nodes that can't reach Hugging Face.
    """
    spec = importlib.util.spec_from_file_location("_check_glm5", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")


def test_clean_dataset_passes(tmp_path: Path, _tokenizer_available: bool) -> None:
    if not _tokenizer_available:
        pytest.skip("GLM-5.1 tokenizer not available")
    dataset = tmp_path / "clean.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "messages": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Q2"},
                    {
                        "role": "assistant",
                        "reasoning_content": "R",
                        "content": "A2",
                    },
                ]
            },
        ],
    )
    result = _run(dataset, "--json")
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["counts"].get("unbalanced_think", 0) == 0
    assert summary["counts"].get("no_eos", 0) == 0
    assert summary["bad_samples"] == []


def test_unbalanced_think_is_flagged(tmp_path: Path, _tokenizer_available: bool) -> None:
    if not _tokenizer_available:
        pytest.skip("GLM-5.1 tokenizer not available")
    dataset = tmp_path / "bad.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "<think>open with no close",
                    },
                ]
            }
        ],
    )
    result = _run(dataset, "--json")
    # Exit 2 means "issues found" (intentional, useful for shell scripting).
    assert result.returncode == 2, result.stderr
    summary = json.loads(result.stdout)
    assert summary["counts"]["unbalanced_think"] == 1
    assert any(
        b.get("unbalanced_think") for b in summary["bad_samples"]
    ), summary["bad_samples"]
