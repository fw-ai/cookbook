#!/usr/bin/env python3
"""Smoke tests for the slim v2.2 training skill entry points."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILLS_DIR.parent


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_validator_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SKILLS_DIR / "validate_skills.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_plugin_versions_match() -> None:
    claude = json.loads(read(REPO_ROOT / ".claude-plugin" / "plugin.json"))
    codex = json.loads(read(REPO_ROOT / ".codex-plugin" / "plugin.json"))
    assert claude["version"] == "2.2.0"
    assert codex["version"] == "2.2.0"


def test_install_includes_reference_carrier() -> None:
    for path in (REPO_ROOT / "README.md", SKILLS_DIR / "GETTING-STARTED.md"):
        text = read(path)
        assert "-s fireworks-training" in text
        for skill in ("research", "configure", "debug"):
            assert f"-s {skill}" in text


def test_entry_routes_and_carrier_references() -> None:
    research = read(SKILLS_DIR / "research" / "SKILL.md")
    configure = read(SKILLS_DIR / "configure" / "SKILL.md")
    debug = read(SKILLS_DIR / "debug" / "SKILL.md")
    carrier = read(SKILLS_DIR / "fireworks-training" / "SKILL.md")

    assert "configure" in research and "debug" in research
    assert "research" in configure and "debug" in configure
    assert "research" in debug and "configure" in debug
    assert "shared reference carrier" in carrier
    assert "../fireworks-training/references/telemetry.md" in research
    assert "../fireworks-training/references/telemetry.md" in configure
    assert "../fireworks-training/references/telemetry.md" in debug


def test_question_ids_are_stable() -> None:
    documents = {
        "welcome-entry": read(SKILLS_DIR / "research" / "SKILL.md"),
        "research-q1": read(
            SKILLS_DIR / "research" / "references" / "interview-questions.md"
        ),
        "configure-q-path": read(
            SKILLS_DIR / "configure" / "references" / "path-intake.md"
        ),
        "debug-q-category": read(
            SKILLS_DIR / "debug" / "references" / "triage-paths.md"
        ),
    }
    for question, text in documents.items():
        assert question in text


def test_debug_reuses_existing_session() -> None:
    triage = read(SKILLS_DIR / "debug" / "references" / "triage-paths.md")
    attribution = triage.split("## Attribution", 1)[1]
    assert "read `skill_session_id` from its `run.md`" in attribution
    assert 'if [ -z "${FIREWORKS_SESSION_ID:-}" ]; then' in attribution
    assert "Never replace an existing run's session UUID." in attribution


TESTS = (
    test_validator_passes,
    test_plugin_versions_match,
    test_install_includes_reference_carrier,
    test_entry_routes_and_carrier_references,
    test_question_ids_are_stable,
    test_debug_reuses_existing_session,
)


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    if failures:
        return 1
    print(f"OK: {len(TESTS)} entry-point tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
