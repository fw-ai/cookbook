#!/usr/bin/env python3
"""Integration smoke: research handoff manifest → configure can consume it."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILLS_DIR.parent
DEMO_MANIFEST = (
    REPO_ROOT / "fireworks-training-runs" / "demo-rag-policy-20260903" / "run.md"
)

REQUIRED_HANDOFF = (
    "case_study:",
    "implied_method:",
    "suggested_path:",
    "matched_case_study:",
    "handoff_choice: plan_configure",
)

REQUIRED_JOURNEY = (
    "journey_schema_version: 2",
    "intake_responses:",
    "task_summary:",
)

CONFIGURE_MUST_READ = (
    "case_study",
    "implied_method",
    "suggested_path",
    "dataset_plan",
    "eval_plan",
    "workflow_path",
    "Q-path",
)


def test_demo_manifest_has_research_handoff() -> None:
    assert DEMO_MANIFEST.exists(), f"missing demo manifest: {DEMO_MANIFEST}"
    text = DEMO_MANIFEST.read_text(encoding="utf-8")
    for marker in REQUIRED_HANDOFF:
        assert marker in text, f"demo manifest missing `{marker}`"
    for marker in REQUIRED_JOURNEY:
        assert marker in text, f"demo manifest missing journey `{marker}`"


def test_configure_skill_reads_handoff_fields() -> None:
    configure = (SKILLS_DIR / "configure" / "SKILL.md").read_text(encoding="utf-8")
    for field in CONFIGURE_MUST_READ:
        assert field in configure, f"configure/SKILL.md must reference `{field}`"


def test_research_does_not_pick_specific_model() -> None:
    research = (SKILLS_DIR / "research" / "SKILL.md").read_text(encoding="utf-8")
    case_studies = (
        SKILLS_DIR / "research" / "references" / "case-studies.md"
    ).read_text(encoding="utf-8")
    assert "cookbook-catalog" in research.lower()
    assert "implied_method" in research or "readiness package" in research.lower()
    assert "does not" in research.lower()
    assert "qwen3-8b" not in case_studies


def test_firectl_preflight_readonly() -> None:
    result = subprocess.run(
        ["firectl", "whoami"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("SKIP test_firectl_preflight_readonly: firectl whoami failed")
        return
    assert "@" in result.stdout or "account" in result.stdout.lower()


def test_handoff_order_documented() -> None:
    getting_started = (SKILLS_DIR / "GETTING-STARTED.md").read_text(encoding="utf-8")
    assert "Research" in getting_started and "Configure" in getting_started
    assert re.search(r"Research.*Configure|research.*configure", getting_started, re.I)


TESTS = (
    test_demo_manifest_has_research_handoff,
    test_configure_skill_reads_handoff_fields,
    test_research_does_not_pick_specific_model,
    test_firectl_preflight_readonly,
    test_handoff_order_documented,
)


def main() -> int:
    failed = 0
    for test in TESTS:
        name = test.__name__
        try:
            test()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    if failed:
        print(f"\n{failed} failed, {len(TESTS) - failed} passed")
        return 1
    print(f"\nOK: {len(TESTS)} research→configure handoff tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
