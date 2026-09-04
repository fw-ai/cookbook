#!/usr/bin/env python3
"""Entry-point routing tests for research / configure / debug (v2.2.0)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILLS_DIR.parent

CASE_STUDY_SLUGS = (
    "sft_prompt_router",
    "sft_cord_receipts",
    "dpo_style",
    "reasoning_rl",
    "embedding_support_search",
    "agentic_rl_text2sql",
)

RESEARCH_ROUTING_CASES: tuple[tuple[str, str], ...] = (
    (
        "gradeable classification task compare dedicated and serverless training paths",
        "sft_prompt_router",
    ),
    (
        "invoice parsing form extraction screenshot to JSON structured output",
        "sft_cord_receipts",
    ),
    (
        "easier to say which of two answers is better than write the ideal one brand voice",
        "dpo_style",
    ),
    (
        "objectively checkable answers grader exists but no gold worked solutions math",
        "reasoning_rl",
    ),
    (
        "search returns topically adjacent article not the one that governs the situation",
        "embedding_support_search",
    ),
    (
        "tool calling sql database schema agent navigate multi-turn reinforcement learning",
        "agentic_rl_text2sql",
    ),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _skill_text(name: str) -> str:
    return _read(SKILLS_DIR / name / "SKILL.md")


def test_validate_skills_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SKILLS_DIR / "validate_skills.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_all_case_study_readmes_have_is_this_you() -> None:
    for slug in CASE_STUDY_SLUGS:
        readme = REPO_ROOT / "training" / "case-studies" / slug / "README.md"
        assert readme.exists(), f"missing README for {slug}"
        text = _read(readme)
        assert "Is this you?" in text, f"{slug} README missing Is this you? block"


def test_case_studies_index_lists_all_slugs() -> None:
    index = _read(SKILLS_DIR / "research" / "references" / "case-studies.md")
    for slug in CASE_STUDY_SLUGS:
        assert f"`{slug}`" in index, f"case-studies.md missing slug {slug}"


def _is_this_you_text(slug: str) -> str:
    readme = _read(REPO_ROOT / "training" / "case-studies" / slug / "README.md")
    match = re.search(r"\*\*Is this you\?\*\*(.+?)(?:\n\n|\*\*)", readme, re.DOTALL)
    assert match, f"{slug} missing Is this you? paragraph"
    return match.group(1).lower()


def test_research_keyword_routing_smoke() -> None:
    slug_keywords = {
        slug: set(re.findall(r"[a-z]{4,}", _is_this_you_text(slug)))
        for slug in CASE_STUDY_SLUGS
    }

    for prompt, expected_slug in RESEARCH_ROUTING_CASES:
        prompt_words = set(re.findall(r"[a-z]{4,}", prompt.lower()))
        scores = {
            slug: len(prompt_words & words)
            for slug, words in slug_keywords.items()
        }
        best_slug = max(scores, key=scores.get)
        assert scores[best_slug] > 0, f"no keyword overlap for prompt: {prompt!r}"
        assert best_slug == expected_slug, (
            f"prompt {prompt!r} routed to {best_slug}, expected {expected_slug} "
            f"(scores={scores})"
        )


def test_welcome_shared_entry() -> None:
    welcome = _read(SKILLS_DIR / "references" / "welcome.md")
    assert "three ways I can help" in welcome
    assert "**Research**" in welcome
    for slug in ("research", "configure", "debug"):
        assert "welcome.md" in _skill_text(slug)


def test_research_interview_and_visibility_markers() -> None:
    research = _skill_text("research")
    interview = _read(SKILLS_DIR / "research" / "references" / "interview-questions.md")
    output = _read(SKILLS_DIR / "research" / "references" / "output-template.md")
    catalog = _read(SKILLS_DIR / "research" / "references" / "cookbook-catalog.md")

    for marker in (
        "skill banner",
        "completion gate",
        "output-template.md",
        "interview-questions.md",
        "cookbook-catalog.md",
        "Never tell the user to paste",
        "Handoff",
    ):
        assert marker.lower() in research.lower() or marker in research, (
            f"research/SKILL.md missing marker `{marker}`"
        )
    assert "Q1b" in interview
    assert "Q-eval" in interview
    assert "completion gate" in interview.lower()
    assert "**Research**" in output
    assert "training/recipes/" in catalog
    assert "training/examples/" in catalog


def test_configure_path_intake_markers() -> None:
    configure = _skill_text("configure")
    path_intake = _read(SKILLS_DIR / "configure" / "references" / "path-intake.md")
    output = _read(SKILLS_DIR / "configure" / "references" / "output-template.md")

    for marker in (
        "path-intake.md",
        "execution_surface",
        "Q-path",
        "completion gate",
        "cost-estimation.md",
        "Do not calculate Dedicated SFT or DPO",
        "docs.fireworks.ai/fine-tuning/cost-estimator",
        "api-key-setup.md",
    ):
        assert marker in configure, f"configure/SKILL.md missing `{marker}`"
    api_key_setup = _read(SKILLS_DIR / "references" / "api-key-setup.md")
    assert "read -s FIREWORKS_API_KEY" in api_key_setup
    assert "managed_firectl" in path_intake
    assert "Never show only `firectl`" in path_intake
    assert "**Configure**" in output


def test_skill_entry_routing_markers() -> None:
    research = _skill_text("research")
    configure = _skill_text("configure")
    debug = _skill_text("debug")
    stub = _skill_text("fireworks-training")
    discover_redirect = _skill_text("discover")

    assert "entry_skill: research" in research
    assert "configure" in research
    assert "entry_skill: configure" in configure
    assert "**research**" in configure
    assert "**debug**" in configure
    assert "always monitoring" in configure.lower()
    assert "entry_skill: debug" in debug
    assert "three-strike" in debug.lower()
    assert "**research**" in stub
    assert "redirect" in stub.lower()
    assert "research" in discover_redirect.lower()


def test_plugin_version_and_skill_names() -> None:
    plugin = json.loads(_read(REPO_ROOT / ".claude-plugin" / "plugin.json"))
    assert plugin["version"] == "2.2.0"
    for skill in ("research", "configure", "debug"):
        frontmatter = _skill_text(skill).split("---", 2)[1]
        assert f"name: {skill}" in frontmatter


def test_telemetry_schema_and_option_ids() -> None:
    schema = _read(SKILLS_DIR / "references" / "telemetry-schema.md")
    interview = _read(SKILLS_DIR / "research" / "references" / "interview-questions.md")
    welcome = _read(SKILLS_DIR / "references" / "welcome.md")
    run_state = _read(
        SKILLS_DIR / "configure" / "references" / "run-state-and-reporting.md"
    )

    for marker in (
        "journey_schema_version",
        "intake_q1_task_shape",
        "handoff_choice",
        "session_outcome",
        "research_only",
        "intake_responses",
        "user_choice",
        "task_summary",
        "response_source",
    ):
        assert marker in schema, f"telemetry-schema.md missing `{marker}`"

    for option_id in ("structured_output", "plan_configure", "query_doc"):
        assert option_id in interview, f"interview missing option_id `{option_id}`"
    assert "Option ID" in interview
    assert "research-q1" in interview

    assert "welcome_choice" in welcome
    assert "telemetry-notice.md" in welcome

    notice = _read(SKILLS_DIR / "references" / "telemetry-notice.md")
    assert "Privacy note" in notice

    assert "## Journey telemetry" in run_state
    assert "intake_responses" in run_state
    assert "task_summary:" in run_state

    for spec in (
        "telemetry/journey-api-spec.md",
        "telemetry/jarvis-funnel-tiles.md",
        "telemetry/sdk-firectl-helpers.md",
    ):
        assert (SKILLS_DIR / "references" / spec).exists(), f"missing {spec}"


def test_getting_started_guide() -> None:
    guide = _read(SKILLS_DIR / "GETTING-STARTED.md")
    for marker in (
        "Quick start (after install)",
        "Fireworks Training",
        "After install — just talk",
        "5-minute smoke test",
        "research",
        "configure",
        "debug",
        "telemetry-schema.md",
        "journey telemetry",
        "telemetry-notice",
        "task summary",
    ):
        assert marker in guide, f"GETTING-STARTED.md missing `{marker}`"
    readme = _read(REPO_ROOT / "README.md")
    assert "GETTING-STARTED.md" in readme


TESTS = (
    test_validate_skills_script_passes,
    test_all_case_study_readmes_have_is_this_you,
    test_case_studies_index_lists_all_slugs,
    test_research_keyword_routing_smoke,
    test_welcome_shared_entry,
    test_research_interview_and_visibility_markers,
    test_configure_path_intake_markers,
    test_skill_entry_routing_markers,
    test_telemetry_schema_and_option_ids,
    test_plugin_version_and_skill_names,
    test_getting_started_guide,
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
    print(f"\nOK: {len(TESTS)} entry-point tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
