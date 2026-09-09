#!/usr/bin/env python3
"""Contract test for the research to configure handoff."""

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
FIXTURES = (
    SKILLS_DIR / "tests" / "fixtures" / "research-handoff-run.md",
    SKILLS_DIR / "tests" / "fixtures" / "research-example-handoff-run.md",
)


def test_fixture_has_required_handoff_fields() -> None:
    for fixture in FIXTURES:
        text = fixture.read_text(encoding="utf-8")
        for field in (
            "cookbook_entry_tier:",
            "cookbook_entry_path:",
            "implied_method:",
            "suggested_path:",
            "dataset_plan:",
            "eval_plan:",
        ):
            assert field in text, f"{fixture.name} missing {field}"
        assert "workflow_path:" not in text
        assert "execution_surface:" not in text


def test_example_handoff_does_not_require_case_study() -> None:
    text = FIXTURES[1].read_text(encoding="utf-8")
    assert "case_study:" not in text
    assert "cookbook_entry_tier: example" in text
    assert "cookbook_entry_path: training/examples/" in text


def test_configure_consumes_handoff_fields() -> None:
    configure = (SKILLS_DIR / "configure" / "SKILL.md").read_text(encoding="utf-8")
    for field in (
        "case_study",
        "cookbook_entry_tier",
        "cookbook_entry_path",
        "notebook",
        "readme",
        "implied_method",
        "dataset_plan",
        "eval_plan",
        "suggested_path",
        "workflow path",
        "execution surface",
    ):
        assert field in configure, f"configure does not consume {field}"
    assert "Run Q-path even when research recommends a coarse path" in configure


def test_configure_persists_complete_handoff() -> None:
    path_intake = (
        SKILLS_DIR / "configure" / "references" / "path-intake.md"
    ).read_text(encoding="utf-8")
    inherit_contract = path_intake.split("## Inherit from research", 1)[1].split(
        "## Q-path", 1
    )[0]
    manifest_contract = path_intake.split("## Record in run manifest", 1)[1].split(
        "## Final plan", 1
    )[0]
    nested_handoff_contract = manifest_contract.split("research_handoff:", 1)[1]
    for field in (
        "case_study",
        "cookbook_entry_tier",
        "cookbook_entry_path",
        "implied_method",
        "suggested_path",
        "dataset_plan",
        "eval_plan",
    ):
        assert field in inherit_contract, f"inherit table missing {field}"
        assert field in manifest_contract, f"run manifest schema missing {field}"
    for field in ("dataset_plan", "eval_plan"):
        assert field in nested_handoff_contract, f"nested handoff missing {field}"


def test_handoff_persists_only_after_confirmation() -> None:
    interview = (
        SKILLS_DIR / "research" / "references" / "interview-questions.md"
    ).read_text(encoding="utf-8")
    ask_index = interview.index("Fire the **Handoff** AskQuestion")
    write_index = interview.index("Write the handoff block")
    assert ask_index < write_index
    assert "only after the user selects\n   `plan_configure`" in interview


def test_all_handoff_schemas_include_planning_fields() -> None:
    for name in ("case-studies.md", "cookbook-catalog.md", "methodology.md"):
        text = (SKILLS_DIR / "research" / "references" / name).read_text(
            encoding="utf-8"
        )
        for field in (
            "cookbook_entry_tier:",
            "cookbook_entry_path:",
            "implied_method:",
            "suggested_path:",
            "dataset_plan:",
            "eval_plan:",
        ):
            assert field in text, f"{name} missing {field}"


def test_research_never_satisfies_configure_path_gate() -> None:
    intake = (SKILLS_DIR / "configure" / "references" / "path-intake.md").read_text(
        encoding="utf-8"
    )
    assert "question_id: configure-q-path" in intake
    assert "`research-q3` answer never satisfies this gate" in intake


def main() -> int:
    tests = (
        test_fixture_has_required_handoff_fields,
        test_example_handoff_does_not_require_case_study,
        test_configure_consumes_handoff_fields,
        test_configure_persists_complete_handoff,
        test_handoff_persists_only_after_confirmation,
        test_all_handoff_schemas_include_planning_fields,
        test_research_never_satisfies_configure_path_gate,
    )
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    if failures:
        return 1
    print(f"OK: {len(tests)} handoff tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
