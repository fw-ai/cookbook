#!/usr/bin/env python3
"""Validate the cookbook's agent skills so an install/attach can't silently break.

Checks, per `skills/*/SKILL.md`:
  1. YAML frontmatter exists and has non-empty `name` + `description`.
  2. Every `references/<file>.md` mentioned in SKILL.md actually exists.
  3. Every relative markdown link in SKILL.md and its references resolves on disk.
  4. Every `references/*.md` file on disk is referenced by SKILL.md (no orphans).

Stdlib only, no third-party deps. Exits non-zero on any error so CI fails.
This is the "skill install keeps working" gate (paved-path bar), the analog of
Tinker's test_install_skills.py.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
# Paved agent skills this gate is responsible for (install/attach must not break).
PAVED_SKILLS = {"fireworks-fine-tuning", "dev"}
# Deprecated stubs kept only for link stability — validate lightly (frontmatter only).
STUB_SKILLS = {"fireworks-agent", "research/fireworks-auto-tune"}
# Older human-facing implementation guides, not auto-attach agent skills — out of scope.
SKIP_SKILLS = {"renderer", "verifier"}

REF_RE = re.compile(r"references/[\w./-]+\.md")
# Markdown links to a relative .md path: [text](path.md) or [text](path.md#anchor)
LINK_RE = re.compile(r"\]\((?!https?://|mailto:)([\w./-]+\.md)(?:#[\w-]+)?\)")


def parse_frontmatter(text: str) -> dict | None:
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    fm: dict[str, str] = {}
    key = None
    for line in text[3:end].splitlines():
        m = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
        if m:
            key = m.group(1)
            fm[key] = m.group(2).strip()
        elif key and line.strip():  # folded/continued value (e.g. `>-`)
            fm[key] += " " + line.strip()
    return fm


def check_skill(skill_md: Path, errors: list[str]) -> None:
    skill_dir = skill_md.parent
    rel = skill_md.relative_to(SKILLS_DIR)
    slug = str(skill_dir.relative_to(SKILLS_DIR))
    text = skill_md.read_text(encoding="utf-8")

    fm = parse_frontmatter(text)
    if not fm:
        errors.append(f"{rel}: missing YAML frontmatter")
        return
    for field in ("name", "description"):
        if not fm.get(field):
            errors.append(f"{rel}: frontmatter missing non-empty `{field}`")

    if slug in STUB_SKILLS:
        return  # deprecated stub: frontmatter is enough

    # 2. references/<file>.md mentioned in SKILL.md must exist
    mentioned = set(REF_RE.findall(text))
    for ref in sorted(mentioned):
        if not (skill_dir / ref).exists():
            errors.append(f"{rel}: routes to `{ref}` but the file is missing")

    # 3. relative .md links resolve (in SKILL.md and each reference)
    md_files = [skill_md] + sorted(skill_dir.glob("references/*.md"))
    for md in md_files:
        for link in LINK_RE.findall(md.read_text(encoding="utf-8")):
            target = (md.parent / link).resolve()
            if not target.exists():
                errors.append(f"{md.relative_to(SKILLS_DIR)}: broken link `{link}`")

    # 4. no orphan references (every reference file is routed to from SKILL.md)
    for ref_file in sorted(skill_dir.glob("references/*.md")):
        ref_rel = f"references/{ref_file.name}"
        if ref_rel not in text:
            errors.append(f"{rel}: `{ref_rel}` exists but SKILL.md never routes to it")

    if slug == "fireworks-fine-tuning":
        check_fine_tuning_contract(skill_dir, errors)


def check_fine_tuning_contract(skill_dir: Path, errors: list[str]) -> None:
    """Protect the safety and method-routing fixes that replace Pilot."""
    orchestrate_path = skill_dir / "references/orchestrate-from-agent.md"
    orchestrate = orchestrate_path.read_text(encoding="utf-8")
    plan_marker = "**1.5 Plan + confirm"
    upload_command = "firectl dataset create <name>"
    if plan_marker not in orchestrate or upload_command not in orchestrate:
        errors.append(
            f"{orchestrate_path.relative_to(SKILLS_DIR)}: missing plan gate or dataset upload command"
        )
    elif orchestrate.index(plan_marker) > orchestrate.index(upload_command):
        errors.append(
            f"{orchestrate_path.relative_to(SKILLS_DIR)}: dataset upload appears before plan approval"
        )

    for command in (
        "firectl sftj create",
        "firectl dpo-job create",
        "firectl rftj create",
        "firectl sftj get",
        "firectl dpo-job get",
        "firectl rftj get",
    ):
        if command not in orchestrate:
            errors.append(
                f"{orchestrate_path.relative_to(SKILLS_DIR)}: missing method-specific command `{command}`"
            )
    for marker in ("--job-id <run-id>", "--deployment-id <run-id>-deploy"):
        if marker not in orchestrate:
            errors.append(
                f"{orchestrate_path.relative_to(SKILLS_DIR)}: idempotent create guidance missing `{marker}`"
            )
    for marker in (
        "dpo-job create --job-id",
        "dpo-job create --loss-method ORPO",
        "rftj create --job-id",
    ):
        if marker not in orchestrate:
            errors.append(
                f"{orchestrate_path.relative_to(SKILLS_DIR)}: method-specific sweep missing `{marker}`"
            )
    if re.search(r"(?m)^\s*firectl sftj export-metrics\b", orchestrate):
        errors.append(
            f"{orchestrate_path.relative_to(SKILLS_DIR)}: uses nonexistent `sftj export-metrics` command"
        )

    choose_path = skill_dir / "references/choose-method.md"
    choose = choose_path.read_text(encoding="utf-8")
    for marker in ('"managed-rft"', '"sdk-rft"', "sdk_reward_required_field"):
        if marker not in choose:
            errors.append(
                f"{choose_path.relative_to(SKILLS_DIR)}: RFT validator missing `{marker}`"
            )
    if re.search(
        r'elif method in \{"managed-rft", "sdk-rft"\}:[\s\S]{0,300}'
        r'assert "ground_truth" in o',
        choose,
    ):
        errors.append(
            f"{choose_path.relative_to(SKILLS_DIR)}: managed RFT must not require `ground_truth`"
        )

    state_path = skill_dir / "references/run-state-and-reporting.md"
    state = state_path.read_text(encoding="utf-8")
    for heading in (
        "## State machine",
        "## Resume safely",
        "### Resume a Training SDK run",
        "## Required final report",
        "planned_evaluator_name:",
        "evaluator_source_sha256:",
        "planned_job_id:",
        "planned_deployment_id:",
        "sdk_trainer_job:",
        "sdk_latest_checkpoint:",
    ):
        if heading not in state:
            errors.append(
                f"{state_path.relative_to(SKILLS_DIR)}: missing contract section `{heading}`"
            )

    training_api_path = skill_dir / "references/training-api.md"
    training_api = training_api_path.read_text(encoding="utf-8")
    if "Plan + confirm before protected work" not in (
        SKILLS_DIR / "dev/SKILL.md"
    ).read_text(encoding="utf-8"):
        errors.append("dev/SKILL.md: Training SDK protected-work approval gate is missing")
    if re.search(
        r"The dataset is JSONL[^.]*with a `ground_truth` field per row",
        training_api,
    ):
        errors.append(
            f"{training_api_path.relative_to(SKILLS_DIR)}: SDK RFT incorrectly requires universal `ground_truth`"
        )
    preference_path = skill_dir / "references/preference-data-and-evaluators.md"
    preference = preference_path.read_text(encoding="utf-8")
    for marker in (
        "offline-only",
        "source hash",
        "do not run it again",
        "Register only after approval",
    ):
        if marker not in preference:
            errors.append(
                f"{preference_path.relative_to(SKILLS_DIR)}: evaluator safety guidance missing `{marker}`"
            )

    marketplace_path = SKILLS_DIR.parent / ".claude-plugin/marketplace.json"
    marketplace = marketplace_path.read_text(encoding="utf-8")
    for plugin in ('"name": "fireworks-fine-tuning"', '"name": "fireworks-training"'):
        if plugin not in marketplace:
            errors.append(
                f"{marketplace_path.relative_to(SKILLS_DIR.parent)}: missing companion plugin `{plugin}`"
            )
    training_plugin_path = SKILLS_DIR / "dev/.claude-plugin/plugin.json"
    if not training_plugin_path.exists():
        errors.append("dev/.claude-plugin/plugin.json: Training SDK plugin manifest is missing")


def main() -> int:
    skill_mds = sorted(SKILLS_DIR.glob("*/SKILL.md")) + sorted(SKILLS_DIR.glob("*/*/SKILL.md"))
    if not skill_mds:
        print("no SKILL.md found under skills/", file=sys.stderr)
        return 1
    errors: list[str] = []
    checked = 0
    for skill_md in skill_mds:
        slug = str(skill_md.parent.relative_to(SKILLS_DIR))
        if slug in SKIP_SKILLS:
            print(f"skip (out of scope): {slug}")
            continue
        if slug not in PAVED_SKILLS and slug not in STUB_SKILLS:
            print(f"skip (not a paved skill): {slug}")
            continue
        check_skill(skill_md, errors)
        checked += 1
    if errors:
        print("Skill validation FAILED:\n", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"OK: validated {checked} paved skill(s), no errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
