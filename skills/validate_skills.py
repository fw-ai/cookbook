#!/usr/bin/env python3
"""Validate the canonical Fireworks training skill and its routed references."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILLS_DIR.parent
CANONICAL_SKILL = "fireworks-training"

REF_RE = re.compile(r"references/[\w./-]+\.md")
LINK_RE = re.compile(r"\]\((?!https?://|mailto:)([^)#]+\.md)(?:#[^)]+)?\)")
COOKBOOK_LINK_RE = re.compile(r"\]\(([^)]+(?:training/(?:recipes|examples|utils)/)[^)]+)\)")
FORBIDDEN_PRODUCT_TERMS = (
    re.compile(r"\bPilot\b", re.IGNORECASE),
    re.compile(r"firectl\s+session", re.IGNORECASE),
    re.compile(r"fireworks-fine-tuning"),
    re.compile(r"skills/dev"),
)


def parse_frontmatter(text: str) -> dict[str, str] | None:
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    result: dict[str, str] = {}
    key: str | None = None
    for line in text[3:end].splitlines():
        match = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
        if match:
            key = match.group(1)
            result[key] = match.group(2).strip()
        elif key and line.strip():
            result[key] += " " + line.strip()
    return result


def check_frontmatter(skill_md: Path, errors: list[str]) -> None:
    text = skill_md.read_text(encoding="utf-8")
    frontmatter = parse_frontmatter(text)
    if not frontmatter:
        errors.append(f"{skill_md.relative_to(SKILLS_DIR)}: missing YAML frontmatter")
        return
    for field in ("name", "description"):
        if not frontmatter.get(field):
            errors.append(
                f"{skill_md.relative_to(SKILLS_DIR)}: missing non-empty `{field}`"
            )
    expected_name = skill_md.parent.name
    if frontmatter.get("name") != expected_name:
        errors.append(
            f"{skill_md.relative_to(SKILLS_DIR)}: name must match directory "
            f"(`{expected_name}`)"
        )


def check_skill(skill_dir: Path, errors: list[str]) -> None:
    skill_md = skill_dir / "SKILL.md"
    check_frontmatter(skill_md, errors)
    root_text = skill_md.read_text(encoding="utf-8")
    references_dir = skill_dir / "references"
    reference_files = sorted(references_dir.rglob("*.md"))

    for ref_file in reference_files:
        if ref_file.parent != references_dir:
            errors.append(
                "SKILL.md: references must stay one level deep: "
                f"`{ref_file.relative_to(skill_dir)}`"
            )

    if (skill_dir / ".claude-plugin").exists():
        errors.append(
            "fireworks-training: plugin metadata belongs at the repository root"
        )

    mentioned = set(REF_RE.findall(root_text))
    for ref in sorted(mentioned):
        if not (skill_dir / ref).exists():
            errors.append(f"SKILL.md: routed reference is missing: `{ref}`")

    for ref_file in reference_files:
        rel = ref_file.relative_to(skill_dir).as_posix()
        if rel not in mentioned:
            errors.append(f"SKILL.md: reference is not routed: `{rel}`")

    markdown_files = [skill_md, *reference_files]
    for markdown_file in markdown_files:
        text = markdown_file.read_text(encoding="utf-8")
        for link in LINK_RE.findall(text):
            target = (markdown_file.parent / link).resolve()
            if not target.exists():
                errors.append(
                    f"{markdown_file.relative_to(SKILLS_DIR)}: broken link `{link}`"
                )
        for link in COOKBOOK_LINK_RE.findall(text):
            if link.startswith("https://github.com/fw-ai/cookbook/"):
                if "/main/" not in link:
                    errors.append(
                        f"{markdown_file.relative_to(SKILLS_DIR)}: cookbook URL "
                        f"must point to public main: `{link}`"
                    )
                continue
            target = (markdown_file.parent / link).resolve()
            if not target.exists():
                errors.append(
                    f"{markdown_file.relative_to(SKILLS_DIR)}: cookbook path "
                    f"does not exist: `{link}`"
                )
        for forbidden in FORBIDDEN_PRODUCT_TERMS:
            if forbidden.search(text):
                errors.append(
                    f"{markdown_file.relative_to(SKILLS_DIR)}: forbidden legacy "
                    f"term matches `{forbidden.pattern}`"
                )
        for cookbook_url in re.finditer(
            r"https://github\.com/fw-ai/cookbook/(?:blob|tree)/([^/\s)]+)",
            text,
        ):
            if cookbook_url.group(1) != "main":
                errors.append(
                    f"{markdown_file.relative_to(SKILLS_DIR)}: cookbook URL "
                    f"must point to public main: `{cookbook_url.group(0)}`"
                )

    check_training_contract(skill_dir, root_text, errors)


def check_training_contract(
    skill_dir: Path, root_text: str, errors: list[str]
) -> None:
    for relative in (
        "training/recipes/sft_loop.py",
        "training/recipes/dpo_loop.py",
        "training/recipes/orpo_loop.py",
        "training/recipes/rl_loop.py",
        "training/recipes/async_rl_loop.py",
        "training/recipes/igpo_loop.py",
        "training/recipes/distillation_loop.py",
        "training/examples/serverless_rl/countdown_rl.py",
        "training/pyproject.toml",
    ):
        if not (REPO_ROOT / relative).exists():
            errors.append(f"canonical cookbook path is missing: `{relative}`")

    plan_marker = "## Mandatory final-plan confirmation"
    first_create = "firectl sftj create"
    if plan_marker not in root_text or first_create not in root_text:
        errors.append("SKILL.md: missing final-plan gate or managed create commands")
    elif root_text.index(plan_marker) > root_text.index(first_create):
        errors.append("SKILL.md: create command appears before the confirmation gate")

    required_markers = (
        "every parameter the user set",
        "every default",
        "--dry-run -o json",
        "platform-resolved, unknown before create",
        "requires renewed confirmation",
        "Promotion and deployment each require",
        "## Agent execution boundary",
        "BLOCKED: mutating command",
        "must never configure",
        "https://docs.fireworks.ai/llms.txt",
        "https://docs.fireworks.ai/fine-tuning/training-api/serverless.md",
        "https://docs.fireworks.ai/fine-tuning/training-api/training-and-sampling.md",
        "git rev-parse HEAD",
        "training/pyproject.toml",
        "training/recipes/sft_loop.py",
        "training/recipes/dpo_loop.py",
        "training/recipes/orpo_loop.py",
        "training/recipes/rl_loop.py",
        "training/recipes/async_rl_loop.py",
        "training/recipes/distillation_loop.py",
        "training/examples/serverless_rl/",
        "--job-id <run-id>",
        "--deployment-id <run-id>-deploy",
        "--deployment-shape accounts/fireworks/deploymentShapes/<resolved-shape>",
        "firectl dpo-job create",
        "firectl dpo-job create --loss-method DPO",
        "firectl dpo-job create --loss-method ORPO",
        "firectl rftj create",
        "firectl rftj create --evaluator <resource>",
    )
    for marker in required_markers:
        if marker not in root_text:
            errors.append(f"SKILL.md: required training contract missing `{marker}`")

    if re.search(r"under\s+~?\$5.*proceed", root_text, re.IGNORECASE):
        errors.append("SKILL.md: small-run auto-proceed exception is forbidden")
    if re.search(r"(?m)^\s*firectl sftj export-metrics\b", root_text):
        errors.append("SKILL.md: nonexistent `sftj export-metrics` command")

    choose = (skill_dir / "references/choose-method.md").read_text(encoding="utf-8")
    for marker in (
        '"managed-rft"',
        '"sdk-rft"',
        "managed_evaluator_required_fields",
        "sdk_reward_required_fields",
        "validate_preference_output",
        "3_000_000",
        "previous_role",
        "DPO input must contain exactly one user turn",
    ):
        if marker not in choose:
            errors.append(f"choose-method.md: RFT validator missing `{marker}`")
    if re.search(
        r'elif method == "managed-rft":[\s\S]{0,500}'
        r'assert "ground_truth" in o',
        choose,
    ):
        errors.append("choose-method.md: managed RFT must not require ground_truth")

    state = (skill_dir / "references/run-state-and-reporting.md").read_text(
        encoding="utf-8"
    )
    for marker in (
        "## State machine",
        "partial_failure_cleanup",
        "## Resume safely",
        "### Resume a Training API dedicated run",
        "## Required final report",
        "planned_evaluator_name:",
        "evaluator_source_sha256:",
        "evaluator_account:",
        "evaluator_registration_started_at_utc:",
        "planned_job_id:",
        "planned_deployment_id:",
        "trainer_job:",
        "latest_checkpoint:",
        "firectl_version:",
        "docs_urls:",
        "cookbook_commit:",
        "sdk_version:",
    ):
        if marker not in state:
            errors.append(f"run-state-and-reporting.md: missing `{marker}`")

    preference = (
        skill_dir / "references/preference-data-and-evaluators.md"
    ).read_text(encoding="utf-8")
    for marker in (
        "offline-only",
        "source hash",
        "do not run it again",
        "Register only after approval",
    ):
        if marker not in preference:
            errors.append(
                "preference-data-and-evaluators.md: evaluator safety guidance "
                f"missing `{marker}`"
            )

    marketplace_path = REPO_ROOT / ".claude-plugin/marketplace.json"
    marketplace = json.loads(marketplace_path.read_text(encoding="utf-8"))
    plugins = marketplace.get("plugins", [])
    if len(plugins) != 1:
        errors.append(".claude-plugin/marketplace.json: expected exactly one plugin")
    elif plugins[0].get("name") != CANONICAL_SKILL:
        errors.append(
            ".claude-plugin/marketplace.json: plugin must be `fireworks-training`"
        )
    elif plugins[0].get("source") != "./":
        errors.append(
            ".claude-plugin/marketplace.json: plugin source must be repository root"
        )

    plugin_path = REPO_ROOT / ".claude-plugin/plugin.json"
    if not plugin_path.exists():
        errors.append(".claude-plugin/plugin.json: missing")
    else:
        plugin = json.loads(plugin_path.read_text(encoding="utf-8"))
        if plugin.get("name") != CANONICAL_SKILL:
            errors.append("plugin.json: name must be `fireworks-training`")

    codex_plugin_path = REPO_ROOT / ".codex-plugin/plugin.json"
    if not codex_plugin_path.exists():
        errors.append(".codex-plugin/plugin.json: missing")
    else:
        codex_plugin = json.loads(codex_plugin_path.read_text(encoding="utf-8"))
        if codex_plugin.get("name") != "cookbook":
            errors.append("Codex plugin name must match the `cookbook` plugin root")
        if codex_plugin.get("skills") != "./skills/":
            errors.append("Codex plugin must package the canonical `./skills/` tree")

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for marker in (
        "claude plugin install fireworks-training@fw-ai-cookbook",
        "-a cursor -y",
        "-a codex -y",
        ".codex-plugin/plugin.json",
        "AI-agent safety guard",
    ):
        if marker not in readme:
            errors.append(f"README.md: portable install guidance missing `{marker}`")


def check_repository_legacy_terms(errors: list[str]) -> None:
    """Keep canonical cookbook entry points free of retired product surfaces."""
    paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "CLAUDE.md",
        *sorted((REPO_ROOT / "training").rglob("*.md")),
        *sorted((REPO_ROOT / "training/recipes").glob("*.py")),
        REPO_ROOT / "training/utils/config.py",
    ]
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_PRODUCT_TERMS:
            if forbidden.search(text):
                errors.append(
                    f"{path.relative_to(REPO_ROOT)}: forbidden legacy term "
                    f"matches `{forbidden.pattern}`"
                )


def check_serverless_example(errors: list[str]) -> None:
    example = (
        REPO_ROOT / "training/examples/serverless_rl/countdown_rl.py"
    ).read_text(encoding="utf-8")
    readme = (
        REPO_ROOT / "training/examples/serverless_rl/README.md"
    ).read_text(encoding="utf-8")
    for path, text in (("countdown_rl.py", example), ("README.md", readme)):
        if "qwen3p6-27b" not in text:
            errors.append(f"serverless_rl/{path}: supported default model is missing")
        if "qwen3p5-27b" in text:
            errors.append(f"serverless_rl/{path}: unsupported stale model remains")
        if "max_seq_len" not in text:
            errors.append(f"serverless_rl/{path}: explicit sequence bound is missing")


def main() -> int:
    errors: list[str] = []
    skill_mds = sorted(SKILLS_DIR.glob("*/SKILL.md"))
    active = []
    for skill_md in skill_mds:
        slug = skill_md.parent.name
        active.append(slug)
        if slug != CANONICAL_SKILL:
            errors.append(f"unexpected installable skill: `{slug}`")

    canonical_dir = SKILLS_DIR / CANONICAL_SKILL
    if not canonical_dir.exists():
        errors.append("canonical skill directory is missing")
    else:
        check_skill(canonical_dir, errors)
    check_repository_legacy_terms(errors)
    check_serverless_example(errors)

    if active != [CANONICAL_SKILL]:
        errors.append(
            f"expected only `{CANONICAL_SKILL}` as active skill, found {active}"
        )

    if errors:
        print("Skill validation FAILED:\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("OK: validated one canonical Fireworks training skill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
