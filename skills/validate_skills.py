#!/usr/bin/env python3
"""Validate Fireworks training skills (research, configure, debug) and redirect stubs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILLS_DIR.parent
PLUGIN_NAME = "fireworks-training"
ACTIVE_SKILLS = ("research", "configure", "debug")
REDIRECT_STUBS = ("fireworks-training", "discover")
EXPECTED_SKILLS = (*ACTIVE_SKILLS, *REDIRECT_STUBS)
SKILL_VERSION = "2.2.0"

REF_RE = re.compile(r"(?:\.\./configure/)?references/[\w./-]+\.md")
LOCAL_REF_RE = re.compile(r"references/[\w./-]+\.md")
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


def resolve_reference(skill_dir: Path, ref: str) -> Path:
    if ref.startswith("../configure/"):
        return (skill_dir / ref).resolve()
    return (skill_dir / ref).resolve()


def check_markdown_files(
    markdown_files: list[Path], skill_dir: Path, errors: list[str]
) -> None:
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


def check_skill_references(skill_dir: Path, root_text: str, errors: list[str]) -> None:
    references_dir = skill_dir / "references"
    if not references_dir.exists():
        return

    reference_files = sorted(references_dir.rglob("*.md"))
    for ref_file in reference_files:
        if ref_file.parent != references_dir:
            errors.append(
                f"{skill_dir.name}/SKILL.md: references must stay one level deep: "
                f"`{ref_file.relative_to(skill_dir)}`"
            )

    mentioned_local = set(LOCAL_REF_RE.findall(root_text))
    for ref in sorted(mentioned_local):
        if (skill_dir / ref).exists():
            continue
        if (SKILLS_DIR / ref).exists():
            continue
        errors.append(f"{skill_dir.name}/SKILL.md: routed reference missing `{ref}`")

    for ref_file in reference_files:
        rel = ref_file.relative_to(skill_dir).as_posix()
        if rel not in mentioned_local:
            errors.append(f"{skill_dir.name}/SKILL.md: reference is not routed: `{rel}`")

    for ref in REF_RE.findall(root_text):
        if ref.startswith("../configure/"):
            if not resolve_reference(skill_dir, ref).exists():
                errors.append(
                    f"{skill_dir.name}/SKILL.md: cross-skill reference missing `{ref}`"
                )

    markdown_files = [skill_dir / "SKILL.md", *reference_files]
    check_markdown_files(markdown_files, skill_dir, errors)


def check_redirect_stub(skill_dir: Path, errors: list[str]) -> None:
    skill_md = skill_dir / "SKILL.md"
    check_frontmatter(skill_md, errors)
    root_text = skill_md.read_text(encoding="utf-8")
    for marker in ("**research**", "**configure**", "**debug**", SKILL_VERSION, "redirect"):
        if marker not in root_text:
            errors.append(f"{skill_dir.name} stub: missing redirect marker `{marker}`")
    if (skill_dir / "references").exists():
        errors.append(f"{skill_dir.name} stub: must not ship a references/ directory")
    check_markdown_files([skill_md], skill_dir, errors)


def check_lightweight_skill(skill_dir: Path, errors: list[str]) -> None:
    skill_md = skill_dir / "SKILL.md"
    check_frontmatter(skill_md, errors)
    root_text = skill_md.read_text(encoding="utf-8")

    if (skill_dir / ".claude-plugin").exists():
        errors.append(f"{skill_dir.name}: plugin metadata belongs at the repository root")

    for marker in ("FIREWORKS_CLIENT_SOURCE", f"fireworks-training-skill/{SKILL_VERSION}", "AskQuestion"):
        if marker not in root_text:
            errors.append(f"{skill_dir.name}/SKILL.md: missing `{marker}`")

    if skill_dir.name == "research":
        for marker in (
            "telemetry-schema.md",
            "Journey telemetry",
            "research_intake_answered",
            "interview-questions.md",
            "cookbook-catalog.md",
            "telemetry-notice.md",
        ):
            if marker not in root_text:
                errors.append(f"research/SKILL.md: missing `{marker}`")

    check_skill_references(skill_dir, root_text, errors)


def check_configure_skill(skill_dir: Path, errors: list[str]) -> None:
    skill_md = skill_dir / "SKILL.md"
    check_frontmatter(skill_md, errors)
    root_text = skill_md.read_text(encoding="utf-8")

    if (skill_dir / ".claude-plugin").exists():
        errors.append("configure: plugin metadata belongs at the repository root")

    for marker in (
        "cost-estimation.md",
        "models-shapes-and-cost.md",
        "Do not calculate Dedicated SFT or DPO",
        "docs.fireworks.ai/fine-tuning/cost-estimator",
        "api-key-setup.md",
        "Never ask for the key in chat",
    ):
        if marker not in root_text:
            errors.append(f"configure/SKILL.md: missing `{marker}`")

    check_skill_references(skill_dir, root_text, errors)
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
        errors.append("configure/SKILL.md: missing final-plan gate or managed create commands")
    elif root_text.index(plan_marker) > root_text.index(first_create):
        errors.append("configure/SKILL.md: create command appears before the confirmation gate")

    required_markers = (
        "## Entry routing",
        "entry_skill: configure",
        "always monitoring",
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
        "https://docs.fireworks.ai/fine-tuning/training-api/dedicated#training-and-sampling.md",
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
        "FIREWORKS_CLIENT_SOURCE",
        "FIREWORKS_SESSION_ID",
        f"fireworks-training-skill/{SKILL_VERSION}",
        "Do not create a separate telemetry file",
        "Never use `PURPOSE_PILOT`",
        "telemetry-schema.md",
        "telemetry-notice.md",
        "configure_path_answered",
        "session_outcome",
    )
    for marker in required_markers:
        if marker not in root_text:
            errors.append(f"configure/SKILL.md: required training contract missing `{marker}`")

    if re.search(r"under\s+~?\$5.*proceed", root_text, re.IGNORECASE):
        errors.append("configure/SKILL.md: small-run auto-proceed exception is forbidden")
    if re.search(r"(?m)^\s*firectl sftj export-metrics\b", root_text):
        errors.append("configure/SKILL.md: nonexistent `sftj export-metrics` command")

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
        "skill_session_id:",
        "skill_client_source:",
        "## Journey telemetry",
        "journey_schema_version:",
        "intake_responses",
        "task_summary:",
        "telemetry_opt_out:",
        "handoff_choice:",
        "session_outcome:",
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


def check_plugin_metadata(errors: list[str]) -> None:
    marketplace_path = REPO_ROOT / ".claude-plugin/marketplace.json"
    marketplace = json.loads(marketplace_path.read_text(encoding="utf-8"))
    plugins = marketplace.get("plugins", [])
    if len(plugins) != 1:
        errors.append(".claude-plugin/marketplace.json: expected exactly one plugin")
    elif plugins[0].get("name") != PLUGIN_NAME:
        errors.append(
            f".claude-plugin/marketplace.json: plugin must be `{PLUGIN_NAME}`"
        )
    elif plugins[0].get("source") != "./":
        errors.append(
            ".claude-plugin/marketplace.json: plugin source must be repository root"
        )
    else:
        description = plugins[0].get("description", "")
        for skill in ACTIVE_SKILLS:
            if skill not in description:
                errors.append(
                    f".claude-plugin/marketplace.json: description must mention `{skill}`"
                )

    plugin_path = REPO_ROOT / ".claude-plugin/plugin.json"
    if not plugin_path.exists():
        errors.append(".claude-plugin/plugin.json: missing")
    else:
        plugin = json.loads(plugin_path.read_text(encoding="utf-8"))
        if plugin.get("name") != PLUGIN_NAME:
            errors.append(f"plugin.json: name must be `{PLUGIN_NAME}`")
        if plugin.get("version") != SKILL_VERSION:
            errors.append(f"plugin.json: version must be `{SKILL_VERSION}`")

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
        "skills/research/",
        "skills/configure/",
        "skills/debug/",
        "skills/GETTING-STARTED.md",
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


def check_telemetry_schema(errors: list[str]) -> None:
    schema = SKILLS_DIR / "references" / "telemetry-schema.md"
    if not schema.exists():
        errors.append("skills/references/telemetry-schema.md: missing")
        return
    text = schema.read_text(encoding="utf-8")
    for marker in (
        "journey_schema_version",
        "welcome_choice",
        "intake_q1_task_shape",
        "intake_q_eval",
        "handoff_choice",
        "session_outcome",
        "research_only",
        "intake_responses",
        "task_summary",
        "user_choice",
        "response_source",
        "telemetry-notice.md",
        "research_intake_answered",
        "discover_intake_answered",
        "telemetry/journey-api-spec.md",
    ):
        if marker not in text:
            errors.append(f"telemetry-schema.md: missing `{marker}`")

    interview = (
        SKILLS_DIR / "research" / "references" / "interview-questions.md"
    ).read_text(encoding="utf-8")
    for option_id in (
        "structured_output",
        "plan_configure",
        "intake_q1_task_shape",
        "research-q-eval",
        "Option ID",
    ):
        if option_id not in interview:
            errors.append(f"interview-questions.md: missing `{option_id}`")

    welcome = (SKILLS_DIR / "references" / "welcome.md").read_text(encoding="utf-8")
    if "welcome_choice" not in welcome:
        errors.append("welcome.md: missing `welcome_choice` telemetry")
    if "telemetry-notice.md" not in welcome:
        errors.append("welcome.md: missing telemetry-notice.md reference")

    notice = (SKILLS_DIR / "references" / "telemetry-notice.md")
    if not notice.exists():
        errors.append("skills/references/telemetry-notice.md: missing")

    api_key_setup = SKILLS_DIR / "references" / "api-key-setup.md"
    if not api_key_setup.exists():
        errors.append("skills/references/api-key-setup.md: missing")
    else:
        setup_text = api_key_setup.read_text(encoding="utf-8")
        for marker in ("read -s FIREWORKS_API_KEY", "Never ask the user to paste"):
            if marker not in setup_text:
                errors.append(f"api-key-setup.md: missing `{marker}`")

    for rel in (
        "references/telemetry/journey-api-spec.md",
        "references/telemetry/jarvis-funnel-tiles.md",
        "references/telemetry/sdk-firectl-helpers.md",
    ):
        if not (SKILLS_DIR / rel).exists():
            errors.append(f"skills/{rel}: missing Phase 2 spec")


def check_welcome_shared(errors: list[str]) -> None:
    welcome = SKILLS_DIR / "references" / "welcome.md"
    if not welcome.exists():
        errors.append("skills/references/welcome.md: missing shared welcome file")
        return
    text = welcome.read_text(encoding="utf-8")
    for marker in (
        "**Fireworks Training**",
        "three ways I can help",
        "Entry AskQuestion",
        "research",
        "configure",
        "debug",
    ):
        if marker not in text:
            errors.append(f"welcome.md: missing `{marker}`")
    for slug in ACTIVE_SKILLS:
        skill_md = SKILLS_DIR / slug / "SKILL.md"
        if not skill_md.exists():
            continue
        root = skill_md.read_text(encoding="utf-8")
        if "welcome.md" not in root:
            errors.append(f"{slug}/SKILL.md: missing welcome.md reference")


def check_serverless_example(errors: list[str]) -> None:
    example = (
        REPO_ROOT / "training/examples/serverless_rl/countdown_rl.py"
    ).read_text(encoding="utf-8")
    readme = (
        REPO_ROOT / "training/examples/serverless_rl/README.md"
    ).read_text(encoding="utf-8")
    for path, text in (("countdown_rl.py", example), ("README.md", readme)):
        if "kimi-k3" not in text:
            errors.append(f"serverless_rl/{path}: supported default model is missing")
        if "qwen3p5-27b" in text:
            errors.append(f"serverless_rl/{path}: unsupported stale model remains")
        if "max_seq_len" not in text:
            errors.append(f"serverless_rl/{path}: explicit sequence bound is missing")


def main() -> int:
    errors: list[str] = []
    skill_mds = sorted(SKILLS_DIR.glob("*/SKILL.md"))
    active = [skill_md.parent.name for skill_md in skill_mds]

    unexpected = sorted(set(active) - set(EXPECTED_SKILLS))
    missing = sorted(set(EXPECTED_SKILLS) - set(active))
    for slug in unexpected:
        errors.append(f"unexpected skill directory: `{slug}`")
    for slug in missing:
        errors.append(f"missing expected skill directory: `{slug}`")

    for slug in ACTIVE_SKILLS:
        skill_dir = SKILLS_DIR / slug
        if not skill_dir.exists():
            continue
        if slug == "configure":
            check_configure_skill(skill_dir, errors)
        else:
            check_lightweight_skill(skill_dir, errors)

    for slug in REDIRECT_STUBS:
        stub_dir = SKILLS_DIR / slug
        if stub_dir.exists():
            check_redirect_stub(stub_dir, errors)

    check_plugin_metadata(errors)
    check_welcome_shared(errors)
    check_telemetry_schema(errors)
    check_repository_legacy_terms(errors)
    check_serverless_example(errors)

    if sorted(active) != sorted(EXPECTED_SKILLS):
        errors.append(
            f"expected skills {sorted(EXPECTED_SKILLS)}, found {sorted(active)}"
        )

    if errors:
        print("Skill validation FAILED:\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(
        "OK: validated Fireworks training skills "
        "(research, configure, debug) and redirect stubs (discover, fireworks-training)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
