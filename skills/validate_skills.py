#!/usr/bin/env python3
"""Validate the slim Fireworks training skill split."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILLS_DIR.parent
ACTIVE_SKILLS = ("research", "configure", "debug")
REDIRECTS = ("discover", "fireworks-training")
EXPECTED_SKILLS = (*ACTIVE_SKILLS, *REDIRECTS)
CONFIGURE_PATH_OPTIONS = {
    "managed_firectl",
    "managed_sdk",
    "serverless",
    "dedicated",
}

LINK_RE = re.compile(r"\]\((?!https?://|mailto:)([^)]+)\)")
CARRIER_REF_RE = re.compile(r"\.\./fireworks-training/references/([\w-]+\.md)")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return ""
    parts = text.split("---", 2)
    return parts[1] if len(parts) == 3 else ""


def check_skill_names(errors: list[str]) -> None:
    found = sorted(path.parent.name for path in SKILLS_DIR.glob("*/SKILL.md"))
    if found != sorted(EXPECTED_SKILLS):
        errors.append(f"expected skills {sorted(EXPECTED_SKILLS)}, found {found}")
    for name in found:
        text = read(SKILLS_DIR / name / "SKILL.md")
        if f"name: {name}" not in frontmatter(text):
            errors.append(f"{name}/SKILL.md: frontmatter name must match directory")


def check_links(errors: list[str]) -> None:
    for path in sorted(SKILLS_DIR.rglob("*.md")):
        for link in LINK_RE.findall(read(path)):
            file_part, _, anchor = link.partition("#")
            if file_part and not file_part.endswith(".md"):
                continue
            target = (path.parent / file_part).resolve() if file_part else path
            if not target.exists():
                errors.append(f"{path.relative_to(SKILLS_DIR)}: broken link `{link}`")
                continue
            if (
                anchor
                and target.name == "SKILL.md"
                and anchor not in markdown_anchors(read(target))
            ):
                errors.append(f"{path.relative_to(SKILLS_DIR)}: broken anchor `{link}`")


def markdown_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        heading = line.lstrip("#").strip().lower()
        slug = re.sub(r"[^\w\s-]", "", heading)
        slug = re.sub(r"[\s-]+", "-", slug).strip("-")
        if slug:
            anchors.add(slug)
    return anchors


def require_markers(
    path: Path,
    markers: tuple[str, ...],
    errors: list[str],
) -> None:
    text = read(path)
    for marker in markers:
        if marker not in text:
            errors.append(f"{path.relative_to(SKILLS_DIR)}: missing `{marker}`")


def check_entry_skills(errors: list[str]) -> None:
    require_markers(
        SKILLS_DIR / "research" / "SKILL.md",
        (
            "entry_skill: research",
            "interview-questions.md",
            "completion gate",
            "firectl skill-journey record --help",
            "configure",
            "debug",
        ),
        errors,
    )
    require_markers(
        SKILLS_DIR / "configure" / "SKILL.md",
        (
            "entry_skill: configure",
            "references/path-intake.md",
            "references/cost-estimation.md",
            "Mandatory final-plan gate",
            "BLOCKED: mutating command",
            "actively monitoring",
            "Do not calculate Dedicated SFT or DPO",
            "fireworks-training/references",
        ),
        errors,
    )
    require_markers(
        SKILLS_DIR / "debug" / "SKILL.md",
        (
            "entry_skill: debug",
            "three failed hypotheses",
            "firectl skill-journey record --help",
            "fireworks-training/references",
        ),
        errors,
    )
    for name in REDIRECTS:
        require_markers(
            SKILLS_DIR / name / "SKILL.md",
            ("redirect", "research", "configure", "debug"),
            errors,
        )


def check_slim_layout(errors: list[str]) -> None:
    configure = SKILLS_DIR / "configure"
    allowed = {
        "cost-estimation.md",
        "output-template.md",
        "path-intake.md",
    }
    found = {path.name for path in (configure / "references").glob("*.md")}
    if found != allowed:
        errors.append(
            f"configure references must be slim: expected {sorted(allowed)}, "
            f"found {sorted(found)}"
        )
    if len(read(configure / "SKILL.md").splitlines()) > 240:
        errors.append("configure/SKILL.md exceeds 240-line slim contract")
    for retired in (
        SKILLS_DIR / "research" / "references" / "reframe-v2-draft.md",
        SKILLS_DIR / "research" / "references" / "external-dataset-discovery-draft.md",
        SKILLS_DIR / "research" / "scripts" / "hf_dataset_search.py",
    ):
        if retired.exists():
            errors.append(f"deferred draft remains: {retired.relative_to(SKILLS_DIR)}")


def check_packaging(errors: list[str]) -> None:
    plugin = json.loads(read(REPO_ROOT / ".claude-plugin" / "plugin.json"))
    codex = json.loads(read(REPO_ROOT / ".codex-plugin" / "plugin.json"))
    if plugin.get("version") != "2.2.0":
        errors.append(".claude-plugin/plugin.json: expected version 2.2.0")
    if codex.get("version") != "2.2.0":
        errors.append(".codex-plugin/plugin.json: expected version 2.2.0")
    if codex.get("skills") != "./skills/":
        errors.append(".codex-plugin/plugin.json: expected skills ./skills/")

    for path in (REPO_ROOT / "README.md", SKILLS_DIR / "GETTING-STARTED.md"):
        text = read(path)
        for marker in (
            "-s fireworks-training",
            "-s research",
            "-s configure",
            "-s debug",
        ):
            if marker not in text:
                errors.append(f"{path.name}: install command missing `{marker}`")


def check_carrier(errors: list[str]) -> None:
    carrier = SKILLS_DIR / "fireworks-training" / "references"
    carrier_files = {path.name for path in carrier.glob("*.md")}
    routed: set[str] = set()
    for skill in ACTIVE_SKILLS:
        routed.update(CARRIER_REF_RE.findall(read(SKILLS_DIR / skill / "SKILL.md")))
    missing_routes = carrier_files - routed
    unknown_routes = routed - carrier_files
    if missing_routes:
        errors.append(
            f"carrier references not routed by active skills: {sorted(missing_routes)}"
        )
    if unknown_routes:
        errors.append(
            f"active skills route missing carrier refs: {sorted(unknown_routes)}"
        )
    for path in carrier.glob("*.md"):
        text = read(path)
        if "`../SKILL.md`" in text or "root `SKILL.md`" in text:
            errors.append(
                f"fireworks-training/references/{path.name}: "
                "stale carrier root backlink"
            )
    telemetry = read(carrier / "telemetry.md")
    for marker in ("symlink", "`0700`", "`0600`", ".gitignore", "skip recording"):
        if marker not in telemetry:
            errors.append(f"telemetry.md: local fallback missing `{marker}`")
    if "repository helper" in telemetry:
        errors.append("telemetry.md: stale removed helper guidance")


def check_cookbook_routes(errors: list[str]) -> None:
    required = (
        "training/case-studies/sft_prompt_router/prompt_router_sft_sdk.ipynb",
        "training/case-studies/sft_cord_receipts/cord_receipt_sft_sdk.ipynb",
        "training/case-studies/dpo_style/dpo_helpsteer3_sdk.ipynb",
        "training/case-studies/reasoning_rl/rft_grpo_math.ipynb",
        "training/case-studies/embedding_support_search/airbnb_policy_embedding.ipynb",
        "training/case-studies/agentic_rl_text2sql/sql_agent_rl_loop.ipynb",
        "training/examples/sft/train_sft.py",
        "training/examples/dpo/train_dpo.py",
        "training/examples/embedding/train_embedding.py",
        "training/examples/orpo/ifeval/train_ifeval_orpo.py",
        "training/examples/serverless_rl/countdown_rl.py",
        "training/examples/serverless_dpo/ultrafeedback_dpo.py",
        "training/examples/rl/deepmath/train_deepmath.py",
        "training/examples/rl/frozen_lake/train_frozen_lake.py",
        "training/examples/rl/eval_protocol_chat/train.py",
        "training/examples/rl/harbor/recipes",
        "training/examples/multihop_qa/train_multihop_qa_igpo.py",
        "training/examples/distillation/gsm8k_privileged/train_gsm8k_privileged.py",
        "training/examples/distillation/routed_mopd/train_two_teacher_lora.py",
        "training/recipes/sft_loop.py",
        "training/recipes/dpo_loop.py",
        "training/recipes/orpo_loop.py",
        "training/recipes/rl_loop.py",
        "training/recipes/async_rl_loop.py",
        "training/recipes/igpo_loop.py",
        "training/recipes/distillation_loop.py",
        "training/recipes/embedding_loop.py",
    )
    for relative in required:
        if not (REPO_ROOT / relative).exists():
            errors.append(f"research cookbook route does not exist: `{relative}`")

    research_refs = "\n".join(
        read(path)
        for path in (
            SKILLS_DIR / "research" / "references" / "case-studies.md",
            SKILLS_DIR / "research" / "references" / "cookbook-catalog.md",
        )
    )
    for stale in (
        "prompt_router_dedicated.ipynb",
        "prompt_router_serverless.ipynb",
        "rl/coding_agent/",
        "rl/multi_turn_message_in/",
        "intake-questions.md",
    ):
        if stale in research_refs:
            errors.append(f"research catalog contains stale route `{stale}`")


def check_semantic_contracts(errors: list[str]) -> None:
    path_options = CONFIGURE_PATH_OPTIONS
    path_intake = read(SKILLS_DIR / "configure" / "references" / "path-intake.md")
    for option in path_options:
        canonical_row = re.compile(
            rf"\| `{re.escape(option)}` \|[^\n]+\| {re.escape(option)} \|"
        )
        if not canonical_row.search(path_intake):
            errors.append(f"path-intake.md: `{option}` must equal its workflow_path")

    research_handoff = "\n".join(
        read(path)
        for path in (
            SKILLS_DIR / "research" / "references" / "case-studies.md",
            SKILLS_DIR / "research" / "references" / "cookbook-catalog.md",
            SKILLS_DIR / "research" / "references" / "methodology.md",
        )
    )
    if re.search(r"(?m)^execution_surface:", research_handoff):
        errors.append("research handoff must not set execution_surface")
    if "always confirms the exact" not in research_handoff:
        errors.append("research handoff must require exact Configure Q-path")
    for field in (
        "cookbook_entry_tier:",
        "cookbook_entry_path:",
        "implied_method:",
        "suggested_path:",
        "dataset_plan:",
        "eval_plan:",
    ):
        for name in ("case-studies.md", "cookbook-catalog.md", "methodology.md"):
            text = read(SKILLS_DIR / "research" / "references" / name)
            if field not in text:
                errors.append(f"{name}: handoff schema missing `{field}`")
    canonical_methods = (
        "sft",
        "dpo",
        "orpo",
        "rft",
        "igpo",
        "distillation",
        "embedding",
    )
    for name in (
        "case-studies.md",
        "cookbook-catalog.md",
        "methodology.md",
    ):
        text = read(SKILLS_DIR / "research" / "references" / name)
        for method in canonical_methods:
            if method not in text:
                errors.append(f"{name}: handoff does not normalize `{method}`")

    questions = read(SKILLS_DIR / "research" / "references" / "interview-questions.md")
    if "Q1, Q2, and Q-eval use the six registered contract options" not in questions:
        errors.append("research questions must document their six-option exception")
    if questions.find("Fire the **Handoff** AskQuestion") > questions.find(
        "Write the handoff block"
    ):
        errors.append("research must ask for handoff approval before persistence")

    triage = read(SKILLS_DIR / "debug" / "references" / "triage-paths.md")
    if "send one real request" in triage:
        errors.append("debug must not send a paid serving request")
    if "explicit approval" not in triage:
        errors.append("debug serving probe must require Configure approval")
    if "Debug does not run the retry" not in triage:
        errors.append("debug retries must hand off to Configure")

    path_intake_markers = (
        "question_id: configure-q-method",
        "`labeled`",
        "`preference_pairs`",
        "`scored_prompts`",
        "`unsure`",
        "DPO unless the user explicitly requests ORPO",
    )
    for marker in path_intake_markers:
        if marker not in path_intake:
            errors.append(f"path-intake.md: missing Q-method contract `{marker}`")
    if "`research-q3` answer never satisfies this gate" not in path_intake:
        errors.append("research-q3 must never skip Configure Q-path")

    research_skill = read(SKILLS_DIR / "research" / "SKILL.md")
    notice_index = research_skill.find("telemetry-notice.md")
    welcome_index = research_skill.find("welcome-entry")
    if notice_index < 0 or welcome_index < 0 or notice_index > welcome_index:
        errors.append("research must show telemetry notice before welcome-entry")

    configure_skill = read(SKILLS_DIR / "configure" / "SKILL.md")
    debug_skill = read(SKILLS_DIR / "debug" / "SKILL.md")
    for name, text in (("configure", configure_skill), ("debug", debug_skill)):
        if "Research owns the `welcome-entry` question" not in text:
            errors.append(f"{name} must route welcome-entry to Research")
    if "platform-resolved,\n   unknown before create" not in configure_skill:
        errors.append("configure must preserve unknowable backend defaults")
    for field in (
        "cookbook_entry_tier",
        "cookbook_entry_path",
        "notebook",
        "readme",
    ):
        if field not in configure_skill:
            errors.append(f"configure must consume handoff field `{field}`")

    run_state = read(
        SKILLS_DIR / "fireworks-training" / "references" / "run-state-and-reporting.md"
    )
    if "fireworks-training-skill/2.2.0" not in run_state:
        errors.append("run manifest attribution must use skill version 2.2.0")
    if "fireworks-training-skill/2.0.0" in run_state:
        errors.append("run manifest retains stale skill version 2.0.0")
    for method in canonical_methods:
        if method not in run_state or method not in path_intake:
            errors.append(f"configure schemas do not represent `{method}`")
    methodology = read(SKILLS_DIR / "research" / "references" / "methodology.md")
    if "external-dataset draft" in methodology:
        errors.append("research methodology references deferred dataset draft")
    if "do not persist the handoff until the user" not in methodology:
        errors.append("research methodology persists handoff before approval")


def check_cost_contract(errors: list[str]) -> None:
    reference = SKILLS_DIR / "configure" / "references" / "cost-estimation.md"
    require_markers(
        reference,
        (
            "docs.fireworks.ai/fine-tuning/cost-estimator",
            "Managed",
            "Serverless",
            "Dedicated",
            "Embedding, IGPO, or distillation",
            "contact the Training team",
        ),
        errors,
    )
    cases = json.loads(
        read(
            SKILLS_DIR
            / "configure"
            / "tests"
            / "fixtures"
            / "cost-estimation-cases.json"
        )
    )
    reference_text = read(reference)
    for case in cases:
        if case["name"] not in reference_text:
            errors.append(f"cost route enum does not represent `{case['name']}`")
    if not (
        SKILLS_DIR / "configure" / "tests" / "test_cost_estimation_contract.py"
    ).exists():
        errors.append("configure cost-estimation contract test is missing")


def main() -> int:
    errors: list[str] = []
    check_skill_names(errors)
    check_links(errors)
    check_entry_skills(errors)
    check_slim_layout(errors)
    check_packaging(errors)
    check_carrier(errors)
    check_cookbook_routes(errors)
    check_semantic_contracts(errors)
    check_cost_contract(errors)

    if errors:
        print("Skill validation FAILED:\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("OK: validated slim research/configure/debug skill split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
