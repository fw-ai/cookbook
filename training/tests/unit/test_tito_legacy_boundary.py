"""Enforce the frozen legacy utility island during the TITO migration."""

from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_PREFIXES = (
    "training.utils.rl.agent",
    "training.utils.rl.rollout.message",
)
_LEGACY_REEXPORTS = {
    "training.utils.rl": {"MessageTrajectoryAssembler", "TITOTokenizer"},
    "training.utils.rl.rollout": {"MessageTrajectoryAssembler", "TITOTokenizer"},
}
_ALLOWLIST = {
    "training/utils/rl/__init__.py",
    "training/utils/rl/rollout/__init__.py",
    "training/utils/rl/rollout/message.py",
    "training/utils/rl/agent/__init__.py",
    "training/utils/rl/agent/openai.py",
    "training/utils/rl/agent/sampling.py",
    "training/utils/rl/agent/session.py",
    "training/utils/rl/agent/trajectory.py",
    "training/utils/rl/agent/turn_matching.py",
    "training/tests/unit/test_rollout_helpers.py",
    "training/tests/unit/test_rollout_message.py",
    "training/tests/unit/test_training_session.py",
    "training/tests/unit/test_turn_matching.py",
    # Existing compatibility coverage for the frozen CookbookTurnRenderer helper.
    "training/tests/unit/test_muse_glimmer_renderer.py",
}


def _legacy_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    output: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(_LEGACY_PREFIXES):
                output.append(f"line {node.lineno}: from {module}")
            else:
                legacy_names = _LEGACY_REEXPORTS.get(module, set())
                imported = sorted(
                    alias.name for alias in node.names if alias.name in legacy_names
                )
                if imported:
                    output.append(
                        f"line {node.lineno}: from {module} import "
                        + ", ".join(imported)
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(_LEGACY_PREFIXES):
                    output.append(f"line {node.lineno}: import {alias.name}")
    return output


def test_only_frozen_legacy_modules_reexports_and_dedicated_tests_import_island() -> (
    None
):
    violations: list[str] = []
    for path in sorted((_ROOT / "training").rglob("*.py")):
        relative = path.relative_to(_ROOT).as_posix()
        if any(part.startswith(".") for part in path.relative_to(_ROOT).parts):
            continue
        imports = _legacy_imports(path)
        if imports and relative not in _ALLOWLIST:
            violations.extend(f"{relative}: {item}" for item in imports)
    assert violations == []


def test_deleted_legacy_examples_and_server_do_not_return() -> None:
    legacy_example = _ROOT / "training/examples/rl/multi_turn_message_in"
    forbidden_sources = [
        *legacy_example.glob("*.py"),
        *legacy_example.glob("*.md"),
        *legacy_example.glob("*.sh"),
        _ROOT / "training/examples/rl/harbor/openai_policy.py",
    ]
    assert [str(path) for path in forbidden_sources if path.exists()] == []


def test_documentation_does_not_link_deleted_legacy_example() -> None:
    references = []
    for path in sorted(_ROOT.rglob("*.md")):
        if any(part.startswith(".") for part in path.relative_to(_ROOT).parts):
            continue
        if "multi_turn_message_in" in path.read_text(encoding="utf-8"):
            references.append(path.relative_to(_ROOT).as_posix())
    assert references == []
