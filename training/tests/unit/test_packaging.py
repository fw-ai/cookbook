"""Guards for the nonstandard ``cookbook/training`` package layout."""

from __future__ import annotations

import tomllib
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_setuptools_package_list_covers_every_python_module_directory() -> None:
    """Keep explicit sdist-safe discovery synchronized with cookbook source."""
    with (PACKAGE_ROOT / "pyproject.toml").open("rb") as file:
        configured = set(tomllib.load(file)["tool"]["setuptools"]["packages"])

    discovered = {"training"}
    for path in PACKAGE_ROOT.rglob("*.py"):
        relative_parent = path.parent.relative_to(PACKAGE_ROOT)
        if any(
            part in {".venv", "build", "dist", "__pycache__"}
            or part.endswith(".egg-info")
            for part in relative_parent.parts
        ):
            continue
        suffix = ".".join(relative_parent.parts)
        discovered.add("training" + (f".{suffix}" if suffix else ""))

    assert configured == discovered, (
        "update [tool.setuptools].packages for added or removed source directories: "
        f"missing={sorted(discovered - configured)}, extra={sorted(configured - discovered)}"
    )
