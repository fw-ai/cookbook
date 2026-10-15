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
            part.startswith(".")
            or part in {"build", "dist", "site-packages", "venv", "__pycache__"}
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


def test_published_minimum_requires_native_wheel_contents(monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    from training.tests import verify_sdk_minimum

    minimum = verify_sdk_minimum.declared_sdk_minimum(PACKAGE_ROOT / "pyproject.toml")
    package = SimpleNamespace(version=minimum, files=["fireworks/training/sdk/tinker_compat.py"])
    monkeypatch.setattr(verify_sdk_minimum, "distribution", lambda _name: package)
    monkeypatch.setattr(
        "sys.argv",
        ["verify_sdk_minimum.py", "--pyproject", str(PACKAGE_ROOT / "pyproject.toml"), "--assert-installed"],
    )
    assert verify_sdk_minimum.main() == 1
    assert "publish SDK first" in capsys.readouterr().out

    package.files = ["fireworks/training/sdk/_core/__init__.py"]
    assert verify_sdk_minimum.main() == 0
    package.version = "0.0.0"
    assert verify_sdk_minimum.main() == 1
