"""Verify that the built cookbook artifact imports with published dependencies.

Run this script from outside the checkout after ``pip install .``. It discovers
the base modules shipped from ``training.recipes`` and ``training.utils`` in the
checkout, then imports them from the installed wheel. No credentials, network,
or GPU are required.
"""

from __future__ import annotations

import importlib
from importlib.metadata import version
from pathlib import Path

import training


CHECKOUT_PACKAGE = Path(__file__).resolve().parents[1]
INSTALLED_PACKAGE = Path(training.__file__).resolve().parent

# This adapter intentionally requires the optional ``eval`` dependency group.
OPTIONAL_MODULES = {"training.utils.rl.rollout.eval_protocol"}


def _checkout_modules(package_dir: Path) -> list[str]:
    modules = []
    for path in package_dir.rglob("*.py"):
        relative = path.relative_to(CHECKOUT_PACKAGE).with_suffix("")
        parts = relative.parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join(("training", *parts)))
    return sorted(set(modules) - OPTIONAL_MODULES)


def main() -> None:
    if INSTALLED_PACKAGE == CHECKOUT_PACKAGE:
        raise RuntimeError(
            "Imported training from the checkout instead of the installed artifact"
        )

    modules = []
    for package_name in ("recipes", "utils"):
        modules.extend(_checkout_modules(CHECKOUT_PACKAGE / package_name))

    for module_name in modules:
        module = importlib.import_module(module_name)
        module_path = getattr(module, "__file__", None)
        if module_path and Path(module_path).resolve().is_relative_to(
            CHECKOUT_PACKAGE
        ):
            raise RuntimeError(f"Imported {module_name} from the checkout: {module_path}")

    print(f"cookbook: {version('fireworks-training-cookbook')} ({INSTALLED_PACKAGE})")
    print(f"fireworks-ai: {version('fireworks-ai')}")
    print(f"imported {len(modules)} installed cookbook modules")


if __name__ == "__main__":
    main()
