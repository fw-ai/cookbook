"""``import training.renderer`` must register every renderer in the package.

Callers reach these renderers through ``training._vendor.tinker_cookbook_0_4_3.renderers.get_renderer``,
whose registry is populated by ``register_renderer`` running at module import.
Nothing else triggers it: putting the cookbook on ``sys.path``, or pip-installing
it, registers nothing. So a renderer module the package ``__init__`` forgets to
import is invisible to every caller, and the only symptom is a ``RendererError``
from whatever tries to use it -- for the serverless release smoke, that surfaces
as workers dying before they create a session, with the real cause buried in
per-session stderr.

``mistral`` was in exactly that state, which is what these tests exist to stop
recurring.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

RENDERER_DIR = pathlib.Path(__file__).resolve().parents[2] / "renderer"


def _modules_that_register() -> set[str]:
    """Renderer modules containing a top-level ``register_renderer(...)`` call.

    Parsed rather than imported so the check reports a missing module instead of
    failing on the import it is trying to prove happens.
    """

    registering = set()
    for path in RENDERER_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "register_renderer":
                registering.add(path.stem)
                break
    return registering


def _modules_imported_by_init() -> set[str]:
    tree = ast.parse((RENDERER_DIR / "__init__.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("training.renderer"):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("training.renderer."):
                    imported.add(alias.name.rsplit(".", 1)[-1])
    return imported


def test_init_imports_every_registering_module() -> None:
    missing = sorted(_modules_that_register() - _modules_imported_by_init())
    assert not missing, (
        "these renderer modules call register_renderer but are not imported by "
        f"training/renderer/__init__.py, so importing the package does not register them: {missing}"
    )


def test_registering_modules_are_actually_found() -> None:
    """Guard the guard: an AST walk that silently matched nothing would make the
    check above pass for every possible package state."""

    assert len(_modules_that_register()) > 5


def test_importing_the_package_registers_a_local_renderer() -> None:
    """The end-to-end property callers depend on, for one renderer upstream does
    not ship. Skipped where the real tinker-cookbook is absent."""

    renderers = pytest.importorskip(
        "training.renderer",
        reason="needs the real tinker-cookbook",
    )
    import training.renderer  # noqa: F401  (registers every local renderer)

    assert renderers.is_renderer_registered("deepseek_v4")
