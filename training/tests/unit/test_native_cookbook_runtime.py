"""Recipes run without external training SDK or cookbook packages."""

import ast
import subprocess
import sys
import tomllib
from pathlib import Path


def test_cookbook_has_no_external_training_sdk_dependency() -> None:
    root = Path(__file__).resolve().parents[2]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    requirements = metadata["project"]["dependencies"]
    for extra in metadata["project"]["optional-dependencies"].values():
        requirements += extra
    assert not any(value.lower().startswith(("tinker", "tml-renderers")) for value in requirements)
    for path in root.rglob("*.py"):
        if any(part in {"build", "dist", ".venv"} for part in path.relative_to(root).parts):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(name.split(".")[0] in {"tinker", "tinker_cookbook"} for name in names), path


def test_all_recipes_import_with_external_packages_blocked() -> None:
    script = """
import asyncio, importlib, importlib.abc, sys
class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tinker', 'tinker_cookbook', 'tml_renderers'}:
            raise AssertionError('Cookbook imported external dependency: ' + fullname)
sys.meta_path.insert(0, BlockExternal())
from training.recipes import sft_loop, dpo_loop, orpo_loop, rl_loop, async_rl_loop
from training.recipes import distillation_loop, embedding_loop, decision_loop
from training.recipes.experiment import ppo_value_head_loop, async_rl_loop_serverless
from fireworks.training.sdk import types
from fireworks.training.sdk._core._client import AsyncTrainingHTTPClient
async def construct_http_clients():
    for use_pyqwest in (False, True):
        client = AsyncTrainingHTTPClient(
            api_key='fw-test', base_url='https://trainer.test',
            _client_config=types.ClientConfigResponse(use_pyqwest_transport=use_pyqwest),
        )
        await client.close()
asyncio.run(construct_http_clients())
assert types.Datum.__module__.startswith('fireworks.training.sdk.')
assert not any(n.split('.')[0] in {'tinker', 'tinker_cookbook'} for n in sys.modules)
assert 'fireworks.training.sdk.patches' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
