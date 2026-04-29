"""Smoke test: gsm8k_async migrated to renderer-backed wiring.

Per AC-11 / AC-12 follow-through (Codex R1 review), the legacy
``ctx.sample_with_tokens(messages=...)`` + hand-packed ``RolloutSample``
path must be replaced with ``single_turn_renderer_rollout`` and
``DeploymentSampler.sample_with_prompt_tokens``.
"""

from __future__ import annotations

import ast
from pathlib import Path


_TRAIN = Path(__file__).resolve().parent / "train.py"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            for alias in node.names:
                out.add(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
    return out


def test_uses_single_turn_renderer_rollout():
    mods = _imports(_TRAIN)
    assert "training.utils.rl.renderer_rollout.single_turn_renderer_rollout" in mods


def test_does_not_use_legacy_sample_with_tokens_messages_kwarg():
    text = _TRAIN.read_text()
    assert "sample_with_tokens(messages=" not in text, (
        "gsm8k_async/train.py must not call sample_with_tokens(messages=...) — "
        "use sample_with_prompt_tokens via single_turn_renderer_rollout instead."
    )


def test_uses_deployment_sampler_sample_with_prompt_tokens():
    text = _TRAIN.read_text()
    assert "sample_with_prompt_tokens" in text, (
        "gsm8k_async/train.py must wire DeploymentSampler.sample_with_prompt_tokens"
    )
