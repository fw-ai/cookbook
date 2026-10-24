"""Run the real upstream command builder through the detached adapter seam."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")
from harbor.agents.installed.base import BaseInstalledAgent
from training.examples.harbor_rl.harnesses.mini_swe import agent as mini


def test_upstream_mini_swe_detaches_only_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = mini.ConfigurableMiniSweAgent(
        logs_dir=tmp_path,
        sidecar_bundle_path=str(tmp_path / "bundle"),
        sidecar_launch_spec="{}",
        context_limit=4096,
        output_limit=128,
        tool_timeout_seconds=60,
    )
    agent._policy_base_url = "http://127.0.0.1:1/v1"
    agent._policy_api_key = "test-scoped-token"
    detached, direct = [], []

    async def execute(_self, environment, *, command, **kwargs):
        direct.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def detach(executor, environment, **kwargs):
        detached.append(kwargs)

    async def disposition(environment):
        return None

    monkeypatch.setattr(BaseInstalledAgent, "exec_as_agent", execute)
    monkeypatch.setattr(mini, "exec_harness", detach)
    monkeypatch.setattr(mini, "sidecar_failure_disposition", disposition)
    asyncio.run(agent._run_mini_swe("solve the task", object(), object()))
    assert len(detached) == 1
    assert detached[0]["command"].startswith("set -o pipefail;")
    assert "mini-swe-agent --yolo" in detached[0]["command"]
    assert detached[0]["env"]["OPENAI_API_KEY"] == "test-scoped-token"
    assert all("mini-swe-agent --yolo" not in command for command in direct)
    assert direct, "upstream config writes retain their short executor"
