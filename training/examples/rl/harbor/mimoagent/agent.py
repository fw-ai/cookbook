"""Harbor mimoagent harness configured for its environment-local TITO sidecar.

mimoagent is Xiaomi MiMo's edited fork of mini-swe-agent 1.9.0 (MIT; see
constants.py for attribution). It is baked into the task image by
prepare_tasks.py; this class installs the TITO sidecar, uploads the task
instruction, and runs the one-shot driver inside the container.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from training.examples.rl.harbor.tito.sidecar import (
    SIDECAR_CONTEXT_OVERFLOW_PATH,
    abandon_sidecar_after_harness_cancellation,
    install_sidecar,
    sidecar_failure_disposition,
    terminalize_sidecar,
    upload_private_text,
)

from .constants import PINNED_MIMOAGENT_VERSION

_DRIVER_PATH = "/opt/mimoagent-driver.py"
_VENV_PYTHON = "/opt/mimoagent/bin/python"
_TASK_PATH = "/logs/agent/mimoagent-task.txt"
_STATUS_PATH = "/tmp/fireworks-tito-mimoagent/agent-status"


class ConfigurableMimoAgent(BaseInstalledAgent):
    """mimoagent inside a Harbor task container, policy traffic via TITO."""

    def __init__(
        self,
        *args: Any,
        sidecar_bundle_path: str,
        sidecar_launch_spec: str,
        context_limit: int,
        output_limit: int,
        tool_timeout_seconds: int,
        step_limit: int = 150,
        arm: str = "bash",
        **kwargs: Any,
    ) -> None:
        del context_limit  # the sidecar enforces the context window
        kwargs.pop("model_name", None)
        super().__init__(*args, model_name="openai/policy", **kwargs)
        if arm not in {"bash", "cc"}:
            raise ValueError(f"unsupported mimoagent arm: {arm!r}")
        self._arm = arm
        self._sidecar_bundle_path = sidecar_bundle_path
        self._sidecar_launch_spec = sidecar_launch_spec
        self._output_limit = int(output_limit)
        self._tool_timeout_seconds = int(tool_timeout_seconds)
        self._step_limit = int(step_limit)
        self._policy_base_url = ""
        self._policy_api_key = ""

    @staticmethod
    def name() -> str:
        return "mimoagent"

    def get_version_command(self) -> str | None:
        # Import creates the mimoagent config dir; /tmp is writable even when
        # the agent user's HOME is not.
        return (
            "MIMOAGENT_GLOBAL_CONFIG_DIR=/tmp/mimoagent-version-check"
            f" {_VENV_PYTHON} -c 'import mimoagent; print(mimoagent.__version__)'"
        )

    async def install(self, environment: BaseEnvironment) -> None:
        present = await environment.exec(
            command=(
                "MIMOAGENT_GLOBAL_CONFIG_DIR=/tmp/mimoagent-version-check"
                f" {_VENV_PYTHON} -c 'import mimoagent; print(mimoagent.__version__)'"
            ),
        )
        if present.return_code != 0:
            raise RuntimeError(
                "mimoagent is not installed in the Harbor task image; prepare "
                "the image with harbor.mimoagent.prepare_tasks before RL"
            )
        installed_version = (present.stdout or "").strip()
        expected = self._version or PINNED_MIMOAGENT_VERSION
        if installed_version != expected:
            raise RuntimeError(
                f"baked mimoagent version mismatch: expected {expected}, "
                f"found {installed_version or 'unknown'}"
            )
        endpoint = await install_sidecar(
            environment,
            bundle_path=self._sidecar_bundle_path,
            launch_spec=self._sidecar_launch_spec,
        )
        self._policy_base_url = endpoint["openai_base_url"]
        self._policy_api_key = endpoint["api_key"]

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context
        try:
            await upload_private_text(
                environment, content=instruction, remote_path=_TASK_PATH
            )
            await self.exec_as_agent(
                environment,
                command=(
                    f"mkdir -p {shlex.quote(str(Path(_STATUS_PATH).parent))}; "
                    f"rm -f {shlex.quote(_STATUS_PATH)}; "
                    f"( {_VENV_PYTHON} {_DRIVER_PATH} </dev/null; "
                    f"printf '%s\\n' \"$?\" > {shlex.quote(_STATUS_PATH)}; "
                    ") 2>&1 | stdbuf -oL tee /logs/agent/mimoagent.txt; "
                    f"test -s {shlex.quote(_STATUS_PATH)} || exit 127; "
                    f"agent_status=$(cat {shlex.quote(_STATUS_PATH)}); "
                    f"if test -s {shlex.quote(SIDECAR_CONTEXT_OVERFLOW_PATH)}; "
                    'then exit 43; fi; exit "$agent_status"'
                ),
                env={
                    "OPENAI_BASE_URL": self._policy_base_url,
                    "OPENAI_API_KEY": self._policy_api_key,
                    "MIMOAGENT_MODEL_NAME": "policy",
                    "MIMOAGENT_TASK_FILE": _TASK_PATH,
                    "MIMOAGENT_ARM": self._arm,
                    "MIMOAGENT_MCP_SERVERS": json.dumps(
                        [
                            {"name": server.name, "url": server.url}
                            for server in self.mcp_servers
                            if server.url
                        ]
                    ),
                    "MIMOAGENT_STEP_LIMIT": str(self._step_limit),
                    "MIMOAGENT_OUTPUT_LIMIT": str(self._output_limit),
                    "MIMOAGENT_TOOL_TIMEOUT_SECONDS": str(self._tool_timeout_seconds),
                    # mimoagent creates its config dir at import; the logs dir
                    # is writable for every agent user, HOME may not be.
                    "MIMOAGENT_GLOBAL_CONFIG_DIR": "/logs/agent/mimoagent-config",
                },
            )
        except asyncio.CancelledError as exc:
            try:
                await asyncio.shield(
                    abandon_sidecar_after_harness_cancellation(
                        environment,
                        process_pattern="[m]imoagent-driver",
                    )
                )
            except Exception as cleanup_error:  # noqa: BLE001
                exc.add_note(
                    f"TITO sidecar cancellation cleanup failed: {cleanup_error}"
                )
            raise
        except BaseException as exc:
            try:
                disposition = await sidecar_failure_disposition(environment)
                await asyncio.shield(
                    terminalize_sidecar(
                        environment,
                        status="failed",
                        reason=disposition or f"{type(exc).__name__}: {exc}",
                    )
                )
            except Exception as cleanup_error:  # noqa: BLE001
                exc.add_note(f"TITO sidecar failure cleanup failed: {cleanup_error}")
            raise
        else:
            await asyncio.shield(terminalize_sidecar(environment, status="completed"))


__all__ = ["ConfigurableMimoAgent"]
