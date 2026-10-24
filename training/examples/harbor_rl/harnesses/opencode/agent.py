"""Harbor OpenCode agent configured for its environment-local TITO sidecar.

OpenCode and the sidecar run inside the same Harbor environment. The custom
provider points the CLI at the trajectory-scoped loopback endpoint and
advertises the trainer's real context limit so OpenCode can manage long task
histories against the same token budget.
"""

from __future__ import annotations

import json
import shlex
from typing import Any

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from training.examples.harbor_rl.tito.harness_exec import exec_harness
from training.examples.harbor_rl.tito.sidecar import (
    harness_command,
    install_sidecar,
    run_harness_with_sidecar,
    upload_private_text,
)
from training.examples.harbor_rl.harnesses.opencode.config import _TOOL_TIMEOUT_PLUGIN

# Keep the per-trajectory credential outside Harbor's collected /logs tree.
_OPENCODE_CONFIG_HOME = "/tmp/fireworks-tito-opencode/xdg-config"
_OPENCODE_CONFIG_PATH = f"{_OPENCODE_CONFIG_HOME}/opencode/opencode.json"
_OPENCODE_PLUGIN_PATH = (
    f"{_OPENCODE_CONFIG_HOME}/opencode/plugins/fireworks-tito-timeout.js"
)
_OPENCODE_PLUGIN_LOCK_PATH = f"{_OPENCODE_CONFIG_HOME}/opencode/package-lock.json"
_PROVIDER_ID = "fireworks-rl"
_MODEL_ID = "policy"
_AGENT_STATUS_PATH = "/tmp/fireworks-tito-opencode/agent-status"
_OFFLINE_PLUGIN_LOCK = {
    "lockfileVersion": 3,
    "packages": {
        "": {
            # OpenCode 1.18.8 adds this development dependency to every
            # writable config directory.  Our local timeout plugin has no npm
            # imports, so the lock entry is sufficient to suppress that
            # unrelated runtime install in an egress-restricted sandbox.
            "dependencies": {"@opencode-ai/plugin": "1.18.8"},
        }
    },
}


class ConfigurableOpenCode(OpenCode):
    """OpenCode against one trajectory-scoped loopback sidecar endpoint."""

    def __init__(
        self,
        *args: Any,
        sidecar_bundle_path: str,
        sidecar_launch_spec: str,
        context_limit: int,
        output_limit: int,
        tool_timeout_seconds: int,
        **kwargs: Any,
    ) -> None:
        # Harbor passes its display model through AgentConfig.  OpenCode needs
        # the custom provider/model pair registered below instead.
        kwargs.pop("model_name", None)
        super().__init__(
            *args,
            model_name=f"{_PROVIDER_ID}/{_MODEL_ID}",
            **kwargs,
        )
        self._sidecar_bundle_path = sidecar_bundle_path
        self._sidecar_launch_spec = sidecar_launch_spec
        self._policy_base_url = ""
        self._policy_api_key = ""
        self._context_limit = int(context_limit)
        self._output_limit = int(output_limit)
        self._tool_timeout_seconds = int(tool_timeout_seconds)

    def _policy_config(self) -> dict[str, Any]:
        return {
            "$schema": "https://opencode.ai/config.json",
            # A rollout does not consume the UI-only session title. Disabling
            # the native title agent removes an unrelated auxiliary inference
            # and its startup transport from every Harbor trial.
            "agent": {"title": {"disable": True}},
            # V1 owns one linear trajectory for the primary OpenCode loop.
            # OpenCode's ``task`` tool starts independent subagent sessions;
            # those must not be multiplexed through the parent's endpoint.
            "tools": {"task": False},
            "provider": {
                _PROVIDER_ID: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Fireworks RL policy",
                    "options": {
                        "baseURL": self._policy_base_url,
                        "apiKey": self._policy_api_key,
                    },
                    "models": {
                        _MODEL_ID: {
                            "name": _MODEL_ID,
                            # OpenCode otherwise omits an empty reasoning part
                            # when it replays an assistant tool call. GLM's
                            # response includes ``reasoning_content: ""``;
                            # declaring the provider's native interleaved field
                            # makes OpenCode preserve that exact round trip.
                            "interleaved": {"field": "reasoning_content"},
                            "limit": {
                                "context": self._context_limit,
                                "output": self._output_limit,
                            },
                        }
                    },
                }
            },
            "compaction": {
                "auto": True,
                "prune": False,
                "reserved": 10000,
            },
            # Harbor collects the final workspace diff itself.  OpenCode's
            # private undo snapshots duplicate that work and, for large repos,
            # can block the first policy call while a second git index is
            # created, before the sidecar sees any policy traffic.
            "snapshot": False,
        }

    async def install(self, environment: BaseEnvironment) -> None:
        present = await environment.exec(
            command="command -v opencode >/dev/null 2>&1 && opencode --version",
        )
        if present.return_code != 0:
            raise RuntimeError(
                "OpenCode is not installed in the Harbor task image; run "
                "prepare_opencode_tasks before starting RL"
            )
        installed_version = self.parse_version(present.stdout or "")
        if self._version and installed_version != self._version:
            raise RuntimeError(
                "baked OpenCode version mismatch: expected "
                f"{self._version}, found {installed_version or 'unknown'}; "
                "rebuild the Harbor task images with "
                f"--opencode-version {self._version}"
            )
        endpoint = await install_sidecar(
            environment,
            bundle_path=self._sidecar_bundle_path,
            launch_spec=self._sidecar_launch_spec,
        )
        self._policy_base_url = endpoint["openai_base_url"]
        self._policy_api_key = endpoint["api_key"]

    async def _write_config(
        self,
        environment: BaseEnvironment,
        *,
        env: dict[str, str],
    ) -> None:
        """Upload private config without placing its bearer in Harbor logs."""

        await self.exec_as_agent(
            environment,
            command=(
                f'mkdir -p "{_OPENCODE_CONFIG_HOME}/opencode/plugins" '
                f'"{_OPENCODE_CONFIG_HOME}/opencode/node_modules"'
            ),
            env=env,
        )
        await upload_private_text(
            environment,
            content=json.dumps(self._policy_config(), indent=2),
            remote_path=_OPENCODE_CONFIG_PATH,
        )
        await upload_private_text(
            environment,
            content=_TOOL_TIMEOUT_PLUGIN,
            remote_path=_OPENCODE_PLUGIN_PATH,
        )
        await upload_private_text(
            environment,
            content=json.dumps(_OFFLINE_PLUGIN_LOCK),
            remote_path=_OPENCODE_PLUGIN_LOCK_PATH,
        )
        await self.exec_as_agent(
            environment,
            command=(
                "chmod 600 "
                f'"{_OPENCODE_CONFIG_PATH}" "{_OPENCODE_PLUGIN_PATH}" '
                f'"{_OPENCODE_PLUGIN_LOCK_PATH}"'
            ),
            env=env,
        )

    def _agent_env(self) -> dict[str, str]:
        env = {
            "OPENCODE_FAKE_VCS": "git",
            # The rollout image is intentionally egress-restricted and the
            # provider/model contract is already pinned in opencode.json.
            # Avoid unrelated catalog/update requests during every container
            # bootstrap, especially when a full Harbor cohort starts at once.
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS": str(
                self._tool_timeout_seconds * 1000
            ),
            "FIREWORKS_TITO_TOOL_TIMEOUT_MS": str(self._tool_timeout_seconds * 1000),
            "XDG_CONFIG_HOME": _OPENCODE_CONFIG_HOME,
            "XDG_DATA_HOME": "/logs/agent/opencode/xdg-data",
            "XDG_STATE_HOME": "/logs/agent/opencode/xdg-state",
        }
        env.update(self.extra_env)
        return env

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context
        self._instruction = instruction
        await run_harness_with_sidecar(
            environment,
            self._run_opencode(instruction, environment),
            process_pattern="[o]pencode --model=",
        )

    async def _run_opencode(
        self, instruction: str, environment: BaseEnvironment
    ) -> None:
        env = self._agent_env()
        cli_flags = self.build_cli_flags()
        cli_flags_arg = f"{cli_flags} " if cli_flags else ""
        resume_flag = "--continue " if self._resume else ""
        await self._write_config(environment, env=env)
        await exec_harness(
            self.exec_as_agent,
            environment,
            command=harness_command(
                'if [ -f "$HOME/.nvm/nvm.sh" ]; then . "$HOME/.nvm/nvm.sh"; fi; '
                f"opencode --model={shlex.quote(str(self.model_name))} "
                "run --format=json "
                f"{resume_flag}{cli_flags_arg}--thinking "
                "--dangerously-skip-permissions -- "
                f"{shlex.quote(instruction)}",
                status_path=_AGENT_STATUS_PATH,
                log_path="/logs/agent/opencode.txt",
            ),
            log_path="/logs/agent/opencode.txt",
            env=env,
        )


__all__ = ["ConfigurableOpenCode"]
