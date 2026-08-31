"""Harbor Pi agent configured for its environment-local TITO sidecar."""

from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.pi import Pi
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

from .constants import PI_OPENAI_COMPAT, PINNED_PI_VERSION

_PROVIDER_ID = "fireworks-tito"
_MODEL_ID = "policy"
_CONFIG_HOME = "/tmp/fireworks-tito-pi"
_MODELS_PATH = f"{_CONFIG_HOME}/models.json"
_SETTINGS_PATH = f"{_CONFIG_HOME}/settings.json"
_EXTENSION_PATH = f"{_CONFIG_HOME}/extension.ts"
_AGENT_STATUS_PATH = f"{_CONFIG_HOME}/agent-status"


def _extension_source() -> str:
    return Path(__file__).with_name("extension.ts").read_text(encoding="utf-8")


class ConfigurablePi(Pi):
    """Pi inside a Harbor task container, with policy traffic sent to TITO."""

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

    async def install(self, environment: BaseEnvironment) -> None:
        present = await environment.exec(
            command=". ~/.nvm/nvm.sh 2>/dev/null || true; command -v pi >/dev/null 2>&1 && pi --version",
        )
        if present.return_code != 0:
            raise RuntimeError(
                "Pi is not installed in the Harbor task image; prepare the "
                "pinned Pi image before starting RL"
            )
        installed_version = self.parse_version(present.stdout or "")
        expected = self._version or PINNED_PI_VERSION
        if installed_version != expected:
            raise RuntimeError(
                f"baked Pi version mismatch: expected {expected}, "
                f"found {installed_version or 'unknown'}"
            )
        endpoint = await install_sidecar(
            environment,
            bundle_path=self._sidecar_bundle_path,
            launch_spec=self._sidecar_launch_spec,
        )
        self._policy_base_url = endpoint["openai_base_url"]
        self._policy_api_key = endpoint["api_key"]

    def _models(self) -> dict[str, Any]:
        return {
            "providers": {
                _PROVIDER_ID: {
                    "name": "Fireworks TITO",
                    "baseUrl": self._policy_base_url,
                    "apiKey": self._policy_api_key,
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": _MODEL_ID,
                            "name": _MODEL_ID,
                            "api": "openai-completions",
                            "contextWindow": self._context_limit,
                            "maxTokens": self._output_limit,
                            "reasoning": True,
                            "input": ["text"],
                            "compat": dict(PI_OPENAI_COMPAT),
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                        }
                    ],
                }
            }
        }

    def _settings(self) -> dict[str, Any]:
        return {
            "defaultProvider": _PROVIDER_ID,
            "defaultModel": _MODEL_ID,
            "quietStartup": True,
            "defaultProjectTrust": "always",
            "retry": {
                "enabled": False,
                "provider": {"maxRetries": 3, "maxRetryDelayMs": 30_000},
            },
            "compaction": {
                "enabled": True,
                "reserveTokens": self._output_limit,
            },
        }

    async def _write_config(self, environment: BaseEnvironment) -> None:
        """Upload private config without placing its bearer in Harbor logs."""

        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {_CONFIG_HOME}",
            env={"PI_CODING_AGENT_DIR": _CONFIG_HOME},
        )
        await upload_private_text(
            environment,
            content=json.dumps(self._models(), sort_keys=True),
            remote_path=_MODELS_PATH,
        )
        await upload_private_text(
            environment,
            content=json.dumps(self._settings(), sort_keys=True),
            remote_path=_SETTINGS_PATH,
        )
        await upload_private_text(
            environment,
            content=_extension_source(),
            remote_path=_EXTENSION_PATH,
        )
        await self.exec_as_agent(
            environment,
            command=(f"chmod 600 {_MODELS_PATH} {_SETTINGS_PATH} {_EXTENSION_PATH}"),
            env={"PI_CODING_AGENT_DIR": _CONFIG_HOME},
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context
        cli_flags = self.build_cli_flags()
        cli_flags_arg = f"{cli_flags} " if cli_flags else ""
        resume_flag = "--continue " if self._resume else ""
        try:
            await self._write_config(environment)
            await self.exec_as_agent(
                environment,
                command=(
                    f"rm -f {shlex.quote(_AGENT_STATUS_PATH)}; "
                    '( if [ -f "$HOME/.nvm/nvm.sh" ]; then '
                    '. "$HOME/.nvm/nvm.sh"; fi; '
                    "pi --print --mode json --session-dir /logs/agent/pi/sessions "
                    f"{resume_flag}--provider {_PROVIDER_ID} --model {_MODEL_ID} "
                    f"{cli_flags_arg}--extension {_EXTENSION_PATH} "
                    "--no-extensions --no-skills --no-context-files --approve "
                    f"{shlex.quote(instruction)} </dev/null; "
                    f"printf '%s\\n' \"$?\" > {shlex.quote(_AGENT_STATUS_PATH)}; "
                    ') 2>&1 | grep -v \'"type":"message_update"\' '
                    "| stdbuf -oL tee /logs/agent/pi.txt; "
                    f"test -s {shlex.quote(_AGENT_STATUS_PATH)} || exit 127; "
                    f"agent_status=$(cat {shlex.quote(_AGENT_STATUS_PATH)}); "
                    f"if test -s {shlex.quote(SIDECAR_CONTEXT_OVERFLOW_PATH)}; "
                    'then exit 43; fi; exit "$agent_status"'
                ),
                env={
                    "FIREWORKS_TITO_TOOL_TIMEOUT_SECONDS": str(
                        self._tool_timeout_seconds
                    ),
                    "PI_CODING_AGENT_DIR": _CONFIG_HOME,
                },
            )
        except asyncio.CancelledError as exc:
            try:
                await asyncio.shield(
                    abandon_sidecar_after_harness_cancellation(
                        environment,
                        process_pattern="[p]i --print --mode json",
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


__all__ = ["ConfigurablePi"]
