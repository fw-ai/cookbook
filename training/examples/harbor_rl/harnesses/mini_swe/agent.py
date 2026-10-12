"""Harbor Mini-SWE agent configured for its environment-local TITO sidecar."""

from __future__ import annotations

from typing import Any

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.agents.model_connection import ResolvedModelConnection
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from training.examples.harbor_rl.tito.sidecar import (
    install_sidecar,
    run_harness_with_sidecar,
    sidecar_failure_disposition,
)


class ConfigurableMiniSweAgent(MiniSweAgent):
    """Run Mini-SWE inside Harbor while keeping policy state in the sandbox."""

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
        del context_limit
        kwargs.pop("model_name", None)
        super().__init__(
            *args,
            model_name="openai/policy",
            max_tokens=int(output_limit),
            config={
                "environment": {
                    "timeout": int(tool_timeout_seconds),
                }
            },
            **kwargs,
        )
        self._sidecar_bundle_path = sidecar_bundle_path
        self._sidecar_launch_spec = sidecar_launch_spec
        self._policy_base_url = ""
        self._policy_api_key = ""

    @property
    def model_connection(self) -> ResolvedModelConnection:
        if not self._policy_base_url or not self._policy_api_key:
            return super().model_connection
        return ResolvedModelConnection(
            provider="openai",
            api_key=self._policy_api_key,
            base_url=self._policy_base_url,
            configured_base_url=self._policy_base_url,
            env={
                "MSWEA_API_KEY": self._policy_api_key,
                "OPENAI_API_KEY": self._policy_api_key,
                "OPENAI_BASE_URL": self._policy_base_url,
                "OPENAI_API_BASE": self._policy_base_url,
            },
        )

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
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
        await run_harness_with_sidecar(
            environment,
            self._run_mini_swe(instruction, environment, context),
            process_pattern="[m]ini-swe-agent --yolo",
        )

    async def _run_mini_swe(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await super().run(instruction, environment, context)
        # Harbor's Mini-SWE command has no context-overflow exit status, so a
        # typed sidecar disposition turns a clean exit into a failed agent.
        disposition = await sidecar_failure_disposition(environment)
        if disposition is not None:
            raise RuntimeError(disposition)


__all__ = ["ConfigurableMiniSweAgent"]
