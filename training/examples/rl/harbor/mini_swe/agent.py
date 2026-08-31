"""Harbor Mini-SWE agent configured for its environment-local TITO sidecar."""

from __future__ import annotations

import asyncio
from typing import Any

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.agents.model_connection import ResolvedModelConnection
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from training.examples.rl.harbor.tito.sidecar import (
    abandon_sidecar_after_harness_cancellation,
    install_sidecar,
    sidecar_failure_disposition,
    terminalize_sidecar,
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
        try:
            await super().run(instruction, environment, context)
        except asyncio.CancelledError as exc:
            try:
                await asyncio.shield(
                    abandon_sidecar_after_harness_cancellation(
                        environment,
                        process_pattern="[m]ini-swe-agent --yolo",
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
            disposition = await sidecar_failure_disposition(environment)
            if disposition is not None:
                await asyncio.shield(
                    terminalize_sidecar(
                        environment,
                        status="failed",
                        reason=disposition,
                    )
                )
                raise RuntimeError(disposition)
            await asyncio.shield(terminalize_sidecar(environment, status="completed"))


__all__ = ["ConfigurableMiniSweAgent"]
