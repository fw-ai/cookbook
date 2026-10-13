"""NeMo Gym correlation adapter for the SDK's exact-token TiTo sidecar.

Only explicitly admitted rollouts can sample. The SDK owns token lineage,
OpenAI response parsing, routing replay, and cancellation of active requests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiohttp import web
from fireworks.training.sdk import (
    TITOSidecar,
    TITOChatRequest,
    TITOTrajectoryArtifact,
    TITOTrajectoryEndpoint,
    TrajectoryDriftPolicy,
)
from fireworks.training.sdk.sampling import DeploymentSampler

from training.renderer.tito.registry import build_sidecar_tito_renderer
from training.utils.rl.rollout import RolloutRun
from training.utils.rl.rollout.tito import materialize_tito_trajectory

_HOST = "127.0.0.1"
_MODEL_ID = "policy"


def _is_user_simulator_call(tools: list[dict[str, Any]]) -> bool:
    """True for a ``toolsandbox`` internal user-simulator call, not the agent-under-test.

    ``toolsandbox``'s resources server drives its own simulated-user role via a
    direct ``/v1/chat/completions`` call to the same rollout's policy model
    server (see ``resources_servers/toolsandbox/app.py``'s ``_ServerClientUser``)
    -- NeMo Gym's rollout-correlation propagates unconditionally by URL path, so
    this call reaches this proxy tagged with the *same* rollout_id as the real
    agent-under-test's turns, and would otherwise get woven into the same
    training-sample token chain (corrupting it: the simulated user's tokens
    would look like the policy's own action). ToolSandbox's own tool-visibility
    rules make this reliably distinguishable without any config: the agent's
    turns always carry the scenario's real (multi-entry) toolset, while the
    user-simulator either gets no tools at all, or exactly one --
    ``end_conversation`` -- which is never visible to the agent role. Other
    environments' agent turns may legitimately carry no tools at all (plain
    single-turn tasks), which this heuristic would misclassify -- so the proxy
    only applies it when constructed with ``exclude_user_simulator=True``
    (``--exclude-user-simulator`` in train.py), never by default.
    """
    if not tools:
        return True
    if len(tools) == 1:
        function = tools[0].get("function") or {}
        if function.get("name") == "end_conversation":
            return True
    return False


@dataclass
class RolloutSession:
    rollout_id: str
    endpoint: TITOTrajectoryEndpoint
    artifact: TITOTrajectoryArtifact | None = None

    @property
    def turn_count(self) -> int:
        return sum(len(segment.turns) for segment in self.artifact.segments) if self.artifact is not None else 0

    def to_run(self, *, reward: float) -> RolloutRun:
        if self.artifact is None:
            return RolloutRun(segments=[], run_id=self.rollout_id)
        return materialize_tito_trajectory(self.artifact, reward=reward)


class RecordingChatProxy:
    def __init__(
        self,
        *,
        tokenizer: Any,
        renderer_name: str,
        sampler: Any = None,
        max_sample_tokens: int = 1024,
        temperature: float = 1.0,
        exclude_user_simulator: bool = False,
    ) -> None:
        self._renderer = build_sidecar_tito_renderer(tokenizer, renderer_name)
        self._initial_sampler = sampler
        self._max_sample_tokens = int(max_sample_tokens)
        self._temperature = float(temperature)
        self._exclude_user_simulator = bool(exclude_user_simulator)
        self._sessions: dict[str, RolloutSession] = {}
        self._sidecar: TITOSidecar | None = None
        self._http: aiohttp.ClientSession | None = None
        self._runner: web.AppRunner | None = None
        self.port = 0
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.router.add_get("/v1/models", self._list_models)
        self.app.router.add_post("/v1/chat/completions", self._chat_completions)

    def _classify(self, request: TITOChatRequest) -> str:
        if self._exclude_user_simulator and _is_user_simulator_call(list(request.tools)):
            return "auxiliary"
        return "policy"

    async def set_sampler(self, sampler: Any, sample_kwargs: dict[str, Any] | None = None) -> None:
        if self._sidecar is not None:
            raise RuntimeError("sampler already bound")
        defaults = {"temperature": self._temperature, "top_p": 1.0, "top_k": 0, **(sample_kwargs or {})}
        max_context = defaults.get("max_seq_len") or getattr(self._renderer, "max_context_tokens", 65536)
        factory = TITOSidecar.from_deployment_sampler if isinstance(sampler, DeploymentSampler) else TITOSidecar
        self._sidecar = factory(
            sampler,
            renderer=self._renderer,
            max_context_tokens=int(max_context),
            max_output_tokens=self._max_sample_tokens,
            sampling_defaults=defaults,
            call_classifier=self._classify,
            default_drift_policy=TrajectoryDriftPolicy(max_masked_tokens=0, on_other_mismatch="new_segment"),
        )
        await self._sidecar.start()

    async def start(self, port: int = 0) -> int:
        self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        self._runner = web.AppRunner(self.app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, _HOST, port)
        await site.start()
        sockets = list(getattr(getattr(site, "_server", None), "sockets", []) or [])
        if not sockets:
            raise RuntimeError("recording proxy failed to bind a socket")
        self.port = int(sockets[0].getsockname()[1])
        if self._initial_sampler is not None:
            await self.set_sampler(self._initial_sampler)
        return self.port

    async def close(self) -> None:
        self._sessions.clear()
        try:
            if self._sidecar is not None:
                await self._sidecar.close()
        finally:
            try:
                if self._http is not None:
                    await self._http.close()
            finally:
                if self._runner is not None:
                    await self._runner.cleanup()
                    self._runner = None
                self.port = 0

    async def open_session(self, rollout_id: str) -> RolloutSession:
        if self._sidecar is None:
            raise RuntimeError("sampler not ready")
        if rollout_id in self._sessions:
            raise ValueError("rollout session already active")
        endpoint = await self._sidecar.create_trajectory_async(trajectory_id=rollout_id)
        session = RolloutSession(rollout_id, endpoint)
        self._sessions[rollout_id] = session
        return session

    async def finish_session(self, rollout_id: str, *, abandon: bool = False) -> RolloutSession | None:
        # Close admission first: delayed Gym calls cannot recreate this session.
        session = self._sessions.pop(rollout_id, None)
        if session is not None and self._sidecar is not None:
            if abandon:
                session.artifact = await self._sidecar.abandon_trajectory(rollout_id)
            else:
                session.artifact = await self._sidecar.finish_trajectory(rollout_id)
        return session

    def active_rollout_ids(self) -> list[str]:
        return list(self._sessions)

    async def _list_models(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": _MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "fireworks"}],
            }
        )

    async def _chat_completions(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return web.json_response({"error": "invalid JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "expected an object"}, status=400)
        rollout_id = body.pop("user", None)
        if not isinstance(rollout_id, str) or not rollout_id:
            return web.json_response({"error": "rollout user is required"}, status=400)
        session = self._sessions.get(rollout_id)
        if session is None:
            return web.json_response({"error": "rollout is unknown or closed"}, status=410)
        if body.get("stream"):
            return web.json_response({"error": "streaming is unsupported"}, status=400)
        # The recipe's on-policy defaults are authoritative. TiTo validates all
        # remaining fields and owns budgets, affinity, and exact-token replay.
        for name in ("temperature", "top_p", "top_k"):
            body.pop(name, None)
        parallel_tools = body.pop("parallel_tool_calls", None)
        if parallel_tools is not None and not isinstance(parallel_tools, bool):
            return web.json_response({"error": "parallel_tool_calls must be boolean"}, status=400)
        if self._http is None:
            return web.json_response({"error": "proxy not ready"}, status=503)
        async with self._http.post(
            session.endpoint.openai_base_url + "/chat/completions",
            json=body,
            headers={"Authorization": "Bearer " + session.endpoint.api_key},
        ) as response:
            payload = await response.json()
            if response.status == 200 and parallel_tools is False:
                calls = payload["choices"][0]["message"].get("tool_calls") or []
                if len(calls) > 1:
                    # TiTo renders the default parallel policy. Enforce the
                    # harness's single-call constraint at the response boundary;
                    # Gym failure then abandons this unaccepted trajectory.
                    return web.json_response({"error": "model returned parallel tool calls"}, status=502)
            return web.json_response(payload, status=response.status)
