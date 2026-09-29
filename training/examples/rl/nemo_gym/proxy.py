"""Recording proxy: an OpenAI-compatible /v1/chat/completions server backed by a
Fireworks Dedicated-deployment sampler (``DeploymentSampler``, via
``async_rl_loop``'s ``RolloutSetup``).

NeMo Gym's ``inference_provider`` model server is pointed at this proxy instead
of Fireworks' public API. Each request is forwarded to the setup's sampler
(the recipe hotloads new weights into the same deployment after every
optimizer batch -- this proxy makes no snapshot/version calls itself) and the
exact prompt/completion token ids + logprobs are recorded per turn, keyed by
NeMo Gym's own per-rollout correlation id (propagated via the standard OpenAI
``user`` field -- see the patch in responses_api_models/inference_provider/app.py).

Per-session trajectory bookkeeping (turn-append/rollback detection, token-exact
stitching) does NOT use ``training.utils.rl.rollout.MessageTrajectoryAssembler``'s
exact-dict-equality checkpointing. That distinction matters here:
``simple_agent``-based harnesses (``example_multi_step_simple_agent``,
``workplace_assistant_simple_agent``) drive NeMo Gym's *Responses API*, not raw
chat messages -- they accumulate Responses-format output items between turns,
and the model server reconstructs a fresh chat-messages list from that history
on every single turn. The reconstruction is not guaranteed byte-identical to
whatever assistant message dict we returned the turn before (tool-call
id/ordering can differ, and -- confirmed live -- ``reasoning_content`` is
silently dropped by NeMo Gym's converter unless ``uses_reasoning_parser=true``
is set on the model server), which made ``MessageTrajectoryAssembler``
hard-raise ``MessageValidationError`` on turn 2 in early testing.

``_HistoryChain`` was first built to degrade gracefully off a content hash
instead of hard-crashing on a mismatch -- but a live run against
``workplace_assistant`` (longer, more tool-heavy conversations than
``example_multi_step``) showed the hash mismatches often enough that *zero*
turns were ever recognized as continuations, silently truncating every
training sample to its last turn. The fix: ``SimpleAgent``'s episode loop
(``responses_api_agents/simple_agent/app.py``) is strictly sequential per
rollout -- it never forks or rewrites history for a given ``rollout_id`` --
so a second request against an in-progress session is *always* a genuine
continuation of that session's current leaf turn, regardless of whether the
reconstructed message dict matches byte-for-byte. ``_HistoryChain.resolve()``
below therefore trusts rollout-session identity (an existing leaf -> APPEND)
as the primary signal, and keeps the content-hash comparison only as a
logged consistency check -- not a gate -- to surface real anomalies without
silently discarding real training data on a cosmetic mismatch.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from training.utils.rl.agent.openai import TurnRenderer, build_turn_renderer, flatten_content
from training.utils.rl.agent.trajectory import (
    SelectedLeaf,
    TokenSegment,
    TrainingSessionTree,
    TurnRecord,
)
from training.utils.rl.agent.turn_matching import (
    MessageHashFingerprinter,
    TurnDecision,
    TurnKind,
    TurnRequest,
    classify,
)
from training.utils.rl.rollout import RolloutSample

logger = logging.getLogger(__name__)
_HOST = "127.0.0.1"
_MODEL_ID = "policy"


def _completion_tokens_and_logprobs(completion: Any) -> tuple[list[int], list[float]] | None:
    """Extract the output-only token ids + logprobs from a ``SampledCompletion``.

    Mirrors ``multi_turn_message_in/rollout.py``'s ``_completion_logprobs`` --
    ``full_tokens``/``prompt_len`` is the ``DeploymentSampler`` contract, and
    logprobs may or may not be echoed over the full sequence depending on
    server config.
    """
    prompt_len = int(completion.prompt_len)
    output_tokens = list(completion.full_tokens[prompt_len:])
    values = getattr(completion, "sampling_logprobs", None)
    if values is None:
        return None
    values = list(values)
    if getattr(completion, "logprobs_echoed", False):
        full_len = len(completion.full_tokens)
        if len(values) == full_len:
            values = values[prompt_len:]
        elif len(values) == max(0, full_len - 1):
            values = values[max(0, prompt_len - 1):]
    if len(values) != len(output_tokens) or any(v is None for v in values):
        return None
    return output_tokens, [float(v) for v in values]


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
    environments' agent turns always carry a non-trivial tool list (or none at
    all, for a plain single-turn task), so this filter is a no-op for them.
    """
    if not tools:
        return True
    if len(tools) == 1:
        function = tools[0].get("function") or {}
        if function.get("name") == "end_conversation":
            return True
    return False


def _fix_tool_calls_for_openai(message: dict[str, Any]) -> None:
    """Coerce ``Renderer.to_openai_message()``'s tool_calls into real OpenAI shape.

    The tinker_cookbook renderer emits ``id=None`` (no call ever gets an id)
    and ``function.arguments`` as a parsed dict, not a JSON string -- NeMo
    Gym's own ``NeMoGymChatCompletion`` validates strictly against the real
    OpenAI spec (non-null string id, JSON-string arguments) and 500s on every
    single tool call otherwise. Confirmed live: this silently killed every
    tool-calling turn.
    """
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return
    for call in tool_calls:
        if not call.get("id"):
            call["id"] = f"call_{uuid.uuid4().hex[:24]}"
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            function["arguments"] = json.dumps(arguments or {})


@dataclass
class _HistoryChain:
    """Which recorded chain (if any) an incoming request continues.

    Mirrors Harbor's ``_HistoryChain`` (training/examples/rl/harbor/openai_policy.py)
    but trimmed to what a single-agent, non-branching rollout needs.
    """

    stored_units: list[Any] = field(default_factory=list)
    leaf_id: int | None = None
    response_units: dict[int, list[Any]] = field(default_factory=dict)

    def resolve(self, incoming_units: list[Any]) -> tuple[TurnDecision, int | None]:
        matches = [
            (len(units), node_id)
            for node_id, units in self.response_units.items()
            if len(units) <= len(incoming_units) and units == incoming_units[: len(units)]
        ]
        matched_len, hash_parent_id = max(matches) if matches else (0, None)

        if self.leaf_id is not None:
            # Rollout-session identity is the primary signal, not the content
            # hash: this chain is one per rollout_id, and SimpleAgent's episode
            # loop is strictly sequential and never rewrites history within a
            # rollout -- so any request against an in-progress session (a chain
            # that already has a leaf) is necessarily a continuation of that
            # leaf, whether or not the reconstructed message dict happens to
            # hash-match byte-for-byte (see module docstring for why it often
            # doesn't). The hash comparison is kept only to log real anomalies.
            if hash_parent_id != self.leaf_id:
                logger.warning(
                    "history-chain hash mismatch on rollout continuation "
                    "(trusting rollout-session identity instead): leaf_id=%s hash-matched parent_id=%s",
                    self.leaf_id, hash_parent_id,
                )
            return TurnDecision(TurnKind.APPEND, matched_len), self.leaf_id

        fallback = classify(self.stored_units, incoming_units)
        kind = TurnKind.NEW if not self.response_units else TurnKind.WIPE
        return TurnDecision(kind, fallback.matched_prefix_len), None


@dataclass
class RolloutSession:
    rollout_id: str
    tree: TrainingSessionTree = field(default_factory=TrainingSessionTree)
    chain: _HistoryChain = field(default_factory=_HistoryChain)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turn_count: int = 0

    def to_sample(self, *, reward: float) -> RolloutSample | None:
        if self.chain.leaf_id is None:
            return None
        segments: list[TokenSegment] = self.tree.materialize(
            [SelectedLeaf(self.chain.leaf_id)], max_context_tokens=0
        )
        if not segments:
            return None
        segment = segments[0]
        tokens = list(segment.prompt_ids) + list(segment.response_ids)
        if len(tokens) < 2:
            return None
        prompt_len = len(segment.prompt_ids)
        loss_mask = [0] * prompt_len + list(segment.loss_mask)
        logprobs = [0.0] * prompt_len + list(segment.rollout_log_probs)
        if not any(loss_mask):
            return None
        return RolloutSample(tokens=tokens, logprobs=logprobs, loss_mask=loss_mask, reward=reward)


class TinkerRecordingProxy:
    """A local OpenAI endpoint backed by a live Fireworks deployment sampler.

    The sampler (``DeploymentSampler``, built once by the rollout factory via
    ``training.examples.rl.vanilla_sampler.build_deployment_sampler``) is fixed
    for the whole run -- ``async_rl_loop`` hotloads new weights into the same
    deployment after every optimizer batch, so this proxy needs no
    snapshot-swap call of its own.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        renderer_name: str,
        sampler: Any = None,
        max_sample_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> None:
        self._sampler = sampler
        self._tokenizer = tokenizer
        self._renderer: TurnRenderer = build_turn_renderer(tokenizer, renderer_name)
        self._fingerprinter = MessageHashFingerprinter()
        self._max_sample_tokens = int(max_sample_tokens)
        self._temperature = float(temperature)
        self._sessions: dict[str, RolloutSession] = {}

        self.app = web.Application()
        self.app.router.add_get("/v1/models", self._list_models)
        self.app.router.add_post("/v1/chat/completions", self._chat_completions)
        self._runner: web.AppRunner | None = None
        self.port = 0

    # --- lifecycle -----------------------------------------------------------

    def set_sampler(self, sampler: Any) -> None:
        """Bind the deployment-backed sampler once ``RolloutSetup`` is available.

        Called exactly once by the rollout factory (``build_deployment_sampler``
        needs ``setup``, which only exists after the recipe's Dedicated
        trainer/deployment are up). No per-step calls after that -- the recipe
        hotloads new weights into this same deployment underneath.
        """
        self._sampler = sampler

    async def start(self, port: int = 0) -> int:
        self._runner = web.AppRunner(self.app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, _HOST, port)
        await site.start()
        server = getattr(site, "_server", None)
        sockets = list(getattr(server, "sockets", []) or [])
        if not sockets:
            raise RuntimeError("recording proxy failed to bind a socket")
        self.port = int(sockets[0].getsockname()[1])
        logger.info("recording proxy listening on %s:%d", _HOST, self.port)
        return self.port

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self.port = 0

    # --- session management ---------------------------------------------------

    def get_or_create_session(self, rollout_id: str) -> RolloutSession:
        session = self._sessions.get(rollout_id)
        if session is None:
            session = RolloutSession(rollout_id=rollout_id)
            self._sessions[rollout_id] = session
        return session

    def pop_session(self, rollout_id: str) -> RolloutSession | None:
        return self._sessions.pop(rollout_id, None)

    def active_rollout_ids(self) -> list[str]:
        return list(self._sessions.keys())

    # --- HTTP handlers ----------------------------------------------------------

    async def _list_models(self, request: web.Request) -> web.Response:
        del request
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": _MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "fireworks"}],
            }
        )

    async def _chat_completions(self, request: web.Request) -> web.Response:
        body = await request.json()
        rollout_id = str(body.get("user") or "default")
        session = self.get_or_create_session(rollout_id)
        t0 = time.monotonic()
        logger.info("proxy: received request rollout_id=%s turn=%d", rollout_id, session.turn_count)

        if self._sampler is None:
            raise web.HTTPServiceUnavailable(text="sampler not ready -- set_sampler() not called yet")

        async with session.lock:
            messages = list(body.get("messages") or [])
            tools = list(body.get("tools") or [])

            system_prompt = ""
            if messages and messages[0].get("role") == "system":
                system_prompt = flatten_content(messages[0].get("content"))
                messages = messages[1:]

            is_user_sim = _is_user_simulator_call(tools)
            decision: TurnDecision | None = None
            parent_id: int | None = None
            incoming_units: list[Any] = []
            if not is_user_sim:
                turn_request = TurnRequest(messages=messages, system=system_prompt)
                incoming_units = self._fingerprinter.units(turn_request)
                decision, parent_id = session.chain.resolve(incoming_units)

            prompt_ids = self._renderer.prompt_tokens(
                messages=messages,
                tools=tools,
                system_prompt=system_prompt,
            )
            max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or self._max_sample_tokens)
            temperature = float(body.get("temperature", self._temperature))
            logger.info(
                "proxy: rollout_id=%s decision=%s prepared prompt (%d tokens), calling sampler (+%.1fs)",
                rollout_id, (decision.kind if decision else "user_sim (excluded)"), len(prompt_ids), time.monotonic() - t0,
            )

            completions = await self._sampler.sample_with_prompt_tokens(
                prompt_ids,
                n=1,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=self._renderer.stop_sequences(),
                # Without this, SampledCompletion.sampling_logprobs is always
                # None -- _completion_tokens_and_logprobs() then returns None
                # on every call, silently 503ing every rollout (access logging
                # is disabled, so this failed with zero visible error).
                logprobs=True,
            )
            logger.info(
                "proxy: rollout_id=%s sampler returned %d completions (+%.1fs)",
                rollout_id, len(completions or []), time.monotonic() - t0,
            )
            if not completions:
                raise web.HTTPServiceUnavailable(text="sampler returned no completions")
            extracted = _completion_tokens_and_logprobs(completions[0])
            if extracted is None:
                raise web.HTTPServiceUnavailable(text="sampler returned no usable logprobs")
            output_tokens, output_logprobs = extracted

            parsed_message = self._renderer.parse_completion(output_tokens)
            _fix_tool_calls_for_openai(parsed_message)
            finish_reason = "tool_calls" if parsed_message.get("tool_calls") else "stop"
            logger.info(
                "proxy: rollout_id=%s parsed completion: %d output_tokens, content=%r, tool_calls=%r",
                rollout_id, len(output_tokens),
                (parsed_message.get("content") or "")[:200],
                parsed_message.get("tool_calls"),
            )

            response_id = f"resp_{uuid.uuid4().hex}"
            turn = TurnRecord(
                prompt_ids=prompt_ids,
                output_ids=output_tokens,
                finish_reason=finish_reason,
                output_log_probs=output_logprobs,
                text=flatten_content(parsed_message.get("content")),
                metadata={"response_id": response_id},
            )
            if is_user_sim:
                # Sampled and served like any other call, but deliberately kept
                # out of the training-sample token chain -- see
                # _is_user_simulator_call()'s docstring.
                logger.info("proxy: rollout_id=%s user-simulator turn (excluded from training chain)", rollout_id)
            else:
                node = session.tree.add_turn(turn, parent_id=parent_id, response_id=response_id)
                session.chain.stored_units = list(incoming_units)
                session.chain.leaf_id = node.node_id

                # Units representing "history + this new assistant turn" so the *next*
                # incoming request (which will include this turn plus a tool result) is
                # recognized as an APPEND rather than a fresh chain.
                response_units = list(incoming_units) + self._fingerprinter.units(
                    TurnRequest(messages=[parsed_message], system=None)
                )[1:]
                session.chain.response_units[node.node_id] = response_units
                session.turn_count += 1

        logger.info("proxy: rollout_id=%s responding (+%.1fs total)", rollout_id, time.monotonic() - t0)
        return web.json_response(self._completion_payload(parsed_message, output_tokens))

    @staticmethod
    def _completion_payload(message: dict[str, Any], output_tokens: list[int]) -> dict[str, Any]:
        finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(output_tokens),
                "total_tokens": len(output_tokens),
            },
        }


__all__ = ["RolloutSession", "TinkerRecordingProxy"]
