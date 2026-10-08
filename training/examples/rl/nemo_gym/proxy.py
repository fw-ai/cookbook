"""Recording proxy: an OpenAI-compatible /v1/chat/completions server backed by a
Fireworks Dedicated-deployment sampler (``DeploymentSampler``, via
``async_rl_loop``'s ``RolloutSetup``).

NeMo Gym's ``inference_provider`` model server is pointed at this proxy instead
of Fireworks' public API. Each request is forwarded to the setup's sampler
(the recipe hotloads new weights into the same deployment after every
optimizer batch -- this proxy makes no snapshot/version calls itself) and the
exact prompt/completion token ids + logprobs are recorded per turn, keyed by
NeMo Gym's own per-rollout correlation id (propagated via the standard OpenAI
``user`` field, enabled by NeMo Gym's ``correlate_via_user_field`` -- NVIDIA-NeMo/Gym#3783).

Per-session trajectory bookkeeping does NOT use ``training.utils.rl.rollout.MessageTrajectoryAssembler``'s
exact-dict-equality checkpointing. ``simple_agent``-based harnesses drive NeMo
Gym's *Responses API*, and the model server rebuilds the chat-messages list
from that history every turn; the rebuild is not guaranteed byte-identical to
the assistant message returned the turn before (tool-call id/ordering can
differ, and ``reasoning_content`` is dropped unless ``uses_reasoning_parser`` is
set), so message-equality checks reject real continuations.

Instead, history is resolved by rollout identity: ``SimpleAgent``'s episode loop
is strictly sequential per rollout and never forks or rewrites history, so each
request carrying an in-progress rollout id is a continuation of that rollout's
current leaf turn (append-only). Harnesses that branch or rewrite history are
out of scope for this example.
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
from training.utils.rl.agent.sampling import completion_values, token_segment_to_sample
from training.utils.rl.agent.session import DeploymentTrainingSession
from training.utils.rl.agent.trajectory import SelectedLeaf, TurnRecord
from training.utils.rl.rollout import RolloutSample

logger = logging.getLogger(__name__)
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


def _finish_reason(completion: Any, message: dict[str, Any]) -> str:
    """OpenAI finish_reason from the sampler's own stop signal, not just tool_calls."""
    if getattr(completion, "finish_reason", None) == "length":
        return "length"
    return "tool_calls" if message.get("tool_calls") else "stop"


def _flatten_message_content(message: dict[str, Any]) -> None:
    """Make ``content`` a plain string, as the OpenAI chat-completions spec requires.

    Some renderers (e.g. ``kimi_k3``) return typed content parts
    (``[{"type": "text", "text": ...}]``) even for plain text, while others
    (e.g. ``qwen3_5``) return a string. NeMo Gym's ``inference_provider`` does
    string arithmetic on ``content`` and 500s on a list.
    """
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        message["content"] = flatten_content(content)


def _fix_tool_calls_for_openai(message: dict[str, Any]) -> None:
    """Coerce ``Renderer.to_openai_message()``'s tool_calls into real OpenAI shape.

    The cookbook renderer emits ``id=None`` (no call ever gets an id)
    and ``function.arguments`` as a parsed dict, not a JSON string -- NeMo
    Gym's own ``NeMoGymChatCompletion`` validates strictly against the real
    OpenAI spec (non-null string id, JSON-string arguments) and 500s on every
    single tool call otherwise.
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
class RolloutSession:
    """Token-exact record of one rollout's model calls (append-only chain)."""

    rollout_id: str
    training: DeploymentTrainingSession = field(default_factory=DeploymentTrainingSession)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    leaf_id: int | None = None
    turn_count: int = 0

    def to_samples(self, *, reward: float) -> list[RolloutSample]:
        """One sample per materialized segment.

        ``materialize`` splits the chain into several segments when a turn's
        prompt is not a token-prefix of the previous turn's prompt + output;
        every segment carries trainable tokens, so none may be dropped.
        """
        if self.leaf_id is None:
            return []
        segments = self.training.tree.materialize([SelectedLeaf(self.leaf_id)], max_context_tokens=0)
        samples = []
        for segment in segments:
            if not any(segment.loss_mask):
                continue
            samples.append(token_segment_to_sample(segment, reward=reward))
        return samples


class RecordingChatProxy:
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
        exclude_user_simulator: bool = False,
    ) -> None:
        self._sampler = sampler
        self._tokenizer = tokenizer
        self._renderer: TurnRenderer = build_turn_renderer(tokenizer, renderer_name)
        self._max_sample_tokens = int(max_sample_tokens)
        self._temperature = float(temperature)
        self._exclude_user_simulator = bool(exclude_user_simulator)
        self._sample_kwargs: dict[str, Any] = {}
        self._sessions: dict[str, RolloutSession] = {}

        self.app = web.Application()
        self.app.router.add_get("/v1/models", self._list_models)
        self.app.router.add_post("/v1/chat/completions", self._chat_completions)
        self._runner: web.AppRunner | None = None
        self.port = 0

    # --- lifecycle -----------------------------------------------------------

    def set_sampler(self, sampler: Any, sample_kwargs: dict[str, Any] | None = None) -> None:
        """Bind the deployment-backed sampler once ``RolloutSetup`` is available.

        Called exactly once by the rollout factory (``build_deployment_sampler``
        needs ``setup``, which only exists after the recipe's Dedicated
        trainer/deployment are up). No per-step calls after that -- the recipe
        hotloads new weights into this same deployment underneath.

        ``sample_kwargs`` should be the recipe's ``RolloutSetup.sample_kwargs``.
        It carries the on-policy sampling settings (``temperature``, ``top_p=1.0``,
        ``top_k=0``, ``max_seq_len``, ``http_timeout``, ...); without ``top_p``/
        ``top_k`` the serving stack applies the model's generation_config
        defaults, which truncate rollouts and bias the policy-gradient estimate.

        Threading: the proxy awaits this sampler on its own event loop/thread,
        not the recipe's loop. That is only safe while the recipe does not drive
        the same sampler object concurrently from its own loop for these rollouts
        (true for the ``rollout_fn`` contract used here). If that ever changes,
        give the proxy its own sampler instance instead of sharing this one.
        """
        self._sampler = sampler
        self._sample_kwargs = dict(sample_kwargs or {})

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
        # The `user` field carries NeMo Gym's per-rollout correlation id
        # (correlate_via_user_field). Without it, concurrent rollouts would
        # silently share one session and corrupt each other's token chains.
        rollout_id = body.get("user")
        if not rollout_id:
            raise web.HTTPBadRequest(
                text="missing `user` field: enable correlate_via_user_field (NVIDIA-NeMo/Gym#3783) "
                "so each rollout's calls carry its rollout id"
            )
        rollout_id = str(rollout_id)
        if self._sampler is None:
            raise web.HTTPServiceUnavailable(text="sampler not ready -- set_sampler() not called yet")
        session = self.get_or_create_session(rollout_id)
        t0 = time.monotonic()
        logger.info("proxy: rollout_id=%s turn=%d request received", rollout_id, session.turn_count)

        async with session.lock:
            messages = list(body.get("messages") or [])
            tools = list(body.get("tools") or [])

            system_prompt = ""
            if messages and messages[0].get("role") == "system":
                system_prompt = flatten_content(messages[0].get("content"))
                messages = messages[1:]

            is_user_sim = self._exclude_user_simulator and _is_user_simulator_call(tools)
            prompt_ids = self._renderer.prompt_tokens(messages=messages, tools=tools, system_prompt=system_prompt)
            # Sampling settings come from the recipe (RolloutSetup.sample_kwargs),
            # not the request: the trainer's importance ratios assume the rollout
            # policy sampled at exactly those settings (temperature, top_p, top_k),
            # so a per-request override would silently skew GRPO. A request's
            # max_tokens may only tighten the configured cap, never raise it.
            sample_kwargs = {"max_tokens": self._max_sample_tokens, "temperature": self._temperature}
            sample_kwargs.update(self._sample_kwargs)
            cap = int(sample_kwargs["max_tokens"])
            requested = body.get("max_tokens") or body.get("max_completion_tokens")
            sample_kwargs["max_tokens"] = min(int(requested), cap) if requested else cap
            # Without logprobs=True, SampledCompletion.sampling_logprobs is None.
            sample_kwargs.update(n=1, stop=self._renderer.stop_sequences(), logprobs=True)

            logger.info(
                "proxy: rollout_id=%s prompt=%d tokens, calling sampler (+%.1fs)",
                rollout_id, len(prompt_ids), time.monotonic() - t0,
            )
            completions = await session.training.sample_with_prompt_tokens(self._sampler, prompt_ids, **sample_kwargs)
            if not completions:
                raise web.HTTPServiceUnavailable(text="sampler returned no completions")
            completion = completions[0]
            output_tokens = list(completion.full_tokens[int(completion.prompt_len) :])
            try:
                output_logprobs = completion_values(
                    completion, attribute="sampling_logprobs", output_len=len(output_tokens)
                )
            except ValueError as exc:
                raise web.HTTPServiceUnavailable(text=str(exc)) from exc
            if output_logprobs is None:
                raise web.HTTPServiceUnavailable(text="sampler returned no usable logprobs")

            parsed_message = self._renderer.parse_completion(output_tokens)
            _fix_tool_calls_for_openai(parsed_message)
            _flatten_message_content(parsed_message)
            finish_reason = _finish_reason(completion, parsed_message)

            if is_user_sim:
                # Served like any other call, but kept out of the training chain
                # -- see _is_user_simulator_call().
                logger.debug("proxy: rollout_id=%s user-simulator turn excluded from training chain", rollout_id)
            else:
                node = session.training.tree.add_turn(
                    TurnRecord(
                        prompt_ids=prompt_ids,
                        output_ids=output_tokens,
                        finish_reason=finish_reason,
                        output_log_probs=output_logprobs,
                        text=flatten_content(parsed_message.get("content")),
                    ),
                    parent_id=session.leaf_id,
                )
                session.leaf_id = node.node_id
                session.turn_count += 1

        logger.info("proxy: rollout_id=%s turn=%d served in %.1fs", rollout_id, session.turn_count, time.monotonic() - t0)
        return web.json_response(self._completion_payload(parsed_message, len(prompt_ids), output_tokens, finish_reason))

    @staticmethod
    def _completion_payload(
        message: dict[str, Any], prompt_len: int, output_tokens: list[int], finish_reason: str
    ) -> dict[str, Any]:
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
                "prompt_tokens": prompt_len,
                "completion_tokens": len(output_tokens),
                "total_tokens": prompt_len + len(output_tokens),
            },
        }


__all__ = ["RolloutSession", "RecordingChatProxy"]
