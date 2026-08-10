"""OpenAI-compatible recording policy server for Harbor/OpenCode rollouts."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from training.utils.rl.async_rl.errors import RecoverableRolloutError
from training.utils.rl.agent.openai import (
    build_turn_renderer,
    flatten_content,
)
from training.utils.rl.agent.sampling import (
    completion_routes,
    completion_values,
)
from training.utils.rl.agent.session import DeploymentTrainingSession
from training.utils.rl.agent.trajectory import (
    SelectedLeaf,
    TokenSegment,
    TurnRecord,
)
from training.utils.rl.agent.turn_matching import (
    MessageHashFingerprinter,
    TurnDecision,
    TurnKind,
    TurnRequest,
    classify,
)

logger = logging.getLogger(__name__)
_MODEL_ID = "policy"
_HOST = "0.0.0.0"


class TraceIntegrityError(RecoverableRolloutError):
    """A sampled turn cannot be represented as token-exact training data."""


def _field(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _wire_message(message: Any) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for index, call in enumerate(_field(message, "tool_calls") or []):
        function = _field(call, "function")
        name = str(_field(function, "name", "") or "")
        arguments = _field(function, "arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        calls.append(
            {
                "index": index,
                "id": _field(call, "id") or f"{name}:{index}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments or "{}",
                },
            }
        )
    wire: dict[str, Any] = {
        "role": "assistant",
        "content": flatten_content(_field(message, "content")),
    }
    if calls:
        wire["tool_calls"] = calls
    return wire


def _strip_volatile(value: Any) -> Any:
    """Remove request metadata that does not represent conversation intent."""
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _strip_volatile(item)
            for key, item in value.items()
            if key not in {"cache_control", "provider_metadata"}
        }
    return value


def _canonical_tool_arguments(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return value
    else:
        parsed = value
    return json.dumps(
        parsed,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _history_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize OpenCode wire details before example-local history matching."""
    normalized = copy.deepcopy(_strip_volatile(messages))
    for message in normalized:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            # ``index`` is required on streamed tool-call deltas but OpenCode
            # does not echo it in the next request's assistant message.
            call.pop("index", None)
            function = call.get("function") or {}
            if "arguments" in function:
                function["arguments"] = _canonical_tool_arguments(function["arguments"])
    return normalized


def _assistant_identity(message: dict[str, Any]) -> str:
    """Identify one returned assistant without its omitted reasoning."""
    tool_calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        tool_calls.append(
            {
                "id": str(call.get("id") or ""),
                "name": str(function.get("name") or ""),
                "arguments": _canonical_tool_arguments(
                    function.get("arguments") or "{}"
                ),
            }
        )
    payload = {
        "content": flatten_content(message.get("content")),
        "tool_calls": tool_calls,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _with_recorded_reasoning(
    messages: list[dict[str, Any]],
    turns: list[TurnRecord],
) -> list[dict[str, Any]]:
    """Restore assistant reasoning that OpenCode omits from later requests.

    Match the visible assistant response and tool calls, not its position.
    A history rewrite may insert an auxiliary summary assistant that was never a
    trainable turn; positional matching would attach task reasoning to it.
    """
    recorded: dict[str, list[str]] = {}
    for turn in turns:
        identity = str(turn.metadata.get("assistant_identity") or "")
        if not identity:
            continue
        recorded.setdefault(identity, []).append(
            str(turn.metadata.get("reasoning_content") or "")
        )
    if not recorded:
        return messages

    restored = list(messages)
    for position in range(len(restored) - 1, -1, -1):
        message = restored[position]
        if message.get("role") != "assistant":
            continue
        candidates = recorded.get(_assistant_identity(message))
        if not candidates:
            continue
        reasoning = candidates.pop()
        if reasoning and not restored[position].get("reasoning_content"):
            restored[position] = {
                **restored[position],
                "reasoning_content": reasoning,
            }
    return restored


@dataclass
class _HistoryResolution:
    decision: TurnDecision
    parent_id: int | None


@dataclass
class _HistoryChain:
    """OpenCode history state; message semantics stay in this example."""

    stored_units: list[Any] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    leaf_id: int | None = None
    response_units: dict[int, list[Any]] = field(default_factory=dict)

    def resolve(self, incoming_units: list[Any]) -> _HistoryResolution:
        matches = [
            (len(units), node_id)
            for node_id, units in self.response_units.items()
            if len(units) <= len(incoming_units)
            and units == incoming_units[: len(units)]
        ]
        if matches:
            matched, parent_id = max(matches)
            kind = TurnKind.APPEND if parent_id == self.leaf_id else TurnKind.WIPE
            return _HistoryResolution(TurnDecision(kind, matched), parent_id)

        fallback = classify(self.stored_units, incoming_units)
        kind = TurnKind.NEW if not self.response_units else TurnKind.WIPE
        return _HistoryResolution(
            TurnDecision(kind, fallback.matched_prefix_len),
            None,
        )

    def reset_active(self) -> None:
        self.stored_units = []
        self.turns = []
        self.leaf_id = None


@dataclass
class OpenCodePolicySession:
    run_id: str
    max_context_tokens: int
    chain: _HistoryChain = field(default_factory=_HistoryChain)
    training_session: DeploymentTrainingSession = field(
        default_factory=DeploymentTrainingSession
    )
    selected_leaves: list[SelectedLeaf] = field(default_factory=list)
    match_events: list[dict[str, Any]] = field(default_factory=list)
    request_traces: list[dict[str, Any]] = field(default_factory=list)
    auxiliary_turns: int = 0
    history_wipes: int = 0
    sampling_failures: int = 0
    trace_integrity_failures: int = 0
    trace_integrity_error: str | None = None
    context_overflows: int = 0
    toolless_tool_turns: int = 0
    last_error: str | None = None
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def note_request_failure(
        self,
        error: BaseException,
        *,
        trace_integrity: bool = False,
    ) -> None:
        detail = f"{type(error).__name__}: {error}"
        self.sampling_failures += 1
        self.last_error = detail
        if trace_integrity:
            self.trace_integrity_failures += 1
            if self.trace_integrity_error is None:
                self.trace_integrity_error = detail

    def recorded_turns(self) -> list[TurnRecord]:
        return self.training_session.tree.recorded_turns

    def freeze(self, kind: str) -> None:
        if self.chain.leaf_id is None:
            return
        self.selected_leaves.append(
            SelectedLeaf(
                node_id=self.chain.leaf_id,
                metadata={
                    "segment_kind": kind,
                    "finish_reason": self.chain.turns[-1].finish_reason,
                    "num_turns": len(self.chain.turns),
                    "text": self.chain.turns[-1].text,
                    "run_id": self.run_id,
                },
            )
        )
        self.chain.reset_active()

    def record(
        self,
        *,
        incoming_units: list[Any],
        response_units: list[Any],
        parent_id: int | None,
        turn: TurnRecord,
        decision: TurnDecision,
    ) -> None:
        actual = self.chain.resolve(incoming_units)
        if decision != actual.decision or parent_id != actual.parent_id:
            raise ValueError("OpenCode history changed between sampling and record")
        if decision.kind is TurnKind.WIPE:
            self.freeze("history_wipe")
        if parent_id is not None:
            self.selected_leaves = [
                leaf for leaf in self.selected_leaves if leaf.node_id != parent_id
            ]
        response_id = str(turn.metadata.get("response_id") or "")
        node = self.training_session.tree.add_turn(
            turn,
            parent_id=parent_id,
            response_id=response_id,
        )
        self.chain.stored_units = list(incoming_units)
        self.chain.turns = [
            path_node.turn
            for path_node in self.training_session.tree.path(node.node_id)
        ]
        self.chain.leaf_id = node.node_id
        self.chain.response_units[node.node_id] = list(response_units)

    def drain(self) -> list[TokenSegment]:
        self.freeze("final")
        return self.training_session.tree.materialize(
            self.selected_leaves,
            max_context_tokens=self.max_context_tokens,
        )


@dataclass(frozen=True)
class _PreparedRequest:
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    system_prompt: str
    prompt_tokens: list[int]
    max_tokens: int
    request_units: list[Any]
    decision: TurnDecision | None
    parent_id: int | None

    @property
    def is_trainable(self) -> bool:
        return bool(self.tools)


class OpenCodePolicyServer:
    """A local OpenAI endpoint backed by the live RLOR deployment sampler."""

    def __init__(
        self,
        *,
        sampler: Any,
        tokenizer: Any,
        sample_kwargs: dict[str, Any],
        renderer_name: str,
        max_seq_len: int,
        max_sample_tokens: int,
        capture_request_traces: bool = False,
    ) -> None:
        self._sampler = sampler
        self._tokenizer = tokenizer
        self._sample_kwargs = dict(sample_kwargs)
        self._renderer = build_turn_renderer(tokenizer, renderer_name)
        self._fingerprinter = MessageHashFingerprinter()
        self._max_seq_len = int(max_seq_len)
        self._max_sample_tokens = int(max_sample_tokens)
        self._capture_request_traces = bool(capture_request_traces)
        self._sessions: dict[str, OpenCodePolicySession] = {}
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self.port = 0

        self.app = web.Application()
        self.app.router.add_get("/v1/models", self._list_models)
        self.app.router.add_post("/v1/chat/completions", self._chat_completions)

    async def start(self) -> None:
        if self._runner is not None and self.port > 0:
            return
        if self._runner is not None:
            await self.close()
        try:
            self._runner = web.AppRunner(self.app, access_log=None)
            await self._runner.setup()
            self._site = web.TCPSite(
                self._runner,
                _HOST,
                0,
            )
            await self._site.start()
            server = getattr(self._site, "_server", None)
            sockets = list(getattr(server, "sockets", []) or [])
            if not sockets:
                raise RuntimeError("OpenCode policy server failed to bind a socket")
            self.port = int(sockets[0].getsockname()[1])
        except BaseException:
            await self.close()
            raise
        logger.info(
            "OpenCode policy server listening on %s:%d",
            _HOST,
            self.port,
        )

    async def close(self) -> None:
        runner = self._runner
        self._runner = None
        self._site = None
        self.port = 0
        if runner is not None:
            await runner.cleanup()

    def register_session(self, run_id: str) -> str:
        key = f"trial-{uuid.uuid4().hex}"
        self._sessions[key] = OpenCodePolicySession(
            run_id=run_id,
            max_context_tokens=self._max_seq_len,
        )
        return key

    async def pop_session(self, key: str) -> OpenCodePolicySession:
        session = self._sessions[key]
        async with session.lock:
            session.closed = True
            return self._sessions.pop(key)

    def discard_session(self, key: str) -> None:
        """Retire a cancelled rollout without waiting on an in-flight request."""
        session = self._sessions.pop(key, None)
        if session is not None:
            session.closed = True

    def _session(self, request: web.Request) -> OpenCodePolicySession:
        token = request.headers.get("Authorization", "")
        token = token.removeprefix("Bearer").strip()
        session = self._sessions.get(token)
        if session is None:
            raise web.HTTPUnauthorized(text="unknown trial token")
        return session

    async def _list_models(self, request: web.Request) -> web.Response:
        del request
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": _MODEL_ID,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "fireworks",
                    }
                ],
            }
        )

    async def _chat_completions(self, request: web.Request) -> web.StreamResponse:
        session = self._session(request)
        body = await request.json()
        async with session.lock:
            if session.closed:
                raise web.HTTPUnauthorized(text="trial session is closed")
            try:
                message, turn, prepared = await self._sample_turn(session, body)
            except web.HTTPException:
                raise
            except TraceIntegrityError as exc:
                session.note_request_failure(exc, trace_integrity=True)
                logger.warning(
                    "OpenCode policy trace integrity failed for %s: %s",
                    session.run_id,
                    exc,
                )
                raise web.HTTPServiceUnavailable(text=str(exc)) from exc
            except Exception as exc:
                session.note_request_failure(exc)
                logger.warning(
                    "OpenCode policy sampling failed for %s: %s",
                    session.run_id,
                    exc,
                )
                raise web.HTTPServiceUnavailable(text=str(exc)) from exc

            payload = self._completion_payload(message, turn)
            if not body.get("stream"):
                try:
                    self._record_turn_or_raise(
                        session,
                        prepared=prepared,
                        message=message,
                        turn=turn,
                    )
                except TraceIntegrityError as exc:
                    raise web.HTTPServiceUnavailable(text=str(exc)) from exc
                return web.json_response(payload)

            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
            )
            await response.prepare(request)
            chunks = self._stream_chunks(payload)
            await response.write(chunks[0].encode("utf-8"))
            self._record_turn_or_raise(
                session,
                prepared=prepared,
                message=message,
                turn=turn,
            )
            for chunk in chunks[1:]:
                await response.write(chunk.encode("utf-8"))
            await response.write_eof()
            return response

    async def _sample_turn(
        self,
        session: OpenCodePolicySession,
        body: dict[str, Any],
    ) -> tuple[dict[str, Any], TurnRecord, _PreparedRequest]:
        prepared = self._prepare_request(session, body)
        completion = await self._sample_completion(session, prepared)
        message, turn = self._turn_from_completion(
            completion,
            prompt_tokens=prepared.prompt_tokens,
        )
        return message, turn, prepared

    def _prepare_request(
        self,
        session: OpenCodePolicySession,
        body: dict[str, Any],
    ) -> _PreparedRequest:
        raw_messages = copy.deepcopy(list(body.get("messages") or []))
        tools = copy.deepcopy(list(body.get("tools") or []))
        system_prompt = ""
        if raw_messages and raw_messages[0].get("role") == "system":
            system_prompt = flatten_content(raw_messages[0].get("content"))
            raw_messages = raw_messages[1:]

        prompt_messages = _with_recorded_reasoning(
            raw_messages,
            session.recorded_turns(),
        )
        prompt_tokens = self._renderer.prompt_tokens(
            messages=prompt_messages,
            tools=tools,
            system_prompt=system_prompt,
        )
        requested_max = int(
            body.get("max_tokens")
            or body.get("max_completion_tokens")
            or self._max_sample_tokens
        )
        max_tokens = min(requested_max, self._max_sample_tokens)
        remaining = self._max_seq_len - len(prompt_tokens)
        if remaining < 1:
            session.context_overflows += 1
            raise web.HTTPBadRequest(
                text=(
                    f"prompt ({len(prompt_tokens)} tokens) exceeds the "
                    f"trainer sequence limit ({self._max_seq_len})"
                )
            )
        max_tokens = min(max_tokens, remaining)

        request_units: list[Any] = []
        decision: TurnDecision | None = None
        parent_id: int | None = None
        if tools:
            request_units = self._fingerprinter.units(
                TurnRequest(
                    messages=_history_messages(raw_messages),
                    system=_strip_volatile(system_prompt),
                )
            )
            resolution = session.chain.resolve(request_units)
            decision = resolution.decision
            parent_id = resolution.parent_id
        return _PreparedRequest(
            tools=tools,
            messages=raw_messages,
            system_prompt=system_prompt,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            request_units=request_units,
            decision=decision,
            parent_id=parent_id,
        )

    def _response_units(
        self,
        prepared: _PreparedRequest,
        message: dict[str, Any],
    ) -> list[Any]:
        return self._fingerprinter.units(
            TurnRequest(
                messages=_history_messages([*prepared.messages, message]),
                system=_strip_volatile(prepared.system_prompt),
            )
        )

    async def _sample_completion(
        self,
        session: OpenCodePolicySession,
        prepared: _PreparedRequest,
    ) -> Any:
        call_kwargs = dict(self._sample_kwargs)
        call_kwargs.update(
            n=1,
            max_tokens=prepared.max_tokens,
            logprobs=True,
            echo=False,
            stop=self._renderer.stop_sequences(),
        )
        completions = await session.training_session.sample_with_prompt_tokens(
            self._sampler,
            prepared.prompt_tokens,
            **call_kwargs,
        )
        if not completions:
            raise RecoverableRolloutError(
                "Fireworks sampler returned no OpenCode completion"
            )
        return completions[0]

    def _turn_from_completion(
        self,
        completion: Any,
        *,
        prompt_tokens: list[int],
    ) -> tuple[dict[str, Any], TurnRecord]:
        try:
            prompt_len = int(completion.prompt_len)
            full_tokens = [int(token) for token in completion.full_tokens]
        except Exception as exc:
            raise TraceIntegrityError(
                f"OpenCode completion tokens are invalid: {type(exc).__name__}: {exc}"
            ) from exc
        if (
            prompt_len != len(prompt_tokens)
            or full_tokens[:prompt_len] != prompt_tokens
        ):
            raise TraceIntegrityError(
                "OpenCode completion prompt tokens do not match the request"
            )
        output_tokens = full_tokens[prompt_len:]
        if not output_tokens:
            raise TraceIntegrityError(
                "Fireworks sampler returned an empty OpenCode completion"
            )

        try:
            logprobs = completion_values(
                completion,
                attribute="sampling_logprobs",
                output_len=len(output_tokens),
            )
            raw_logprobs = completion_values(
                completion,
                attribute="inference_logprobs",
                output_len=len(output_tokens),
            )
            routes = completion_routes(
                completion,
                output_len=len(output_tokens),
            )
        except (TypeError, ValueError) as exc:
            raise TraceIntegrityError(f"Fireworks {exc}") from exc
        if logprobs is None:
            raise TraceIntegrityError(
                "Fireworks completion is missing sampling_logprobs"
            )
        if self._sample_kwargs.get("include_routing_matrix") and routes is None:
            raise TraceIntegrityError(
                "OpenCode completion is missing requested routing matrices"
            )

        text = str(getattr(completion, "text", "") or "")
        if not text:
            text = self._tokenizer.decode(
                output_tokens,
                skip_special_tokens=False,
            )
        parser_fallback = False
        try:
            parsed = self._renderer.parse_completion(output_tokens)
            message = _wire_message(parsed)
        except Exception as exc:
            parser_fallback = True
            parsed = {}
            message = {
                "role": "assistant",
                "content": self._tokenizer.decode(
                    output_tokens,
                    skip_special_tokens=True,
                ),
            }
            logger.warning(
                "OpenCode completion parsing failed; returning raw text: %s: %s",
                type(exc).__name__,
                exc,
            )

        try:
            turn = TurnRecord(
                prompt_ids=prompt_tokens,
                output_ids=output_tokens,
                finish_reason=str(
                    getattr(completion, "finish_reason", "stop") or "stop"
                ),
                output_log_probs=logprobs,
                output_raw_log_probs=raw_logprobs,
                output_routing_matrices=routes,
                text=text,
                metadata={
                    "reasoning_content": str(_field(parsed, "reasoning_content") or ""),
                    "assistant_identity": _assistant_identity(message),
                    "parser_fallback": parser_fallback,
                    "response_id": f"chatcmpl-{uuid.uuid4().hex}",
                },
            )
        except Exception as exc:
            raise TraceIntegrityError(
                "OpenCode completion could not be represented: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return message, turn

    def _record_turn(
        self,
        session: OpenCodePolicySession,
        *,
        prepared: _PreparedRequest,
        message: dict[str, Any],
        turn: TurnRecord,
    ) -> None:
        decision = prepared.decision
        if self._capture_request_traces:
            session.request_traces.append(
                {
                    "trainable": prepared.is_trainable,
                    "system_prompt": prepared.system_prompt,
                    "messages": copy.deepcopy(prepared.messages),
                    "tools": copy.deepcopy(prepared.tools),
                    "prompt_ids": list(turn.prompt_ids),
                    "completion_ids": list(turn.output_ids),
                    "completion_logprobs": list(turn.output_log_probs),
                    "completion_raw_logprobs": (
                        list(turn.output_raw_log_probs)
                        if turn.output_raw_log_probs is not None
                        else None
                    ),
                    "completion_routing_matrices": (
                        list(turn.output_routing_matrices)
                        if turn.output_routing_matrices is not None
                        else None
                    ),
                    "completion_text": turn.text,
                    "decoded_completion_text": self._tokenizer.decode(
                        turn.output_ids,
                        skip_special_tokens=False,
                    ),
                    "assistant_message": copy.deepcopy(message),
                    "finish_reason": turn.finish_reason,
                    "parser_fallback": bool(
                        turn.metadata.get("parser_fallback", False)
                    ),
                    "turn_kind": decision.kind.value
                    if decision is not None
                    else "auxiliary",
                    "matched_prefix_len": (
                        decision.matched_prefix_len if decision is not None else 0
                    ),
                }
            )
        # OpenCode title/summary requests are served but do not advance the
        # trainable chain. A later tool-bearing rewrite is classified separately.
        if not prepared.is_trainable:
            session.auxiliary_turns += 1
            return

        assert decision is not None
        if decision.kind is TurnKind.WIPE and session.chain.turns:
            session.history_wipes += 1
        session.record(
            incoming_units=prepared.request_units,
            response_units=self._response_units(prepared, message),
            parent_id=prepared.parent_id,
            turn=turn,
            decision=decision,
        )
        session.match_events.append(
            {
                "kind": decision.kind.value,
                "matched_prefix_len": decision.matched_prefix_len,
                "incoming_units": len(prepared.request_units),
                "prompt_tokens": len(prepared.prompt_tokens),
            }
        )
        if prepared.tools and not message.get("tool_calls"):
            session.toolless_tool_turns += 1
        logger.info(
            "[harbor-opencode] run=%s turn=%d kind=%s prompt=%d output=%d tools=%d",
            session.run_id,
            len(session.match_events),
            decision.kind.value,
            len(prepared.prompt_tokens),
            len(turn.output_ids),
            len(prepared.tools),
        )

    def _record_turn_or_raise(
        self,
        session: OpenCodePolicySession,
        *,
        prepared: _PreparedRequest,
        message: dict[str, Any],
        turn: TurnRecord,
    ) -> None:
        try:
            self._record_turn(
                session,
                prepared=prepared,
                message=message,
                turn=turn,
            )
        except Exception as exc:
            integrity_error = TraceIntegrityError(
                f"OpenCode trace recording failed: {type(exc).__name__}: {exc}"
            )
            session.note_request_failure(
                integrity_error,
                trace_integrity=True,
            )
            logger.warning(
                "OpenCode policy trace recording failed for %s: %s",
                session.run_id,
                exc,
            )
            raise integrity_error from exc

    def _completion_payload(
        self,
        message: dict[str, Any],
        turn: TurnRecord,
    ) -> dict[str, Any]:
        finish_reason = turn.finish_reason
        if message.get("tool_calls"):
            finish_reason = "tool_calls"
        elif finish_reason not in {"stop", "length"}:
            finish_reason = "stop"
        return {
            "id": str(turn.metadata["response_id"]),
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
                "prompt_tokens": len(turn.prompt_ids),
                "completion_tokens": len(turn.output_ids),
                "total_tokens": len(turn.prompt_ids) + len(turn.output_ids),
            },
        }

    @staticmethod
    def _stream_chunks(payload: dict[str, Any]) -> list[str]:
        choice = payload["choices"][0]
        first = {
            "id": payload["id"],
            "object": "chat.completion.chunk",
            "created": payload["created"],
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": choice["message"],
                    "finish_reason": None,
                }
            ],
        }
        final = {
            "id": payload["id"],
            "object": "chat.completion.chunk",
            "created": payload["created"],
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": choice["finish_reason"],
                }
            ],
            "usage": payload["usage"],
        }
        return [
            f"data: {json.dumps(first)}\n\n",
            f"data: {json.dumps(final)}\n\n",
            "data: [DONE]\n\n",
        ]


__all__ = [
    "OpenCodePolicyServer",
    "OpenCodePolicySession",
]
