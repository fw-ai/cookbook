"""Anthropic ``/v1/messages`` shim -> Fireworks token-in-token-out inference.

The unmodified claude-code CLI (running in a SWE-Gym Docker runtime) dials this shim as
if it were the Anthropic API. Per turn the shim:

1. groups the request by ``Authorization: Bearer <session_id>`` into a
   :class:`Session` and a single :class:`Chain` of :class:`TurnRecord`s;
2. translates Anthropic messages + tool schemas into chat-template messages
   and tool schemas into chat-template messages;
3. renders them to token ids with the **deployment tokenizer's HF chat
   template** (`apply_chat_template(..., tools=...)`) -- the tinker renderer
   cannot emit tool *schemas*, and the HF template is what the deployment was
   served with, so this is the right parity target for the RL rollout;
4. runs a **token-in-token-out** completion via
   :meth:`DeploymentSampler.sample_with_prompt_tokens` and records the exact
   prompt ids + output ids + per-token logprobs;
5. parses the output tokens back into Anthropic content blocks using the
   Fireworks **tinker renderer** ``parse_response`` (robust tool_call / thinking
   extraction), and streams an Anthropic Messages SSE response.

Multi-segment trajectories: a request whose message prefix no longer matches
the stored chain is a *prefix break*.  We do NOT discard the dropped prefix --
we freeze it into a segment and start a fresh one, so every assistant token is
still trained on.  Two causes:
* **context compaction / history rewrite** -> the current chain is frozen as a
  ``wipe`` segment, then rebuilt from the new prefix;
* **sub-agent excursion** -> a ``Task``/``Agent`` tool_use opens a sub-chain;
  when main later shows that dispatch's ``tool_result`` the sub-chain is frozen
  as a ``subagent`` segment.
``pop_session_split`` returns ALL such segments (>=1) plus the live ``final``
chain.  ``rollout_fn`` returns one ``RolloutRun`` containing one
``RolloutSample`` per segment; they share one run reward and the run's GRPO
advantage is broadcast across them (the run is still one group member -- see
``rollout.py`` and ``Rollout``).

The Fireworks-specific substitutions are intentional: inference is
``DeploymentSampler`` HTTP TITO, output parsing uses the tinker renderer, and
``close_session`` is a synchronous tombstone because the agent process has
already exited by the time we drain.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import secrets
from collections import OrderedDict
from typing import Any

from aiohttp import web

from training.examples.rl.coding_agent.trajectory import (
    TokenSegment,
    TurnRecord,
    TurnSegment,
    make_turn_segment,
    merge_turn_segments,
)
from training.utils.rl.rollout.turn_matching import (
    DEFAULT_TURN_MATCHING,
    TurnKind,
    TurnRequest,
    classify,
    make_fingerprinter,
)

logger = logging.getLogger(__name__)

# Tool names claude-code uses to dispatch a sub-agent.  A reply containing a
# tool_use with one of these names opens a sub-chain: the sub-agent's own
# inference turns arrive on the same session (same bearer) but rewrite the
# message history, so they form their own segment rather than appending to
# main.  When main later shows the dispatch's tool_result, the sub-chain is
# frozen into a ``subagent`` segment (see ``_close_finished_subagent``).
_SUBAGENT_TOOLS = {"Task", "Agent"}

# claude-code can send large message bodies (full file contents in tool
# results); cap the aiohttp request size generously.
_MAX_REQUEST_BYTES = 64 * 1024 * 1024
_MAX_CLOSED_SESSIONS = 10_000


# =============================================================================
# 1. Session / chain state
# =============================================================================
@dataclasses.dataclass
class Chain:
    """One conversation chain (main, or an active sub-agent).

    ``turns`` is the online log of engine calls (drained into training
    segments at rollout end).  ``stored_units`` is the turn-matching
    fingerprint of the last accepted turn -- per-message hashes or rendered
    token ids depending on the active fingerprinter -- against which the next
    turn is classified new / append / wipe (see
    :mod:`training.utils.rl.rollout.turn_matching`).
    """

    turns: list[TurnRecord] = dataclasses.field(default_factory=list)
    stored_units: list[Any] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Session:
    main: Chain = dataclasses.field(default_factory=Chain)
    # At most one sub-agent excursion is tracked at a time; ``pending_dispatch_id``
    # is the ``Task``/``Agent`` tool_use id we watch for on main to know when the
    # sub-agent has returned.  Frozen chains accumulate in ``segments`` (one
    # ``TurnSegment`` per wipe / finished sub-agent); ``pop_session_split``
    # appends the still-live chains and merges them all into training tensors.
    active_sub: Chain | None = None
    pending_dispatch_id: str = ""
    segments: list[TurnSegment] = dataclasses.field(default_factory=list)
    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    # Thread-safety/concurrency: aiohttp runs requests on one event loop, but
    # one session may still receive overlapping HTTP requests. ``tail`` forms a
    # per-session FIFO so only one turn mutates chain state at a time, while
    # ``lock`` is held only for tail bookkeeping and never across inference I/O.
    tail: asyncio.Future[None] | None = None
    # Sticky-routing key: forwarded as the completions ``user`` field so every
    # turn of one trajectory lands on the same replica and reuses the KV /
    # prefix cache instead of re-prefilling the whole growing context each turn
    # (needs deployment ``enableSessionAffinity=true``).
    user_key: str = ""


_Store = dict[str, Session]
_closed: OrderedDict[str, None] = OrderedDict()


def _is_closed(sid: str) -> bool:
    return sid in _closed


def _complete_tail(tail: asyncio.Future[None]) -> None:
    if not tail.done():
        tail.set_result(None)


# =============================================================================
# 2. Anthropic <-> chat-template translation
# =============================================================================
def _strip_cache_control(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache_control(x) for x in obj]
    return obj


def _flatten(c: Any) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return str(c)
    parts: list[str] = []
    for b in c:
        if isinstance(b, dict):
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text", ""))
            elif t == "tool_result":
                parts.append(_flatten(b.get("content")))
            elif t == "image":
                parts.append("[image omitted]")
        elif isinstance(b, str):
            parts.append(b)
    return "\n".join(p for p in parts if p)


def _translate_anthropic(msgs: list[dict], system: Any) -> list[dict]:
    """Anthropic messages + system -> chat-template messages. Pure function."""
    system_parts: list[str] = []
    translated: list[dict] = []
    if system:
        system_parts.append(_flatten(system))
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role == "system":
            system_parts.append(_flatten(content))
            continue
        if role == "user":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": _flatten(content)}]
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    translated.append({"role": "tool", "content": _flatten(b.get("content"))})
                elif isinstance(b, dict) and b.get("type") == "text":
                    translated.append({"role": "user", "content": b.get("text", "")})
                else:
                    translated.append({"role": "user", "content": _flatten(b)})
        elif role == "assistant":
            texts, thinkings, tcs = [], [], []
            blocks = content if isinstance(content, list) else [{"type": "text", "text": _flatten(content)}]
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "thinking":
                    thinkings.append(b.get("thinking", ""))
                elif b.get("type") == "tool_use":
                    tcs.append({"function": {"name": b.get("name", "tool"), "arguments": b.get("input") or {}}})
            mo: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if thinkings:
                mo["reasoning_content"] = "".join(thinkings)
            if tcs:
                mo["tool_calls"] = tcs
            translated.append(mo)
    if system_parts:
        translated.insert(0, {"role": "system", "content": "\n\n".join(p for p in system_parts if p)})
    return translated


def _anthropic_tools_to_chat_tools(anth_tools: list[dict] | None) -> list[dict] | None:
    if not anth_tools:
        return None
    ts: list[dict] = []
    for t in anth_tools:
        if not isinstance(t, dict) or "name" not in t:
            continue
        ts.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return ts or None


# =============================================================================
# 3. Per-turn stages
# =============================================================================
def _snapshot(turns: list[TurnRecord], kind: str) -> TurnSegment:
    """Freeze a chain's turns into a segment, carrying per-turn stats."""
    return make_turn_segment(
        turns, kind=kind, metadata={"num_turns": len(turns), "turns": _turn_stats(turns)},
    )


def _message_carries_tool_result(message: Any, tool_use_id: str) -> bool:
    """True iff ``message`` is a user message bearing a ``tool_result`` block
    for ``tool_use_id`` (i.e. the dispatched sub-agent has returned)."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("tool_use_id") == tool_use_id
        for block in content
    )


def _close_finished_subagent(s: Session, all_msgs: list[dict]) -> None:
    """If main now carries the pending dispatch's ``tool_result``, freeze the
    sub-chain into a ``subagent`` segment and clear the sub state."""
    if not s.pending_dispatch_id or s.active_sub is None:
        return
    if not any(_message_carries_tool_result(m, s.pending_dispatch_id) for m in all_msgs):
        return
    if s.active_sub.turns:
        s.segments.append(_snapshot(s.active_sub.turns, kind="subagent"))
    s.active_sub = None
    s.pending_dispatch_id = ""


def _render_token_ids(chat_messages: list[dict], tools_schema: list[dict] | None, tokenizer) -> list[int]:
    """Render chat-template messages into model input ids via the HF chat
    template.  Pure: tokenizes as-is on every request; later trajectory merge
    masks/drops any history a later prompt no longer token-matches."""
    enc = tokenizer.apply_chat_template(
        chat_messages,
        tools=tools_schema,
        tokenize=True,
        add_generation_prompt=True,
    )
    token_ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(token_ids)


def _build_turn_request(all_msgs: list[dict], system: Any, prompt_ids: list[int]) -> TurnRequest:
    """Strip Anthropic ``cache_control`` and pack the turn-matching inputs.

    Both fingerprinter strategies read from this; message-hash uses the stripped
    messages / system, token-prefix uses ``prompt_ids``."""
    return TurnRequest(
        messages=[_strip_cache_control(m) for m in all_msgs],
        system=_strip_cache_control(system),
        prompt_ids=prompt_ids,
    )


def _route_chain(s: Session, request_units: list[Any], all_msgs: list[dict]) -> tuple[Chain, bool]:
    """Pick the chain this turn extends: main, or the active sub-agent.

    Harness-specific (Anthropic ``Task`` / ``Agent`` sub-agents).  First closes
    a finished sub-agent (its dispatch ``tool_result`` has arrived on main),
    then routes to main iff the request still continues main's fingerprint;
    otherwise to the active sub-chain.
    """
    _close_finished_subagent(s, all_msgs)
    if s.active_sub is None:
        return s.main, False
    continues_main = classify(s.main.stored_units, request_units).kind is not TurnKind.WIPE
    return (s.main, False) if continues_main else (s.active_sub, True)


async def _generate(prompt_ids: list[int], s: Session, body: dict, app) -> TurnRecord:
    """Token-in-token-out completion via DeploymentSampler; capture exact ids."""
    sampler = app["sampler"]
    sample_kwargs: dict[str, Any] = dict(app["sample_kwargs"] or {})
    sample_kwargs.update(s.sampling_defaults or {})

    # Per-turn max_tokens: honor the request cap and the session context budget.
    max_tokens = int(sample_kwargs.get("max_tokens", 1024))
    if "max_tokens" in body:
        max_tokens = min(max_tokens, int(body["max_tokens"]))
    if s.max_context_tokens > 0:
        remaining = s.max_context_tokens - len(prompt_ids)
        if remaining <= 0:
            logger.warning(
                "[shim] prompt %d >= max_context_tokens %d; returning length stop",
                len(prompt_ids), s.max_context_tokens,
            )
            return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[], finish_reason="length")
        max_tokens = min(max_tokens, remaining)

    stop = list(app.get("stop") or [])
    for sseq in body.get("stop_sequences") or []:
        if isinstance(sseq, str):
            stop.append(sseq)

    call_kwargs = dict(sample_kwargs)
    call_kwargs.pop("max_tokens", None)
    call_kwargs.pop("n", None)
    call_kwargs.pop("stop", None)
    # Sticky routing: pin this trajectory's turns to one replica for KV reuse.
    if s.user_key:
        call_kwargs.setdefault("user", s.user_key)
    completions = await sampler.sample_with_prompt_tokens(
        prompt_ids, n=1, max_tokens=max_tokens, stop=(stop or None), **call_kwargs
    )
    # Per-turn KV-reuse signal: the deployment returns ``cached-prompt-tokens``
    # in response headers (parsed into ``ServerMetrics`` by the SDK and buffered
    # on the sampler). A high cached/prompt ratio on turn >1 means the sticky
    # replica reused the prefix cache instead of re-prefilling the whole context.
    # Reliable only at rollout concurrency 1 (concurrent sessions interleave in
    # the shared sampler buffer); the multi-replica sticky-routing check reads
    # the deployment serving log instead. Env-gated so normal runs don't drain.
    if os.environ.get("SWE_SHIM_METRICS") and hasattr(sampler, "drain_metrics"):
        pl = len(prompt_ids)
        for m in sampler.drain_metrics():
            if getattr(m, "prompt_tokens", None) == pl:
                cached = m.cached_prompt_tokens or 0
                logger.info(
                    "[shim-kv] user=%s prompt_tokens=%s cached_prompt_tokens=%s reuse=%.2f",
                    s.user_key, m.prompt_tokens, cached,
                    (cached / m.prompt_tokens) if m.prompt_tokens else 0.0,
                )
                break
    if not completions:
        return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[], finish_reason="stop")

    c = completions[0]
    prompt_len = int(c.prompt_len)
    out_tokens = list(c.full_tokens[prompt_len:])
    out_lp_raw = getattr(c, "inference_logprobs", None)
    out_lp = list(out_lp_raw) if out_lp_raw is not None else []
    if getattr(c, "logprobs_echoed", False) and len(out_lp) == prompt_len + len(out_tokens):
        out_lp = out_lp[prompt_len:]
    if len(out_lp) != len(out_tokens):
        logger.warning(
            "[shim] logprob/token mismatch (%d vs %d); zeroing turn logprobs",
            len(out_lp), len(out_tokens),
        )
        out_lp = [0.0] * len(out_tokens)
    finish = getattr(c, "finish_reason", "stop") or "stop"
    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=out_tokens,
        finish_reason=str(finish),
        output_log_probs=out_lp,
    )


def _anthropic_blocks(message: Any, finish: str) -> tuple[list[dict], str, str]:
    """Map a tinker-renderer-parsed assistant Message -> Anthropic blocks.

    Also returns the ``dispatch_id`` of a sub-agent (``Task``/``Agent``)
    tool_use when one is present, so the handler can open a sub-chain.
    """
    blocks: list[dict] = []
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("type")
            if t == "thinking":
                blocks.append({"type": "thinking", "thinking": part.get("thinking", "")})
            elif t == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})

    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    has_tool = False
    dispatch_id = ""
    for tc in tool_calls or []:
        fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)
        name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else "tool")
        raw_args = getattr(fn, "arguments", None)
        if raw_args is None and isinstance(fn, dict):
            raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                inp = json.loads(raw_args)
            except json.JSONDecodeError:
                inp = {}
        else:
            inp = raw_args or {}
        tool_use_id = f"toolu_{secrets.token_hex(8)}"
        blocks.append({"type": "tool_use", "id": tool_use_id, "name": name, "input": inp})
        has_tool = True
        if name in _SUBAGENT_TOOLS:
            dispatch_id = tool_use_id

    if not blocks:
        blocks.append({"type": "text", "text": ""})
    if has_tool:
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    return blocks, stop_reason, dispatch_id


def _build_reply(output_ids: list[int], finish: str, app) -> tuple[list[dict], str, str]:
    if not output_ids:
        return [{"type": "text", "text": ""}], ("max_tokens" if finish == "length" else "end_turn"), ""
    renderer = app["renderer"]
    try:
        message, _term = renderer.parse_response(list(output_ids))
    except Exception as e:  # parsing failure must not kill the agent turn
        logger.warning("[shim] parse_response failed (%s); falling back to decoded text", e)
        text = app["tokenizer"].decode(output_ids, skip_special_tokens=True)
        return [{"type": "text", "text": text}], ("max_tokens" if finish == "length" else "end_turn"), ""
    return _anthropic_blocks(message, finish)


def _start_sub_chain(s: Session, dispatch_id: str) -> None:
    """Open a fresh sub-chain on this session and record the tool_use id we
    watch on main to know when the sub-agent has returned (closed inside
    :func:`_route_chain` via :func:`_close_finished_subagent`)."""
    s.pending_dispatch_id = dispatch_id
    if s.active_sub is None:
        s.active_sub = Chain()


def _log_turn_match(
    *,
    sid: str,
    chain: str,
    strategy: str,
    decision,
    stored_len: int,
    incoming_len: int,
    prompt_len: int,
) -> None:
    logger.info(
        "[shim-prefix] sid=%s chain=%s strategy=%s kind=%s matched=%d stored=%d incoming=%d prompt=%d",
        sid, chain, strategy, decision.kind.value, decision.matched_prefix_len,
        stored_len, incoming_len, prompt_len,
    )
    if decision.kind is TurnKind.APPEND:
        return
    path = os.environ.get("SWE_PREFIX_MISMATCH_LOG")
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(json.dumps({
                "session_id": sid,
                "chain": chain,
                "strategy": strategy,
                "kind": decision.kind.value,
                "matched_prefix_len": decision.matched_prefix_len,
                "stored_len": stored_len,
                "incoming_len": incoming_len,
                "prompt_len": prompt_len,
            }) + "\n")
    except OSError as e:
        logger.warning("[shim-prefix] failed to write mismatch log %s: %s", path, e)


# =============================================================================
# 4. Request handling + SSE wrap
# =============================================================================
async def _run_turn(s: Session, body: dict, app, sid: str) -> tuple[list[dict], str, int, int]:
    """Render -> classify -> generate one turn under the session lock.

    Returns ``(blocks, stop_reason, in_tokens, out_tokens)``.  Renders the full
    request first (so token-prefix matching can see the prompt), routes it to a
    chain, classifies new/append/wipe, snapshots a diverged chain into a
    segment, then records the engine call.
    """
    all_msgs = body.get("messages") or []
    system = body.get("system")
    chat_messages = _translate_anthropic(all_msgs, system)
    tools_schema = _anthropic_tools_to_chat_tools(body.get("tools"))
    prompt_ids = _render_token_ids(chat_messages, tools_schema, app["tokenizer"])

    request_units = app["fingerprinter"].units(_build_turn_request(all_msgs, system, prompt_ids))
    target, is_sub = _route_chain(s, request_units, all_msgs)
    stored_len = len(target.stored_units)
    decision = classify(target.stored_units, request_units)
    chain_name = "sub" if is_sub else "main"
    _log_turn_match(
        sid=sid,
        chain=chain_name,
        strategy=getattr(app["fingerprinter"], "name", "unknown"),
        decision=decision,
        stored_len=stored_len,
        incoming_len=len(request_units),
        prompt_len=len(prompt_ids),
    )
    if decision.kind is TurnKind.WIPE and target.turns:
        s.segments.append(_snapshot(target.turns, kind="wipe"))
    if decision.kind is not TurnKind.APPEND:
        target.turns = []  # new / wipe start a fresh segment

    prev_full = (
        len(target.turns[-1].prompt_ids) + len(target.turns[-1].output_ids)
        if target.turns else 0
    )
    turn_index = len(target.turns) + 1
    target.stored_units = request_units

    turn = await _generate(prompt_ids, s, body, app)
    blocks, stop_reason, dispatch_id = _build_reply(turn.output_ids, turn.finish_reason, app)
    target.turns.append(turn)
    if dispatch_id and not is_sub:  # a sub-agent dispatched from main; sub doesn't nest
        _start_sub_chain(s, dispatch_id)

    in_tokens, out_tokens = len(prompt_ids), len(turn.output_ids)
    # ``suffix`` is the new prefix this turn adds on top of what the deployment
    # saw last turn -- the only span a sticky replica must (re-)prefill when KV
    # reuse works.  Correlate by sid / user with the [shim-kv] cache line.
    suffix = max(0, in_tokens - prev_full) if decision.kind is TurnKind.APPEND else in_tokens
    logger.info(
        "[shim] sid=%s turn=%d chain=%s kind=%s prompt=%d suffix=%d out=%d finish=%s user=%s",
        sid, turn_index, chain_name, decision.kind.value, in_tokens, suffix,
        out_tokens, turn.finish_reason, s.user_key,
    )
    return blocks, stop_reason, in_tokens, out_tokens


async def _handle_request(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    auth = request.headers.get("Authorization", "")
    sid = auth.removeprefix("Bearer ").strip()
    if not sid or _is_closed(sid):
        return web.Response(status=503, text="session closed")
    s = request.app["store"].setdefault(sid, Session())
    if not s.user_key:
        s.user_key = sid
    async with s.lock:
        previous_tail = s.tail
        current_tail: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        s.tail = current_tail
    turn_task: asyncio.Task[tuple[list[dict], str, int, int]] | None = None
    try:
        if previous_tail is not None:
            await previous_tail
        turn_task = asyncio.create_task(_run_turn(s, body, request.app, sid))
        blocks, stop_reason, in_tokens, out_tokens = await asyncio.shield(turn_task)
    finally:
        if turn_task is not None and not turn_task.done():
            try:
                await turn_task
            except Exception:
                pass
        if previous_tail is not None and not previous_tail.done() and turn_task is None:
            previous_tail.add_done_callback(lambda _future: _complete_tail(current_tail))
        else:
            _complete_tail(current_tail)
    return await _stream_response(request, blocks, stop_reason, in_tokens, out_tokens)


async def _stream_response(request, blocks, stop_reason, in_tok, out_tok) -> web.StreamResponse:
    out = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
    await out.prepare(request)

    ms = {
        "type": "message_start",
        "message": {
            "id": f"msg_{secrets.token_hex(12)}", "type": "message", "role": "assistant",
            "model": "fireworks-actor", "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": in_tok, "output_tokens": 0},
        },
    }
    await out.write(f"event: message_start\ndata: {json.dumps(ms, ensure_ascii=False)}\n\n".encode())

    for idx, block in enumerate(blocks):
        bt = block["type"]
        if bt == "thinking":
            start = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        elif bt == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"], ensure_ascii=False)}
        await out.write(
            f"event: content_block_start\ndata: "
            f"{json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': start}, ensure_ascii=False)}\n\n".encode()
        )
        await out.write(
            f"event: content_block_delta\ndata: "
            f"{json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': delta}, ensure_ascii=False)}\n\n".encode()
        )
        await out.write(
            f"event: content_block_stop\ndata: "
            f"{json.dumps({'type': 'content_block_stop', 'index': idx}, ensure_ascii=False)}\n\n".encode()
        )

    md = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }
    await out.write(f"event: message_delta\ndata: {json.dumps(md, ensure_ascii=False)}\n\n".encode())
    await out.write(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'}, ensure_ascii=False)}\n\n".encode())
    return out


async def _count_tokens(request: web.Request) -> web.Response:
    return web.json_response({"input_tokens": 0})


async def _ok(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


# =============================================================================
# 5. Public API
# =============================================================================
def open_session(store: _Store, sid: str, *, sampling_defaults: dict | None = None, max_context_tokens: int = 0) -> None:
    """Register a session before claude-code dials in. Fail-fast on duplicate."""
    if sid in store:
        raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
    s = store[sid] = Session()
    s.sampling_defaults = dict(sampling_defaults or {})
    s.max_context_tokens = int(max_context_tokens or 0)
    s.user_key = sid


def _turn_stats(turns: list[TurnRecord]) -> list[dict]:
    """Per-turn shape used by the rollout trajectory artifact / batching check.

    ``suffix_len`` is the new prefix a turn adds on top of the prior turn's full
    sequence (what a sticky replica must prefill if the KV cache is reused).
    """
    stats: list[dict] = []
    prev_full = 0
    for i, t in enumerate(turns):
        p, o = len(t.prompt_ids), len(t.output_ids)
        stats.append(
            {
                "turn": i + 1,
                "prompt_len": p,
                "suffix_len": (max(0, p - prev_full) if i > 0 else p),
                "out_len": o,
                "finish": t.finish_reason,
            }
        )
        prev_full = p + o
    return stats


def pop_session_split(store: _Store, sid: str) -> list[TokenSegment]:
    """Drain the session into >=1 merged training segments.

    Segments frozen mid-run (``wipe`` / finished ``subagent`` snapshots) live
    in ``s.segments``; this appends whatever chains are still live -- a
    dangling sub-agent and main -- then merges them all.  A run that never
    compacted or dispatched a sub-agent yields exactly one ``final`` segment.
    """
    s = store.pop(sid, None)
    if s is None:
        return []
    if s.active_sub is not None and s.active_sub.turns:
        s.segments.append(_snapshot(s.active_sub.turns, kind="subagent"))
    if s.main.turns:
        s.segments.append(_snapshot(s.main.turns, kind="final"))
    return merge_turn_segments(s.segments, max_context_tokens=s.max_context_tokens)


def close_session(sid: str) -> None:
    """Tombstone the session (late requests get 503). Synchronous: the agent
    has already exited by drain time, so there is nothing in flight to cancel."""
    _closed[sid] = None
    _closed.move_to_end(sid)
    while len(_closed) > _MAX_CLOSED_SESSIONS:
        _closed.popitem(last=False)


def build_stop_strings(renderer, tokenizer) -> list[str]:
    """Decode the renderer's stop sequences to strings (the completions API
    rejects integer stop token ids)."""
    seqs = renderer.get_stop_sequences() or []
    out: list[str] = []
    for sseq in seqs:
        if isinstance(sseq, int):
            out.append(tokenizer.decode([sseq], skip_special_tokens=False))
        elif isinstance(sseq, str):
            out.append(sseq)
    return out


def start(
    *,
    tokenizer,
    renderer,
    sampler,
    sample_kwargs: dict,
    stop: list[str] | None = None,
    turn_matching: str = DEFAULT_TURN_MATCHING,
) -> tuple[web.Application, _Store]:
    """Build the aiohttp app + session store.

    Caller runs the server (e.g. ``aiohttp_threaded.run_app_in_thread``). Use
    ``open_session`` before the agent dials in and ``pop_session_split`` to
    drain its trajectory at rollout end.  ``turn_matching`` selects the
    turn-routing strategy (``message_hash`` | ``token_prefix``); raises
    ``ValueError`` on an unknown value.
    """
    store: _Store = {}
    app = web.Application(client_max_size=_MAX_REQUEST_BYTES)
    app["tokenizer"] = tokenizer
    app["renderer"] = renderer
    app["sampler"] = sampler
    app["sample_kwargs"] = dict(sample_kwargs or {})
    app["stop"] = list(stop or [])
    app["store"] = store
    app["fingerprinter"] = make_fingerprinter(turn_matching)
    app.router.add_post("/v1/messages", _handle_request)
    app.router.add_post("/v1/messages/count_tokens", _count_tokens)
    app.router.add_get("/healthz", _ok)
    app.router.add_get("/v1/models", _ok)
    return app, store
