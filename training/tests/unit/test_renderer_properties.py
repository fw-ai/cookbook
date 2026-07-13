"""Unified property-based QA harness for cookbook model renderers.

This is the CPU CI gate for renderer correctness. It mirrors the four
invariants upstream tinker asserts per renderer, but factors the renderer
set (:mod:`renderer_matrix`) and the conversation bank
(:mod:`renderer_scenarios`) into shared data so a new renderer joins CI by
adding a single :class:`~renderer_matrix.RendererCase` row.

The four invariants:

1. HF generation parity — ``build_generation_prompt`` tokens equal
   ``apply_chat_template(add_generation_prompt=True)`` for every
   generation-shaped scenario (the renderer<->upstream tokenization
   contract). Tool declarations are byte-compared too: the HF reference
   gets ``tools=`` and the renderer gets its own tool block (there is no
   ``hf_safe`` opt-out for tool or thinking scenarios).
2. HF supervised parity — ``build_supervised_example`` tokens equal
   ``apply_chat_template(add_generation_prompt=False)`` for renderers whose
   supervised header matches the generation header.
3. supervised<->generation<->parse consistency — the supervised weight mask
   is ``000...111``, its observation equals the generation prompt of the
   prefix (for renderers with ``observation_equals_generation``), and the
   trained action tokens parse back to a clean assistant turn whose text
   matches the scenario.
4. sequence extension — for renderers that claim it, each turn's full
   sequence is a prefix of the next turn's observation (KV-cache-safe).

Every test skips cleanly when the upstream tokenizer cannot be loaded
(network outage, gated repo, missing local checkpoint) so CI without HF Hub
access completes; the moment a tokenizer is present the invariant runs and
asserts. Known divergences are tracked per ``(renderer, scenario)`` in
:mod:`renderer_expected_divergences` and applied as
``pytest.mark.xfail(strict=True)``. Because the marks are strict, fixing a
divergence turns the xfail into an XPASS and fails the suite until the stale
entry is removed — the divergence maps can never silently rot. Set
``RENDERER_QA_STRICT=1`` (CI) to also fail when a REQUIRED public renderer's
tokenizer silently stops loading.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

# Importing the cookbook renderer package registers cookbook-local renderer
# names (glm5, gemma4, minimax_m2, nemotron, deepseek_v4, mistral,
# kimi_k27_code, ...) under tinker_cookbook.renderers.get_renderer.
import training.renderer  # noqa: F401
from tinker_cookbook.renderers import Renderer, TrainOnWhat, get_renderer
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from training.renderer.verifier.utils.hf_parity import (
    compare_renderer_to_hf,
    format_divergence,
)
from training.utils.supervised import build_tool_prefixed_messages

from training.tests.unit.renderer_expected_divergences import (
    EXTENSION_EXPECTED_DIVERGENCES,
    HF_EXPECTED_DIVERGENCES,
    HISTORICAL_PARSE_EXPECTED_DIVERGENCES,
    OBSERVATION_EXPECTED_DIVERGENCES,
    PARSE_EXPECTED_DIVERGENCES,
    TEXT_EXPECTED_DIVERGENCES,
)
from training.tests.unit.renderer_matrix import (
    REQUIRED_RENDERERS,
    RENDERER_MATRIX,
    RendererCase,
)
from training.tests.unit.renderer_scenarios import ALL_SCENARIOS, Scenario

# Exceptions raised when an upstream tokenizer / renderer cannot be
# materialized (offline, gated repo, missing chat_template, or a tokenizer
# registration issue). These map to a clean skip, not a failure.
_UNAVAILABLE = (AttributeError, OSError, ValueError, RuntimeError)


# ---------------------------------------------------------------------------
# Shared helpers (ported from tinker_cookbook.renderers.renderers_test)
# ---------------------------------------------------------------------------
def _content_text(content: Any) -> str:
    """Join the text parts of message content, ignoring non-text parts.

    Renderers may return content as a plain string or as a list of parts
    (text, thinking, tool). For round-trip comparison we care only about the
    visible text, so thinking/tool parts are dropped — this keeps the parse
    assertion robust to how each renderer structures its parsed content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
        return "".join(texts)
    return ""


def _visible_text(content: Any) -> str:
    """Extract the visible answer text from message content for comparison.

    Some renderers emit the opening ``<think>`` in the generation suffix, so
    the trained/parsed action begins with the reasoning and its ``</think>``
    close rather than a fully wrapped block. The visible answer is whatever
    follows the final ``</think>``; when there is none, the whole text is the
    answer. This normalizes both the parsed message and the scenario's
    expected text so the round-trip check compares answers, not the
    model-specific placement of thinking delimiters.
    """
    text = _content_text(content)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def _thinking_parts(content: Any) -> list[str]:
    """Return the non-empty thinking strings from structured message content.

    ``parse_response`` represents recovered reasoning as
    ``{"type": "thinking", "thinking": ...}`` parts when the renderer separates
    thinking from visible text. A plain-string content (or a list with no
    thinking part) yields ``[]``, meaning the renderer did NOT round-trip a
    structured thinking part — the signal the invariant-3 structured check
    asserts against.
    """
    if not isinstance(content, list):
        return []
    return [
        str(part.get("thinking", ""))
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "thinking"
        and part.get("thinking")
    ]


def _input_tool_calls(message: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Extract ``(name, arguments)`` for each tool call in an INPUT message.

    The input side always uses the OpenAI schema (``function.name`` plus
    ``function.arguments`` as a dict or JSON string), so argument keys are
    unambiguous here; the parsed side is the one that varies by renderer.
    """
    calls: list[tuple[str, dict[str, Any]]] = []
    for call in message.get("tool_calls") or []:
        function = call["function"]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        calls.append((function["name"], arguments))
    return calls


def _parsed_tool_call_name_and_args(tool_call: Any) -> tuple[str, Any]:
    """Extract ``(name, arguments)`` from a parsed tool call.

    ``parse_response`` returns either a Tinker ``ToolCall`` (``.function.name``
    plus a JSON-string ``.function.arguments``) or an HF-style dict. Arguments
    are returned as a dict when JSON-decodable and as the raw string otherwise
    (some renderers, e.g. gemma4, emit a non-JSON ``key:value`` argument
    encoding whose keys are still recoverable), letting the caller check key
    presence uniformly.
    """
    function = getattr(tool_call, "function", None)
    if function is not None:
        name = function.name
        raw = function.arguments
    elif isinstance(tool_call, dict):
        nested = tool_call.get("function")
        if isinstance(nested, dict):
            name = nested["name"]
            raw = nested.get("arguments", {})
        else:
            name = tool_call["name"]
            raw = tool_call.get("arguments", {})
    else:
        raise TypeError(f"Unsupported tool_call shape: {type(tool_call)!r}")
    if isinstance(raw, str):
        try:
            return name, json.loads(raw)
        except json.JSONDecodeError:
            return name, raw
    return name, raw


def _assert_arguments_round_trip(
    expected: dict[str, Any],
    recovered: Any,
    *,
    renderer_name: str,
    scenario_id: str,
) -> None:
    """Assert tool argument values survive, including nested dicts and lists.

    Most parsers return JSON-decodable dictionaries, for which exact equality
    is the contract. Gemma's parser exposes its native ``key:value`` encoding;
    for that representation, require every recursively nested scalar value and
    key to survive instead of weakening all renderers to a keys-only check.
    """
    if isinstance(recovered, dict):
        assert recovered == expected, (
            f"parsed tool arguments changed for {renderer_name} / {scenario_id}.\n"
            f"  expected: {expected!r}\n  recovered: {recovered!r}"
        )
        return

    rendered = str(recovered)

    def assert_present(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                assert str(key) in rendered
                assert_present(nested)
        elif isinstance(value, list):
            for item in value:
                assert_present(item)
        else:
            assert str(value).lower() in rendered.lower()

    assert_present(expected)


def _expected_thinking(message: dict[str, Any]) -> str:
    reasoning = message.get("reasoning_content")
    if reasoning:
        return str(reasoning).strip()
    return "".join(_thinking_parts(message.get("content"))).strip()


def _is_clean_termination(termination: Any) -> bool:
    """Normalize a renderer's termination signal to a clean/not-clean bool.

    ``parse_response`` returns either a ``ParseTermination`` (with an
    ``is_clean`` property) or, for some renderers, a plain truthy/falsy value.
    Support both per the renderer contract.
    """
    is_clean = getattr(termination, "is_clean", None)
    if is_clean is not None:
        return bool(is_clean)
    return bool(termination)


def _assert_parsed_text(
    *,
    renderer_name: str,
    scenario_id: str,
    expected_message: dict[str, Any],
    parsed_message: dict[str, Any],
) -> None:
    # For MULTI-PART text content, the visible-text join is renderer-defined
    # (gemma-4 trims each part before joining; others concatenate raw) and is
    # already verified byte-for-byte by HF parity (invariants 1&2). Comparing a
    # parsed answer to a naive raw join would flag that legitimate per-part
    # transform as a divergence, so skip the strict text-equality here (tool
    # calls / thinking are still checked by the caller).
    content = expected_message.get("content")
    text_parts = (
        sum(1 for p in content if isinstance(p, dict) and p.get("type") == "text")
        if isinstance(content, list)
        else 0
    )
    if text_parts > 1:
        return

    expected_text = _visible_text(expected_message.get("content", ""))
    parsed_text = _visible_text(parsed_message.get("content", ""))
    assert parsed_text == expected_text, (
        f"Parsed assistant text does not match for {renderer_name} / "
        f"{scenario_id}.\n  expected: {expected_text!r}\n  parsed:   {parsed_text!r}"
    )


def _assert_parsed_assistant(
    *,
    renderer_name: str,
    scenario_id: str,
    expected_message: dict[str, Any],
    parsed_message: dict[str, Any],
    action_tokens: list[int],
    tokenizer: Tokenizer,
) -> None:
    """Assert text, tool calls, and thinking all survive render then parse."""
    _assert_parsed_text(
        renderer_name=renderer_name,
        scenario_id=scenario_id,
        expected_message=expected_message,
        parsed_message=parsed_message,
    )

    expected_calls = _input_tool_calls(expected_message)
    parsed_calls = [
        _parsed_tool_call_name_and_args(tool_call)
        for tool_call in (parsed_message.get("tool_calls") or [])
    ]
    if expected_calls:
        assert len(parsed_calls) == len(expected_calls), (
            f"parse_response recovered {len(parsed_calls)} tool calls, expected "
            f"{len(expected_calls)} for {renderer_name} / {scenario_id}.\n"
            f"  action: {tokenizer.decode(action_tokens)!r}"
        )
        for (expected_name, expected_args), (parsed_name, parsed_args) in zip(
            expected_calls, parsed_calls
        ):
            assert parsed_name == expected_name, (
                f"parsed tool call order/name changed for {renderer_name} / "
                f"{scenario_id}: expected {expected_name!r}, got {parsed_name!r}"
            )
            _assert_arguments_round_trip(
                expected_args,
                parsed_args,
                renderer_name=renderer_name,
                scenario_id=scenario_id,
            )
    else:
        assert not parsed_calls, (
            f"parse_response invented tool calls for {renderer_name} / "
            f"{scenario_id}: {parsed_calls!r}"
        )

    expected_thinking = _expected_thinking(expected_message)
    parsed_thinking = "".join(
        _thinking_parts(parsed_message.get("content"))
    ).strip()
    if expected_thinking:
        # Only require the thinking to round-trip if the renderer actually
        # EMITTED it. Some templates (gemma-4) emit the thinking channel ONLY on
        # tool-calling turns and strip reasoning on plain answer turns, matching
        # apply_chat_template exactly. If the reasoning text is not in the
        # rendered action, the renderer legitimately dropped it (nothing to
        # recover) — but the parser still must not INVENT thinking.
        renderer_emitted_thinking = expected_thinking in tokenizer.decode(action_tokens)
        if renderer_emitted_thinking:
            assert parsed_thinking == expected_thinking, (
                f"parsed thinking changed for {renderer_name} / {scenario_id}.\n"
                f"  expected: {expected_thinking!r}\n"
                f"  parsed:   {parsed_thinking!r}"
            )
        else:
            assert not parsed_thinking, (
                f"parse_response produced thinking the renderer never emitted "
                f"for {renderer_name} / {scenario_id}: {parsed_thinking!r}"
            )
    else:
        assert not parsed_thinking, (
            f"parse_response invented thinking for {renderer_name} / "
            f"{scenario_id}: {parsed_thinking!r}"
        )


def _split_by_weights(
    tokens: list[int], weights: list[float]
) -> tuple[list[int], list[int]]:
    """Split tokens into (observation, action) asserting a ``000...111`` mask.

    The supervised loss mask for ``LAST_ASSISTANT_MESSAGE`` must be a run of
    zeros (the prompt the model conditions on) followed by a run of ones (the
    tokens it is trained to produce). Any interior transition back to zero
    means the mask is malformed. Returns ``(ob, ac)`` where ``ob`` are the
    weight-0 tokens and ``ac`` the weight-1 tokens.
    """
    assert len(tokens) == len(
        weights
    ), f"Token/weight length mismatch: {len(tokens)} vs {len(weights)}"

    first_nonzero: int | None = None
    for i, w in enumerate(weights):
        if w > 0:
            first_nonzero = i
            break

    if first_nonzero is None:
        return tokens, []

    for i, w in enumerate(weights):
        if i < first_nonzero:
            assert w == 0, f"Expected weight=0 at index {i}, got {w}"
        else:
            assert w == 1, f"Expected weight=1 at index {i}, got {w}"

    return tokens[:first_nonzero], tokens[first_nonzero:]


def _verify_extension_property(
    renderer: Renderer, messages: list[dict], tokenizer: Tokenizer
) -> None:
    """Assert successive assistant-boundary sequences are prefix extensions.

    For a conversation with assistant turns at t and t+1, the full rendered
    sequence through assistant t (observation + completion) must be a prefix
    of the observation before assistant t+1. This is what makes O(T) KV-cache
    reuse correct for multi-turn trajectories.
    """
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if len(assistant_indices) < 2:
        return

    for i in range(len(assistant_indices) - 1):
        asst_idx = assistant_indices[i]
        next_asst_idx = assistant_indices[i + 1]

        model_input_through_asst, _ = renderer.build_supervised_example(
            messages[: asst_idx + 1]
        )
        seq_through_asst = model_input_through_asst.to_ints()

        prompt_before_next = renderer.build_generation_prompt(
            messages[:next_asst_idx]
        ).to_ints()

        is_prefix = prompt_before_next[: len(seq_through_asst)] == seq_through_asst
        if not is_prefix:
            diverge_idx = min(len(seq_through_asst), len(prompt_before_next))
            for j in range(min(len(seq_through_asst), len(prompt_before_next))):
                if seq_through_asst[j] != prompt_before_next[j]:
                    diverge_idx = j
                    break
            raise AssertionError(
                f"Extension property violated between assistant {i} and "
                f"{i + 1}. Full sequence through asst {i} "
                f"(len={len(seq_through_asst)}) is NOT a prefix of the prompt "
                f"before asst {i + 1} (len={len(prompt_before_next)}). "
                f"Divergence at token {diverge_idx}:\n"
                f"  seq[{diverge_idx}:]:  "
                f"{seq_through_asst[diverge_idx : diverge_idx + 10]} = "
                f"{tokenizer.decode(seq_through_asst[diverge_idx : diverge_idx + 10])!r}\n"
                f"  next[{diverge_idx}:]: "
                f"{prompt_before_next[diverge_idx : diverge_idx + 10]} = "
                f"{tokenizer.decode(prompt_before_next[diverge_idx : diverge_idx + 10])!r}"
            )


def _load_renderer(case: RendererCase) -> tuple[Tokenizer, Renderer]:
    """Load the case's tokenizer and renderer, or raise to trigger a skip."""
    tokenizer = get_tokenizer(case.tokenizer_model)
    renderer = get_renderer(case.renderer, tokenizer)
    return tokenizer, renderer


def _native_tool_call_ids(
    messages: list[dict], tool_call_id_style: str | None
) -> list[dict]:
    """Rewrite tool-call ids to a renderer's native convention (round-trip legs).

    Kimi's wire format encodes the function NAME *inside* the tool-call id
    (``functions.<name>:<index>``) and carries no separate name field, so an
    OpenAI-style opaque id (``call_weather_1``) renders tokens from which the
    name is literally absent — no parser could recover it. Feeding Kimi the id
    shape its own generation emits makes the render->parse round-trip a faithful
    check of the real parse path instead of an unsatisfiable one.

    The rewrite is deterministic per assistant turn (``functions.{name}:{i}``
    for the i-th call in that turn, mirroring ``KimiK2Renderer.render_message``'s
    own id fallback), and matching ``tool`` results are re-keyed via the same
    old->new map so tool-result folding stays consistent. Slicing a scenario is
    safe: an assistant turn always enumerates to the same ids, and a call always
    precedes its result. Renderers without a declared style are returned
    unchanged (identity), so no other renderer is affected.
    """
    if tool_call_id_style != "kimi":
        return messages
    id_map: dict[str, str] = {}
    rewritten: list[dict] = []
    for message in messages:
        calls = message.get("tool_calls")
        if calls:
            new_calls = []
            for index, call in enumerate(calls):
                native = f"functions.{call['function']['name']}:{index}"
                old = call.get("id")
                if old is not None:
                    id_map[old] = native
                new_calls.append({**call, "id": native})
            rewritten.append({**message, "tool_calls": new_calls})
        elif message.get("role") == "tool" and message.get("tool_call_id") in id_map:
            rewritten.append(
                {**message, "tool_call_id": id_map[message["tool_call_id"]]}
            )
        else:
            rewritten.append(message)
    return rewritten


def _assemble(
    renderer: Renderer,
    messages: list[dict],
    tools: list[dict] | None,
    *,
    tool_call_id_style: str | None = None,
) -> list[dict]:
    """Assemble the renderer input exactly as production SFT does.

    Uses the shared ``build_tool_prefixed_messages`` from ``training.utils.supervised``
    (the same path ``render_messages_to_datums`` takes), so tool scenarios are
    rendered WITH their declaration prefix — identical to training and to the HF
    parity legs — instead of an ad-hoc harness-local assembly.

    ``tool_call_id_style`` (from the case) rewrites tool-call ids to a renderer's
    native convention first; it is a no-op for every renderer except Kimi. Only
    the invariant-3 legs route through here, so HF byte-parity is unaffected.
    """
    messages = _native_tool_call_ids(messages, tool_call_id_style)
    return build_tool_prefixed_messages(messages, renderer=renderer, tools=tools)


def _case_id(case: RendererCase, scenario: Scenario) -> str:
    return f"{case.renderer}-{scenario.id}"


def _scenario_supported(case: RendererCase, scenario: Scenario) -> bool:
    """Whether the case advertises every capability the scenario needs."""
    if scenario.requires_thinking and not case.supports_thinking:
        return False
    if scenario.requires_tools and not case.supports_tools:
        return False
    return True


# kimi_k25 injects Kimi's default system prompt whenever a conversation has no
# system message, while apply_chat_template does not — so every system-less
# scenario diverges on HF byte-parity. Real training rows carry a system turn
# (see the system_user scenario, where the two agree). Rather than enumerate
# every system-less scenario, gate this by structure.
_KIMI_DEFAULT_SYSTEM_XFAIL = (
    "kimi_k25 injects Kimi's default system prompt when a conversation has no "
    "system message; apply_chat_template does not, so system-less scenarios "
    "diverge on HF byte-parity. Covered by scenarios that carry a system turn."
)


def _lacks_system_message(scenario: Scenario) -> bool:
    return not (scenario.messages and scenario.messages[0].get("role") == "system")


def _hf_xfail_reason(case: RendererCase, scenario: Scenario) -> str | None:
    """Resolve a documented HF-parity divergence for this pair.

    Precedence (all resolve to a documented xfail reason, or ``None`` when the
    pair is expected to match):

    1. kimi_k25's structural default-system-prompt injection (gated by
       conversation shape, so one predicate covers every system-less scenario);
    2. a matrix-wide ``case.xfail_hf`` divergence;
    3. a scenario-wide ``scenario.xfail_reason`` divergence;
    4. the empirically-built per-(renderer, scenario) HF divergence map
       map — the common case, since almost every real divergence is
       renderer-specific rather than scenario-wide.
    """
    if case.renderer == "kimi_k25" and _lacks_system_message(scenario):
        return _KIMI_DEFAULT_SYSTEM_XFAIL
    return (
        case.xfail_hf
        or scenario.xfail_reason
        or HF_EXPECTED_DIVERGENCES.get((case.renderer, scenario.id))
    )


def _parse_xfail_reason(case: RendererCase, scenario: Scenario) -> str | None:
    """Resolve only renderer self-consistency divergences.

    HF formatting gaps must not suppress the independent loss-mask and parse
    checks. This distinction is especially important for Kimi's default-system
    mismatch and whitespace-only template differences.
    """
    return PARSE_EXPECTED_DIVERGENCES.get((case.renderer, scenario.id))


def _historical_parse_xfail_reason(
    case: RendererCase,
    scenario: Scenario,
    message: dict[str, Any],
) -> str | None:
    # Historical-turn parse gaps are tracked independently of final-turn parse
    # gaps (PARSE_EXPECTED_DIVERGENCES): a renderer may fail to round-trip its
    # FINAL reasoning turn yet parse an intermediate tool-call turn fine (or
    # vice versa). Only consult the historical map here so a final-turn xfail
    # cannot mask (or falsely XPASS) a historical turn.
    if not (_input_tool_calls(message) or _expected_thinking(message)):
        return None
    return HISTORICAL_PARSE_EXPECTED_DIVERGENCES.get((case.renderer, scenario.id))


def _supervised_tokens_match(result: Any) -> bool:
    """Whether supervised tokens match HF modulo the trailing-newline convention.

    ``im_end``-style renderers (Qwen, Nemotron, ...) intentionally omit the
    single trailing ``\\n`` the HF template emits after the final turn, since
    nothing is trained on it. We forgive exactly that one trailing whitespace
    token — the same tolerance the dedicated per-renderer suites apply (see
    ``test_nemotron_renderer.py`` / ``test_glm5_renderer.py``) — but any other
    difference is a genuine parity failure.
    """
    if result.match:
        return True
    renderer_tokens = result.renderer_tokens
    hf_tokens = result.hf_tokens
    if len(hf_tokens) == len(renderer_tokens) + 1 and (
        hf_tokens[: len(renderer_tokens)] == renderer_tokens
    ):
        dropped = (
            result.hf_decoded[len(renderer_tokens)]
            if len(renderer_tokens) < len(result.hf_decoded)
            else ""
        )
        return dropped.strip() == ""
    return False


def _case_scenario_param(
    case: RendererCase,
    scenario: Scenario,
    *,
    xfail_reason: str | None = None,
) -> Any:
    marks = ()
    if xfail_reason:
        marks = (pytest.mark.xfail(reason=xfail_reason, strict=True),)
    return pytest.param(
        case,
        scenario,
        id=_case_id(case, scenario),
        marks=marks,
    )


# ---------------------------------------------------------------------------
# Parameter sets (built once; capability gating is decided at collection).
# ---------------------------------------------------------------------------
# Generation parity is only meaningful for generation-shaped conversations
# (a model turn is about to be produced). Running add_generation_prompt=True on
# an assistant-terminated conversation is ill-defined and produces a spurious
# trailing-newline divergence; those scenarios are covered by supervised parity
# (invariant 2) and the consistency round-trip (invariant 3) instead.
_HF_GEN_PARAMS = [
    _case_scenario_param(
        case,
        scenario,
        xfail_reason=_hf_xfail_reason(case, scenario),
    )
    for case in RENDERER_MATRIX
    for scenario in ALL_SCENARIOS
    if case.has_hf_chat_template
    and not scenario.ends_with_assistant
    and _scenario_supported(case, scenario)
]

_HF_SUPERVISED_PARAMS = [
    _case_scenario_param(
        case,
        scenario,
        xfail_reason=_hf_xfail_reason(case, scenario),
    )
    for case in RENDERER_MATRIX
    for scenario in ALL_SCENARIOS
    if case.supervised_hf_parity
    and scenario.ends_with_assistant
    and _scenario_supported(case, scenario)
]

_CONSISTENCY_PARAMS = [
    _case_scenario_param(
        case,
        scenario,
        xfail_reason=TEXT_EXPECTED_DIVERGENCES.get(
            (case.renderer, scenario.id)
        ),
    )
    for case in RENDERER_MATRIX
    for scenario in ALL_SCENARIOS
    if scenario.ends_with_assistant and _scenario_supported(case, scenario)
]

_STRUCTURED_FINAL_PARSE_PARAMS = [
    _case_scenario_param(
        case,
        scenario,
        xfail_reason=_parse_xfail_reason(case, scenario),
    )
    for case in RENDERER_MATRIX
    for scenario in ALL_SCENARIOS
    if scenario.ends_with_assistant
    and (
        _input_tool_calls(scenario.messages[-1])
        or _expected_thinking(scenario.messages[-1])
    )
    and _scenario_supported(case, scenario)
]

_HISTORICAL_PARSE_PARAMS = [
    pytest.param(
        case,
        scenario,
        index,
        id=f"{_case_id(case, scenario)}-message-{index}",
        marks=(
            pytest.mark.xfail(reason=reason, strict=True),
        )
        if reason
        else (),
    )
    for case in RENDERER_MATRIX
    for scenario in ALL_SCENARIOS
    for index, message in enumerate(scenario.messages)
    if message.get("role") == "assistant"
    and not (scenario.ends_with_assistant and index == len(scenario.messages) - 1)
    and _scenario_supported(case, scenario)
    for reason in (_historical_parse_xfail_reason(case, scenario, message),)
]

_EXTENSION_SCENARIOS = [
    scenario
    for scenario in ALL_SCENARIOS
    if scenario.id
    in {
        "multi_turn_sft",
        "history_thinking_two_turns",
        "react_two_round_tool_results",
    }
]

_EXTENSION_PARAMS = [
    _case_scenario_param(
        case,
        scenario,
        xfail_reason=EXTENSION_EXPECTED_DIVERGENCES.get((case.renderer, scenario.id)),
    )
    for case in RENDERER_MATRIX
    for scenario in _EXTENSION_SCENARIOS
    if case.has_extension_property and _scenario_supported(case, scenario)
]

_MULTI_ASSISTANT_PARAMS = [
    pytest.param(case, scenario, id=_case_id(case, scenario))
    for case in RENDERER_MATRIX
    for scenario in ALL_SCENARIOS
    if scenario.id
    in {
        "multi_turn_sft",
        "reasoning_then_two_assistants",
        "react_two_round_tool_results",
    }
    and _scenario_supported(case, scenario)
]

_OBSERVATION_PARAMS = [
    _case_scenario_param(
        case,
        scenario,
        xfail_reason=OBSERVATION_EXPECTED_DIVERGENCES.get(
            (case.renderer, scenario.id)
        ),
    )
    for case in RENDERER_MATRIX
    for scenario in ALL_SCENARIOS
    if case.observation_equals_generation
    and scenario.ends_with_assistant
    and _scenario_supported(case, scenario)
]


# ---------------------------------------------------------------------------
# Invariant 1 — HF generation parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case,scenario", _HF_GEN_PARAMS)
@pytest.mark.timeout(180)
def test_hf_generation_parity(case: RendererCase, scenario: Scenario) -> None:
    try:
        _load_renderer(case)
    except _UNAVAILABLE as exc:
        pytest.skip(f"tokenizer unavailable for {case.tokenizer_model!r}: {exc}")

    result = compare_renderer_to_hf(
        renderer_name=case.renderer,
        tokenizer_model=case.tokenizer_model,
        messages=scenario.messages,
        add_generation_prompt=True,
        apply_chat_template_kwargs=case.hf_kwargs,
        tools=scenario.tools,
    )
    assert result.match, format_divergence(result)


# ---------------------------------------------------------------------------
# Invariant 2 — HF supervised parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case,scenario", _HF_SUPERVISED_PARAMS)
@pytest.mark.timeout(180)
def test_hf_supervised_parity(case: RendererCase, scenario: Scenario) -> None:
    try:
        _load_renderer(case)
    except _UNAVAILABLE as exc:
        pytest.skip(f"tokenizer unavailable for {case.tokenizer_model!r}: {exc}")

    result = compare_renderer_to_hf(
        renderer_name=case.renderer,
        tokenizer_model=case.tokenizer_model,
        messages=scenario.messages,
        add_generation_prompt=False,
        apply_chat_template_kwargs=case.hf_kwargs,
        tools=scenario.tools,
    )
    assert _supervised_tokens_match(result), format_divergence(result)


# ---------------------------------------------------------------------------
# Invariant 3 — supervised <-> generation <-> parse consistency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case,scenario", _CONSISTENCY_PARAMS)
@pytest.mark.timeout(180)
def test_supervised_generation_parse_consistency(
    case: RendererCase, scenario: Scenario
) -> None:
    try:
        tokenizer, renderer = _load_renderer(case)
    except _UNAVAILABLE as exc:
        pytest.skip(f"tokenizer unavailable for {case.tokenizer_model!r}: {exc}")

    normalized = _assemble(
        renderer,
        scenario.messages,
        scenario.tools,
        tool_call_id_style=case.tool_call_id_style,
    )
    model_input, weights = renderer.build_supervised_example(normalized)
    tokens = [int(t) for t in model_input.to_ints()]
    weights_list = [float(w) for w in weights.tolist()]

    _, ac = _split_by_weights(tokens, weights_list)
    assert ac, "Expected a non-empty trained action span for the last assistant turn"

    parsed_message, termination = renderer.parse_response(ac)
    assert _is_clean_termination(termination), (
        f"parse_response did not terminate cleanly for {case.renderer} / "
        f"{scenario.id}. Decoded action: {tokenizer.decode(ac)!r}; "
        f"parsed: {parsed_message}"
    )

    final_message = scenario.messages[-1]
    if _input_tool_calls(final_message) or _expected_thinking(final_message):
        _assert_parsed_text(
            renderer_name=case.renderer,
            scenario_id=scenario.id,
            expected_message=final_message,
            parsed_message=parsed_message,
        )
    else:
        _assert_parsed_assistant(
            renderer_name=case.renderer,
            scenario_id=scenario.id,
            expected_message=final_message,
            parsed_message=parsed_message,
            action_tokens=ac,
            tokenizer=tokenizer,
        )


@pytest.mark.parametrize("case,scenario", _STRUCTURED_FINAL_PARSE_PARAMS)
@pytest.mark.timeout(180)
def test_final_structured_assistant_content_round_trips(
    case: RendererCase, scenario: Scenario
) -> None:
    """Track tool/thinking parse gaps without masking loss-mask contracts."""
    try:
        tokenizer, renderer = _load_renderer(case)
    except _UNAVAILABLE as exc:
        pytest.skip(f"tokenizer unavailable for {case.tokenizer_model!r}: {exc}")

    model_input, weights = renderer.build_supervised_example(
        _assemble(
            renderer,
            scenario.messages,
            scenario.tools,
            tool_call_id_style=case.tool_call_id_style,
        )
    )
    _, action = _split_by_weights(
        [int(token) for token in model_input.to_ints()],
        [float(weight) for weight in weights.tolist()],
    )
    assert action, "Expected a non-empty trained action span for the last assistant turn"
    parsed_message, termination = renderer.parse_response(action)
    assert _is_clean_termination(termination)
    _assert_parsed_assistant(
        renderer_name=case.renderer,
        scenario_id=scenario.id,
        expected_message=scenario.messages[-1],
        parsed_message=parsed_message,
        action_tokens=action,
        tokenizer=tokenizer,
    )


@pytest.mark.parametrize("case,scenario", _OBSERVATION_PARAMS)
@pytest.mark.timeout(180)
def test_supervised_observation_equals_generation_prompt(
    case: RendererCase, scenario: Scenario
) -> None:
    """Check the observation leg independently so parse xfails cannot hide it."""
    try:
        tokenizer, renderer = _load_renderer(case)
    except _UNAVAILABLE as exc:
        pytest.skip(f"tokenizer unavailable for {case.tokenizer_model!r}: {exc}")

    normalized = _assemble(
        renderer,
        scenario.messages,
        scenario.tools,
        tool_call_id_style=case.tool_call_id_style,
    )
    model_input, weights = renderer.build_supervised_example(normalized)
    observation, action = _split_by_weights(
        [int(token) for token in model_input.to_ints()],
        [float(weight) for weight in weights.tolist()],
    )
    assert action, "Expected a non-empty trained action span for the last assistant turn"
    generation = [
        int(token)
        for token in renderer.build_generation_prompt(
            _assemble(
                renderer,
                scenario.messages[:-1],
                scenario.tools,
                tool_call_id_style=case.tool_call_id_style,
            )
        ).to_ints()
    ]
    assert observation == generation, (
        "Observation tokens do not match the generation prompt for "
        f"{case.renderer} / {scenario.id}.\n"
        f"  observation ({len(observation)} tok): {tokenizer.decode(observation)!r}\n"
        f"  generation  ({len(generation)} tok): {tokenizer.decode(generation)!r}"
    )


@pytest.mark.parametrize("case,scenario,index", _HISTORICAL_PARSE_PARAMS)
@pytest.mark.timeout(180)
def test_historical_assistant_turns_parse_consistently(
    case: RendererCase, scenario: Scenario, index: int
) -> None:
    """Parse every historical assistant turn, not only the final answer.

    Multi-round tool scenarios often end in plain text, so checking only the
    final action would never exercise the parser on their intermediate tool
    calls or reasoning turns.
    """
    try:
        tokenizer, renderer = _load_renderer(case)
    except _UNAVAILABLE as exc:
        pytest.skip(f"tokenizer unavailable for {case.tokenizer_model!r}: {exc}")

    expected_message = scenario.messages[index]
    model_input, weights = renderer.build_supervised_example(
        _assemble(
            renderer,
            scenario.messages[: index + 1],
            scenario.tools,
            tool_call_id_style=case.tool_call_id_style,
        ),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    _, action = _split_by_weights(
        [int(token) for token in model_input.to_ints()],
        [float(weight) for weight in weights.tolist()],
    )
    assert action, (
        f"empty historical assistant action for {case.renderer} / "
        f"{scenario.id} at message {index}"
    )
    parsed_message, termination = renderer.parse_response(action)
    assert _is_clean_termination(termination), (
        f"historical assistant did not terminate cleanly for {case.renderer} / "
        f"{scenario.id} at message {index}: {tokenizer.decode(action)!r}"
    )
    _assert_parsed_assistant(
        renderer_name=case.renderer,
        scenario_id=f"{scenario.id}[{index}]",
        expected_message=expected_message,
        parsed_message=parsed_message,
        action_tokens=action,
        tokenizer=tokenizer,
    )


@pytest.mark.parametrize("case,scenario", _MULTI_ASSISTANT_PARAMS)
@pytest.mark.timeout(180)
def test_all_assistant_messages_loss_mask_is_a_superset(
    case: RendererCase, scenario: Scenario
) -> None:
    """ALL_ASSISTANT_MESSAGES must train history in addition to the last turn."""
    try:
        _, renderer = _load_renderer(case)
    except _UNAVAILABLE as exc:
        pytest.skip(f"tokenizer unavailable for {case.tokenizer_model!r}: {exc}")

    normalized = _assemble(
        renderer,
        scenario.messages,
        scenario.tools,
        tool_call_id_style=case.tool_call_id_style,
    )
    last_input, last_weights = renderer.build_supervised_example(
        normalized,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    all_input, all_weights = renderer.build_supervised_example(
        normalized,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert list(last_input.to_ints()) == list(all_input.to_ints()), (
        f"changing only train_on_what changed rendered tokens for {case.renderer} / "
        f"{scenario.id}"
    )

    last = [float(weight) for weight in last_weights.tolist()]
    all_assistant = [float(weight) for weight in all_weights.tolist()]
    assert set(last) <= {0.0, 1.0}
    assert set(all_assistant) <= {0.0, 1.0}
    assert all(
        all_weight >= last_weight
        for last_weight, all_weight in zip(last, all_assistant)
    ), (
        f"ALL_ASSISTANT_MESSAGES untrained a token selected by "
        f"LAST_ASSISTANT_MESSAGE for {case.renderer} / {scenario.id}"
    )
    assert sum(all_assistant) > sum(last), (
        f"ALL_ASSISTANT_MESSAGES did not add any historical assistant tokens for "
        f"{case.renderer} / {scenario.id}"
    )


# ---------------------------------------------------------------------------
# Invariant 4 — sequence extension property
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case,scenario", _EXTENSION_PARAMS)
@pytest.mark.timeout(180)
def test_sequence_extension_property(
    case: RendererCase, scenario: Scenario
) -> None:
    try:
        tokenizer, renderer = _load_renderer(case)
    except _UNAVAILABLE as exc:
        pytest.skip(f"tokenizer unavailable for {case.tokenizer_model!r}: {exc}")

    assert renderer.has_extension_property, (
        f"{case.renderer} is listed with has_extension_property=True but its "
        "renderer reports False"
    )

    messages = _assemble(
        renderer,
        scenario.messages,
        scenario.tools,
        tool_call_id_style=case.tool_call_id_style,
    )
    _verify_extension_property(renderer, messages, tokenizer)


# ---------------------------------------------------------------------------
# Harness integrity — prevent stale ids, flags, and divergence entries
# ---------------------------------------------------------------------------
def test_scenario_metadata_is_self_consistent() -> None:
    scenario_ids = [scenario.id for scenario in ALL_SCENARIOS]
    assert len(scenario_ids) == len(set(scenario_ids)), "scenario ids must be unique"

    for scenario in ALL_SCENARIOS:
        assert scenario.messages, f"{scenario.id}: messages must not be empty"
        assert scenario.ends_with_assistant == (
            scenario.messages[-1].get("role") == "assistant"
        ), f"{scenario.id}: ends_with_assistant is stale"

        # A scenario is tool-shaped if it carries tool calls / tool results OR
        # declares tools (a declaration-only scenario still needs a renderer
        # that can serialize the tool block).
        has_tools = bool(scenario.tools) or any(
            message.get("tool_calls") or message.get("role") == "tool"
            for message in scenario.messages
        )
        assert scenario.requires_tools == has_tools, (
            f"{scenario.id}: requires_tools={scenario.requires_tools}, "
            f"but tool-shaped (messages or tools=)={has_tools}"
        )

        has_thinking = any(
            message.get("reasoning_content")
            or _thinking_parts(message.get("content"))
            for message in scenario.messages
        )
        assert scenario.requires_thinking == has_thinking, (
            f"{scenario.id}: requires_thinking={scenario.requires_thinking}, "
            f"but thinking-shaped messages={has_thinking}"
        )


def test_expected_divergence_entries_reference_the_matrix() -> None:
    renderer_names = {case.renderer for case in RENDERER_MATRIX}
    scenario_ids = {scenario.id for scenario in ALL_SCENARIOS}
    # Maps are intentionally NOT disjoint: a pair can diverge on more than one
    # invariant (e.g. a renderer whose tool declaration diverges from HF *and*
    # whose parser does not read the tool call back). Each map still gates a
    # distinct invariant, so the only integrity requirement is that every entry
    # references a real (renderer, scenario) in the matrix/bank.
    divergence_maps = (
        HF_EXPECTED_DIVERGENCES,
        PARSE_EXPECTED_DIVERGENCES,
        HISTORICAL_PARSE_EXPECTED_DIVERGENCES,
        TEXT_EXPECTED_DIVERGENCES,
        OBSERVATION_EXPECTED_DIVERGENCES,
        EXTENSION_EXPECTED_DIVERGENCES,
    )
    for renderer_name, scenario_id in (
        set().union(*(set(divergences) for divergences in divergence_maps))
    ):
        assert renderer_name in renderer_names, f"unknown renderer {renderer_name!r}"
        assert scenario_id in scenario_ids, f"unknown scenario {scenario_id!r}"


# ---------------------------------------------------------------------------
# Coverage guard — public renderers must not silently drop to skip
# ---------------------------------------------------------------------------
# By default the invariant tests skip when a tokenizer can't load (offline dev
# boxes stay green). That resilience hides a real failure mode: if a *public*
# model's tokenizer stops loading (HF outage, renamed repo, tokenizer-class
# break), its coverage silently vanishes while CI stays green. This guard turns
# that into a loud failure — but only in strict mode, so it never flakes on an
# offline dev box.
#
# Enable in CI with RENDERER_QA_STRICT=1 (any of 1/true/yes). Preview renderers
# (kimi_k27_code and deepseek_v4) are intentionally not REQUIRED and are exempt.
_STRICT_MODE = os.environ.get("RENDERER_QA_STRICT", "").lower() in ("1", "true", "yes")

_REQUIRED_CASES = [c for c in RENDERER_MATRIX if c.renderer in REQUIRED_RENDERERS]


@pytest.mark.skipif(
    not _STRICT_MODE,
    reason="coverage guard runs only under RENDERER_QA_STRICT=1 (set in CI)",
)
@pytest.mark.parametrize(
    "case", _REQUIRED_CASES, ids=[c.renderer for c in _REQUIRED_CASES]
)
@pytest.mark.timeout(180)
def test_required_renderers_load_in_strict_mode(case: RendererCase) -> None:
    """A REQUIRED (public, ungated) renderer's tokenizer MUST load in strict
    mode. Failing here means a public model lost coverage — fix the tokenizer
    resolution rather than letting the invariant tests skip it into a false
    green.
    """
    tokenizer, renderer = _load_renderer(case)
    assert tokenizer is not None
    assert renderer is not None
    assert renderer.has_extension_property == case.has_extension_property, (
        f"{case.renderer}: matrix has_extension_property="
        f"{case.has_extension_property}, renderer reports "
        f"{renderer.has_extension_property}"
    )

    canonical = next(scenario for scenario in ALL_SCENARIOS if scenario.id == "system_user")
    result = compare_renderer_to_hf(
        renderer_name=case.renderer,
        tokenizer_model=case.tokenizer_model,
        messages=canonical.messages,
        add_generation_prompt=True,
        apply_chat_template_kwargs=case.hf_kwargs,
        tools=canonical.tools,
    )
    expected_gap = _hf_xfail_reason(case, canonical)
    if expected_gap:
        assert not result.match, (
            f"{case.renderer}'s strict smoke test now passes; remove its stale "
            f"expected divergence: {expected_gap}"
        )
        return
    assert result.match, (
        f"{case.renderer} loaded but failed the canonical HF parity smoke test:\n"
        f"{format_divergence(result)}"
    )
