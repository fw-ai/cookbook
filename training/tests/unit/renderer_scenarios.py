"""Reusable conversation scenarios for the renderer QA harness.

This module is the *data* half of the property-based renderer suite. It
holds a bank of OpenAI-style message dicts (``list[dict]``) shaped to
exercise the structural, thinking, and tool-calling behaviours every
cookbook renderer must get right, plus the metadata the harness needs to
decide which invariants each scenario is eligible for.

Design intent / tradeoffs:

* Scenarios are plain OpenAI-style dicts (``role``/``content``, optional
  ``reasoning_content`` and ``tool_calls``) so a single bank feeds both
  ``tokenizer.apply_chat_template`` (HF byte-parity) and
  ``training.utils.supervised.normalize_messages`` (the renderer path).
  Keeping one representation avoids drift between the two comparison
  legs.
* **Every** scenario is byte-compared against ``apply_chat_template``.
  There is no ``hf_safe`` opt-out: tool and thinking scenarios go through
  the same HF-parity contract as structural ones. Tool *declarations* are
  rendered on both sides — the HF reference receives ``tools=`` and the
  renderer receives the block from
  ``create_conversation_prefix_with_tools`` (see
  ``renderer.verifier.utils.hf_parity``) — so a renderer that serializes a
  tool declaration or a chain-of-thought differently from its own template
  is a divergence we surface (as a tracked ``xfail``), not one we hide by
  exempting the scenario shape. ``tools`` on a scenario names the
  declaration schemas passed to both sides.

Content is intentionally short but structurally meaningful: enough to
tokenize into several tokens and hit real boundaries (system prompts,
multi-turn history, unicode, code fences, control-token-looking text)
without bloating CI runtime.

Scenario catalog (keep in sync with the lists below):

Structural (role layout, whitespace, unicode, control text):

* ``single_user`` — one bare user turn (generation prompt, no assistant).
* ``system_user`` — a leading system turn followed by a user turn.
* ``multi_turn_gen`` — system + user/assistant history ending on a user turn.
* ``multi_turn_sft`` — full multi-turn conversation ending on an assistant turn.
* ``empty_system`` — an empty-string system message ahead of a user turn.
* ``whitespace_trim`` — user content wrapped in spaces (probes Jinja ``| trim``).
* ``internal_newlines`` — assistant answer with a blank line kept intact.
* ``unicode_emoji`` — assistant answer mixing CJK, emoji, and accented text.
* ``literal_control_text`` — user text containing literal control-token strings.
* ``long_content`` — a long repeated user prompt with a short assistant answer.
* ``code_block`` — assistant answer containing a fenced Python code block.
* ``leading_newline_user`` — user content with leading newlines (header merge).
* ``trailing_newline_user`` — user content with a trailing newline.
* ``tab_indented_user`` — tab-wrapped user content.
* ``whitespace_only_user`` — user content that is only whitespace.
* ``multipart_user_text`` — user content as a list of text parts (join vs drop).
* ``multipart_assistant_text`` — assistant content as a list of text parts.
* ``developer_first`` — a leading ``developer`` role (remap-to-system or reject).
* ``developer_midconversation`` — a ``developer`` role after the first user turn.
* ``consecutive_system`` — two system messages back to back.
* ``mid_conversation_system`` — a system message after a user turn.
* ``consecutive_user`` — two user turns back to back.
* ``consecutive_assistant`` — two assistant turns back to back.
* ``empty_assistant_content`` — a terminal assistant turn with empty content.
* ``whitespace_only_assistant`` — a terminal assistant containing only whitespace.

Thinking (chain-of-thought via ``reasoning_content``):

* ``single_thinking`` — one terminal assistant turn carrying reasoning + answer.
* ``history_thinking`` — reasoning confined to history, ending on a user turn
  (strip-mode drops it, preserve-mode keeps it).
* ``reasoning_then_two_assistants`` — two assistant turns each carrying
  reasoning, ending on the terminal reasoning turn.
* ``history_thinking_two_turns`` — reasoning across two historical assistant
  turns, ending on a user turn (multi-turn strip/preserve).
* ``reasoning_only`` — terminal reasoning with an empty visible answer.
* ``multipart_thinking_text`` — structured thinking and text content parts.

Tool (tool calls and tool results):

* ``single_tool_call`` — one terminal assistant tool call.
* ``tool_call_then_answer`` — tool call, tool result, then an assistant answer.
* ``dangling_tool_call`` — a terminal tool call (nested args) with no result.
* ``react_two_round_tool_results`` — two full ReAct rounds (call -> result ->
  answer), exercising multi-round tool history.
* ``parallel_tool_calls`` — one assistant turn requesting two tools in parallel,
  both results, then a summarizing answer.
* ``assistant_content_plus_tool_call`` — a terminal assistant turn carrying both
  visible content and a tool call.
* ``nested_tool_call_args`` — a terminal tool call whose arguments nest a dict
  and a list.
* ``tool_only_empty_content`` — a terminal tool call with no narration.
* ``json_string_tool_args`` — a terminal tool call whose arguments are JSON text.
* ``dict_tool_args`` — the DICT-arguments twin of ``json_string_tool_args`` (the
  template renders dict vs string args differently; controlled A/B pair).
* ``tool_with_response_schema`` — declares a tool whose schema carries a
  ``response`` return-type block, exercising the template's ``response:``
  declaration branch (generation-shaped).
* ``parallel_tool_results_reversed`` — parallel results returned out of call order.
* ``tool_result_pending_answer`` — call + result, ending on the result so the
  generation prompt is exercised (generation-shaped tool byte-parity).
* ``empty_args_tool_call`` — a tool call with an empty ``{}`` argument object.
* ``nested_json_tool_result`` — a tool result whose content is nested JSON.

Thinking + tool combinations (reasoning and tool calls in one turn):

* ``thinking_tool_call`` — a terminal assistant turn carrying both reasoning
  and a tool call.
* ``thinking_tool_then_answer`` — reasoning + tool call, a tool result, then a
  reasoning + answer turn (both assistant turns carry chain-of-thought).
* ``reasoning_only_tool_call`` — reasoning with no visible text, then a call.
* ``thinking_parallel_tool_calls`` — reasoning followed by two parallel calls.
* ``multipart_content_tool_call`` — multi-part text content plus a tool call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Scenario:
    """A single reusable conversation plus its harness eligibility metadata.

    Attributes:
        id: Stable identifier used in pytest parametrization ids.
        messages: OpenAI-style message dicts for this conversation.
        ends_with_assistant: True when the final message is an assistant
            turn, making the scenario eligible for the supervised HF-parity
            and supervised<->generation<->parse round-trip invariants.
        requires_thinking: Scenario carries reasoning content; only run it
            against renderers whose case advertises thinking support.
        requires_tools: Scenario carries tool calls / tool results; only
            run it against renderers whose case advertises tool support.
        tools: Tool declaration schemas (OpenAI function-calling shape) to
            pass to ``apply_chat_template(tools=...)`` and to the renderer's
            ``create_conversation_prefix_with_tools`` so tool *declarations*
            are byte-compared too. ``None`` means no declarations are passed.
        xfail_reason: When set, the HF-parity invariant is xfailed with
            this reason for every case (a documented, scenario-wide known
            divergence).
    """

    id: str
    messages: list[dict]
    ends_with_assistant: bool
    requires_thinking: bool = False
    requires_tools: bool = False
    tools: list[dict] | None = None
    xfail_reason: str | None = None


# ---------------------------------------------------------------------------
# Tool declarations (OpenAI function-calling schema)
# ---------------------------------------------------------------------------
# A flat, typed tool and a nested-argument tool. These are the declaration
# schemas a caller would pass as ``tools=`` at inference; scenarios below
# reference these names in their ``tool_calls``.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create a calendar event with attendees and a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "location": {
                        "type": "object",
                        "properties": {
                            "building": {"type": "string"},
                            "room": {"type": "string"},
                        },
                    },
                },
                "required": ["title"],
            },
        },
    },
]


# A tool declaration that also carries a ``response`` return-type schema. The
# gemma-4 template has a dedicated ``response:`` branch in its declaration macro
# (and gemma4's ``_format_function_declaration`` mirrors it); the ``TOOLS`` above
# never exercise it, so ``tool_with_response_schema`` pins that branch.
RESPONSE_SCHEMA_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
        "response": {
            "type": "object",
            "description": "The weather result.",
            "properties": {"temp": {"type": "string"}},
        },
    },
}


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------
def _system(content: str) -> dict:
    """Build a system-role message."""
    return {"role": "system", "content": content}


def _user(content: str) -> dict:
    """Build a user-role message."""
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    """Build a plain assistant-role message."""
    return {"role": "assistant", "content": content}


def _developer(content: str) -> dict:
    """Build a developer-role message.

    The ``developer`` role is an OpenAI-lineage role most templates remap to
    ``system`` (or reject). It is deliberately kept here to exercise how each
    renderer handles a role its Jinja template may not know about.
    """
    return {"role": "developer", "content": content}


def _text_parts(*parts: str) -> list[dict[str, str]]:
    """Build a multi-part text content list (OpenAI content-parts schema)."""
    return [{"type": "text", "text": part} for part in parts]


def _user_parts(*parts: str) -> dict:
    """Build a user-role message whose content is a list of text parts."""
    return {"role": "user", "content": _text_parts(*parts)}


def _assistant_parts(*parts: str) -> dict:
    """Build an assistant-role message whose content is a list of text parts."""
    return {"role": "assistant", "content": _text_parts(*parts)}


def _assistant_thinking(reasoning: str, content: str) -> dict:
    """Build an assistant turn carrying a CoT via ``reasoning_content``.

    ``normalize_messages`` promotes ``reasoning_content`` into a
    ``ThinkingPart`` for renderers that understand thinking, so this is the
    portable way to express chain-of-thought across the cookbook renderers.
    """
    return {"role": "assistant", "reasoning_content": reasoning, "content": content}


def _assistant_tool_call(
    content: str,
    *,
    name: str,
    arguments: dict[str, Any] | str,
    call_id: str,
) -> dict:
    """Build an assistant turn that emits a single OpenAI-style tool call."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _assistant_tool_calls(
    content: str,
    *,
    calls: list[tuple[str, dict[str, Any], str]],
) -> dict:
    """Build an assistant turn emitting several parallel tool calls.

    ``calls`` is a list of ``(name, arguments, call_id)`` triples. Used to
    exercise the parallel-tool-call shape where one assistant turn requests
    multiple functions before any results come back.
    """
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            for name, arguments, call_id in calls
        ],
    }


def _assistant_thinking_tool_call(
    content: str,
    *,
    reasoning: str,
    name: str,
    arguments: dict[str, Any],
    call_id: str,
) -> dict:
    """Build an assistant turn carrying BOTH chain-of-thought and a tool call.

    Exercises the reasoning+tool combination in one message: the renderer must
    emit the thinking channel and the tool call together and read both back.
    """
    return {
        "role": "assistant",
        "reasoning_content": reasoning,
        "content": content,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _tool_result(
    content: str, *, call_id: str, name: str = "get_weather"
) -> dict:
    """Build a tool-result message answering a prior tool call."""
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": call_id,
        "name": name,
    }


# ---------------------------------------------------------------------------
# Structural scenarios — role layout, whitespace, unicode, control text.
# ---------------------------------------------------------------------------
_LONG_CONTENT = (
    "Summarize the following note in one sentence. "
    + (
        "The quick brown fox "
        "jumps over the lazy dog while the diligent engineer reviews yet another "
        "tokenizer edge case. "
    )
    * 30
).strip()

_CODE_BLOCK_ANSWER = (
    "Here is a minimal example:\n\n"
    "```python\n"
    "def add(a: int, b: int) -> int:\n"
    "    return a + b\n"
    "```"
)

STRUCTURAL_SCENARIOS: list[Scenario] = [
    Scenario(
        id="single_user",
        messages=[_user("What is 2 + 2?")],
        ends_with_assistant=False,
    ),
    Scenario(
        id="system_user",
        messages=[
            _system("Answer with a single integer and nothing else."),
            _user("2 + 2 = ?"),
        ],
        ends_with_assistant=False,
    ),
    Scenario(
        id="multi_turn_gen",
        messages=[
            _system("Answer briefly."),
            _user("What is 2+2?"),
            _assistant("4."),
            _user("And 3+3?"),
        ],
        ends_with_assistant=False,
    ),
    Scenario(
        id="multi_turn_sft",
        messages=[
            _system("Answer briefly."),
            _user("What is 2+2?"),
            _assistant("4."),
            _user("And 3+3?"),
            _assistant("6."),
        ],
        ends_with_assistant=True,
    ),
    Scenario(
        id="empty_system",
        messages=[_system(""), _user("Hello there.")],
        ends_with_assistant=False,
    ),
    Scenario(
        id="whitespace_trim",
        messages=[
            _system("Be concise."),
            _user("   surrounded by spaces   "),
        ],
        ends_with_assistant=False,
    ),
    Scenario(
        id="internal_newlines",
        messages=[
            _user("Give me a two-line couplet."),
            _assistant("Roses are red,\n\nviolets are blue."),
        ],
        ends_with_assistant=True,
    ),
    Scenario(
        id="unicode_emoji",
        messages=[
            _user("Reply with a friendly greeting and an emoji."),
            _assistant("Hello, 世界! 👋🌍 café naïve résumé."),
        ],
        ends_with_assistant=True,
    ),
    Scenario(
        id="literal_control_text",
        messages=[
            _user(
                "Explain what this literal text means: <|im_end|> and <think> "
                "and <|endoftext|> are just characters here, not real tokens."
            ),
        ],
        ends_with_assistant=False,
    ),
    Scenario(
        id="long_content",
        messages=[
            _user(_LONG_CONTENT),
            _assistant("A fox jumps while an engineer reviews tokenizer edge cases."),
        ],
        ends_with_assistant=True,
    ),
    Scenario(
        id="code_block",
        messages=[
            _user("Show me how to add two integers in Python."),
            _assistant(_CODE_BLOCK_ANSWER),
        ],
        ends_with_assistant=True,
    ),
    # -- Whitespace-at-content-boundary edge cases ------------------------
    # These probe templates that apply Jinja ``| trim`` (or per-part
    # trimming) to message content: the renderer must match whatever the
    # canonical template does with leading/trailing/tab/whitespace-only text.
    Scenario(
        id="leading_newline_user",
        messages=[_user("\n\nHello there.")],
        ends_with_assistant=False,
    ),
    Scenario(
        id="trailing_newline_user",
        messages=[_user("Hello there.\n")],
        ends_with_assistant=False,
    ),
    Scenario(
        id="tab_indented_user",
        messages=[_user("\tindented\t")],
        ends_with_assistant=False,
    ),
    Scenario(
        id="whitespace_only_user",
        messages=[_user("   \n  ")],
        ends_with_assistant=False,
    ),
    # -- Multi-part text content -----------------------------------------
    # OpenAI content-parts lists; templates differ on whether they join
    # parts verbatim or trim each part before joining.
    Scenario(
        id="multipart_user_text",
        messages=[_user_parts("part one ", "part two")],
        ends_with_assistant=False,
    ),
    Scenario(
        id="multipart_assistant_text",
        messages=[
            _user("Say two things."),
            _assistant_parts("first thing ", "second thing"),
        ],
        ends_with_assistant=True,
    ),
    # -- Developer role ---------------------------------------------------
    # Most templates remap ``developer`` to ``system`` (or raise); the
    # renderer must agree with whatever the canonical template does.
    Scenario(
        id="developer_first",
        messages=[
            _developer("Be extremely terse."),
            _user("What is 2+2?"),
        ],
        ends_with_assistant=False,
    ),
    Scenario(
        id="developer_midconversation",
        messages=[
            _user("What is 2+2?"),
            _developer("From now on, answer with a single digit."),
            _user("And 3+3?"),
        ],
        ends_with_assistant=False,
    ),
    # -- Unusual role layouts --------------------------------------------
    # Consecutive same-role turns and mid-conversation system messages are
    # legal OpenAI input but many templates only expect strict alternation.
    Scenario(
        id="consecutive_system",
        messages=[
            _system("You are a helpful assistant."),
            _system("Always answer in one word."),
            _user("What is 2+2?"),
        ],
        ends_with_assistant=False,
    ),
    Scenario(
        id="mid_conversation_system",
        messages=[
            _user("What is 2+2?"),
            _system("Now switch to answering with words, not digits."),
            _user("What is 3+3?"),
        ],
        ends_with_assistant=False,
    ),
    Scenario(
        id="consecutive_user",
        messages=[
            _user("What is 2+2?"),
            _user("Actually, what is 3+3?"),
        ],
        ends_with_assistant=False,
    ),
    Scenario(
        id="consecutive_assistant",
        messages=[
            _user("Give me two facts."),
            _assistant("The sky is blue."),
            _assistant("Grass is green."),
        ],
        ends_with_assistant=True,
    ),
    Scenario(
        id="empty_assistant_content",
        messages=[
            _user("Reply with nothing."),
            _assistant(""),
        ],
        ends_with_assistant=True,
    ),
    Scenario(
        id="whitespace_only_assistant",
        messages=[
            _user("Reply with whitespace only."),
            _assistant("   \n  "),
        ],
        ends_with_assistant=True,
    ),
]


# ---------------------------------------------------------------------------
# Thinking scenarios — chain-of-thought via ``reasoning_content``.
# ---------------------------------------------------------------------------
THINKING_SCENARIOS: list[Scenario] = [
    Scenario(
        id="single_thinking",
        messages=[
            _user("What is 12 * 12?"),
            _assistant_thinking(
                "12 times 12 is 144, a standard multiplication.",
                "144.",
            ),
        ],
        ends_with_assistant=True,
        requires_thinking=True,
    ),
    Scenario(
        id="history_thinking",
        messages=[
            _user("What is 2+2?"),
            _assistant_thinking("2+2 is basic arithmetic.", "4."),
            _user("And 3+3?"),
        ],
        ends_with_assistant=False,
        requires_thinking=True,
    ),
    # Two assistant turns each carrying reasoning, ending on a terminal
    # reasoning turn: exercises how history reasoning is stripped/preserved
    # across multiple turns AND the terminal reasoning turn.
    Scenario(
        id="reasoning_then_two_assistants",
        messages=[
            _user("What is 2+2?"),
            _assistant_thinking("2+2 is basic arithmetic.", "4."),
            _user("And 3+3?"),
            _assistant_thinking("3+3 is also basic arithmetic.", "6."),
        ],
        ends_with_assistant=True,
        requires_thinking=True,
    ),
    # Reasoning confined to HISTORY only (final turn is a user prompt), so
    # strip-mode renderers drop it and preserve-mode renderers keep it,
    # mirroring the canonical template.
    Scenario(
        id="history_thinking_two_turns",
        messages=[
            _system("Answer briefly."),
            _user("What is 2+2?"),
            _assistant_thinking("2+2 is basic arithmetic.", "4."),
            _user("And 3+3?"),
            _assistant_thinking("3+3 is basic arithmetic too.", "6."),
            _user("And 4+4?"),
        ],
        ends_with_assistant=False,
        requires_thinking=True,
    ),
    Scenario(
        id="reasoning_only",
        messages=[
            _user("Think through the answer without a visible response."),
            _assistant_thinking("The requested result is intentionally hidden.", ""),
        ],
        ends_with_assistant=True,
        requires_thinking=True,
    ),
    Scenario(
        id="multipart_thinking_text",
        messages=[
            _user("What is 3 * 3?"),
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Three groups of three make nine."},
                    {"type": "text", "text": "9."},
                ],
            },
        ],
        ends_with_assistant=True,
        requires_thinking=True,
    ),
]


# ---------------------------------------------------------------------------
# Tool scenarios — tool calls and tool results. Declarations (``tools=TOOLS``)
# are byte-compared against ``apply_chat_template(tools=...)`` too.
# ---------------------------------------------------------------------------
TOOL_SCENARIOS: list[Scenario] = [
    Scenario(
        id="single_tool_call",
        messages=[
            _user("What's the weather in San Francisco?"),
            _assistant_tool_call(
                "Let me check the weather for you.",
                name="get_weather",
                arguments={"city": "San Francisco", "unit": "celsius"},
                call_id="call_weather_1",
            ),
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    Scenario(
        id="tool_call_then_answer",
        messages=[
            _user("What's the weather in San Francisco?"),
            _assistant_tool_call(
                "Checking now.",
                name="get_weather",
                arguments={"city": "San Francisco"},
                call_id="call_weather_2",
            ),
            _tool_result("18°C and sunny.", call_id="call_weather_2"),
            _assistant("It's 18°C and sunny in San Francisco."),
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    Scenario(
        id="dangling_tool_call",
        messages=[
            _user("Schedule a team sync tomorrow with Alice and Bob."),
            _assistant_tool_call(
                "Creating the event.",
                name="create_event",
                arguments={
                    "title": "Team Sync",
                    "attendees": ["alice", "bob"],
                    "location": {"building": "HQ", "room": "101"},
                },
                call_id="call_event_1",
            ),
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    # Full two-round ReAct trajectory: each round is a tool call followed by
    # a tool result and an assistant answer. Exercises multi-round tool
    # history handling and the weight mask over the final assistant turn.
    Scenario(
        id="react_two_round_tool_results",
        messages=[
            _system("You can call tools to answer questions."),
            _user("What's the weather in San Francisco, then in Paris?"),
            _assistant_tool_call(
                "Checking San Francisco first.",
                name="get_weather",
                arguments={"city": "San Francisco", "unit": "celsius"},
                call_id="call_react_1",
            ),
            _tool_result("18°C and sunny.", call_id="call_react_1"),
            _assistant("It's 18°C and sunny in San Francisco."),
            _user("Now Paris."),
            _assistant_tool_call(
                "Checking Paris now.",
                name="get_weather",
                arguments={"city": "Paris", "unit": "celsius"},
                call_id="call_react_2",
            ),
            _tool_result("12°C and cloudy.", call_id="call_react_2"),
            _assistant("It's 12°C and cloudy in Paris."),
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    # One assistant turn requesting TWO tools in parallel, both results, then
    # a summarizing answer.
    Scenario(
        id="parallel_tool_calls",
        messages=[
            _user("What's the weather in San Francisco and in Paris?"),
            _assistant_tool_calls(
                "Checking both cities.",
                calls=[
                    (
                        "get_weather",
                        {"city": "San Francisco", "unit": "celsius"},
                        "call_par_1",
                    ),
                    (
                        "get_weather",
                        {"city": "Paris", "unit": "celsius"},
                        "call_par_2",
                    ),
                ],
            ),
            _tool_result("18°C and sunny.", call_id="call_par_1"),
            _tool_result("12°C and cloudy.", call_id="call_par_2"),
            _assistant("San Francisco is 18°C and sunny; Paris is 12°C and cloudy."),
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    # Assistant turn carrying BOTH visible content and a tool call in the
    # same message (the model narrates while also invoking a function).
    Scenario(
        id="assistant_content_plus_tool_call",
        messages=[
            _user("Book a weather-aware trip to San Francisco."),
            _assistant_tool_call(
                "Sure — let me look up the current weather there.",
                name="get_weather",
                arguments={"city": "San Francisco", "unit": "celsius"},
                call_id="call_content_1",
            ),
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    # Tool call whose arguments contain nested dict + list structures, to
    # exercise JSON serialization of non-flat arguments in the round-trip.
    Scenario(
        id="nested_tool_call_args",
        messages=[
            _user("Schedule a team sync with Alice and Bob in HQ room 101."),
            _assistant_tool_call(
                "Creating the event.",
                name="create_event",
                arguments={
                    "title": "Team Sync",
                    "attendees": ["alice", "bob"],
                    "location": {"building": "HQ", "room": "101"},
                },
                call_id="call_nested_1",
            ),
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    Scenario(
        id="tool_only_empty_content",
        messages=[
            _user("What's the weather in Tokyo?"),
            _assistant_tool_call(
                "",
                name="get_weather",
                arguments={"city": "Tokyo", "unit": "celsius"},
                call_id="call_empty_content_1",
            ),
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    # Controlled A/B pair with ``dict_tool_args``: identical logical tool call,
    # one with DICT arguments and one with the equivalent JSON-STRING arguments.
    # The gemma-4 template renders these two DIFFERENTLY (dict -> native
    # key:<|"|>val<|"|>; string -> verbatim JSON inside the braces), so the pair
    # pins whether a renderer preserves that distinction. NOTE: cookbook
    # ``normalize_messages`` homogenizes both to a JSON string before the
    # renderer runs, so the renderer cannot tell them apart post-normalize.
    Scenario(
        id="json_string_tool_args",
        messages=[
            _user("What's the weather in Paris?"),
            _assistant_tool_call(
                "Checking Paris.",
                name="get_weather",
                arguments='{"city":"Paris","unit":"celsius"}',
                call_id="call_json_args_1",
            ),
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    Scenario(
        id="dict_tool_args",
        messages=[
            _user("What's the weather in Paris?"),
            _assistant_tool_call(
                "Checking Paris.",
                name="get_weather",
                arguments={"city": "Paris", "unit": "celsius"},
                call_id="call_dict_args_1",
            ),
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    # Declares a tool whose schema includes a ``response`` return-type block,
    # exercising the template's ``response:`` declaration branch. Generation-
    # shaped (ends on the user) so it runs invariant-1 byte-parity for every
    # renderer with a chat template.
    Scenario(
        id="tool_with_response_schema",
        messages=[_user("What's the weather in Paris?")],
        ends_with_assistant=False,
        requires_tools=True,
        tools=[RESPONSE_SCHEMA_TOOL],
    ),
    Scenario(
        id="parallel_tool_results_reversed",
        messages=[
            _user("Compare the weather in San Francisco and Paris."),
            _assistant_tool_calls(
                "Checking both cities.",
                calls=[
                    ("get_weather", {"city": "San Francisco"}, "call_reverse_1"),
                    ("get_weather", {"city": "Paris"}, "call_reverse_2"),
                ],
            ),
            _tool_result("12°C and cloudy.", call_id="call_reverse_2"),
            _tool_result("18°C and sunny.", call_id="call_reverse_1"),
            _assistant("San Francisco is sunny; Paris is cloudy."),
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    # Generation-shaped tool conversation: ends on the tool RESULT so the
    # generation prompt (invariant 1) exercises tool-declaration + tool-call +
    # tool-result rendering for every renderer, not just the supervised leg.
    Scenario(
        id="tool_result_pending_answer",
        messages=[
            _user("What's the weather in San Francisco?"),
            _assistant_tool_call(
                "Checking now.",
                name="get_weather",
                arguments={"city": "San Francisco", "unit": "celsius"},
                call_id="call_pending_1",
            ),
            _tool_result("18°C and sunny.", call_id="call_pending_1"),
        ],
        ends_with_assistant=False,
        requires_tools=True,
        tools=TOOLS,
    ),
    # Tool call with an EMPTY argument object — probes the ``{}``/no-args path.
    Scenario(
        id="empty_args_tool_call",
        messages=[
            _user("What's the weather?"),
            _assistant_tool_call(
                "Checking.",
                name="get_weather",
                arguments={},
                call_id="call_empty_args_1",
            ),
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    # Tool RESULT whose content is nested JSON (not a scalar string) — probes
    # how a structured tool result is folded back into history.
    Scenario(
        id="nested_json_tool_result",
        messages=[
            _user("What's the weather in San Francisco?"),
            _assistant_tool_call(
                "Checking now.",
                name="get_weather",
                arguments={"city": "San Francisco", "unit": "celsius"},
                call_id="call_struct_result_1",
            ),
            _tool_result(
                '{"temp_c": 18, "conditions": ["sunny", "breezy"], '
                '"detail": {"humidity": 0.4}}',
                call_id="call_struct_result_1",
            ),
            _assistant("It's 18°C, sunny and breezy in San Francisco."),
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
]


# ---------------------------------------------------------------------------
# Thinking + tool combinations — reasoning and tool calls in one turn.
# ---------------------------------------------------------------------------
COMBO_SCENARIOS: list[Scenario] = [
    Scenario(
        id="thinking_tool_call",
        messages=[
            _user("What's the weather in San Francisco?"),
            _assistant_thinking_tool_call(
                "Let me look that up.",
                reasoning="The user wants current SF weather; call get_weather.",
                name="get_weather",
                arguments={"city": "San Francisco", "unit": "celsius"},
                call_id="call_combo_1",
            ),
        ],
        ends_with_assistant=True,
        requires_thinking=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    Scenario(
        id="thinking_tool_then_answer",
        messages=[
            _system("You can call tools to answer questions."),
            _user("What's the weather in San Francisco?"),
            _assistant_thinking_tool_call(
                "Checking now.",
                reasoning="Call get_weather for San Francisco.",
                name="get_weather",
                arguments={"city": "San Francisco", "unit": "celsius"},
                call_id="call_combo_2",
            ),
            _tool_result("18°C and sunny.", call_id="call_combo_2"),
            _assistant_thinking(
                "The tool reports 18°C and sunny; summarize it.",
                "It's 18°C and sunny in San Francisco.",
            ),
        ],
        ends_with_assistant=True,
        requires_thinking=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    # Reasoning with NO visible text, then a tool call: the model thinks and
    # calls without narrating.
    Scenario(
        id="reasoning_only_tool_call",
        messages=[
            _user("What's the weather in San Francisco?"),
            _assistant_thinking_tool_call(
                "",
                reasoning="No preamble needed; just call get_weather.",
                name="get_weather",
                arguments={"city": "San Francisco", "unit": "celsius"},
                call_id="call_combo_3",
            ),
        ],
        ends_with_assistant=True,
        requires_thinking=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    # Reasoning + TWO parallel tool calls in one turn.
    Scenario(
        id="thinking_parallel_tool_calls",
        messages=[
            _user("Weather in San Francisco and Paris?"),
            {
                "role": "assistant",
                "reasoning_content": "Both cities are independent; call both.",
                "content": "Checking both.",
                "tool_calls": [
                    {
                        "id": "call_combo_par_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "San Francisco", "unit": "celsius"},
                        },
                    },
                    {
                        "id": "call_combo_par_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris", "unit": "celsius"},
                        },
                    },
                ],
            },
        ],
        ends_with_assistant=True,
        requires_thinking=True,
        requires_tools=True,
        tools=TOOLS,
    ),
    # Assistant content as multi-part TEXT plus a tool call in the same turn.
    Scenario(
        id="multipart_content_tool_call",
        messages=[
            _user("Look up the SF weather and note it."),
            {
                "role": "assistant",
                "content": _text_parts("Sure — ", "looking it up now."),
                "tool_calls": [
                    {
                        "id": "call_combo_mp_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "San Francisco", "unit": "celsius"},
                        },
                    }
                ],
            },
        ],
        ends_with_assistant=True,
        requires_tools=True,
        tools=TOOLS,
    ),
]


ALL_SCENARIOS: list[Scenario] = [
    *STRUCTURAL_SCENARIOS,
    *THINKING_SCENARIOS,
    *TOOL_SCENARIOS,
    *COMBO_SCENARIOS,
]
