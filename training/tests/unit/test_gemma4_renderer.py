"""Verify ``Gemma4Renderer`` matches the official Gemma 4 chat template.

Two layers of verification:

1. **Token-level parity** vs ``tokenizer.apply_chat_template`` for a
   battery of conversation shapes. This catches the kinds of bugs the
   previous implementation had (wrong role name, wrong generation
   suffix, missing trim) without ever loading the model.

2. **End-to-end model parity** on GPU: load the actual Gemma 4 model,
   render the same prompt via (a) ``Gemma4Renderer.build_generation_prompt``
   and (b) HuggingFace ``apply_chat_template``, then greedy-decode with
   both. The two outputs MUST be byte-identical. This is the test that
   would have caught the original "wrong role name" bug instantly —
   the model would emit garbage for an `assistant` role token because
   it was trained on `model`.

To run, set ``GEMMA4_MODEL_PATH`` to a directory containing the official
``google/gemma-4-*-it`` files (must include ``chat_template.jinja``,
``tokenizer.json``, ``tokenizer_config.json``, and the safetensors
weights for the GPU test). All tests are skipped automatically if the
path is unset.

    GEMMA4_MODEL_PATH=/path/to/gemma-4-E2B-it \
        PYTHONPATH=../.. python -m pytest tests/unit/test_gemma4_renderer.py -v
"""

from __future__ import annotations

import os

import pytest
import transformers

from training.renderer.gemma4 import Gemma4Renderer

_MODEL_PATH_ENV = "GEMMA4_MODEL_PATH"


def _resolve_model_path() -> str:
    path = os.environ.get(_MODEL_PATH_ENV)
    if not path:
        pytest.skip(
            f"Set {_MODEL_PATH_ENV} to a directory containing an official "
            "google/gemma-4-*-it checkpoint to run these tests."
        )
    return path


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(
        _resolve_model_path(), trust_remote_code=True,
    )


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return Gemma4Renderer(tokenizer)


def _hf_tokens(
    tokenizer,
    messages,
    *,
    add_generation_prompt: bool = True,
    tools=None,
    enable_thinking: bool = False,
) -> list[int]:
    kwargs: dict = {"add_generation_prompt": add_generation_prompt}
    if tools is not None:
        kwargs["tools"] = tools
    if enable_thinking:
        kwargs["enable_thinking"] = True
    result = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    return list(result)


def _renderer_tokens(renderer, messages) -> list[int]:
    return list(renderer.build_generation_prompt(messages, role="assistant").to_ints())


def _assert_match(tokenizer, hf, ours):
    if hf != ours:
        raise AssertionError(
            "Token mismatch.\n"
            f"HF   ({len(hf)} toks): {hf}\n"
            f"Ours ({len(ours)} toks): {ours}\n"
            f"--- HF text ---\n{tokenizer.decode(hf)}\n"
            f"--- Our text ---\n{tokenizer.decode(ours)}"
        )


# ── Token-level parity vs the official HF chat template ─────────────────────


@pytest.mark.parametrize(
    "messages",
    [
        # Single user turn (generation prompt as the very first turn).
        [{"role": "user", "content": "Hello"}],
        # System + multi-turn dialogue.
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ],
        # User content with leading & trailing whitespace — verifies the
        # template's `| trim` is reproduced by the renderer.
        [{"role": "user", "content": "  spaced  "}],
        # User content with internal whitespace runs that survive trimming.
        [{"role": "user", "content": "line one\nline two\nline three"}],
        # Content that looks like the chat template tokens (must be plain
        # text, not interpreted as control tokens).
        [
            {"role": "user", "content": "<|turn>not a turn<turn|>"},
            {"role": "assistant", "content": "ack"},
        ],
        # Multi-line code block.
        [
            {"role": "user", "content": "```python\nprint('hi')\n```"},
            {"role": "assistant", "content": "```\nhi\n```"},
        ],
        # Mixed unicode and emoji.
        [
            {"role": "system", "content": "Réponds en français."},
            {"role": "user", "content": "Bonjour 👋"},
            {"role": "assistant", "content": "Salut! Ça va? 😊"},
        ],
        # Tool role rendered verbatim (NOT rewritten to user).
        [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "checking"},
            {"role": "tool", "content": "sunny, 72F"},
            {"role": "assistant", "content": "It's sunny."},
        ],
        # Assistant content with a thinking block — must be stripped from
        # history to match the template's strip_thinking macro.
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<|channel>thought\nI think...<channel|>The answer is 4."},
            {"role": "user", "content": "next"},
        ],
    ],
    ids=[
        "user_only",
        "system_multi_turn",
        "trim_whitespace",
        "multiline",
        "literal_template_syntax",
        "code_block",
        "unicode",
        "tool_role_verbatim",
        "model_thinking_stripped",
    ],
)
def test_parity_with_hf_chat_template(tokenizer, renderer, messages):
    """Renderer tokens must match HF apply_chat_template byte-for-byte."""
    _assert_match(tokenizer, _hf_tokens(tokenizer, messages), _renderer_tokens(renderer, messages))


# ── Public API contract ──────────────────────────────────────────────────────


def test_stop_sequences_include_turn_close_and_tool_response(renderer, tokenizer):
    """Both `<turn|>` (normal close) and `<|tool_response>` (tool-call
    handoff) are valid completion signals — Gemma 4 emits the latter
    immediately after `<tool_call|>` when it wants the runner to inject
    a tool result, and both vLLM and SGLang treat it as the end of the
    assistant turn for tool-calling outputs."""
    turn_close = tokenizer.encode("<turn|>", add_special_tokens=False)[0]
    tool_resp_open = tokenizer.encode("<|tool_response>", add_special_tokens=False)[0]
    assert renderer.get_stop_sequences() == [turn_close, tool_resp_open]


def test_parse_response_plain_text_roundtrip(tokenizer, renderer):
    """A rendered model turn parses back to the original text."""
    sample = "Sure, here is the answer."
    encoded = tokenizer.encode(sample + "<turn|>", add_special_tokens=False)
    msg, ok = renderer.parse_response(encoded)
    assert (ok, msg["role"], msg["content"]) == (True, "assistant", sample)


def test_parse_response_extracts_thinking(tokenizer, renderer):
    """A model turn containing a `<|channel>thought<channel|>` block parses
    out into separate ThinkingPart + TextPart content."""
    body = "<|channel>I'm thinking<channel|>The answer is 4.<turn|>"
    encoded = tokenizer.encode(body, add_special_tokens=False)
    msg, ok = renderer.parse_response(encoded)
    assert ok
    assert isinstance(msg["content"], list)
    types = [p["type"] for p in msg["content"]]
    assert types == ["thinking", "text"]
    assert msg["content"][0]["thinking"] == "I'm thinking"
    assert msg["content"][1]["text"] == "The answer is 4."


# ── build_supervised_example parity ─────────────────────────────────────────


def test_supervised_example_matches_hf_no_generation_prompt(tokenizer, renderer):
    """``build_supervised_example`` must produce a token sequence that
    matches ``apply_chat_template(..., add_generation_prompt=False)``.

    SFT training uses build_supervised_example, not build_generation_prompt,
    so this is the parity test that actually pins what the model is
    trained on.
    """
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well."},
    ]
    hf_ids = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    model_input, _weights = renderer.build_supervised_example(messages)
    our_ids = list(model_input.to_ints())
    _assert_match(tokenizer, hf_ids, our_ids)


# ── Sequence-extension property ─────────────────────────────────────────────


def test_sequence_extension_property_holds_across_assistant_turns(renderer):
    """``Gemma4Renderer.has_extension_property`` returns True, claiming
    that successive **observations at assistant-turn boundaries** are
    prefix extensions of each other. Concretely: for a multi-turn
    transcript, ``build_generation_prompt(messages[:k])`` for each k
    where ``messages[k]`` is the next assistant turn must be a strict
    prefix of the same call at the next such k.

    This is the property RL training relies on for KV-cache reuse and
    O(T) compute scaling across multi-turn rollouts. If it breaks,
    multi-turn RL recomputes from scratch on every assistant step.
    """
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "third question"},
        {"role": "assistant", "content": "third answer"},
    ]
    # Indices where the NEXT message is an assistant turn — i.e. the
    # observations the model gets right before generating its response.
    boundaries = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    assert len(boundaries) >= 2, "need at least two assistant turns to test extension"

    prev: list[int] = []
    for k in boundaries:
        cur = list(
            renderer.build_generation_prompt(messages[:k], role="assistant").to_ints()
        )
        assert cur[: len(prev)] == prev, (
            f"observation at boundary k={k} is not a prefix extension of the previous "
            f"observation; len(prev)={len(prev)} len(cur)={len(cur)}"
        )
        prev = cur


# ── Tool-call / tool-response parity ────────────────────────────────────────


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City"},
                "unit": {
                    "type": "string",
                    "description": "Unit",
                    "enum": ["c", "f"],
                },
            },
            "required": ["location"],
        },
    },
}

_NESTED_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "q"},
                "filters": {
                    "type": "object",
                    "description": "f",
                    "properties": {
                        "year": {"type": "string", "description": "y"},
                    },
                    "required": ["year"],
                },
                "tags": {
                    "type": "array",
                    "description": "t",
                    "items": {"type": "string"},
                },
            },
            "required": ["query"],
        },
    },
}

_NO_PROPS_TOOLS = [
    {"type": "function", "function": {"name": "a", "description": "d1",
     "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "b", "description": "d2",
     "parameters": {"type": "object", "properties": {}, "required": []}}},
]


def _renderer_tokens_with_tools(renderer, messages, tools, *, system_prompt: str = ""):
    """Build a renderer prompt that mirrors HF's ``tools=...`` argument by
    prepending the synthetic system message returned by
    ``create_conversation_prefix_with_tools``.

    If the conversation already starts with a system message we fold its
    content into the synthetic system message via ``system_prompt`` and drop
    the original — the official template only emits one system block.
    """
    if messages and messages[0]["role"] in ("system", "developer"):
        sys_text = messages[0]["content"]
        rest = messages[1:]
    else:
        sys_text = system_prompt
        rest = list(messages)
    prefix = renderer.create_conversation_prefix_with_tools(tools, system_prompt=sys_text)
    return list(renderer.build_generation_prompt(prefix + rest, role="assistant").to_ints())


@pytest.mark.parametrize(
    ("messages", "tools"),
    [
        (
            [{"role": "user", "content": "weather in paris?"}],
            [_WEATHER_TOOL],
        ),
        (
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "weather?"},
            ],
            [_WEATHER_TOOL],
        ),
        (
            [{"role": "user", "content": "go"}],
            [_NESTED_TOOL],
        ),
        (
            [{"role": "user", "content": "go"}],
            _NO_PROPS_TOOLS,
        ),
    ],
    ids=[
        "single_tool_user_only",
        "single_tool_with_system",
        "nested_tool_def",
        "two_tools_no_props",
    ],
)
def test_tool_definitions_parity(tokenizer, renderer, messages, tools):
    """Renderer must match HF's ``apply_chat_template(..., tools=...)`` token
    sequence for the supported tool-definition shapes."""
    hf = _hf_tokens(tokenizer, messages, tools=tools)
    ours = _renderer_tokens_with_tools(renderer, messages, tools)
    _assert_match(tokenizer, hf, ours)


@pytest.mark.parametrize(
    ("messages", "tools"),
    [
        # Assistant emits a tool call with two scalar args (string, string).
        (
            [
                {"role": "user", "content": "weather in paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"location": "paris", "unit": "c"},
                            },
                        }
                    ],
                },
            ],
            [_WEATHER_TOOL],
        ),
        # Assistant tool call with mixed types (bool, int, string) — exercises
        # the format_argument type branches.
        (
            [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "fn",
                                "arguments": {"flag": True, "count": 3, "name": "z"},
                            },
                        }
                    ],
                },
            ],
            None,
        ),
        # Nested-object & array arguments — exercises recursive format_argument.
        (
            [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "fn",
                                "arguments": {
                                    "obj": {"a": 1, "b": "two"},
                                    "arr": [1, 2, 3],
                                },
                            },
                        }
                    ],
                },
            ],
            None,
        ),
    ],
    ids=["string_args", "scalar_mixed_args", "nested_args"],
)
def test_tool_call_parity(tokenizer, renderer, messages, tools):
    """Assistant ``tool_calls`` must serialise to the same tokens as the HF
    template's ``<|tool_call>call:name{...}<tool_call|>`` block."""
    if tools is not None:
        hf = _hf_tokens(tokenizer, messages, tools=tools, add_generation_prompt=False)
        ours = _renderer_tokens_with_tools(
            renderer,
            messages + [],  # explicit copy for clarity
            tools,
        )
        # _renderer_tokens_with_tools always appends a generation suffix; for
        # this no-genprompt parity check use the manual no-suffix call.
        if messages[0]["role"] in ("system", "developer"):
            sys_text, rest = messages[0]["content"], messages[1:]
        else:
            sys_text, rest = "", list(messages)
        prefix = renderer.create_conversation_prefix_with_tools(tools, system_prompt=sys_text)
        ours = list(
            renderer.build_supervised_example(prefix + rest)[0].to_ints()
        )
    else:
        hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
        ours = list(renderer.build_supervised_example(messages)[0].to_ints())
    _assert_match(tokenizer, hf, ours)


@pytest.mark.parametrize(
    "messages",
    [
        # Tool response with mapping payload, then a generation prompt.
        # Per the template, the prompt suffix `<|turn>model\n` is suppressed
        # because the previous message was a pure tool response.
        [
            {"role": "user", "content": "weather in paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "paris"},
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": "",
                "tool_responses": [
                    {"name": "get_weather", "response": {"temp": 18, "unit": "c"}}
                ],
            },
        ],
        # Tool response with a scalar payload.
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "fn", "arguments": {}}}
                ],
            },
            {
                "role": "user",
                "content": "",
                "tool_responses": [{"name": "fn", "response": "hello"}],
            },
        ],
        # Tool response followed by another user turn — the next turn DOES get
        # a `<|turn>model\n` generation suffix because the *immediately
        # preceding* message is no longer a tool response.
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "fn", "arguments": {"x": 1}},
                    }
                ],
            },
            {
                "role": "user",
                "content": "",
                "tool_responses": [{"name": "fn", "response": {"v": 2}}],
            },
            {"role": "user", "content": "now what"},
        ],
    ],
    ids=["mapping_response", "scalar_response", "response_then_user"],
)
def test_tool_response_parity(tokenizer, renderer, messages):
    """``tool_responses`` must round-trip the template's
    ``<|tool_response>response:name{...}<tool_response|>`` block, including
    the trailing-``<turn|>`` suppression and the generation-suffix
    suppression that the template applies after a pure tool-response turn."""
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_tokens(renderer, messages)
    _assert_match(tokenizer, hf, ours)


# ── enable_thinking parity ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def thinking_renderer(tokenizer):
    return Gemma4Renderer(tokenizer, enable_thinking=True)


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "hi"}],
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ],
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ans"},
            {"role": "user", "content": "second"},
        ],
    ],
    ids=["no_system", "with_system", "multi_turn"],
)
def test_enable_thinking_parity(tokenizer, thinking_renderer, messages):
    """When ``enable_thinking=True``, the renderer must inject the same
    ``<|think|>`` system marker the HF template emits."""
    hf = _hf_tokens(tokenizer, messages, enable_thinking=True)
    ours = list(
        thinking_renderer.build_generation_prompt(messages, role="assistant").to_ints()
    )
    _assert_match(tokenizer, hf, ours)


def test_enable_thinking_with_tools_parity(tokenizer, thinking_renderer):
    """``enable_thinking`` and ``tools`` should compose: the system block
    contains ``<|think|>`` followed by the tool declarations, with no
    intervening whitespace, exactly like the template."""
    messages = [{"role": "user", "content": "weather?"}]
    tools = [_WEATHER_TOOL]
    hf = _hf_tokens(tokenizer, messages, tools=tools, enable_thinking=True)

    prefix = thinking_renderer.create_conversation_prefix_with_tools(tools, system_prompt="")
    ours = list(
        thinking_renderer.build_generation_prompt(prefix + messages, role="assistant").to_ints()
    )
    _assert_match(tokenizer, hf, ours)


# ── Tool-call parse_response round trip ─────────────────────────────────────


def test_parse_response_extracts_tool_calls(tokenizer, renderer):
    """A model turn containing one or more tool-call blocks parses out into
    the ``tool_calls`` field, leaving any surrounding text on ``content``."""
    body = (
        '<|tool_call>call:get_weather{location:<|"|>paris<|"|>}<tool_call|>'
        "<turn|>"
    )
    encoded = tokenizer.encode(body, add_special_tokens=False)
    msg, ok = renderer.parse_response(encoded)
    assert ok
    assert msg.get("tool_calls"), "expected tool_calls to be populated"
    assert msg["tool_calls"][0]["name"] == "get_weather"  # type: ignore[index]
    # Surrounding content (none in this case) should be empty string.
    assert msg["content"] == ""


# ── End-to-end model parity on GPU ──────────────────────────────────────────


@pytest.fixture(scope="module")
def gpu_model(tokenizer):
    """Load the Gemma 4 model on the first available GPU.

    Skipped if torch / CUDA is not available, or if the model directory
    only contains tokenizer files (e.g. CI without weights).
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available for end-to-end model test.")
    path = _resolve_model_path()
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        )
    except (OSError, ValueError) as e:
        pytest.skip(f"Could not load Gemma 4 model from {path}: {e}")
    model.eval()
    return model


def _greedy_complete(model, tokenizer, prompt_ids: list[int], max_new_tokens: int = 32) -> str:
    import torch

    device = next(model.parameters()).device
    input_t = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        out = model.generate(
            input_t,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, len(prompt_ids):], skip_special_tokens=True)


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "What is 2+2? Answer in one word."}],
        [
            {"role": "system", "content": "Reply in one short sentence."},
            {"role": "user", "content": "Capital of France?"},
        ],
    ],
    ids=["arithmetic", "system_geography"],
)
def test_e2e_model_parity_greedy(tokenizer, renderer, gpu_model, messages):
    """End-to-end model parity: feed both the renderer's prompt and HF's
    prompt through the actual Gemma 4 model and assert the greedy
    completions are byte-identical.

    This is the test that would have caught the previous "wrong role name"
    bug instantly — a `<|turn>assistant\\n` prompt is out-of-distribution
    for a model trained on `<|turn>model\\n`, and the model would have
    emitted garbage on token 1.
    """
    hf_ids = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    our_ids = _renderer_tokens(renderer, messages)
    assert hf_ids == our_ids, "Renderer prompt diverges from HF prompt at the token level"

    hf_completion = _greedy_complete(gpu_model, tokenizer, hf_ids)
    our_completion = _greedy_complete(gpu_model, tokenizer, our_ids)

    # Sanity: model produced real text, not an immediate stop token
    # (the failure mode of an out-of-distribution prompt).
    assert hf_completion.strip(), f"Model produced empty completion: {hf_completion!r}"
    assert "<turn|>" not in hf_completion, (
        f"Model immediately emitted stop token — prompt is out-of-distribution. "
        f"completion={hf_completion!r}"
    )
    # The two completions must be byte-identical because the prompts are
    # token-identical and decoding is greedy.
    assert hf_completion == our_completion, (
        f"Completion mismatch:\n  HF:   {hf_completion!r}\n  Ours: {our_completion!r}"
    )


# ── End-to-end tool-calling: real model → renderer → real parsers ──────────


_E2E_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "description": "Temperature unit (c or f)",
                    "enum": ["c", "f"],
                },
            },
            "required": ["location"],
        },
    },
}


def test_e2e_tool_call_renderer_to_model_to_renderer_parser(
    tokenizer, renderer, gpu_model
):
    """End-to-end loop with tools:

    1. Build a tool-aware prompt with the renderer.
    2. Greedy-decode through the actual Gemma 4 model.
    3. Assert the model emits a properly-formed ``<|tool_call>...
       <tool_call|>`` block — proves our prompt is in-distribution for
       tool calling.
    4. Round-trip the raw token IDs through ``Gemma4Renderer.parse_response``
       and assert it extracts ``tool_calls`` with ``name='get_weather'``
       and a ``location`` argument.

    This is the integration test that catches the failure mode where the
    renderer output looks right token-for-token but doesn't actually
    drive the trained model to emit a tool call.
    """
    import torch  # noqa: F401  (skip-guarded by gpu_model fixture)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. When the user asks about "
                "weather, call the get_weather function."
            ),
        },
        {"role": "user", "content": "What is the weather in Paris in celsius?"},
    ]
    tools = [_E2E_TOOL]

    prefix = renderer.create_conversation_prefix_with_tools(
        tools, system_prompt=messages[0]["content"]
    )
    our_ids = list(
        renderer.build_generation_prompt(prefix + messages[1:], role="assistant").to_ints()
    )
    # Sanity: matches HF's tools= prompt token-for-token (the unit-test
    # tier already covers this, but pinning it here too makes failure
    # diagnosis easier when the model output is wrong).
    hf_ids = _hf_tokens(tokenizer, messages, tools=tools, add_generation_prompt=True)
    assert hf_ids == our_ids

    device = next(gpu_model.parameters()).device
    input_t = torch.tensor([our_ids], device=device)
    with torch.no_grad():
        out = gpu_model.generate(
            input_t,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, len(our_ids):].tolist()

    decoded_with_specials = tokenizer.decode(gen_ids, skip_special_tokens=False)
    assert "<|tool_call>" in decoded_with_specials, (
        "Model did NOT emit a tool call from our prompt — wire format wrong? "
        f"Got: {decoded_with_specials!r}"
    )

    msg, ok = renderer.parse_response(gen_ids)
    assert ok, (
        f"parse_response did not find a stop token in real model output: "
        f"{decoded_with_specials!r}"
    )
    assert msg.get("tool_calls"), (
        f"parse_response did not extract tool_calls from: {decoded_with_specials!r}"
    )
    tc = msg["tool_calls"][0]  # type: ignore[index]
    assert tc["name"] == "get_weather"
    assert "location" in tc["arguments"], (
        f"Missing location arg in parsed tool call: {tc!r}"
    )
