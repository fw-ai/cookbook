"""Exhaustive contract tests for the Muse Glimmer ATEM renderer."""

from __future__ import annotations

import json

import pytest
import tinker
from jinja2.exceptions import TemplateError
from tinker._compat import model_dump
from tinker.lib._pydantic_conv import to_pydantic_request
from training.renderer import RendererError

from training.renderer.muse_glimmer import (
    _TOOL_ARGUMENT_ERROR,
    MuseGlimmerOptions,
    MuseGlimmerImageTokenCounter,
    MuseGlimmerRenderer,
    _content,
    _factory,
    _render_atem,
    _render_tool_defs,
)
from training.utils.supervised import (
    normalize_messages,
    render_messages_to_datum,
    resolve_renderer_name,
)
from training.utils.supervised import build_tool_prefixed_messages
from training.utils.rl.agent.openai import CookbookTurnRenderer
from training.utils.rl.rollout.renderer import (
    build_multimodal_completions_prompt_token_ids,
)


class _Tokenizer:
    bos_token = "<|begin_of_text|>"
    model_name = "meta-models/Muse-Glimmer-30B"

    _special = {
        "<|begin_of_text|>": 300000,
        "<|start|>": 300001,
        "<|message|>": 300002,
        "<|eom|>": 300003,
        "<|eot|>": 300004,
        "<|patch|>": 300005,
        "<|video|>": 300006,
        "<|image_start|>": 300007,
        "<|image_end|>": 300008,
    }
    _reverse = {value: key for key, value in _special.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        result: list[int] = []
        while text:
            match = next((token for token in self._special if text.startswith(token)), None)
            if match is not None:
                result.append(self._special[match])
                text = text[len(match) :]
            else:
                result.append(ord(text[0]))
                text = text[1:]
        return result

    def decode(self, tokens, **_kwargs) -> str:
        return "".join(self._reverse.get(int(token), chr(int(token))) for token in tokens)


def _renderer(**kwargs) -> MuseGlimmerRenderer:
    return MuseGlimmerRenderer(_Tokenizer(), options=MuseGlimmerOptions(**kwargs))


class _ImageProcessor:
    merge_size = 2

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
        images_kwargs: dict,
    ) -> int:
        del images_kwargs
        assert (height, width) == (56, 56)
        return 16


_VISION_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAADgAAAA4CAIAAAAn5KxJAAAAaklEQVR42u3YMQoAIQwEwETu34cvz1VX2AdBnO1sloFYbVZVnJARhwS0O8/"
    "yymyo/D99zoa2esvpQUFBQUFBQUFBQUFBQXcnTTq3QpelpHUoiYiOurCUgIKCgoKCgoKCgoKCgm6PpeRa6AeiBRFpZuNsuAAAAABJRU5ErkJggg=="
)


def _render(renderer: MuseGlimmerRenderer, messages: list[dict], *, generation=True) -> str:
    if generation:
        tokens = renderer.build_generation_prompt(messages).to_ints()
    else:
        model_input, _weights = renderer.build_supervised_example(messages)
        tokens = model_input.to_ints()
    return renderer.tokenizer.decode(tokens)


def _call(name: str, arguments, call_id: str = "call-1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_default_system_and_generation_suffix_exact() -> None:
    actual = _render(
        _renderer(current_date="2030-02-03"),
        [{"role": "user", "content": " hi "}],
    )
    assert actual == (
        "<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\n"
        "Knowledge cutoff: 2026-01-04.\nCurrent date: 2030-02-03.\n\n"
        'Reasoning strength: high.\n\n# Valid recipients: "self", "user".<|eot|>'
        "<|start|>user<|message|> hi <|eot|><|start|>assistant"
    )


def test_default_system_options_and_date_omission() -> None:
    actual = _render(
        _renderer(
            reasoning_strength="low",
            knowledge_cutoff="2025-07",
            include_current_date=False,
        ),
        [{"role": "user", "content": "x"}],
    )
    assert "Knowledge cutoff: 2025-07." in actual
    assert "Current date:" not in actual
    assert "Reasoning strength: low." in actual


def test_falsy_default_options_fall_back() -> None:
    actual = _render(
        MuseGlimmerRenderer(
            _Tokenizer(),
            options=MuseGlimmerOptions(
                reasoning_strength="",
                knowledge_cutoff="",
                current_date="",
                include_current_date=True,
            ),
        ),
        [{"role": "user", "content": "x"}],
    )
    assert "Knowledge cutoff: 2026-01-04." in actual
    assert "Reasoning strength: high." in actual
    assert "Current date: " in actual


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Reasoning effort", "Reasoning strength"),
        ("Reasoning Effort", "Reasoning Strength"),
        ("reasoning effort", "reasoning strength"),
        ("REASONING EFFORT", "REASONING STRENGTH"),
    ],
)
def test_explicit_system_reasoning_normalization(source: str, expected: str) -> None:
    actual = _render(_renderer(), [{"role": "system", "content": source}])
    assert expected in actual
    assert actual.lower().count("reasoning strength") == 1
    assert "Knowledge cutoff:" not in actual


def test_system_anywhere_suppresses_default_and_each_system_is_enriched() -> None:
    actual = _render(
        _renderer(),
        [
            {"role": "user", "content": "before"},
            {"role": "system", "content": "late"},
            {"role": "system", "content": "Reasoning strength: medium."},
        ],
    )
    assert "You are a helpful AI assistant." not in actual
    assert actual.count("# Valid recipients:") == 2
    assert actual.count("Reasoning strength: high.") == 1
    assert "Reasoning strength: medium." in actual


def test_unrelated_mixed_case_is_not_normalized_and_gets_directive() -> None:
    actual = _render(
        _renderer(include_current_date=False),
        [{"role": "system", "content": "ReAsOnInG EfFoRt: low"}],
    )
    assert "ReAsOnInG EfFoRt: low\n\nReasoning strength: high." in actual


def test_content_parts_match_template_and_ignore_unknowns() -> None:
    assert _content(" raw \n") == " raw \n"
    assert _content(None) == ""
    assert (
        _content(
            [
                {"type": "text", "text": "a"},
                {"type": "image", "image": "ignored"},
                {"type": "image_url", "image_url": "ignored"},
                {"type": "video", "video": "ignored"},
                {"type": "thinking", "thinking": "hidden"},
                {"type": "unknown", "value": "hidden"},
                {"type": "text", "text": "b"},
            ]
        )
        == "a<|patch|><|video|>b"
    )


def test_image_content_materializes_processor_boundaries_and_chunk() -> None:
    processor = _ImageProcessor()
    renderer = MuseGlimmerRenderer(_Tokenizer(), image_processor=processor)
    model_input = renderer.build_generation_prompt(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image", "image": _VISION_IMAGE},
                    {"type": "text", "text": "after"},
                ],
            }
        ]
    )

    image_chunks = [
        chunk
        for chunk in model_input.chunks
        if isinstance(chunk, tinker.types.ImageChunk)
    ]
    assert len(image_chunks) == 1
    assert image_chunks[0].expected_tokens == 4
    assert image_chunks[0].format == "jpeg"

    rendered_text = "".join(
        renderer.tokenizer.decode(chunk.tokens)
        for chunk in model_input.chunks
        if isinstance(chunk, tinker.types.EncodedTextChunk)
    )
    assert "before<|image_start|><|image_end|>after<|eot|>" in rendered_text
    assert "<|patch|>" not in rendered_text
    assert renderer.image_processor is processor
    assert renderer.image_placeholder_token_id == _Tokenizer._special["<|patch|>"]


def test_image_content_without_processor_fails_instead_of_dropping_payload() -> None:
    renderer = _renderer()
    with pytest.raises(RendererError, match="requires an image processor"):
        renderer.build_generation_prompt(
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": _VISION_IMAGE}],
                }
            ]
        )


def test_factory_forwards_image_processor() -> None:
    processor = _ImageProcessor()
    renderer = _factory(_Tokenizer(), processor)
    assert renderer.image_processor is processor


def test_multipart_system_text_is_coalesced_before_normalization() -> None:
    actual = _render(
        _renderer(include_current_date=False),
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Reasoning "},
                    {"type": "text", "text": "effort: low"},
                ],
            }
        ],
    )
    assert "Reasoning strength: low" in actual
    assert "Reasoning effort" not in actual
    assert actual.lower().count("reasoning strength") == 1


def test_sft_and_tinker_wire_preserve_muse_image_chunk() -> None:
    renderer = MuseGlimmerRenderer(_Tokenizer(), image_processor=_ImageProcessor())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Read this chart: "},
                {"type": "image_url", "image_url": {"url": _VISION_IMAGE}},
            ],
        },
        {"role": "assistant", "content": "four"},
    ]
    rendered = render_messages_to_datum(messages, renderer=renderer)
    image_chunks = [
        chunk
        for chunk in rendered.datum.model_input.chunks
        if isinstance(chunk, tinker.types.ImageChunk)
    ]
    assert len(image_chunks) == 1
    assert image_chunks[0].expected_tokens == 4

    wire_request = tinker.types.ForwardBackwardRequest(
        forward_backward_input=tinker.types.ForwardBackwardInput(
            data=[rendered.datum],
            loss_fn="cross_entropy",
            loss_fn_config=None,
        ),
        model_id="muse-glimmer-test",
        seq_id=1,
    )
    wire_payload = model_dump(
        to_pydantic_request(wire_request),
        exclude_unset=False,
        exclude_none=True,
        mode="json",
    )
    wire_chunks = wire_payload["forward_backward_input"]["data"][0]["model_input"][
        "chunks"
    ]
    wire_images = [chunk for chunk in wire_chunks if chunk["type"] == "image"]
    assert len(wire_images) == 1
    assert wire_images[0]["expected_tokens"] == 4
    assert wire_images[0]["format"] == "jpeg"
    assert wire_images[0]["data"]

    prompt = renderer.build_generation_prompt(normalize_messages(messages[:1]))
    prompt_ids, images = build_multimodal_completions_prompt_token_ids(
        messages[:1],
        prompt,
        renderer.tokenizer,
        renderer=renderer,
    )
    assert prompt_ids.count(renderer.image_placeholder_token_id) == 1
    assert len(images) == 1
    assert images[0].startswith("data:image/jpeg;base64,")


def test_image_token_counter_loads_staged_processor_config(tmp_path) -> None:
    (tmp_path / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "MuseGlimmerProcessor",
                "image_processor": {
                    "image_processor_type": "MuseGlimmerImageProcessor",
                    "patch_size": 14,
                    "merge_size": 2,
                    "max_image_tokens": 17,
                },
            }
        )
    )
    processor = MuseGlimmerImageTokenCounter.from_pretrained(str(tmp_path))
    assert processor.patch_size == 14
    assert processor.merge_size == 2
    assert processor.max_image_tokens == 17
    assert processor.get_number_of_image_patches(56, 56, {}) == 16


def test_generation_suffix_is_always_assistant() -> None:
    renderer = _renderer(include_current_date=False)
    rendered = renderer.build_generation_prompt([{"role": "user", "content": "x"}], role="tool")
    assert renderer.tokenizer.decode(rendered.to_ints()).endswith("<|start|>assistant")


def test_atem_argument_types_order_and_exact_whitespace() -> None:
    actual = _render_atem(
        _call(
            "ns.run",
            {
                "truth": True,
                "falsehood": False,
                "nothing": None,
                "raw": "  spaced  ",
                "number": 2.5,
                "nested": {"city": "Zürich", "items": [1, None]},
            },
        )
    )
    assert actual == (
        '<atem:function_calls>\n<atem:invoke name="ns.run">\n'
        '<atem:parameter name="truth">true</atem:parameter>\n'
        '<atem:parameter name="falsehood">false</atem:parameter>\n'
        '<atem:parameter name="nothing">null</atem:parameter>\n'
        '<atem:parameter name="raw">  spaced  </atem:parameter>\n'
        '<atem:parameter name="number">2.5</atem:parameter>\n'
        '<atem:parameter name="nested">{"city": "Zürich", "items": [1, null]}'
        "</atem:parameter>\n</atem:invoke>\n</atem:function_calls>"
    )
    assert _render_atem(_call("empty", {})) == (
        '<atem:function_calls>\n<atem:invoke name="empty">\n' "</atem:invoke>\n</atem:function_calls>"
    )
    with pytest.raises(TypeError):
        _render_atem(_call("not-json", {"value": {1, 2}}))


def test_raw_json_string_tool_arguments_are_rejected_exactly() -> None:
    with pytest.raises(TemplateError, match="requires tool_call.function.arguments") as exc:
        _render_atem(_call("bad", '{"x": 1}'))
    assert str(exc.value) == _TOOL_ARGUMENT_ERROR


def test_tool_definitions_namespaces_descriptions_and_schema_order() -> None:
    tools = [
        {
            "function": {
                "name": "alpha.first",
                "description": "one",
                "parameters": {"type": "object", "properties": {}},
            }
        },
        {
            "name": "alpha.second",
            "description": "two",
            "parameters": {"type": "object"},
        },
        {"name": "solo", "description": "three", "parameters": {}},
    ]
    actual = _render_tool_defs(tools, {"alpha": "Alpha tools"})
    assert '// Tool metadata\n{"name": "alpha", "description": "Alpha tools"}\n' in actual
    assert '{"name": "solo", "description": ""}\n// Function schemas' in actual
    assert actual.count('{"name": "alpha", "description": "Alpha tools"}') == 1
    assert actual.index('"alpha.first"') < actual.index('"alpha.second"')
    assert actual.endswith("</atem:function_calls>")


def test_explicit_empty_system_with_tools_is_not_replaced_by_default() -> None:
    renderer = _renderer(include_current_date=False)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ns.run",
                "description": "Run it",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assembled = build_tool_prefixed_messages(
        [{"role": "system", "content": ""}, {"role": "user", "content": "go"}],
        renderer=renderer,
        tools=tools,
    )
    actual = _render(renderer, assembled)
    assert "You are a helpful AI assistant." not in actual
    assert actual.startswith("<|begin_of_text|><|start|>system<|message|>\n\nReasoning strength: high.")
    assert '# Valid recipients: "self", "ns.*", "user".' in actual


def test_assistant_reasoning_tools_consecutive_boundary_and_content_ignored() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "reasoning_content": "plan",
            "content": "discard me",
            "tool_calls": [_call("a.one", {"x": 1}), _call("b.two", {})],
        },
        {"role": "assistant", "recipient": "self", "content": "continue"},
    ]
    actual = _render(_renderer(current_date="2030-01-01"), messages, generation=False)
    assistant = actual[actual.index("<|start|>assistant to=self") :]
    assert "discard me" not in assistant
    assert assistant == (
        "<|start|>assistant to=self<|message|>plan<|eom|>"
        "<|start|>assistant to=a.one<|message|><atem:function_calls>\n"
        '<atem:invoke name="a.one">\n<atem:parameter name="x">1</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eom|>"
        "<|start|>assistant to=b.two<|message|><atem:function_calls>\n"
        '<atem:invoke name="b.two">\n</atem:invoke>\n</atem:function_calls><|eom|>'
        "<|start|>assistant to=self<|message|>continue<|eom|>"
    )


@pytest.mark.parametrize(
    ("recipient", "end_turn", "ending"),
    [
        (None, None, "<|eot|>"),
        ("", None, "<|eot|>"),
        ("self", None, "<|eom|>"),
        ("tool.x", None, "<|eom|>"),
        ("user", False, "<|eom|>"),
        ("self", True, "<|eot|>"),
    ],
)
def test_recipient_and_end_turn_defaults(recipient, end_turn, ending) -> None:
    message = {"role": "assistant", "content": "answer"}
    if recipient is not None:
        message["recipient"] = recipient
    if end_turn is not None:
        message["end_turn"] = end_turn
    actual = _render(_renderer(include_current_date=False), [message], generation=False)
    expected_recipient = recipient or "user"
    assert actual.endswith(f"<|start|>assistant to={expected_recipient}<|message|>answer{ending}")


def test_tool_result_name_priority_lookup_last_match_and_empty() -> None:
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [_call("first", {}, "id")]},
        {"role": "assistant", "content": "", "tool_calls": [_call("last", {}, "id")]},
        {"role": "tool", "tool_call_id": "id", "content": "matched"},
        {"role": "tool", "tool_call_id": "missing", "content": "fallback"},
        {"role": "tool", "content": "empty"},
        {"role": "tool", "name": "direct", "tool_call_id": "id", "content": "named"},
    ]
    actual = _render(_renderer(include_current_date=False), messages, generation=False)
    assert '<|start|>tool last<|message|><tool_output name="last">\nmatched\n' in actual
    assert '<|start|>tool missing<|message|><tool_output name="missing">\nfallback\n' in actual
    assert '<|start|>tool <|message|><tool_output name="">\nempty\n' in actual
    assert '<|start|>tool direct<|message|><tool_output name="direct">\nnamed\n' in actual
    assert actual.count("</tool_output><|eot|>") == 4


def test_unknown_roles_omitted_and_user_tool_always_eot() -> None:
    actual = _render(
        _renderer(include_current_date=False),
        [
            {"role": "developer", "content": "drop"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "tool", "content": "c"},
            {"role": "tool", "content": "d"},
        ],
        generation=False,
    )
    assert "developer" not in actual and "drop" not in actual
    assert actual.count("<|eot|>") == 5  # default system + 2 users + 2 tools
    assert "<|eom|>" not in actual


def test_common_normalization_preserves_muse_protocol_fields() -> None:
    [message] = normalize_messages(
        [
            {
                "role": "assistant",
                "recipient": "self",
                "end_turn": False,
                "content": [{"type": "text", "text": "caption"}],
            }
        ]
    )
    assert message["recipient"] == "self"
    assert message["end_turn"] is False
    assert message["content"][0] == {"type": "text", "text": "caption"}


def test_common_normalization_rejects_unsupported_video() -> None:
    with pytest.raises(TypeError, match="Unsupported message content part"):
        normalize_messages(
            [
                {
                    "role": "user",
                    "content": [{"type": "video", "video": "clip"}],
                }
            ]
        )


def test_model_resolution() -> None:
    assert resolve_renderer_name("meta-models/Muse-Glimmer-30B") == "muse_glimmer"
    assert resolve_renderer_name("accounts/fireworks/models/muse_glimmer-30b") == "muse_glimmer"


def test_parse_reasoning_tool_call_and_final_answer() -> None:
    renderer = _renderer()
    sampled = (
        " to=self<|message|>thinking<|eom|><|start|>assistant to=weather.get"
        '<|message|><atem:function_calls>\n<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eom|><|start|>assistant to=user"
        "<|message|>sunny<|eot|>"
    )
    parsed, termination = renderer.parse_response(renderer.tokenizer.encode(sampled))
    assert termination.value == "stop_sequence"
    assert parsed["content"] == [
        {"type": "thinking", "thinking": "thinking"},
        {"type": "text", "text": "sunny"},
    ]
    assert parsed["tool_calls"][0].function.name == "weather.get"
    assert parsed["tool_calls"][0].id == "weather.get:0"
    assert json.loads(parsed["tool_calls"][0].function.arguments) == {"city": "Paris"}


def test_rl_openai_adapter_preserves_reasoning_and_routes_parse_fields() -> None:
    renderer = _renderer(include_current_date=False)
    adapter = CookbookTurnRenderer(renderer)
    tokens = adapter.prompt_tokens(
        messages=[
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning_content": "reason",
                "recipient": "self",
                "end_turn": False,
                "content": "draft",
            },
        ],
        tools=[],
        system_prompt="",
    )
    prompt = renderer.tokenizer.decode(tokens)
    assert (
        "<|start|>assistant to=self<|message|>reason<|eom|>" "<|start|>assistant to=self<|message|>draft<|eom|>"
    ) in prompt

    sampled = " to=self<|message|>reason<|eom|><|start|>assistant to=user" "<|message|>answer<|eot|>"
    parsed = adapter.parse_completion(renderer.tokenizer.encode(sampled))
    assert parsed["reasoning_content"] == "reason"
    assert parsed["content"] == "answer"


def test_rl_openai_adapter_correlates_sampled_tool_result_name() -> None:
    renderer = _renderer(include_current_date=False)
    adapter = CookbookTurnRenderer(renderer)
    sampled = (
        ' to=weather.get<|message|><atem:function_calls>\n<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    )
    parsed = adapter.parse_completion(renderer.tokenizer.encode(sampled))
    assert parsed["tool_calls"][0]["id"] == "weather.get:0"

    tokens = adapter.prompt_tokens(
        messages=[
            {"role": "user", "content": "weather?"},
            parsed,
            {
                "role": "tool",
                "tool_call_id": parsed["tool_calls"][0]["id"],
                "content": "sunny",
            },
        ],
        tools=[],
        system_prompt="",
    )
    prompt = renderer.tokenizer.decode(tokens)
    assert '<|start|>tool weather.get<|message|><tool_output name="weather.get">\nsunny' in prompt
