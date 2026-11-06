"""Beam adapter contract against an immutable, real HF tokenizer/template."""

import os
from typing import Any

from fireworks.training import sdk

import pytest
from transformers import AutoConfig
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from training.renderer import get_renderer
from training.renderer.base import ParseTermination, ToolCall, TrainOnWhat, get_text_content
from training.renderer.beam import BeamRenderer
from training.utils.supervised import normalize_messages, resolve_renderer_name
from training.utils.tokenizers import load_tokenizer
from training.renderer.verifier.utils.hf_parity import _load_tokenizer

MODEL = "reflection/Beam-1.0-501B-A23B-NVFP4"
REVISION = "f8ecac0e16045babf60156f6ba996a77fcb48cf9"


@pytest.fixture(scope="module")
def renderer() -> BeamRenderer:
    path = os.environ.get("BEAM_TOKENIZER_PATH", MODEL)
    tokenizer = TokenizersBackend.from_pretrained(
        path, revision=REVISION if path == MODEL else None, trust_remote_code=False
    )
    return get_renderer("beam", tokenizer)


def tokens(model_input: sdk.ModelInput) -> list[int]:
    return [token for chunk in model_input.chunks for token in chunk.tokens]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "Hi 👋"}],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "developer", "content": "Use English."},
            {"role": "user", "content": "Hi"},
        ],
        [
            {"role": "user", "content": "1+1?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Add one."},
                    {"type": "text", "text": "2"},
                ],
            },
            {"role": "user", "content": "Why?"},
        ],
        [{"role": "user", "content": "Hi"}, {"role": "system", "content": "ignored"}],
    ],
)
def test_hf_parity_and_loss_boundary(renderer: BeamRenderer, messages: list[dict[str, Any]]) -> None:
    reference = [renderer.to_openai_message(m) for m in messages]
    prompt = tokens(renderer.build_generation_prompt(messages))
    assert prompt == renderer.tokenizer.apply_chat_template(reference, add_generation_prompt=True, return_dict=False)
    answer = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Think."},
            {"type": "text", "text": "Hello"},
        ],
    }
    model_input, weights = renderer.build_supervised_example(messages + [answer])
    rendered = tokens(model_input)
    assert rendered == renderer.tokenizer.apply_chat_template(
        reference + [renderer.to_openai_message(answer)],
        add_generation_prompt=False,
        return_dict=False,
    )
    assert rendered[: len(prompt)] == prompt
    assert weights[: len(prompt)].sum() == 0
    assert weights[len(prompt) :].tolist() == [1] * (len(rendered) - len(prompt))
    parsed, termination = renderer.parse_response(rendered[len(prompt) :])
    assert termination == ParseTermination.STOP_SEQUENCE
    assert parsed == answer


def test_tool_calls_rejected(renderer: BeamRenderer) -> None:
    call = ToolCall(function=ToolCall.FunctionBody(name="lookup", arguments='{"city": "123"}'))
    with pytest.raises(ValueError, match="tool calling"):
        renderer.build_generation_prompt([{"role": "assistant", "content": "", "tool_calls": [call]}])
    with pytest.raises(ValueError, match="tool calling"):
        renderer.create_conversation_prefix_with_tools([{"name": "lookup"}])


@pytest.mark.parametrize(
    "fields",
    [
        {"reasoning": "fallback", "reasoning_content": "preferred"},
        {"reasoning": "fallback", "reasoning_content": ""},
        {"thinking": "explicit", "reasoning_content": "ignored"},
        {"thinking": "", "reasoning_content": "fallback"},
    ],
)
def test_reasoning_precedence(renderer: BeamRenderer, fields: dict[str, str]) -> None:
    raw = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello", **fields},
    ]
    expected = renderer.tokenizer.apply_chat_template(raw, add_generation_prompt=False, return_dict=False)
    actual, _ = renderer.build_supervised_example(normalize_messages(raw))
    assert tokens(actual) == expected


@pytest.mark.parametrize("suffix", ["", "<|return|>extra", "<|return|><|return|>", "<|end|>"])
def test_malformed_termination(renderer: BeamRenderer, suffix: str) -> None:
    response = renderer.tokenizer.encode("<|channel|>final<|message|>Hello" + suffix, add_special_tokens=False)
    assert renderer.parse_response(response)[1] == ParseTermination.MALFORMED


def test_custom_weights_and_extension(renderer: BeamRenderer) -> None:
    messages = [
        {"role": "user", "content": "Hi", "trainable": False},
        {"role": "assistant", "content": "Hello", "trainable": True},
    ]
    model_input, weights = renderer.build_supervised_example(messages, TrainOnWhat.CUSTOMIZED)
    assert weights.sum() > 0
    next_prompt = tokens(renderer.build_generation_prompt(messages + [{"role": "user", "content": "Again"}]))
    assert next_prompt[: len(tokens(model_input))] == tokens(model_input)


@pytest.mark.parametrize(
    "model",
    [
        MODEL,
        "reflection/Beam-1.0-501B-A23B",
        "accounts/fireworks/models/beam-1p0-501b-a23b-nvfp4",
    ],
)
def test_automatic_selection(model: str) -> None:
    assert resolve_renderer_name(model) == "beam"
    assert resolve_renderer_name(model, "explicit") == "explicit"


@pytest.mark.parametrize("channel", ["analysis", "final"])
@pytest.mark.parametrize("suffix", ["", "<|en"])
def test_truncated_body_keeps_channel(renderer: BeamRenderer, channel: str, suffix: str) -> None:
    wire = f"<|channel|>{channel}<|message|>Partial content" + suffix
    message, termination = renderer.parse_response(renderer.tokenizer.encode(wire, add_special_tokens=False))
    expected = (
        {"type": "thinking", "thinking": "Partial content" + suffix}
        if channel == "analysis"
        else {"type": "text", "text": "Partial content" + suffix}
    )
    assert message["content"] == [expected]
    assert get_text_content(message) == ("" if channel == "analysis" else "Partial content" + suffix)
    assert termination == ParseTermination.MALFORMED


@pytest.mark.parametrize(
    "tail", ["<|start|>assistant<|channel|>fin", "<|start|>assistant<|channel|>final<|message|>Answer"]
)
def test_truncation_after_complete_analysis(renderer: BeamRenderer, tail: str) -> None:
    wire = "<|channel|>analysis<|message|>Reasoning<|end|>" + tail
    message, termination = renderer.parse_response(renderer.tokenizer.encode(wire, add_special_tokens=False))
    assert message["content"][0] == {"type": "thinking", "thinking": "Reasoning"}
    assert get_text_content(message) == ("Answer" if tail.endswith("Answer") else "")
    assert termination == ParseTermination.MALFORMED


@pytest.mark.parametrize("wire", ["<|chan", "<|channel|>analysis<|mess", "<|channel|>unknown<|message|>Unknown"])
def test_incomplete_or_unknown_header_is_not_answer(renderer: BeamRenderer, wire: str) -> None:
    message, termination = renderer.parse_response(renderer.tokenizer.encode(wire, add_special_tokens=False))
    assert message["content"] == []
    assert termination == ParseTermination.MALFORMED


@pytest.mark.parametrize("channel", ["analysis", "final"])
@pytest.mark.parametrize("body", ["Print <|foo and continue", "Print <|", "Print <|en", "a <|foo|> b <|bar"])
def test_truncated_literal_angle_pipe_is_preserved(renderer: BeamRenderer, channel: str, body: str) -> None:
    wire = f"<|channel|>{channel}<|message|>{body}"
    message, termination = renderer.parse_response(renderer.tokenizer.encode(wire, add_special_tokens=False))
    key = "thinking" if channel == "analysis" else "text"
    assert message["content"] == [{"type": key, key: body}]
    assert termination == ParseTermination.MALFORMED


def test_invalid_control_marker_is_not_answer_text(renderer: BeamRenderer) -> None:
    wire = "<|channel|>final<|message|>Answer<|call|>invalid tail"
    message, termination = renderer.parse_response(renderer.tokenizer.encode(wire, add_special_tokens=False))
    assert get_text_content(message) == "Answer"
    assert termination == ParseTermination.MALFORMED


def test_beam_qa_loader_does_not_construct_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    path = os.environ.get("BEAM_TOKENIZER_PATH", MODEL)
    revision = REVISION if path == MODEL else None

    def reject_model_config(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Tokenizer QA must not instantiate Beam's model architecture config")

    monkeypatch.setattr(AutoConfig, "from_pretrained", reject_model_config)
    _load_tokenizer.cache_clear()
    try:
        oracle = _load_tokenizer(path, revision, False, renderer_name="beam")
        production = load_tokenizer(path, revision, False)
        messages = [{"role": "user", "content": "Hello 👋"}]
        assert oracle.apply_chat_template(messages, return_dict=False) == production.apply_chat_template(
            messages, return_dict=False
        )
        assert oracle.eos_token_id == production.eos_token_id
    finally:
        _load_tokenizer.cache_clear()
