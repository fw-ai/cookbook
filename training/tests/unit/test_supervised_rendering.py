from __future__ import annotations

import pytest
import torch
import tinker
from tinker_cookbook.renderers import TrainOnWhat

from training.utils.losses import make_batch_weighted_sft_loss_fn
from training.utils.data import prepare_sampling_messages
from training.utils.supervised import (
    build_renderer,
    build_datum_from_token_mask,
    resolve_renderer_name,
    render_preference_pair,
    normalize_messages,
    render_messages_to_datum,
)


class StubRenderer:
    def __init__(self, tokens: list[int], weights: list[float]):
        self.tokens = torch.tensor(tokens, dtype=torch.int64)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.calls: list[tuple[list[dict], TrainOnWhat]] = []

    def build_supervised_example(self, messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE):
        self.calls.append((messages, train_on_what))
        return self.tokens, self.weights


class SequenceRenderer:
    def __init__(self, outputs: list[tuple[list[int], list[float]]]):
        self.outputs = [
            (torch.tensor(tokens, dtype=torch.int64), torch.tensor(weights, dtype=torch.float32))
            for tokens, weights in outputs
        ]
        self.calls: list[tuple[list[dict], TrainOnWhat]] = []

    def build_supervised_example(self, messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE):
        self.calls.append((messages, train_on_what))
        return self.outputs[len(self.calls) - 1]


class ModelInputRenderer:
    def __init__(self):
        self.calls: list[tuple[list[dict], TrainOnWhat]] = []

    def build_supervised_example(self, messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE):
        self.calls.append((messages, train_on_what))
        model_input = tinker.ModelInput(
            chunks=[
                tinker.EncodedTextChunk(tokens=[10, 11]),
                tinker.types.ImageAssetPointerChunk(
                    location="https://example.com/cat.png",
                    format="png",
                    expected_tokens=3,
                ),
                tinker.EncodedTextChunk(tokens=[12, 13]),
            ]
        )
        weights = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.float32)
        return model_input, weights


def test_render_messages_to_datum_preserves_multi_turn_weights():
    renderer = StubRenderer(
        tokens=[10, 11, 12, 13, 14, 15, 16, 17, 18],
        weights=[0, 0, 1, 1, 0, 0, 1, 1, 1],
    )

    rendered = render_messages_to_datum(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ],
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )

    normalized_messages, train_on_what = renderer.calls[0]
    assert [m["role"] for m in normalized_messages] == ["user", "assistant", "user", "assistant"]
    assert train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES

    assert rendered.token_ids == [10, 11, 12, 13, 14, 15, 16, 17, 18]
    assert rendered.token_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert rendered.datum.loss_fn_inputs["target_tokens"].data == [11, 12, 13, 14, 15, 16, 17, 18]
    assert rendered.datum.loss_fn_inputs["weights"].data == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_render_messages_to_datum_supports_multimodal_model_input():
    renderer = ModelInputRenderer()

    rendered = render_messages_to_datum(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                    {"type": "text", "text": " now"},
                ],
            },
            {"role": "assistant", "content": "cat"},
        ],
        renderer=renderer,
    )

    normalized_messages, train_on_what = renderer.calls[0]
    assert train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES
    assert normalized_messages[0]["content"] == [
        {"type": "text", "text": "look at this"},
        {"type": "image", "image": "https://example.com/cat.png"},
        {"type": "text", "text": " now"},
    ]
    assert rendered.token_ids[:2] == [10, 11]
    assert len(rendered.token_ids) == 7
    assert len(rendered.token_weights) == 7
    assert rendered.token_weights == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert rendered.datum.loss_fn_inputs["target_tokens"].data == [11, 12, 13]
    assert rendered.datum.loss_fn_inputs["weights"].data == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert len(rendered.datum.model_input.chunks) == 3


def test_build_datum_from_token_mask_reuses_ui_mask_semantics():
    rendered = build_datum_from_token_mask(
        token_ids=[100, 101, 102, 103, 104, 105],
        token_mask=[0, 0, 1, 1, 0, 2],
        include_loss_mask=True,
    )

    assert rendered.token_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert rendered.datum.loss_fn_inputs["target_tokens"].data == [101, 102, 103, 104, 105]
    assert rendered.datum.loss_fn_inputs["weights"].data == [0.0, 1.0, 1.0, 0.0, 1.0]
    assert rendered.datum.loss_fn_inputs["loss_mask"].data == [0.0, 1.0, 1.0, 0.0, 1.0]


def test_normalize_messages_supports_openai_tool_call_shape():
    normalized = normalize_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lake_move",
                            "arguments": '{"action":"RIGHT"}',
                        },
                    }
                ],
            }
        ]
    )

    tool_call = normalized[0]["tool_calls"][0]
    assert tool_call.function.name == "lake_move"
    assert tool_call.function.arguments == '{"action": "RIGHT"}'


def test_normalize_messages_keeps_tool_metadata_and_thinking_parts():
    normalized = normalize_messages(
        [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "lake_move",
                "content": "board state",
            },
            {
                "role": "assistant",
                "thinking": "consider options",
                "content": "RIGHT",
            },
        ]
    )

    assert normalized[0]["tool_call_id"] == "call_1"
    assert normalized[0]["name"] == "lake_move"
    assert normalized[1]["content"] == [
        {"type": "thinking", "thinking": "consider options"},
        {"type": "text", "text": "RIGHT"},
    ]


def test_build_renderer_uses_image_processor_for_vl_renderers(monkeypatch):
    calls: list[tuple[str, object | None]] = []

    def fake_get_image_processor(model_name):
        assert model_name == "Qwen/Qwen3-VL-30B-A3B-Instruct"
        return "image-processor"

    def fake_get_renderer(name, tokenizer, image_processor=None):
        calls.append((name, image_processor))
        return "renderer"

    monkeypatch.setattr("training.utils.supervised.get_image_processor", fake_get_image_processor)
    monkeypatch.setattr("training.utils.supervised.get_renderer", fake_get_renderer)

    renderer = build_renderer(
        tokenizer="tok",
        hf_tokenizer_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        renderer_name="qwen3_vl_instruct",
    )

    assert renderer == "renderer"
    assert calls == [("qwen3_vl_instruct", "image-processor")]


def test_resolve_renderer_name_prefers_kimi_k25_for_kimi_k2_5():
    assert resolve_renderer_name("moonshotai/Kimi-K2.5") == "kimi_k25"


def test_weighted_sft_loss_uses_sparse_weights():
    datum_a = build_datum_from_token_mask(
        token_ids=[10, 11, 12, 13],
        token_mask=[0, 0, 1, 0],
    ).datum
    datum_b = build_datum_from_token_mask(
        token_ids=[20, 21, 22],
        token_mask=[0, 1, 1],
    ).datum

    loss_fn = make_batch_weighted_sft_loss_fn()
    loss, metrics = loss_fn(
        [datum_a, datum_b],
        [
            torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float32),
            torch.tensor([-0.4, -0.5], dtype=torch.float32),
        ],
    )

    assert loss.item() == pytest.approx(1.1 / 3.0)
    assert metrics["ce_loss_sum"] == pytest.approx(1.1)
    assert metrics["response_tokens"] == pytest.approx(3.0)
    assert metrics["weighted_tokens"] == pytest.approx(3.0)


def test_render_preference_pair_uses_shared_renderer_path():
    renderer = SequenceRenderer(
        outputs=[
            ([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1]),
            ([1, 2, 3, 9, 10], [0, 0, 0, 1, 1]),
        ]
    )

    pair = render_preference_pair(
        {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "good"}]},
        {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "bad"}]},
        renderer=renderer,
        tokenizer=None,
    )

    assert pair is not None
    assert pair.chosen_tokens == [1, 2, 3, 4, 5, 6]
    assert pair.rejected_tokens == [1, 2, 3, 9, 10]
    assert pair.response_start == 3
    assert pair.chosen_datum.loss_fn_inputs["target_tokens"].data == [2, 3, 4, 5, 6]
    assert len(renderer.calls) == 2


def test_render_preference_pair_preserves_multi_turn_history():
    renderer = SequenceRenderer(
        outputs=[
            ([1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 1, 1, 1]),
            ([1, 2, 3, 4, 9, 10], [0, 0, 0, 0, 1, 1]),
        ]
    )

    pair = render_preference_pair(
        {
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "chosen"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "rejected"},
            ]
        },
        renderer=renderer,
        tokenizer=None,
    )

    assert pair is not None
    assert pair.response_start == 4
    chosen_messages, _ = renderer.calls[0]
    rejected_messages, _ = renderer.calls[1]
    assert [m["role"] for m in chosen_messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in chosen_messages] == ["u1", "a1", "u2", "chosen"]
    assert [m["role"] for m in rejected_messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in rejected_messages] == ["u1", "a1", "u2", "rejected"]


def test_prepare_sampling_messages_only_strips_trailing_assistant():
    prepared = prepare_sampling_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
    )

    assert [m["role"] for m in prepared] == ["system", "user", "assistant", "user"]
