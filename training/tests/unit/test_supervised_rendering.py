from __future__ import annotations

import os

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
    render_messages_to_datums,
)


class StubRenderer:
    def __init__(self, tokens: list[int], weights: list[float]):
        self.tokens = torch.tensor(tokens, dtype=torch.int64)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.calls: list[tuple[list[dict], TrainOnWhat]] = []

    def build_supervised_example(
        self, messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    ):
        self.calls.append((messages, train_on_what))
        return self.tokens, self.weights


class SequenceRenderer:
    def __init__(self, outputs: list[tuple[list[int], list[float]]]):
        self.outputs = [
            (
                torch.tensor(tokens, dtype=torch.int64),
                torch.tensor(weights, dtype=torch.float32),
            )
            for tokens, weights in outputs
        ]
        self.calls: list[tuple[list[dict], TrainOnWhat]] = []

    def build_supervised_example(
        self, messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    ):
        self.calls.append((messages, train_on_what))
        return self.outputs[len(self.calls) - 1]


class MultiExampleStubRenderer:
    """Renderer mimicking KimiK2.build_supervised_examples: yields N examples
    for a multi-turn conversation."""

    def __init__(self, outputs: list[tuple[list[int], list[float]]]):
        self.outputs = [
            (
                torch.tensor(tokens, dtype=torch.int64),
                torch.tensor(weights, dtype=torch.float32),
            )
            for tokens, weights in outputs
        ]
        self.single_calls: list[tuple[list[dict], TrainOnWhat]] = []
        self.plural_calls: list[tuple[list[dict], TrainOnWhat]] = []

    def build_supervised_example(
        self, messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    ):
        self.single_calls.append((messages, train_on_what))
        return self.outputs[0]

    def build_supervised_examples(
        self, messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    ):
        self.plural_calls.append((messages, train_on_what))
        return list(self.outputs)


class ModelInputRenderer:
    def __init__(self):
        self.calls: list[tuple[list[dict], TrainOnWhat]] = []

    def build_supervised_example(
        self, messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    ):
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
    assert [m["role"] for m in normalized_messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES

    assert rendered.token_ids == [10, 11, 12, 13, 14, 15, 16, 17, 18]
    assert rendered.token_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert rendered.datum.loss_fn_inputs["target_tokens"].data == [
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    ]
    assert rendered.datum.loss_fn_inputs["weights"].data == [
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]


def test_render_messages_to_datums_uses_plural_build_supervised_examples():
    """``render_messages_to_datums`` must call ``build_supervised_examples``
    (plural) so renderers that strip thinking from history (KimiK2/K2.5/K2.6,
    Qwen3-thinking, DeepSeekV3-thinking) yield one training example per
    assistant turn. Calling the singular ``build_supervised_example`` with
    ALL_ASSISTANT_MESSAGES produces a single sequence in which every
    intermediate assistant turn has an empty ``<think></think>`` block, which
    is exactly the regression that caused fine-tuned Kimi models to stop
    emitting reasoning traces at inference.
    """
    renderer = MultiExampleStubRenderer(
        outputs=[
            ([10, 11, 12, 13], [0, 0, 1, 1]),
            ([10, 11, 12, 13, 14, 15, 16, 17], [0, 0, 0, 0, 0, 0, 1, 1]),
        ]
    )

    rendered = render_messages_to_datums(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "reasoning_content": "t1", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "reasoning_content": "t2", "content": "a2"},
        ],
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )

    assert renderer.single_calls == []
    assert len(renderer.plural_calls) == 1
    _, train_on_what = renderer.plural_calls[0]
    assert train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES

    assert len(rendered) == 2
    assert rendered[0].token_ids == [10, 11, 12, 13]
    assert rendered[1].token_ids == [10, 11, 12, 13, 14, 15, 16, 17]


def test_render_messages_to_datums_filters_normalized_reasoning_content():
    """``reasoning_content`` must be promoted to a ThinkingPart before the
    normalized messages reach ``build_supervised_examples`` so the renderer's
    per-turn rendering can fill the ``<think>...</think>`` block for every
    assistant turn instead of emitting an empty placeholder."""
    renderer = MultiExampleStubRenderer(
        outputs=[
            ([10, 11, 12, 13], [0, 0, 1, 1]),
            ([10, 11, 12, 13, 14, 15, 16, 17], [0, 0, 0, 0, 0, 0, 1, 1]),
        ]
    )

    render_messages_to_datums(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "reasoning_content": "t1", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "reasoning_content": "t2", "content": "a2"},
        ],
        renderer=renderer,
    )

    normalized_messages, _ = renderer.plural_calls[0]
    assistant_messages = [m for m in normalized_messages if m["role"] == "assistant"]
    assert len(assistant_messages) == 2
    for msg in assistant_messages:
        assert msg["content"][0] == {"type": "thinking", "thinking": msg["content"][0]["thinking"]}
        assert msg["content"][0]["thinking"] in {"t1", "t2"}


def test_render_messages_to_datum_still_uses_singular_for_back_compat():
    """``render_messages_to_datum`` (singular) must keep calling
    ``build_supervised_example`` for callers that expect a single datum; the
    fix for multi-turn thinking lives in ``render_messages_to_datums``
    (plural)."""
    renderer = MultiExampleStubRenderer(
        outputs=[
            ([10, 11, 12, 13], [0, 0, 1, 1]),
        ]
    )

    rendered = render_messages_to_datum(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
        renderer=renderer,
    )

    assert renderer.plural_calls == []
    assert len(renderer.single_calls) == 1
    assert rendered.token_ids == [10, 11, 12, 13]


def test_render_messages_to_datum_supports_multimodal_model_input():
    renderer = ModelInputRenderer()

    rendered = render_messages_to_datum(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    },
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
    assert rendered.datum.loss_fn_inputs["weights"].data == [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]
    assert len(rendered.datum.model_input.chunks) == 3


def test_build_datum_from_token_mask_reuses_ui_mask_semantics():
    rendered = build_datum_from_token_mask(
        token_ids=[100, 101, 102, 103, 104, 105],
        token_mask=[0, 0, 1, 1, 0, 2],
        include_loss_mask=True,
    )

    assert rendered.token_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert rendered.datum.loss_fn_inputs["target_tokens"].data == [
        101,
        102,
        103,
        104,
        105,
    ]
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


def test_normalize_messages_promotes_reasoning_content_to_thinking_part():
    """OpenAI-style ``reasoning_content`` should become a ThinkingPart.

    Datasets produced by Fireworks/OpenAI-compatible APIs store the
    assistant's chain-of-thought in a top-level ``reasoning_content``
    field rather than Tinker's ``thinking`` field. Without this alias,
    renderers like KimiK2Renderer see an empty ``thinking_content``
    string and emit an empty ``<think></think>`` block, so the model
    never learns to produce reasoning traces.
    """
    normalized = normalize_messages(
        [
            {
                "role": "assistant",
                "reasoning_content": "let me compute 2+2",
                "content": "The answer is 4",
            },
        ]
    )

    assert normalized[0]["content"] == [
        {"type": "thinking", "thinking": "let me compute 2+2"},
        {"type": "text", "text": "The answer is 4"},
    ]


def test_normalize_messages_reasoning_content_with_no_text_content():
    """``reasoning_content`` alone should still produce a ThinkingPart.

    Some reasoning-only turns may carry an empty ``content`` string but
    a non-empty ``reasoning_content``. The resulting content must keep
    the ThinkingPart so downstream renderers can still fill the
    ``<think>...</think>`` block during training.
    """
    normalized = normalize_messages(
        [
            {
                "role": "assistant",
                "reasoning_content": "some thoughts",
                "content": "",
            },
        ]
    )

    assert normalized[0]["content"] == [
        {"type": "thinking", "thinking": "some thoughts"},
        {"type": "text", "text": ""},
    ]


def test_normalize_messages_thinking_wins_over_reasoning_content():
    """If both fields are present, ``thinking`` is preserved as-is.

    Keeps a single source of truth per message to avoid duplicating the
    chain-of-thought when a caller supplies both the Tinker-native
    ``thinking`` field and the OpenAI-style ``reasoning_content``.
    """
    normalized = normalize_messages(
        [
            {
                "role": "assistant",
                "thinking": "native thinking",
                "reasoning_content": "openai reasoning",
                "content": "answer",
            },
        ]
    )

    assert normalized[0]["content"] == [
        {"type": "thinking", "thinking": "native thinking"},
        {"type": "text", "text": "answer"},
    ]


def test_normalize_messages_rejects_non_string_reasoning_content():
    """Non-string ``reasoning_content`` values should raise TypeError."""
    with pytest.raises(TypeError):
        normalize_messages(
            [
                {
                    "role": "assistant",
                    "reasoning_content": ["not", "a", "string"],
                    "content": "answer",
                },
            ]
        )


def test_build_renderer_uses_image_processor_for_vl_renderers(monkeypatch):
    calls: list[tuple[str, object | None]] = []

    def fake_get_image_processor(model_name):
        assert model_name == "Qwen/Qwen3-VL-30B-A3B-Instruct"
        return "image-processor"

    def fake_get_renderer(name, tokenizer, image_processor=None):
        calls.append((name, image_processor))
        return "renderer"

    monkeypatch.setattr(
        "training.utils.supervised.get_image_processor", fake_get_image_processor
    )
    monkeypatch.setattr("training.utils.supervised.get_renderer", fake_get_renderer)

    renderer = build_renderer(
        tokenizer="tok",
        tokenizer_model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        renderer_name="qwen3_vl_instruct",
    )

    assert renderer == "renderer"
    assert calls == [("qwen3_vl_instruct", "image-processor")]


def test_build_renderer_opts_in_trust_remote_code_for_kimi_k2_6(monkeypatch):
    """Kimi-K2.6 ships a custom image processor not covered by tinker_cookbook's
    hardcoded trust_remote_code=True list; build_renderer must set the
    HF_TRUST_REMOTE_CODE env var before calling get_image_processor so the
    cached AutoImageProcessor load succeeds non-interactively in CI."""
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)

    env_at_call: list[str | None] = []

    def fake_get_image_processor(model_name):
        env_at_call.append(os.environ.get("HF_TRUST_REMOTE_CODE"))
        return "image-processor"

    def fake_get_renderer(name, tokenizer, image_processor=None):
        return ("renderer", name, image_processor)

    monkeypatch.setattr(
        "training.utils.supervised.get_image_processor", fake_get_image_processor
    )
    monkeypatch.setattr("training.utils.supervised.get_renderer", fake_get_renderer)

    result = build_renderer(
        tokenizer="tok",
        tokenizer_model="moonshotai/Kimi-K2.6",
    )

    assert env_at_call == ["1"]
    assert result == ("renderer", "kimi_k25", "image-processor")


def test_build_renderer_does_not_touch_trust_remote_code_for_kimi_k2_5(monkeypatch):
    """K2.5 is already covered by tinker_cookbook's hardcoded trust_remote_code
    branch, so our opt-in helper must leave HF_TRUST_REMOTE_CODE unset for it."""
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)

    env_at_call: list[str | None] = []

    def fake_get_image_processor(model_name):
        env_at_call.append(os.environ.get("HF_TRUST_REMOTE_CODE"))
        return "image-processor"

    def fake_get_renderer(name, tokenizer, image_processor=None):
        return "renderer"

    monkeypatch.setattr(
        "training.utils.supervised.get_image_processor", fake_get_image_processor
    )
    monkeypatch.setattr("training.utils.supervised.get_renderer", fake_get_renderer)

    build_renderer(
        tokenizer="tok",
        tokenizer_model="moonshotai/Kimi-K2.5",
    )

    assert env_at_call == [None]


def test_build_renderer_preserves_existing_trust_remote_code_value(monkeypatch):
    """Don't stomp a user-set HF_TRUST_REMOTE_CODE — setdefault semantics."""
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "0")

    env_at_call: list[str | None] = []

    def fake_get_image_processor(model_name):
        env_at_call.append(os.environ.get("HF_TRUST_REMOTE_CODE"))
        return "image-processor"

    def fake_get_renderer(name, tokenizer, image_processor=None):
        return "renderer"

    monkeypatch.setattr(
        "training.utils.supervised.get_image_processor", fake_get_image_processor
    )
    monkeypatch.setattr("training.utils.supervised.get_renderer", fake_get_renderer)

    build_renderer(
        tokenizer="tok",
        tokenizer_model="moonshotai/Kimi-K2.6",
    )

    assert env_at_call == ["0"]


def test_resolve_renderer_name_prefers_kimi_k25_for_kimi_k2_5():
    assert resolve_renderer_name("moonshotai/Kimi-K2.5") == "kimi_k25"


def test_resolve_renderer_name_prefers_kimi_k25_for_kimi_k2_6():
    assert resolve_renderer_name("moonshotai/Kimi-K2.6") == "kimi_k25"


def test_resolve_renderer_name_prefers_minimax_m2() -> None:
    """MiniMax M2 tokenizers should resolve to the custom renderer."""
    assert resolve_renderer_name("MiniMaxAI/MiniMax-M2") == "minimax_m2"


def test_resolve_renderer_name_prefers_qwen3_5() -> None:
    """Qwen3.5 models should resolve to the qwen3_5 renderer."""
    assert resolve_renderer_name("Qwen/Qwen3.5-9B") == "qwen3_5"
    assert resolve_renderer_name("Qwen/Qwen3.5-4B") == "qwen3_5"
    assert resolve_renderer_name("Qwen/Qwen3.5-27B") == "qwen3_5"
    assert resolve_renderer_name("Qwen/Qwen3.5-35B-A3B") == "qwen3_5"
    assert resolve_renderer_name("Qwen/Qwen3.5-397B-A17B") == "qwen3_5"


def test_resolve_renderer_name_prefers_gemma4() -> None:
    """Gemma 4 models should resolve to the gemma4 renderer."""
    assert resolve_renderer_name("google/gemma-4-12b-it") == "gemma4"
    assert resolve_renderer_name("google/gemma-4-27b-it") == "gemma4"


def test_build_renderer_resolves_minimax_m2(monkeypatch) -> None:
    """build_renderer should resolve minimax_m2 and dispatch to get_renderer."""
    calls: list[tuple[str, object]] = []

    def fake_get_renderer(name: str, tokenizer, image_processor=None):
        calls.append(("get", name))
        assert tokenizer == "tok"
        assert image_processor is None
        return "renderer"

    monkeypatch.setattr("training.utils.supervised.get_renderer", fake_get_renderer)

    renderer = build_renderer(
        tokenizer="tok",
        tokenizer_model="MiniMaxAI/MiniMax-M2",
    )

    assert renderer == "renderer"
    assert ("get", "minimax_m2") in calls


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
        {
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "good"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "bad"},
            ]
        },
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
    assert [m["role"] for m in chosen_messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [m["content"] for m in chosen_messages] == ["u1", "a1", "u2", "chosen"]
    assert [m["role"] for m in rejected_messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
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
