from __future__ import annotations

import os

import pytest
import torch
import tinker
from tinker_cookbook.renderers import (
    Message,
    ParseTermination,
    RenderContext,
    Renderer,
    TrainOnWhat,
)
from tinker_cookbook.renderers.base import RenderedMessage

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin
from training.utils.losses import make_batch_weighted_sft_loss_fn
from training.utils.data import prepare_sampling_messages
from training.utils.supervised import (
    build_renderer,
    build_renderer_from_resolved_name,
    build_datum_from_token_mask,
    populate_render_worker_state,
    resolve_renderer_name,
    render_preference_pair,
    normalize_messages,
    render_messages_to_datum,
    render_messages_to_datums,
    renderer_supports_images,
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


class AtomicPreferenceRenderer(SequenceRenderer):
    def build_supervised_examples(self, *args, **kwargs):
        raise AssertionError("preference pairs must not use SFT unrolling")


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


class SplitRenderer:
    has_extension_property = False

    def __init__(self):
        self.calls: list[tuple[list[dict], TrainOnWhat]] = []

    def build_supervised_examples(
        self,
        messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    ):
        self.calls.append((messages, train_on_what))
        return [
            (
                torch.tensor([10, 11, 12], dtype=torch.int64),
                torch.tensor([0, 1, 1], dtype=torch.float32),
            ),
            (
                torch.tensor([20, 21, 22], dtype=torch.int64),
                torch.tensor([0, 1, 1], dtype=torch.float32),
            ),
        ]

    def build_supervised_example(self, messages, train_on_what):
        raise AssertionError("render_messages_to_datums should use split examples")


class DisaggregateRecordingRenderer(DisaggregateMultiTurnMixin):
    """Real unroll mixin over a renderer that records each per-split call."""

    has_extension_property = False

    def __init__(self):
        self.calls: list[tuple[list[dict], TrainOnWhat]] = []

    def build_supervised_example(
        self, messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    ):
        self.calls.append(([dict(message) for message in messages], train_on_what))
        # Tokens depend only on the conversation, like a real renderer: the mask
        # mode never moves them, and rendering the same prefix twice matches.
        token = len(messages)
        return (
            torch.tensor([token, token, token], dtype=torch.int64),
            torch.tensor([0, 1, 1], dtype=torch.float32),
        )


class BaseToolPrefixRenderer(Renderer):
    def __init__(self):
        super().__init__(tokenizer=None)
        self.calls: list[tuple[list[dict], TrainOnWhat]] = []

    def get_stop_sequences(self):
        return []

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        raise AssertionError("test uses build_supervised_example directly")

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        return Message(role="assistant", content=""), ParseTermination.EOS

    def build_supervised_example(
        self, messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    ):
        self.calls.append((messages, train_on_what))
        return (
            torch.tensor([10, 11, 12], dtype=torch.int64),
            torch.tensor([0, 1, 1], dtype=torch.float32),
        )


class BaseWarningRenderer(Renderer):
    def __init__(self):
        super().__init__(tokenizer=None)
        self.calls: list[TrainOnWhat] = []

    def get_stop_sequences(self):
        return []

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role_token = 10 if message["role"] == "user" else 20
        return RenderedMessage(
            output=[
                tinker.EncodedTextChunk(
                    tokens=[role_token + (ctx.idx * 2), role_token + (ctx.idx * 2) + 1]
                )
            ]
        )

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        return Message(role="assistant", content=""), ParseTermination.EOS

    def build_supervised_example(
        self, messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    ):
        self.calls.append(train_on_what)
        return super().build_supervised_example(messages, train_on_what=train_on_what)


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


def test_render_messages_to_datums_supports_per_example_mean_reduction():
    renderer = StubRenderer(
        tokens=[10, 11, 12, 13],
        weights=[0, 1, 1, 1],
    )

    [rendered] = render_messages_to_datums(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ],
        renderer=renderer,
        train_on_what="last_assistant_turn",
        reduction="mean",
    )

    assert rendered.datum.loss_fn_inputs["weights"].data == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3]
    )
    assert sum(rendered.datum.loss_fn_inputs["weights"].data) == pytest.approx(1.0)


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
    assert train_on_what == TrainOnWhat.LAST_ASSISTANT_TURN
    assert normalized_messages[0]["content"] == [
        {"type": "text", "text": "look at this"},
        {"type": "image", "image": "https://example.com/cat.png"},
        {"type": "text", "text": " now"},
    ]
    assert rendered.token_ids[:2] == [10, 11]
    assert len(rendered.token_ids) == 7
    assert len(rendered.token_weights) == 7
    assert rendered.token_weights == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert rendered.datum.loss_fn_inputs["target_tokens"].data == [
        11,
        0,
        0,
        0,
        12,
        13,
    ]
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


def test_render_messages_to_datums_uses_renderer_split_for_all_assistant_messages():
    renderer = SplitRenderer()

    rendered = render_messages_to_datums(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ],
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )

    assert len(rendered) == 2
    assert [example.token_ids for example in rendered] == [[10, 11, 12], [20, 21, 22]]
    assert renderer.calls[0][1] == TrainOnWhat.ALL_ASSISTANT_MESSAGES


def test_render_messages_to_datums_suppresses_fake_single_target_warning(caplog):
    renderer = BaseWarningRenderer()
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    expected = render_messages_to_datums(
        messages,
        renderer=renderer,
        train_on_what="last_assistant_turn",
    )
    caplog.clear()

    actual = render_messages_to_datums(
        messages,
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )

    assert [example.token_ids for example in actual] == [
        example.token_ids for example in expected
    ]
    assert [example.token_weights for example in actual] == [
        example.token_weights for example in expected
    ]
    assert renderer.calls[-1] == TrainOnWhat.LAST_ASSISTANT_TURN
    assert "does not satisfy the extension property" not in caplog.text


def test_render_messages_to_datums_preserves_last_turn_with_multiple_assistants(caplog):
    renderer = BaseWarningRenderer()
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},
    ]

    expected = render_messages_to_datums(
        messages,
        renderer=renderer,
        train_on_what="last_assistant_turn",
    )
    caplog.clear()

    actual = render_messages_to_datums(
        messages,
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )

    assert [example.token_ids for example in actual] == [
        example.token_ids for example in expected
    ]
    assert [example.token_weights for example in actual] == [
        example.token_weights for example in expected
    ]
    assert renderer.calls[-1] == TrainOnWhat.LAST_ASSISTANT_TURN
    assert "does not satisfy the extension property" not in caplog.text


def test_render_messages_to_datums_keeps_real_extension_warning(caplog):
    renderer = BaseWarningRenderer()
    messages = [
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    all_assistant = render_messages_to_datums(
        messages,
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )
    last_turn = render_messages_to_datums(
        messages,
        renderer=renderer,
        train_on_what="last_assistant_turn",
    )

    assert [example.token_weights for example in all_assistant] != [
        example.token_weights for example in last_turn
    ]
    assert renderer.calls[0] == TrainOnWhat.ALL_ASSISTANT_MESSAGES
    assert "does not satisfy the extension property" in caplog.text


def test_render_messages_to_datums_uses_renderer_split_for_weighted_rows():
    renderer = SplitRenderer()

    rendered = render_messages_to_datums(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1", "weight": 0},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ],
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )

    normalized_messages, train_on_what = renderer.calls[0]
    assert len(rendered) == 2
    assert [example.token_ids for example in rendered] == [[10, 11, 12], [20, 21, 22]]
    assert train_on_what == TrainOnWhat.CUSTOMIZED
    assert [message["trainable"] for message in normalized_messages] == [
        False,
        False,
        False,
        True,
    ]


def test_weighted_row_trains_each_assistant_turn_in_exactly_one_split():
    """A weighted row renders with ``CUSTOMIZED``, so each per-user-turn
    split must be reduced to its own terminal turn. Otherwise ``CUSTOMIZED``
    re-weights every earlier assistant turn in every later split, training
    history whose thinking has already been stripped."""

    renderer = DisaggregateRecordingRenderer()

    render_messages_to_datums(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3", "weight": 0},
            {"role": "user", "content": "u4"},
            {"role": "assistant", "content": "a4"},
        ],
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )

    # The a3 round is skipped entirely (its terminal assistant is masked),
    # and each remaining round trains only its own answer.
    assert [
        [message["content"] for message in messages] for messages, _ in renderer.calls
    ] == [
        ["u1", "a1"],
        ["u1", "a1", "u2", "a2"],
        ["u1", "a1", "u2", "a2", "u3", "a3", "u4", "a4"],
    ]
    # Every terminal turn here is fully trainable, so the per-message flags
    # carry no information and the split renders like an unweighted row.
    assert all(
        train_on_what == TrainOnWhat.LAST_ASSISTANT_TURN
        for _, train_on_what in renderer.calls
    )
    assert all(
        "trainable" not in message
        for messages, _ in renderer.calls
        for message in messages
    )


def test_weighted_row_masks_terminal_turn_down_to_its_answer():
    """A terminal turn masked down to its final answer states exactly what
    ``LAST_ASSISTANT_MESSAGE`` selects, so the split renders with that mode and
    drops the flags instead of relying on the renderer's CUSTOMIZED branch."""

    renderer = DisaggregateRecordingRenderer()

    render_messages_to_datums(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2-draft", "weight": 0},
            {"role": "assistant", "content": "a2"},
        ],
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )

    messages, train_on_what = renderer.calls[-1]
    assert train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE
    assert [message["content"] for message in messages] == [
        "u1",
        "a1",
        "u2",
        "a2-draft",
        "a2",
    ]
    assert all("trainable" not in message for message in messages)


def test_weighted_row_keeps_customized_for_a_mask_no_builtin_mode_expresses():
    """Masking a middle message of the terminal turn selects a set no built-in
    mode expresses, so per-message weights are required — but only for that
    turn. Everything before it is demoted to context so no earlier turn is
    trained twice, and the flagged render is intersected with the unweighted one
    so masking cannot start training a message the row would otherwise skip."""

    renderer = DisaggregateRecordingRenderer()

    render_messages_to_datums(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2-call"},
            {"role": "assistant", "content": "a2-retry", "weight": 0},
            {"role": "assistant", "content": "a2"},
        ],
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )

    flagged_messages, flagged_mode = renderer.calls[-2]
    default_messages, default_mode = renderer.calls[-1]
    assert flagged_mode == TrainOnWhat.CUSTOMIZED
    assert [
        (message["content"], message["trainable"]) for message in flagged_messages
    ] == [
        ("u1", False),
        ("a1", False),
        ("u2", False),
        ("a2-call", True),
        ("a2-retry", False),
        ("a2", True),
    ]
    assert default_mode == TrainOnWhat.LAST_ASSISTANT_TURN
    assert all("trainable" not in message for message in default_messages)


def test_weighted_row_split_does_not_warn_about_extension_property(recwarn):
    renderer = DisaggregateRecordingRenderer()

    render_messages_to_datums(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1", "weight": 0},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ],
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )

    assert not [
        warning
        for warning in recwarn
        if "does not satisfy the extension property" in str(warning.message)
    ]


def test_render_messages_to_datums_fails_fast_without_split_implementation():
    class UnimplementedSplitRenderer:
        has_extension_property = False

        def build_supervised_examples(self, messages, train_on_what):
            raise NotImplementedError("split rendering is required")

        def build_supervised_example(self, messages, train_on_what):
            raise AssertionError("should not fall back to a single datum")

    with pytest.raises(NotImplementedError, match="split rendering is required"):
        render_messages_to_datums(
            [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
            ],
            renderer=UnimplementedSplitRenderer(),
            train_on_what="all_assistant_messages",
        )


def test_render_messages_to_datums_skips_unimplemented_base_tool_prefix():
    renderer = BaseToolPrefixRenderer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]

    rendered = render_messages_to_datums(
        messages,
        renderer=renderer,
        train_on_what="all_assistant_messages",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert len(rendered) == 1
    assert [m["role"] for m in renderer.calls[0][0]] == [
        "system",
        "user",
        "assistant",
    ]


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


def test_normalize_messages_preserves_dynamic_tools_without_aliasing():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "parameters": {"type": "object"},
            },
        }
    ]
    normalized = normalize_messages([{"role": "system", "content": "", "tools": tools}])

    assert normalized[0]["tools"] == tools
    assert normalized[0]["tools"] is not tools
    assert normalized[0]["tools"][0] is not tools[0]


def test_normalize_messages_preserves_multipart_text_parts():
    """All-text content parts are preserved as a list rather than pre-joined into
    one string. Pre-joining is lossy: it discards per-part boundaries, and each
    model's chat template trims parts differently (gemma-4 trims EACH part;
    others concatenate raw). Preserving the list lets every renderer apply its
    own per-part policy so training tokens match the template.
    """
    normalized = normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one "},
                    {"type": "text", "text": "part two"},
                ],
            }
        ]
    )
    assert normalized[0]["content"] == [
        {"type": "text", "text": "part one "},
        {"type": "text", "text": "part two"},
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


def test_normalize_messages_promotes_reasoning_to_thinking_part():
    """Gemma 4 jinja ``reasoning`` field should become a ThinkingPart."""
    normalized = normalize_messages(
        [
            {
                "role": "assistant",
                "reasoning": "step by step",
                "content": "The answer is 4",
            },
        ]
    )

    assert normalized[0]["content"] == [
        {"type": "thinking", "thinking": "step by step"},
        {"type": "text", "text": "The answer is 4"},
    ]


def test_normalize_messages_reasoning_wins_over_reasoning_content():
    normalized = normalize_messages(
        [
            {
                "role": "assistant",
                "reasoning": "jinja reasoning",
                "reasoning_content": "openai reasoning",
                "content": "answer",
            },
        ]
    )

    assert normalized[0]["content"] == [
        {"type": "thinking", "thinking": "jinja reasoning"},
        {"type": "text", "text": "answer"},
    ]


def test_normalize_messages_empty_reasoning_falls_back_to_reasoning_content():
    normalized = normalize_messages(
        [
            {
                "role": "assistant",
                "reasoning": "",
                "reasoning_content": "openai reasoning",
                "content": "answer",
            },
        ]
    )

    assert normalized[0]["content"] == [
        {"type": "thinking", "thinking": "openai reasoning"},
        {"type": "text", "text": "answer"},
    ]


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_normalize_messages_empty_reasoning_is_not_a_generic_thinking_part(
    field: str,
):
    normalized = normalize_messages(
        [{"role": "assistant", field: "", "content": "answer"}]
    )

    assert normalized[0]["content"] == "answer"


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


def test_normalize_messages_translates_weight_zero_to_trainable_false():
    """Fireworks V1 SFT datasets mark context-only assistant messages with
    ``weight=0``. Without translating that to Tinker's ``trainable`` field,
    ``train_on_what=all_assistant_messages`` silently trains on every
    assistant — including the context-only ones — which teaches thinking
    models to emit empty ``<think></think>`` for the majority of turns
    because historical assistants on Kimi/Qwen3/DeepSeek-thinking are
    rendered with their thinking stripped."""
    normalized = normalize_messages(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "ctx", "weight": 0},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "train me"},
        ]
    )

    assert normalized[0]["trainable"] is False  # user
    assert normalized[1]["trainable"] is False  # weight=0
    assert normalized[2]["trainable"] is False  # user
    assert normalized[3]["trainable"] is True  # weight absent -> assistant default


def test_normalize_messages_translates_weight_one_to_trainable_true():
    """``weight=1`` (explicit trainable marker) must map to ``trainable=True``."""
    normalized = normalize_messages(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1", "weight": 1},
        ]
    )

    assert normalized[1]["trainable"] is True


def test_normalize_messages_prefers_explicit_trainable_over_weight():
    """If both ``trainable`` and ``weight`` are set, ``trainable`` wins."""
    normalized = normalize_messages(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "weight": 1, "trainable": False},
        ]
    )

    assert normalized[1]["trainable"] is False


def test_normalize_messages_does_not_add_trainable_when_no_weight_or_trainable():
    """Datasets without ``weight``/``trainable`` on any message must not
    gain a ``trainable`` field, so renderers still see a back-compatible
    schema and default ``train_on_what`` modes keep working unchanged."""
    normalized = normalize_messages(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
    )

    assert "trainable" not in normalized[0]
    assert "trainable" not in normalized[1]


def test_normalize_messages_rejects_non_numeric_weight():
    """Non-numeric ``weight`` values must raise TypeError to avoid silently
    accepting garbage data."""
    with pytest.raises(TypeError):
        normalize_messages(
            [
                {"role": "assistant", "content": "a", "weight": "yes"},
            ]
        )


class _SingularRecordingRenderer:
    def __init__(self):
        self.calls: list[tuple[list[dict], TrainOnWhat]] = []

    def build_supervised_example(self, messages, train_on_what):
        self.calls.append(([dict(message) for message in messages], train_on_what))
        return (
            torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            torch.tensor([0, 0, 1, 1], dtype=torch.float32),
        )


def test_render_messages_to_datum_honors_weight_through_an_equivalent_mode():
    """A ``weight`` field must never degrade into "train on everything". When
    the flags select exactly what a built-in mode selects, that mode carries
    the intent and the flags are dropped, so the row renders byte-for-byte like
    an unweighted row through the renderer's well-trodden path."""

    renderer = _SingularRecordingRenderer()
    render_messages_to_datum(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "weight": 0},
            {"role": "assistant", "content": "b"},
        ],
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )
    messages, resolved_train_on_what = renderer.calls[0]
    assert resolved_train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE
    assert all("trainable" not in message for message in messages)


def test_render_messages_to_datum_uses_customized_when_no_mode_matches():
    """Masking a middle assistant message selects a set no built-in mode
    expresses, so the render must fall back to per-message weights."""

    renderer = _SingularRecordingRenderer()
    render_messages_to_datum(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b", "weight": 0},
            {"role": "assistant", "content": "c"},
        ],
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )
    messages, resolved_train_on_what = renderer.calls[0]
    assert resolved_train_on_what == TrainOnWhat.CUSTOMIZED
    assert [message["trainable"] for message in messages] == [False, True, False, True]


def test_explicitly_requested_customized_renders_once_without_a_reference():
    """A caller that asks for ``CUSTOMIZED`` outright states the loss directly,
    so there is no unweighted render to intersect with — rendering it against
    one would hand the renderer a flagless conversation under ``CUSTOMIZED``."""

    renderer = _SingularRecordingRenderer()
    render_messages_to_datum(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b", "weight": 0},
            {"role": "assistant", "content": "c"},
        ],
        renderer=renderer,
        train_on_what="customized",
    )
    assert len(renderer.calls) == 1
    messages, resolved_train_on_what = renderer.calls[0]
    assert resolved_train_on_what == TrainOnWhat.CUSTOMIZED
    assert [message["trainable"] for message in messages] == [False, True, False, True]


def test_render_messages_to_datum_uses_equivalent_single_example_mode():
    renderer = BaseWarningRenderer()
    messages = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},
    ]

    expected = render_messages_to_datum(
        messages,
        renderer=renderer,
        train_on_what="last_assistant_turn",
    )
    actual = render_messages_to_datum(
        messages,
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )

    assert actual.token_ids == expected.token_ids
    assert actual.token_weights == expected.token_weights
    assert renderer.calls[-1] == TrainOnWhat.LAST_ASSISTANT_TURN


def test_render_messages_to_datum_keeps_non_equivalent_train_on_what():
    renderer = BaseWarningRenderer()
    render_messages_to_datum(
        [
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
        ],
        renderer=renderer,
        train_on_what="all_assistant_messages",
    )

    assert renderer.calls[-1] == TrainOnWhat.ALL_ASSISTANT_MESSAGES


def test_render_preference_pair_uses_equivalent_single_example_mode(caplog):
    renderer = BaseWarningRenderer()
    item = {
        "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    }

    rendered = render_preference_pair(
        item,
        item,
        renderer=renderer,
        tokenizer=object(),
    )

    assert rendered is not None
    assert renderer.calls == [
        TrainOnWhat.LAST_ASSISTANT_TURN,
        TrainOnWhat.LAST_ASSISTANT_TURN,
    ]
    assert "does not satisfy the extension property" not in caplog.text


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


def test_build_renderer_from_resolved_name_loads_image_processor_by_default(
    monkeypatch,
):
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

    renderer = build_renderer_from_resolved_name(
        tokenizer="tok",
        tokenizer_model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        renderer_name="qwen3_vl_instruct",
    )

    assert renderer == "renderer"
    assert calls == [("qwen3_vl_instruct", "image-processor")]


def test_build_renderer_from_resolved_name_loads_muse_token_counter(monkeypatch):
    calls: list[tuple[str, object | None]] = []
    counter = object()

    def fake_from_pretrained(model_name):
        assert model_name == "meta-models/Muse-Glimmer-30B"
        return counter

    def fake_get_renderer(name, tokenizer, image_processor=None):
        assert tokenizer == "tok"
        calls.append((name, image_processor))
        return "renderer"

    monkeypatch.setattr(
        "training.utils.supervised._muse_glimmer_renderer."
        "MuseGlimmerImageTokenCounter.from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr("training.utils.supervised.get_renderer", fake_get_renderer)

    renderer = build_renderer_from_resolved_name(
        tokenizer="tok",
        tokenizer_model="meta-models/Muse-Glimmer-30B",
        renderer_name="muse_glimmer",
    )

    assert renderer == "renderer"
    assert calls == [("muse_glimmer", counter)]


def test_build_renderer_from_resolved_name_can_skip_image_processor(monkeypatch):
    calls: list[tuple[str, object | None]] = []

    def fail_get_image_processor(_model_name):
        pytest.fail("image processor was loaded")

    def fake_get_renderer(name, tokenizer, image_processor=None):
        calls.append((name, image_processor))
        return "renderer"

    monkeypatch.setattr(
        "training.utils.supervised.get_image_processor", fail_get_image_processor
    )
    monkeypatch.setattr("training.utils.supervised.get_renderer", fake_get_renderer)

    renderer = build_renderer_from_resolved_name(
        tokenizer="tok",
        tokenizer_model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        renderer_name="qwen3_vl_instruct",
        load_image_processor=False,
    )

    assert renderer == "renderer"
    assert calls == [("qwen3_vl_instruct", None)]


def test_build_renderer_uses_tokenizer_remote_code_default_for_image_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)

    env_at_call: list[str | None] = []

    def fake_get_image_processor(_model_name: str) -> str:
        env_at_call.append(os.environ.get("HF_TRUST_REMOTE_CODE"))
        return "image-processor"

    def fake_get_renderer(
        name: str, _tokenizer: object, image_processor: object | None = None
    ) -> tuple[str, str, object | None]:
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
    assert result == ("renderer", "kimi_k26_interleaved", "image-processor")
    assert "HF_TRUST_REMOTE_CODE" not in os.environ


@pytest.mark.parametrize(
    "renderer_name",
    [
        "qwen3_vl_instruct",
        "qwen3_5",
        "qwen3_6",
        "qwen3_8",
        "qwen3_8_interleaved",
        "qwen3_8_disable_thinking_interleaved",
        "qwen3_8_preserved",
        "kimi_k25",
        "kimi_k25_disable_thinking",
        "kimi_k25_interleaved",
        "kimi_k26",
        "kimi_k26_disable_thinking",
        "kimi_k26_interleaved",
        "kimi_k26_preserve_thinking",
        "kimi_k27_code",
        "kimi_k27_code_preserved",
        "kimi_k3",
        "kimi_k3_disable_thinking",
    ],
)
def test_resolved_image_renderer_trusts_opaque_tokenizer_path(
    monkeypatch: pytest.MonkeyPatch,
    renderer_name: str,
) -> None:
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)

    env_at_call: list[str | None] = []

    def fake_get_image_processor(model_name: str) -> str:
        assert model_name == "gs://model-bucket/uploads/checkpoint/hf"
        env_at_call.append(os.environ.get("HF_TRUST_REMOTE_CODE"))
        return "image-processor"

    def fake_get_renderer(
        name: str, _tokenizer: object, image_processor: object | None = None
    ) -> tuple[str, str, object | None]:
        return ("renderer", name, image_processor)

    monkeypatch.setattr(
        "training.utils.supervised.get_image_processor", fake_get_image_processor
    )
    monkeypatch.setattr("training.utils.supervised.get_renderer", fake_get_renderer)

    result = build_renderer_from_resolved_name(
        tokenizer="tok",
        tokenizer_model="gs://model-bucket/uploads/checkpoint/hf",
        renderer_name=renderer_name,
    )

    assert env_at_call == ["1"]
    assert result == ("renderer", renderer_name, "image-processor")
    assert "HF_TRUST_REMOTE_CODE" not in os.environ


def test_image_processor_failure_restores_remote_code_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)

    def fail_get_image_processor(_model_name: str) -> None:
        assert os.environ["HF_TRUST_REMOTE_CODE"] == "1"
        raise RuntimeError("processor load failed")

    monkeypatch.setattr(
        "training.utils.supervised.get_image_processor", fail_get_image_processor
    )

    with pytest.raises(RuntimeError, match="processor load failed"):
        build_renderer_from_resolved_name(
            tokenizer="tok",
            tokenizer_model="gs://model-bucket/uploads/checkpoint/hf",
            renderer_name="kimi_k3",
        )

    assert "HF_TRUST_REMOTE_CODE" not in os.environ


def test_new_image_renderer_inherits_remote_code_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)

    env_at_call: list[str | None] = []

    def fake_get_image_processor(_model_name: str) -> str:
        env_at_call.append(os.environ.get("HF_TRUST_REMOTE_CODE"))
        return "image-processor"

    monkeypatch.setattr(
        "training.utils.supervised.get_image_processor", fake_get_image_processor
    )
    monkeypatch.setattr(
        "training.utils.supervised.get_renderer",
        lambda *_args, **_kwargs: "renderer",
    )

    build_renderer_from_resolved_name(
        tokenizer="tok",
        tokenizer_model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        renderer_name="future_model_vl",
    )

    assert env_at_call == ["1"]
    assert "HF_TRUST_REMOTE_CODE" not in os.environ


def test_build_renderer_uses_remote_code_default_for_kimi_k2_5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)

    env_at_call: list[str | None] = []

    def fake_get_image_processor(_model_name: str) -> str:
        env_at_call.append(os.environ.get("HF_TRUST_REMOTE_CODE"))
        return "image-processor"

    def fake_get_renderer(
        _name: str, _tokenizer: object, image_processor: object | None = None
    ) -> str:
        return "renderer"

    monkeypatch.setattr(
        "training.utils.supervised.get_image_processor", fake_get_image_processor
    )
    monkeypatch.setattr("training.utils.supervised.get_renderer", fake_get_renderer)

    build_renderer(
        tokenizer="tok",
        tokenizer_model="moonshotai/Kimi-K2.5",
    )

    assert env_at_call == ["1"]
    assert "HF_TRUST_REMOTE_CODE" not in os.environ


def test_build_renderer_preserves_existing_trust_remote_code_value(monkeypatch):
    """Don't stomp a user-set HF_TRUST_REMOTE_CODE policy."""
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


def test_resolve_renderer_name_prefers_mistral_small_24b() -> None:
    assert (
        resolve_renderer_name("mistralai/Mistral-Small-24B-Instruct-2501") == "mistral"
    )
    assert (
        resolve_renderer_name(
            "accounts/fireworks/models/mistral-small-24b-instruct-2501"
        )
        == "mistral"
    )


def test_resolve_renderer_name_prefers_kimi_k26_for_kimi_k2_6():
    assert resolve_renderer_name("moonshotai/Kimi-K2.6") == "kimi_k25"


def test_resolve_renderer_name_prefers_preserve_thinking_for_kimi_k2_7_code():
    assert resolve_renderer_name("moonshotai/Kimi-K2.7-Code") == "kimi_k27_code"


def test_resolve_renderer_name_prefers_minimax_m2() -> None:
    """MiniMax M2 tokenizers should resolve to the custom renderer."""
    assert resolve_renderer_name("MiniMaxAI/MiniMax-M2") == "minimax_m2"


def test_resolve_renderer_name_prefers_minimax_m3() -> None:
    """Released MiniMax-M3 tokenizers use the dedicated M3 renderer."""
    assert resolve_renderer_name("MiniMaxAI/MiniMax-M3") == "minimax_m3"


def test_resolve_renderer_name_prefers_upstream_nemotron3() -> None:
    """Nemotron models use Tinker's upstream renderer with parse normalization."""
    assert resolve_renderer_name("nvidia/NVIDIA-Nemotron-3-Super-120B") == "nemotron3"
    assert resolve_renderer_name("nvidia/NVIDIA-Nemotron-H-8B") == "nemotron3"


def test_resolve_renderer_name_supports_qwen3_family_fallback() -> None:
    assert resolve_renderer_name("Qwen/Qwen3-0.6B") == "qwen3"
    assert resolve_renderer_name("accounts/fireworks/models/qwen3-0p6b") == "qwen3"
    assert resolve_renderer_name("Qwen/Qwen3-1.7B") == "qwen3"
    assert resolve_renderer_name("accounts/fireworks/models/qwen3-1p7b") == "qwen3"
    assert resolve_renderer_name("Qwen/Qwen3-4B") == "qwen3"
    assert resolve_renderer_name("accounts/fireworks/models/qwen3-4b") == "qwen3"
    assert resolve_renderer_name("Qwen/Qwen3-14B") == "qwen3"
    assert resolve_renderer_name("accounts/fireworks/models/qwen3-14b") == "qwen3"
    assert resolve_renderer_name("Qwen/Qwen3-4B-Instruct-2507") == "qwen3_instruct"
    assert resolve_renderer_name("Qwen/Qwen3-8B-Base") == "role_colon"


def test_resolve_renderer_name_targets_qwen2_5_32b_v1_contract() -> None:
    assert resolve_renderer_name("Qwen/Qwen2.5-32B-Instruct") == "qwen2_5"
    assert (
        resolve_renderer_name("accounts/fireworks/models/qwen2p5-32b-instruct")
        == "qwen2_5"
    )


def test_resolve_renderer_name_supports_qwen3_235b_instruct_fp8_alias() -> None:
    assert (
        resolve_renderer_name("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
        == "qwen3_instruct"
    )
    assert (
        resolve_renderer_name("Qwen/Qwen3-235B-A22B-Instruct-2507")
        == "qwen3_instruct"
    )


@pytest.mark.parametrize(
    "tokenizer_model",
    [
        "Qwen/Qwen3-0.6B-Base",
        "custom/Qwen3-0.6B-Base",
        "/models/base/fireworks/qwen3-0p6b/hf",
        "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-1.7B-Instruct-2507",
        "custom/Qwen3-1.7B-Base",
        "custom/Qwen3-1.7B-Instruct-2507",
        "/models/base/fireworks/qwen3-1p7b/hf",
        "/cache/instruct-2507/qwen3-1p7b/hf",
        "Qwen/Qwen3-4B-Base",
        "custom/Qwen3-4B-Base",
        "/models/base/fireworks/qwen3-4b/hf",
        "Qwen/Qwen3-14B-Base",
        "custom/Qwen3-14B-Base",
        "/models/base/fireworks/qwen3-14b/hf",
    ],
)
def test_resolve_renderer_name_qwen3_fallback_fails_closed(
    tokenizer_model: str,
) -> None:
    with pytest.raises(ValueError, match="Set Config.renderer_name explicitly"):
        resolve_renderer_name(tokenizer_model)


def test_resolve_renderer_name_prefers_qwen3_5() -> None:
    """Qwen3.5 models should resolve to the qwen3_5 renderer."""
    assert resolve_renderer_name("Qwen/Qwen3.5-9B") == "qwen3_5"
    assert resolve_renderer_name("Qwen/Qwen3.5-4B") == "qwen3_5"
    assert resolve_renderer_name("Qwen/Qwen3.5-27B") == "qwen3_5"
    assert resolve_renderer_name("Qwen/Qwen3.5-35B-A3B") == "qwen3_5"
    assert resolve_renderer_name("Qwen/Qwen3.5-397B-A17B") == "qwen3_5"


def test_resolve_renderer_name_prefers_qwen3_6() -> None:
    """Qwen3.6 models should resolve to the qwen3_6 renderer (alias of qwen3_5)."""
    assert resolve_renderer_name("Qwen/Qwen3.6-27B") == "qwen3_6"
    assert resolve_renderer_name("Qwen/Qwen3.6-9B") == "qwen3_6"
    assert resolve_renderer_name("custom/qwen3_6-finetune") == "qwen3_6"


def test_resolve_renderer_name_prefers_qwen3_8_27b() -> None:
    """Qwen3.8-27B defaults to the preserved renderer; Max/Plus must not match."""
    assert resolve_renderer_name("Qwen/Qwen3.8-27B") == "qwen3_8"
    assert resolve_renderer_name("accounts/fireworks/models/qwen3p8-27b") == "qwen3_8"
    try:
        resolved_max = resolve_renderer_name("accounts/fireworks/models/qwen3p8-max")
    except ValueError:
        resolved_max = None
    assert resolved_max != "qwen3_8"


def test_resolve_renderer_name_prefers_gemma4() -> None:
    """Gemma 4 models should resolve to the gemma4 renderer."""
    assert resolve_renderer_name("google/gemma-4-12b-it") == "gemma4"
    assert resolve_renderer_name("google/gemma-4-27b-it") == "gemma4"


def test_resolve_renderer_name_supports_gemma4_thinking_override() -> None:
    assert resolve_renderer_name("google/gemma-4-12b-it", "gemma4_thinking") == (
        "gemma4_thinking"
    )


def test_resolve_renderer_name_prefers_deepseek_v4() -> None:
    """DeepSeek-V4 tokenizers should resolve to the custom deepseek_v4 renderer."""
    assert resolve_renderer_name("deepseek-ai/DeepSeek-V4-Flash") == "deepseek_v4"
    assert resolve_renderer_name("deepseek-ai/deepseek_v4") == "deepseek_v4"
    assert resolve_renderer_name("custom/DeepSeekV4-finetune") == "deepseek_v4"


def test_resolve_renderer_name_prefers_glm5_variants_for_glm_5_family() -> None:
    """GLM-5.x tokenizers should resolve to versioned GLM renderers."""
    assert resolve_renderer_name("zai-org/GLM-5.1") == "glm5"
    assert resolve_renderer_name("zai-org/GLM-5.1-FP8") == "glm5"
    assert resolve_renderer_name("zai-org/GLM-5.2") == "glm_moe_dsa"
    assert resolve_renderer_name("zai-org/GLM-5.2-FP8") == "glm_moe_dsa"
    assert resolve_renderer_name("custom/glm-5p2-finetune") == "glm_moe_dsa"


@pytest.mark.parametrize(
    ("renderer_name", "expected"),
    [
        ("qwen3", False),
        ("qwen3_vl_instruct", True),
        ("qwen3_5", True),
        ("qwen3_6", True),
        ("qwen3_8", True),
        ("kimi_k25", True),
        ("muse_glimmer", True),
        ("glm_moe_dsa", False),
        ("deepseek_v4", False),
    ],
)
def test_renderer_supports_images_matches_renderer_capability(
    renderer_name: str,
    expected: bool,
) -> None:
    assert renderer_supports_images(renderer_name) is expected


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
    renderer = AtomicPreferenceRenderer(
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


# ---------------------------------------------------------------------------
# populate_render_worker_state
# ---------------------------------------------------------------------------


def test_populate_render_worker_state_writes_canonical_keys(monkeypatch):
    """Common keys (tokenizer, renderer, max_seq_len) plus extras land in state."""
    from training.utils import supervised as sup

    fake_tokenizer = object()
    fake_renderer = object()
    monkeypatch.setattr(
        sup,
        "load_tokenizer",
        lambda model, revision=None, trust_remote_code=None: fake_tokenizer,
    )
    monkeypatch.setattr(sup, "build_renderer", lambda *a, **k: fake_renderer)

    state: dict = {}
    populate_render_worker_state(
        state,
        tokenizer_model="acme/llama",
        renderer_name="llama-3",
        max_seq_len=4096,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        custom_extra="hello",
    )

    assert state["tokenizer"] is fake_tokenizer
    assert state["renderer"] is fake_renderer
    assert state["max_seq_len"] == 4096
    assert state["train_on_what"] == TrainOnWhat.LAST_ASSISTANT_MESSAGE
    assert state["custom_extra"] == "hello"


def test_populate_render_worker_state_forwards_materialized_tokenizer_plan(monkeypatch):
    from training.utils import supervised as sup

    captured: dict = {}

    def fake_load_tokenizer(model, revision=None, trust_remote_code=None):
        captured.update(
            model=model,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        return object()

    monkeypatch.setattr(sup, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(sup, "build_renderer", lambda *a, **k: object())

    populate_render_worker_state(
        {},
        tokenizer_model="m",
        tokenizer_revision="abc123",
        tokenizer_trust_remote_code=False,
        renderer_name="r",
        max_seq_len=1,
    )
    assert captured["model"] == "m"
    assert captured["revision"] == "abc123"
    assert captured["trust_remote_code"] is False


def test_populate_render_worker_state_uses_resolved_name_without_live_resolution(
    monkeypatch,
):
    from training.utils import supervised as sup

    fake_tokenizer = object()
    fake_renderer = object()
    captured: dict = {}
    monkeypatch.setattr(
        sup,
        "load_tokenizer",
        lambda _model, _revision=None, _trust_remote_code=None: fake_tokenizer,
    )
    monkeypatch.setattr(
        sup,
        "build_renderer",
        lambda *_args, **_kwargs: pytest.fail("live renderer resolution was used"),
    )

    def fake_build_resolved(tokenizer, tokenizer_model, renderer_name):
        captured.update(
            tokenizer=tokenizer,
            tokenizer_model=tokenizer_model,
            renderer_name=renderer_name,
        )
        return fake_renderer

    monkeypatch.setattr(sup, "build_renderer_from_resolved_name", fake_build_resolved)

    state: dict = {}
    populate_render_worker_state(
        state,
        tokenizer_model="Qwen/Qwen3.6-27B",
        renderer_name="qwen3_6_preserve_thinking",
        renderer_name_is_resolved=True,
        thinking_trace_history_mode="preserved",
        max_seq_len=4096,
    )

    assert captured == {
        "tokenizer": fake_tokenizer,
        "tokenizer_model": "Qwen/Qwen3.6-27B",
        "renderer_name": "qwen3_6_preserve_thinking",
    }
    assert state["renderer"] is fake_renderer


def test_build_renderer_from_resolved_name_bypasses_semantic_resolution(monkeypatch):
    from training.utils import supervised as sup

    tokenizer = object()
    monkeypatch.setattr(
        sup,
        "resolve_renderer_plan",
        lambda *_args, **_kwargs: pytest.fail("semantic renderer resolution was used"),
    )
    monkeypatch.setattr(
        sup,
        "get_renderer",
        lambda name, actual_tokenizer, **kwargs: (name, actual_tokenizer),
    )
    monkeypatch.setattr(sup, "get_image_processor", None)

    assert build_renderer_from_resolved_name(
        tokenizer,
        "Qwen/Qwen3.6-27B",
        "qwen3_6_preserve_thinking",
    ) == ("qwen3_6_preserve_thinking", tokenizer)
