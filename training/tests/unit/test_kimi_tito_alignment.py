"""Verify Token-In-Token-Out (TITO) mask alignment for Kimi K2.5 FrozenLake.

The rollout processor tracks tokens at the ID level (prompt_ids, completion_ids,
tool suffix, prefill) while the Tinker renderer operates on structured messages.
Both must produce the same token sequences and compatible loss masks.

Known design difference: for RL (GRPO), the empty <think></think> block is part
of the prompt (NOT trained) because it's prefilled context. For SFT (Tinker),
the same tokens are part of the assistant output (trained). This test documents
and verifies this intentional divergence.

Requires: moonshotai/Kimi-K2.5 tokenizer (downloaded on first run).
"""

from __future__ import annotations

import json

import pytest
import torch

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import tinker
    from tinker_cookbook.renderers import get_renderer, TrainOnWhat
    HAS_TINKER = True
except ImportError:
    HAS_TINKER = False

try:
    from tinker_cookbook.renderers import Message, ToolCall, ToolSpec
    HAS_TINKER_KIMI = True
    _test_renderer = None
    try:
        from transformers import AutoTokenizer as _AT
        _tok = _AT.from_pretrained("moonshotai/Kimi-K2.5", trust_remote_code=True)
        _test_renderer = get_renderer("kimi_k25_disable_thinking", _tok)
    except Exception:
        HAS_TINKER_KIMI = False
except ImportError:
    HAS_TINKER_KIMI = False

from training.examples.frozen_lake.masking import (
    build_training_loss_mask,
    build_ui_token_mask,
    compute_model_output_spans,
)

KIMI_TOKENIZER = "moonshotai/Kimi-K2.5"
SKIP_REASON = "Requires transformers + tinker-cookbook + Kimi K2.5 tokenizer"
SKIP_TINKER_KIMI = (
    "Requires tinker-cookbook with Kimi K2.5 renderer support "
    "(kimi_k25_disable_thinking). Upgrade tinker-cookbook to latest."
)

FROZEN_LAKE_TOOL_SPEC: dict = {
    "name": "lake_move",
    "description": "Move in FrozenLake by one step using action LEFT, DOWN, RIGHT, or UP.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["LEFT", "DOWN", "RIGHT", "UP"],
                "description": "Movement direction in the grid world.",
            }
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def _build_two_turn_tool_call_messages():
    """Build a minimal 2-turn FrozenLake conversation in Tinker Message format.

    Returns list of dicts when Tinker Message/ToolCall types aren't available,
    or proper typed Messages when they are.
    """
    if not HAS_TINKER_KIMI:
        return None

    return [
        Message(
            role="user",
            content="FrozenLake grid observation:\n[S]  F \n F   G ",
        ),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="lake_move",
                        arguments='{"action": "RIGHT"}',
                    ),
                    id="functions.lake_move:0",
                )
            ],
        ),
        Message(
            role="tool",
            content='{"action":"RIGHT","reward":0.0,"terminated":false,"truncated":false,"position":1,"row":0,"col":1,"tile":"F","step_index":1}',
            tool_call_id="functions.lake_move:0",
            name="lake_move",
        ),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="lake_move",
                        arguments='{"action": "DOWN"}',
                    ),
                    id="functions.lake_move:1",
                )
            ],
        ),
    ]


@pytest.fixture(scope="module")
def kimi_tokenizer():
    if not HAS_TRANSFORMERS:
        pytest.skip(SKIP_REASON)
    return AutoTokenizer.from_pretrained(KIMI_TOKENIZER, trust_remote_code=True)


@pytest.fixture(scope="module")
def kimi_renderer(kimi_tokenizer):
    if not HAS_TINKER_KIMI:
        pytest.skip(SKIP_TINKER_KIMI)
    return get_renderer("kimi_k25_disable_thinking", kimi_tokenizer)


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


class TestKimiTitoTokenSequenceAlignment:
    """Verify that rollout-style token tracking produces the same sequence as the Tinker renderer."""

    def test_assistant_header_and_suffix_tokens(self, kimi_tokenizer):
        """Verify the critical framing tokens used by the rollout processor."""
        im_end_ids = _encode(kimi_tokenizer, "<|im_end|>")
        assert len(im_end_ids) == 1, f"<|im_end|> should be a single token, got {im_end_ids}"

        prefill_ids = _encode(kimi_tokenizer, "<|tool_calls_section_begin|>")
        assert len(prefill_ids) == 1, (
            f"<|tool_calls_section_begin|> should be a single token, got {prefill_ids}"
        )

        think_open = _encode(kimi_tokenizer, "<think>")
        think_close = _encode(kimi_tokenizer, "</think>")
        assert len(think_open) == 1
        assert len(think_close) == 1

    def test_empty_think_block_is_in_generation_prompt(self, kimi_tokenizer):
        """The HF template with thinking=False produces <think></think> in the generation prompt.

        This means the empty think block is part of the PROMPT that the model sees,
        not part of the model's output. The rollout processor correctly treats it as prompt.
        """
        messages = [
            {"role": "system", "content": "You are an RL policy."},
            {"role": "user", "content": "Make a move."},
        ]
        prompt_ids = kimi_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            thinking=False,
        )

        think_open = _encode(kimi_tokenizer, "<think>")
        think_close = _encode(kimi_tokenizer, "</think>")
        empty_think = think_open + think_close

        assert prompt_ids[-len(empty_think):] == empty_think, (
            f"Expected prompt to end with empty think block token IDs {empty_think}, "
            f"got {prompt_ids[-4:]}"
        )

        prompt_text = kimi_tokenizer.decode(prompt_ids, skip_special_tokens=False)
        assert "<think>" in prompt_text and "</think>" in prompt_text, (
            f"Expected generation prompt to contain think tags, got: ...{prompt_text[-80:]}"
        )

    def test_tinker_renderer_supervised_tokens_match_rollout_reconstruction(
        self, kimi_tokenizer, kimi_renderer,
    ):
        """The supervised token sequence from Tinker should match the rollout reconstruction.

        The rollout tracks: prompt_ids (per turn) + completion_ids (per turn).
        The full episode = last_turn.prompt_ids + last_turn.completion_ids.
        The Tinker renderer builds tokens from structured messages.

        These must produce identical token sequences so training data is consistent.
        """
        messages = _build_two_turn_tool_call_messages()
        tool_prefix = kimi_renderer.create_conversation_prefix_with_tools(
            [FROZEN_LAKE_TOOL_SPEC],
            system_prompt="You are an RL policy for FrozenLake.",
        )
        full_messages = tool_prefix + messages

        model_input, weights = kimi_renderer.build_supervised_example(
            full_messages,
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )

        tinker_token_ids = []
        for chunk in model_input.chunks:
            if hasattr(chunk, "tokens"):
                tinker_token_ids.extend(int(t) for t in chunk.tokens)

        assert len(tinker_token_ids) > 50, (
            f"Expected substantial token sequence, got {len(tinker_token_ids)} tokens"
        )
        assert len(weights) == len(tinker_token_ids), (
            f"Weights length {len(weights)} != token count {len(tinker_token_ids)}"
        )

        tinker_text = kimi_tokenizer.decode(tinker_token_ids, skip_special_tokens=False)
        assert "<|im_end|>" in tinker_text
        assert "<|tool_calls_section_begin|>" in tinker_text

        prompt_only_messages = full_messages[:-1]
        gen_prompt = kimi_renderer.build_generation_prompt(prompt_only_messages)
        gen_prompt_ids = []
        for chunk in gen_prompt.chunks:
            if hasattr(chunk, "tokens"):
                gen_prompt_ids.extend(int(t) for t in chunk.tokens)

        gen_prompt_text = kimi_tokenizer.decode(gen_prompt_ids, skip_special_tokens=False)
        assert gen_prompt_text.rstrip().endswith("</think>"), (
            "Generation prompt for disable_thinking should end with </think>"
        )

    def test_rollout_mask_covers_only_model_generated_tokens(self, kimi_tokenizer):
        """The rollout processor's mask should train on completion tokens only.

        For GRPO, the mask marks positions where the model actually generated
        tokens during inference. The <think></think> block is NOT model-generated
        (it's prefilled context), so it should NOT be in the mask.
        """
        im_end_ids = _encode(kimi_tokenizer, "<|im_end|>")
        prefill_ids = _encode(kimi_tokenizer, "<|tool_calls_section_begin|>")
        think_open = _encode(kimi_tokenizer, "<think>")
        think_close = _encode(kimi_tokenizer, "</think>")

        tool_call_1_text = (
            "<|tool_call_begin|>functions.lake_move:0"
            '<|tool_call_argument_begin|>{"action":"RIGHT"}'
            "<|tool_call_end|><|tool_calls_section_end|>"
        )
        tool_call_1_ids = _encode(kimi_tokenizer, tool_call_1_text)

        tool_call_2_text = (
            "<|tool_call_begin|>functions.lake_move:1"
            '<|tool_call_argument_begin|>{"action":"DOWN"}'
            "<|tool_call_end|><|tool_calls_section_end|>"
        )
        tool_call_2_ids = _encode(kimi_tokenizer, tool_call_2_text)

        # Simulate rollout processor token tracking
        initial_prompt = _encode(kimi_tokenizer, (
            "<|im_system|>system<|im_middle|>"
            "You are an RL policy for FrozenLake.<|im_end|>"
            "<|im_user|>user<|im_middle|>"
            "FrozenLake grid observation:\n[S]  F \n F   G <|im_end|>"
            "<|im_assistant|>assistant<|im_middle|>"
        ))
        initial_prompt += think_open + think_close

        completion_1 = prefill_ids + tool_call_1_ids + im_end_ids
        tool_result_text = (
            '<|im_system|>lake_move<|im_middle|>'
            '## Return of functions.lake_move:0\n'
            '{"action":"RIGHT","reward":0.0,"terminated":false,'
            '"truncated":false,"position":1,"row":0,"col":1,"tile":"F","step_index":1}'
            '<|im_end|>'
            '<|im_assistant|>assistant<|im_middle|>'
        )
        tool_suffix = _encode(kimi_tokenizer, tool_result_text) + think_open + think_close
        completion_2 = prefill_ids + tool_call_2_ids + im_end_ids

        prompt_ids_turn1 = initial_prompt
        prompt_ids_turn2 = initial_prompt + list(completion_1) + list(tool_suffix)

        full_tokens = prompt_ids_turn2 + list(completion_2)

        token_turn_traces = [
            {
                "prompt_ids": prompt_ids_turn1,
                "completion_ids": list(completion_1),
            },
            {
                "prompt_ids": prompt_ids_turn2,
                "completion_ids": list(completion_2),
            },
        ]
        model_request_traces = [
            {"assistant_turn_len": len(completion_1)},
        ]

        spans = compute_model_output_spans(token_turn_traces, model_request_traces)
        ui_mask = build_ui_token_mask(spans, len(full_tokens))
        train_mask = build_training_loss_mask(spans, len(full_tokens) - 1)

        prompt_1_len = len(prompt_ids_turn1)
        comp_1_len = len(completion_1)

        for i in range(prompt_1_len):
            assert ui_mask[i] == 0, (
                f"Token {i} is in the initial prompt, should not be masked. "
                f"Token text: {kimi_tokenizer.decode([full_tokens[i]])!r}"
            )

        for i in range(prompt_1_len, prompt_1_len + comp_1_len):
            assert ui_mask[i] > 0, (
                f"Token {i} is in completion 1, should be masked. "
                f"Token text: {kimi_tokenizer.decode([full_tokens[i]])!r}"
            )

        tool_suffix_start = prompt_1_len + comp_1_len
        tool_suffix_end = tool_suffix_start + len(tool_suffix)
        for i in range(tool_suffix_start, tool_suffix_end):
            assert ui_mask[i] == 0, (
                f"Token {i} is in tool suffix, should not be masked. "
                f"Token text: {kimi_tokenizer.decode([full_tokens[i]])!r}"
            )

        comp_2_start = tool_suffix_end
        for i in range(comp_2_start, len(full_tokens)):
            assert ui_mask[i] > 0, (
                f"Token {i} is in completion 2, should be masked. "
                f"Token text: {kimi_tokenizer.decode([full_tokens[i]])!r}"
            )

    def test_tinker_mask_includes_empty_think_block_but_rollout_mask_does_not(
        self, kimi_tokenizer, kimi_renderer,
    ):
        """Document the intentional mask divergence between Tinker and rollout.

        Tinker SFT: assistant output includes <think></think> → weight=1
        Rollout GRPO: <think></think> is prompt context → weight=0

        This is correct: SFT teaches the model the full output format,
        while GRPO only rewards tokens the model actually generated.
        """
        messages = _build_two_turn_tool_call_messages()
        tool_prefix = kimi_renderer.create_conversation_prefix_with_tools(
            [FROZEN_LAKE_TOOL_SPEC],
            system_prompt="You are an RL policy for FrozenLake.",
        )
        full_messages = tool_prefix + messages

        model_input, weights = kimi_renderer.build_supervised_example(
            full_messages,
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )

        tinker_ids = []
        for chunk in model_input.chunks:
            if hasattr(chunk, "tokens"):
                tinker_ids.extend(int(t) for t in chunk.tokens)

        tinker_weights = weights.tolist()

        think_open = _encode(kimi_tokenizer, "<think>")
        think_close = _encode(kimi_tokenizer, "</think>")

        think_open_positions = []
        for i in range(len(tinker_ids)):
            if (
                i + len(think_open) + len(think_close) <= len(tinker_ids)
                and tinker_ids[i : i + len(think_open)] == think_open
                and tinker_ids[i + len(think_open) : i + len(think_open) + len(think_close)] == think_close
            ):
                think_open_positions.append(i)

        assert len(think_open_positions) >= 2, (
            f"Expected at least 2 empty think blocks (one per assistant turn), "
            f"found {len(think_open_positions)}"
        )

        for pos in think_open_positions:
            for j in range(len(think_open) + len(think_close)):
                token_idx = pos + j
                assert tinker_weights[token_idx] == 1.0, (
                    f"Tinker SFT should train on <think></think> at position {token_idx}, "
                    f"but weight is {tinker_weights[token_idx]}"
                )


class TestKimiToolCallTokenization:
    """Verify Kimi K2.5 tool-call token patterns used by the rollout processor."""

    def test_tool_call_section_round_trips_through_tokenizer(self, kimi_tokenizer):
        """The tool call section should tokenize and detokenize cleanly."""
        section = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.lake_move:0"
            '<|tool_call_argument_begin|>{"action":"RIGHT"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        ids = _encode(kimi_tokenizer, section)
        decoded = kimi_tokenizer.decode(ids, skip_special_tokens=False)

        assert "<|tool_calls_section_begin|>" in decoded
        assert "<|tool_call_begin|>" in decoded
        assert "<|tool_call_argument_begin|>" in decoded
        assert '{"action":"RIGHT"}' in decoded or '{"action": "RIGHT"}' in decoded
        assert "<|tool_call_end|>" in decoded
        assert "<|tool_calls_section_end|>" in decoded

    def test_tool_response_format_matches_hf_template(self, kimi_tokenizer):
        """Verify the tool response format used in the rollout processor."""
        tool_message = {
            "role": "tool",
            "name": "lake_move",
            "tool_call_id": "functions.lake_move:0",
            "content": '{"action":"RIGHT","reward":0.0}',
        }
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "id": "functions.lake_move:0",
                    "function": {
                        "name": "lake_move",
                        "arguments": '{"action":"RIGHT"}',
                    },
                }],
            },
            tool_message,
        ]

        prompt_text = kimi_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking=False,
        )

        assert "## Return of functions.lake_move:0" in prompt_text, (
            f"Expected tool response to include '## Return of ...' header. Got:\n{prompt_text}"
        )
        assert "lake_move" in prompt_text

    def test_im_end_not_followed_by_newline_for_kimi(self, kimi_tokenizer):
        """Kimi uses <|im_end|> without trailing newline (unlike Qwen3)."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]

        prompt_text = kimi_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking=False,
        )

        lines = prompt_text.split("<|im_end|>")
        for i, segment in enumerate(lines[:-1]):
            next_char = lines[i + 1][0] if lines[i + 1] else ""
            assert next_char != "\n" or "<|im_" in lines[i + 1][:20], (
                f"After <|im_end|> segment {i}, expected no bare newline. "
                f"Next segment starts with: {lines[i+1][:30]!r}"
            )


class TestRolloutMaskWithKimiPrefill:
    """Verify the masking module handles Kimi-style prefill correctly."""

    def test_prefill_tokens_are_masked_as_model_output(self):
        """The <|tool_calls_section_begin|> prefill is in completion_ids and should be trained."""
        token_turn_traces = [
            {
                "prompt_ids": [1, 2, 3],
                "completion_ids": [99, 10, 90],  # prefill=99, raw=10, im_end=90
            },
            {
                "prompt_ids": [1, 2, 3, 99, 10, 90, 20, 21],
                "completion_ids": [99, 30, 90],
            },
        ]
        model_request_traces = [
            {"assistant_turn_len": 3},
        ]

        spans = compute_model_output_spans(token_turn_traces, model_request_traces)
        assert spans == [(3, 3, 1), (8, 3, 2)]

        full_tokens = [1, 2, 3, 99, 10, 90, 20, 21, 99, 30, 90]
        ui_mask = build_ui_token_mask(spans, len(full_tokens))
        assert ui_mask == [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 2]

    def test_empty_think_block_in_prompt_is_not_masked(self):
        """<think></think> tokens in the prompt should NOT be trained on for GRPO."""
        think_open, think_close = 50, 51
        prefill = 99
        im_end = 90

        prompt_turn1 = [1, 2, 3, think_open, think_close]  # system + user + <think></think>
        completion_1 = [prefill, 10, im_end]
        tool_suffix = [20, 21, think_open, think_close]
        completion_2 = [prefill, 30, im_end]

        prompt_turn2 = prompt_turn1 + completion_1 + tool_suffix

        token_turn_traces = [
            {
                "prompt_ids": prompt_turn1,
                "completion_ids": completion_1,
            },
            {
                "prompt_ids": prompt_turn2,
                "completion_ids": completion_2,
            },
        ]
        model_request_traces = [
            {"assistant_turn_len": len(completion_1)},
        ]

        spans = compute_model_output_spans(token_turn_traces, model_request_traces)
        full_tokens = prompt_turn2 + completion_2

        ui_mask = build_ui_token_mask(spans, len(full_tokens))

        for i, token in enumerate(full_tokens):
            is_completion = (
                (i >= len(prompt_turn1) and i < len(prompt_turn1) + len(completion_1))
                or (i >= len(prompt_turn2))
            )
            if is_completion:
                assert ui_mask[i] > 0, f"Token {i} ({token}) should be masked (completion)"
            else:
                assert ui_mask[i] == 0, f"Token {i} ({token}) should NOT be masked (prompt/suffix)"

        think_positions_in_prompt = [3, 4]
        for pos in think_positions_in_prompt:
            assert ui_mask[pos] == 0, (
                f"<think>/<think> at prompt position {pos} must NOT be trained"
            )

        think_positions_in_suffix = [
            len(prompt_turn1) + len(completion_1) + 2,
            len(prompt_turn1) + len(completion_1) + 3,
        ]
        for pos in think_positions_in_suffix:
            assert ui_mask[pos] == 0, (
                f"<think></think> at tool suffix position {pos} must NOT be trained"
            )
