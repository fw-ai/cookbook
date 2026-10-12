from __future__ import annotations

import pytest
from jinja2 import TemplateError

from training.utils.rl.rollout import PrefixMismatch
from training.utils.rl.rollout.turn_traces import (
    append_turn_close,
    tool_reply_ids,
    turn_spans,
    turn_traces_to_sample,
)
from training.utils.tokenizers import load_tokenizer

TRACES = [
    {"prompt_ids": [1, 2], "completion_ids": [10], "completion_logprobs": [-0.5]},
    {"prompt_ids": [1, 2, 10, 7, 8], "completion_ids": [11, 12], "completion_logprobs": [-0.1, -0.2]},
]


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ([10, 11], [10, 11, 90, 91]),  # truncated: whole close appended
        ([10, 90], [10, 90, 91]),  # emitted the first close token
        ([10, 90, 91], [10, 90, 91]),  # emitted the whole close
        ([], [90, 91]),
    ],
)
def test_append_turn_close_never_doubles_the_close(completion, expected):
    assert append_turn_close(completion, [90, 91]) == expected


def test_turn_traces_to_sample_trains_completions_only():
    sample = turn_traces_to_sample(TRACES, reward=0.5, finish_reason="length")

    assert sample.tokens == [1, 2, 10, 7, 8, 11, 12]
    assert sample.loss_mask == [0, 0, 1, 0, 0, 1, 1]
    assert sample.logprobs == [0.0, 0.0, -0.5, 0.0, 0.0, -0.1, -0.2]
    assert (sample.reward, sample.finish_reason) == (0.5, "length")


def test_turn_spans_locate_each_completion_in_the_packed_episode():
    assert turn_spans(TRACES) == [(2, 3), (5, 7)]


def test_turn_traces_to_sample_rejects_broken_traces():
    with pytest.raises(ValueError, match="no turn traces"):
        turn_traces_to_sample([], reward=0.0)
    with pytest.raises(PrefixMismatch):
        turn_traces_to_sample(
            [TRACES[0], {**TRACES[1], "prompt_ids": [1, 2, 13, 7, 8]}], reward=0.0
        )


class _UserFirstTemplate:
    """Rejects a tool reply without a user turn before it, like Qwen3.5/3.6."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, **kwargs):
        if messages[0]["role"] == "tool":
            raise TemplateError("No user query found in messages.")
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def test_tool_reply_ids_renders_after_a_placeholder_user_turn_when_required():
    tokenizer = _UserFirstTemplate()

    ids = tool_reply_ids(tokenizer, {"role": "tool", "content": "r", "tool_call_id": None})

    assert ids == tokenizer.encode("<tool>r</tool><assistant>")


def test_tool_reply_ids_matches_the_qwen3_6_template():
    tokenizer = load_tokenizer("Qwen/Qwen3.6-35B-A3B")

    ids = tool_reply_ids(tokenizer, {"role": "tool", "content": "ok"}, enable_thinking=False)

    assert tokenizer.decode(ids) == (
        "<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
