from __future__ import annotations

import json

from eval_protocol.models import EvaluationRow, InputMetadata, Message

from training.examples.rl.frozen_lake.frozen_lake_schema import FROZEN_LAKE_TOOLS
from training.examples.rl.frozen_lake.verify_rollout import build_debug_report_html, enrich_rows


def _build_visual_row(*, row_id: str, image_counts: list[int]) -> EvaluationRow:
    row = EvaluationRow(
        input_metadata=InputMetadata(
            row_id=row_id,
            dataset_info={"environment_context": {"desc": ["SF", "FG"]}},
        )
    )
    row.tools = list(FROZEN_LAKE_TOOLS)
    row.messages = [
        Message(
            role="system",
            content="You are an RL policy for FrozenLake.\nAlways respond with exactly one tool call, no text.",
        ),
        Message(
            role="user",
            content="You are playing FrozenLake. The image shows the current grid and the text below gives the same observation.\n[S]  F\n F   G\n\nReply with exactly one token: LEFT, DOWN, RIGHT, or UP.",
        ),
        Message(
            role="assistant",
            content="RIGHT",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lake_move", "arguments": '{"action":"RIGHT"}'},
                }
            ],
        ),
        Message(
            role="tool",
            name="lake_move",
            tool_call_id="call_1",
            content=json.dumps(
                {
                    "observation": " S  [F]\n F   G",
                    "action": "RIGHT",
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                },
                separators=(",", ":"),
            ),
        ),
        Message(
            role="assistant",
            content="DOWN",
            tool_calls=[
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "lake_move", "arguments": '{"action":"DOWN"}'},
                }
            ],
        ),
        Message(
            role="tool",
            name="lake_move",
            tool_call_id="call_2",
            content=json.dumps(
                {
                    "observation": " S   F\n F  [G]",
                    "action": "DOWN",
                    "reward": 1.0,
                    "terminated": True,
                    "truncated": False,
                },
                separators=(",", ":"),
            ),
        ),
    ]
    row.execution_metadata.extra = {
        "observation_mode": "image",
        "step_rewards": [0.0, 1.0],
        "token_turn_traces": [
            {
                "step_index": 1,
                "prompt_ids": [10, 11, 12],
                "completion_ids": [99, 101, 90],
                "completion_logprobs": [-0.1],
                "prompt_text": "prompt:initial<think></think>",
            },
            {
                "step_index": 2,
                "prompt_ids": [10, 11, 12, 99, 101, 90, 91, 92],
                "completion_ids": [99, 102, 90],
                "completion_logprobs": [-0.2],
                "prompt_text": "prompt:stitched<think></think>",
            },
        ],
        "model_request_traces": [
            {
                "step_index": 1,
                "prompt_ids": [10, 11, 12, 99],
                "prompt_token_count": 4,
                "prompt_text": "prompt:initial<prefill>",
                "assistant_prefill_len": 1,
                "assistant_turn_len": 3,
                "tool_suffix_len": 3,
                "image_count": image_counts[0],
            },
            {
                "step_index": 2,
                "prompt_ids": [10, 11, 12, 99, 101, 90, 91, 92, 99],
                "prompt_token_count": 9,
                "prompt_text": "prompt:stitched<prefill>",
                "assistant_prefill_len": 1,
                "assistant_turn_len": 3,
                "tool_suffix_len": 0,
                "image_count": image_counts[1],
            },
        ],
        "tool_call_traces": [
            {"step_index": 1, "tool_name": "lake_move", "action": "RIGHT", "reward": 0.0},
            {"step_index": 2, "tool_name": "lake_move", "action": "DOWN", "reward": 1.0},
        ],
    }
    return row


def test_enrich_rows_adds_passing_visual_validation_summary():
    row = _build_visual_row(row_id="passing", image_counts=[1, 2])

    enrich_rows(
        [row],
        "missing-tokenizer-for-test",
        model_id="accounts/fireworks/models/kimi-k2p5-vl",
        visual=True,
    )

    extra = row.execution_metadata.extra or {}
    validation_summary = extra["validation_summary"]
    validation_checks = extra["validation_checks"]
    full_episode = extra["full_episode"]

    assert validation_summary["target"] == "kimi-k2.5-vl"
    assert validation_summary["passed"] is True
    assert validation_summary["failing_checks"] == []
    assert validation_summary["image_counts"] == [1, 2]
    assert len(full_episode["token_ids"]) == len(full_episode["mask"]) == len(full_episode["logprobs"])
    assert full_episode["mask"][3] == 1
    assert full_episode["logprobs"][3] is None
    assert full_episode["logprobs"][4] == -0.1
    assert full_episode["mask"][5] == 1
    assert full_episode["logprobs"][5] is None
    assert any(check["name"] == "image_counts_increment_per_turn" and check["status"] == "pass" for check in validation_checks)
    assert any(
        check["name"] == "kimi_instant_prompt_contains_empty_think_block" and check["status"] == "pass"
        for check in validation_checks
    )


def test_enrich_rows_flags_broken_image_count_validation():
    row = _build_visual_row(row_id="failing", image_counts=[1, 1])

    enrich_rows(
        [row],
        "missing-tokenizer-for-test",
        model_id="accounts/fireworks/models/kimi-k2p5-vl",
        visual=True,
    )

    extra = row.execution_metadata.extra or {}
    validation_summary = extra["validation_summary"]

    assert validation_summary["passed"] is False
    assert "image_counts_increment_per_turn" in validation_summary["failing_checks"]


def test_debug_report_renders_validation_section():
    row = _build_visual_row(row_id="report_row", image_counts=[1, 2])

    enrich_rows(
        [row],
        "missing-tokenizer-for-test",
        model_id="accounts/fireworks/models/kimi-k2p5-vl",
        visual=True,
    )

    report_html = build_debug_report_html([row])

    assert "Validation Checks" in report_html
    assert "kimi-k2.5-vl" in report_html
    assert "image_counts_increment_per_turn" in report_html
