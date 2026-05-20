from __future__ import annotations

from training.utils.rl.rollout.trace import (
    analyze_flat_sample,
    analyze_token_turn_traces,
    analyze_turns,
)
from training.utils.rl.rollout.assembler import InferenceCall, TrajectoryAssembler


class _Tokenizer:
    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(f"<{int(i)}>" for i in ids)


def test_token_turn_trace_is_native_trajectory_not_artifact():
    trajectory = analyze_token_turn_traces(
        [
            {"prompt_ids": [1, 2], "completion_ids": [3], "completion_logprobs": [-0.1]},
            {"prompt_ids": [1, 2, 3, 4], "completion_ids": [5], "completion_logprobs": [-0.2]},
        ],
        tokenizer=_Tokenizer(),
    )

    data = trajectory.to_dict()
    assert data["kind"] == "rollout_trajectory"
    assert data["summary"]["prefix_mismatch_count"] == 0
    assert [row["token_id"] for row in data["tokens"]] == [1, 2, 3, 4, 5]
    assert [row["chunk_source"] for row in data["tokens"]] == [
        "prompt_delta",
        "prompt_delta",
        "completion",
        "prompt_delta",
        "completion",
    ]


def test_token_turn_trace_surfaces_prefix_mismatch_on_trajectory_tokens():
    trajectory = analyze_token_turn_traces(
        [
            {"prompt_ids": [1], "completion_ids": [2]},
            {"prompt_ids": [9], "completion_ids": [10]},
        ],
        tokenizer=_Tokenizer(),
    )

    data = trajectory.to_dict()
    assert data["summary"]["prefix_mismatch_count"] == 1
    assert data["issues"][0]["code"] == "prefix_mismatch"
    diverged = [row for row in data["tokens"] if "prefix_mismatch" in row["issues"]]
    assert [row["token_id"] for row in diverged] == [9]


def test_turn_trace_keeps_roles_and_logprob_issue():
    trajectory = analyze_turns(
        [
            {"role": "user", "token_ids": [1, 2]},
            {"role": "assistant", "token_ids": [3], "logprobs": []},
        ],
        tokenizer=_Tokenizer(),
    )

    data = trajectory.to_dict()
    assert data["summary"]["logprob_mismatch_count"] == 1
    assert [row["role"] for row in data["tokens"]] == ["user", "user", "assistant"]
    assert [row["renderer_claim_weight"] for row in data["tokens"]] == [0.0, 0.0, 1.0]


def test_flat_sample_trace_uses_loss_mask_as_training_boundary():
    trajectory = analyze_flat_sample(
        {"tokens": [1, 2, 3, 4], "loss_mask": [0, 1, 0, 1]},
        tokenizer=_Tokenizer(),
    )

    data = trajectory.to_dict()
    assert data["summary"]["generated_token_count"] == 2
    assert [row["role"] for row in data["tokens"]] == [
        "context",
        "assistant",
        "context",
        "assistant",
    ]


def test_trajectory_assembler_emits_native_trajectory():
    asm = TrajectoryAssembler()
    asm.add_call(
        InferenceCall(
            input_tokens=[1, 2],
            output_tokens=[3],
            output_logprobs=[-0.1],
        )
    )

    data = asm.to_trajectory(tokenizer=_Tokenizer()).to_dict()
    assert data["kind"] == "rollout_trajectory"
    assert [row["token_id"] for row in data["tokens"]] == [1, 2, 3]
    assert [row["role"] for row in data["tokens"]] == ["user", "user", "assistant"]
