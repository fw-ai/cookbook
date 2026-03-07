from __future__ import annotations

import json
import sys

from eval_protocol.models import EvaluationRow, InputMetadata

from training.examples.frozen_lake.masking import (
    build_training_loss_mask,
    build_ui_token_mask,
    compute_model_output_spans,
)
from training.examples.frozen_lake.train_frozen_lake import (
    FrozenLakeConfig,
    evaluation_row_to_training_data,
    load_seed_contexts,
    parse_args,
)


def test_parse_args_applies_cli_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_frozen_lake.py",
            "--base-model",
            "accounts/test/models/qwen3-4b",
            "--tokenizer-model",
            "Qwen/Qwen3-4B",
            "--max-seeds",
            "3",
            "--max-steps",
            "9",
            "--observation-mode",
            "image",
            "--allow-plaintext-action-fallback",
            "--training-shape",
            "ts-qwen3-4b-smoke-v1",
        ],
    )

    cfg = parse_args()

    assert isinstance(cfg, FrozenLakeConfig)
    assert cfg.base_model == "accounts/test/models/qwen3-4b"
    assert cfg.tokenizer_model == "Qwen/Qwen3-4B"
    assert cfg.max_seeds == 3
    assert cfg.max_steps == 9
    assert cfg.observation_mode == "image"
    assert cfg.allow_plaintext_action_fallback is True
    assert cfg.training_shape == "ts-qwen3-4b-smoke-v1"


def test_load_seed_contexts_respects_limit_and_defaults(tmp_path):
    seed_path = tmp_path / "seeds.jsonl"
    seed_path.write_text(
        "\n".join(
            [
                json.dumps({"seed": 101, "map_name": "4x4"}),
                json.dumps({"seed": 202}),
                json.dumps({"seed": 303, "map_name": "8x8"}),
            ]
        )
    )

    contexts = load_seed_contexts(str(seed_path), max_seeds=2)

    assert contexts == [
        {"map_name": "4x4", "use_random_map": True, "seed": 101},
        {"map_name": "4x4", "use_random_map": True, "seed": 202},
    ]


def test_evaluation_row_to_training_data_returns_empty_without_traces():
    row = EvaluationRow(input_metadata=InputMetadata(row_id="empty"))
    row.execution_metadata.extra = {}

    datums, prompt_len, inf_logprobs, rewards = evaluation_row_to_training_data(row)

    assert datums == []
    assert prompt_len == 0
    assert inf_logprobs == []
    assert rewards == []


def test_evaluation_row_to_training_data_builds_weighted_episode():
    token_turn_traces = [
        {
            "prompt_ids": [1, 2, 3],
            "completion_ids": [10, 11],
            "completion_logprobs": [-0.1, -0.2],
        },
        {
            "prompt_ids": [1, 2, 3, 10, 11, 20, 21, 22],
            "completion_ids": [30, 31, 32],
            "completion_logprobs": [-0.3, -0.4, -0.5],
        },
    ]
    model_request_traces = [{"assistant_turn_len": 2}, {}]
    step_rewards = [0.0, 1.0]

    row = EvaluationRow(input_metadata=InputMetadata(row_id="episode"))
    row.execution_metadata.extra = {
        "token_turn_traces": token_turn_traces,
        "model_request_traces": model_request_traces,
        "step_rewards": step_rewards,
    }

    datums, prompt_len, inf_logprobs, rewards = evaluation_row_to_training_data(row)

    full_tokens = token_turn_traces[-1]["prompt_ids"] + token_turn_traces[-1]["completion_ids"]
    spans = compute_model_output_spans(token_turn_traces, model_request_traces)
    ui_mask = build_ui_token_mask(spans, len(full_tokens))
    loss_mask = build_training_loss_mask(spans, len(full_tokens) - 1)

    assert prompt_len == 3
    assert rewards == [1.0]
    assert inf_logprobs == [[0.0, 0.0, -0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4, -0.5]]
    assert len(datums) == 1
    assert datums[0].loss_fn_inputs["target_tokens"].data == full_tokens[1:]
    assert datums[0].loss_fn_inputs["weights"].data == loss_mask
    assert datums[0].loss_fn_inputs["loss_mask"].data == loss_mask
    assert ui_mask == [0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 2]
