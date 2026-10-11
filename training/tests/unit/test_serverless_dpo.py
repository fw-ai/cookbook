"""Tests for the serverless DPO example: data prep and its dpo_loop config."""

from __future__ import annotations

import pytest

import training.recipes.dpo_loop as dpo_loop
from training.examples.dpo import train_dpo_serverless
from training.examples.dpo.prepare_data import row_to_preference


def test_serverless_dpo_defaults_use_supported_model_and_matching_tokenizer(tmp_path):
    cfg = train_dpo_serverless.build_config(
        train_dpo_serverless.parse_args(["--log-path", str(tmp_path)])
    )

    assert cfg.serverless is True
    assert cfg.base_model == "accounts/fireworks/models/kimi-k3"
    assert cfg.tokenizer_model == "moonshotai/Kimi-K3"
    assert cfg.max_seq_len > 0
    assert cfg.lora_rank > 0
    assert 0 < cfg.beta < 0.5
    assert cfg.max_pairs == 32
    assert cfg.output_model_id is None


def test_serverless_dpo_records_frozen_reference_under_log_path(tmp_path):
    # Resume needs the original frozen reference; it lives beside the
    # checkpoints' dataloader state so reusing --log-path restores both.
    cfg = train_dpo_serverless.build_config(
        train_dpo_serverless.parse_args(["--log-path", str(tmp_path / "run")])
    )

    assert cfg.runner.reference_checkpoint_file == str(
        tmp_path / "run" / "reference_checkpoint.json"
    )


def test_serverless_dpo_points_at_prepare_data_when_dataset_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(dpo_loop, "main", lambda cfg: pytest.fail("recipe must not start"))

    with pytest.raises(SystemExit, match="prepare_data"):
        train_dpo_serverless.main(["--dataset", str(tmp_path / "missing.jsonl")])


def test_row_to_preference_builds_shared_prompt_pair():
    pref = row_to_preference({
        "instruction": "Explain caching in one sentence.",
        "chosen_response": "Caching stores results so later requests are faster.",
        "rejected_response": "Caching is a thing computers do.",
    })
    assert pref is not None
    assert pref["chosen"]["messages"][:1] == pref["rejected"]["messages"][:1]
    assert pref["chosen"]["messages"][-1]["role"] == "assistant"
    assert pref["chosen"]["messages"][-1]["content"] != pref["rejected"]["messages"][-1]["content"]


@pytest.mark.parametrize(
    "example",
    [
        {"instruction": "", "chosen_response": "a", "rejected_response": "b"},
        {"instruction": "q", "chosen_response": "", "rejected_response": "b"},
        {"instruction": "q", "chosen_response": "same", "rejected_response": "same"},
    ],
)
def test_row_to_preference_drops_invalid_rows(example):
    assert row_to_preference(example) is None
