from __future__ import annotations

import importlib
import os
import sys

import pytest


def _load_module(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    import training.examples.text2sql_sft.train_sft as module

    return importlib.reload(module)


def test_parse_args_reads_overrides(monkeypatch):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sft.py",
            "--base-model",
            "accounts/test/models/qwen3-4b",
            "--tokenizer-model",
            "Qwen/Qwen3-4B",
            "--dataset-path",
            "/tmp/text2sql.jsonl",
            "--training-shape",
            "ts-qwen3-4b-smoke-v1",
            "--max-examples",
            "16",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--grad-accum",
            "3",
            "--learning-rate",
            "2e-5",
            "--lora-rank",
            "8",
        ],
    )

    args = module.parse_args()

    assert args.base_model == "accounts/test/models/qwen3-4b"
    assert args.tokenizer_model == "Qwen/Qwen3-4B"
    assert args.dataset_path == "/tmp/text2sql.jsonl"
    assert args.training_shape == "ts-qwen3-4b-smoke-v1"
    assert args.max_examples == 16
    assert args.epochs == 2
    assert args.batch_size == 4
    assert args.grad_accum == 3
    assert args.learning_rate == 2e-5
    assert args.lora_rank == 8


def test_main_raises_when_dataset_is_missing(monkeypatch):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--dataset-path", "/tmp/missing.jsonl"])
    monkeypatch.setattr(module.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        module.main()


def test_main_builds_sft_config_and_calls_recipe(monkeypatch):
    module = _load_module(monkeypatch)
    dataset_path = "/tmp/text2sql.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sft.py",
            "--base-model",
            "accounts/test/models/qwen3-4b",
            "--tokenizer-model",
            "Qwen/Qwen3-4B",
            "--dataset-path",
            dataset_path,
            "--training-shape",
            "ts-qwen3-4b-smoke-v1",
            "--region",
            "US_OHIO_1",
            "--max-examples",
            "8",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--grad-accum",
            "2",
            "--learning-rate",
            "2e-5",
            "--lora-rank",
            "16",
        ],
    )
    monkeypatch.setattr(module.os.path, "exists", lambda path: path == dataset_path)

    created = {}

    class FakeTrainerJobManager:
        def __init__(self, *, api_key, account_id, base_url):
            created["mgr_init"] = {
                "api_key": api_key,
                "account_id": account_id,
                "base_url": base_url,
            }

    def fake_sft_main(config, *, rlor_mgr=None):
        created["config"] = config
        created["rlor_mgr"] = rlor_mgr
        return {"train/loss": 0.123}

    monkeypatch.setattr(module, "TrainerJobManager", FakeTrainerJobManager)
    monkeypatch.setattr(module.sft_loop, "main", fake_sft_main)

    module.main()

    cfg = created["config"]
    assert cfg.base_model == "accounts/test/models/qwen3-4b"
    assert cfg.dataset == dataset_path
    assert cfg.tokenizer_model == "Qwen/Qwen3-4B"
    assert cfg.learning_rate == 2e-5
    assert cfg.epochs == 2
    assert cfg.batch_size == 4
    assert cfg.grad_accum == 2
    assert cfg.max_examples == 8
    assert cfg.lora_rank == 16
    assert cfg.infra.training_shape_id == "ts-qwen3-4b-smoke-v1"
    assert cfg.infra.region == "US_OHIO_1"
    assert cfg.infra.skip_validations is True
    assert created["mgr_init"] == {
        "api_key": "test-key",
        "account_id": "acct",
        "base_url": "https://unit.test",
    }
