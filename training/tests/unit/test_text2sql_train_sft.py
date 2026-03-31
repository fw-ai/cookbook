from __future__ import annotations

import importlib
import os
import sys

import pytest


def _load_module(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    import training.examples.sft.train_sft as module

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
            "--output-model-id",
            "out-model",
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


def test_parse_args_uses_bundled_text2sql_dataset_by_default(monkeypatch):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--output-model-id", "out-model"])

    args = module.parse_args()

    assert os.path.basename(args.dataset_path) == "text2sql_dataset.jsonl"


def test_main_raises_when_dataset_is_missing(monkeypatch):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--dataset-path", "/tmp/missing.jsonl", "--output-model-id", "out"])
    monkeypatch.setattr(module.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError, match="text2sql_dataset.jsonl|Dataset not found"):
        module.main()


