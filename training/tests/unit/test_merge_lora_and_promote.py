"""Tests for LoRA merge-and-promote export format selection."""

from __future__ import annotations

import pytest

from training.examples.tools import merge_lora_and_promote as merge_tool


@pytest.mark.parametrize(
    "precision",
    ["source", "bf16", "nvfp4", "mxfp8", "fp8_block128"],
)
def test_parse_args_accepts_final_export_precisions(monkeypatch, precision):
    monkeypatch.setattr(
        merge_tool.sys,
        "argv",
        [
            "merge_lora_and_promote.py",
            "--base-model",
            "accounts/a/models/base",
            "--adapter-gcs",
            "gs://bucket/adapter",
            "--lora-rank",
            "8",
            "--output-model-id",
            "merged",
            "--export-precision",
            precision,
        ],
    )

    assert merge_tool.parse_args().export_precision == precision


def test_parse_args_defaults_final_export_to_source(monkeypatch):
    monkeypatch.setattr(
        merge_tool.sys,
        "argv",
        [
            "merge_lora_and_promote.py",
            "--base-model",
            "accounts/a/models/base",
            "--adapter-gcs",
            "gs://bucket/adapter",
            "--lora-rank",
            "8",
            "--output-model-id",
            "merged",
        ],
    )

    assert merge_tool.parse_args().export_precision == "source"


@pytest.mark.parametrize("precision", ["source", "bf16", "nvfp4", "mxfp8", "fp8_block128"])
def test_export_precision_does_not_force_full_model_dequantization(precision):
    assert merge_tool._training_quant_extra_args(precision) == []
