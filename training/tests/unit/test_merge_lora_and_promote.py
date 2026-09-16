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
            "--adapter-model",
            "accounts/a/models/lora",
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
            "--adapter-model",
            "accounts/a/models/lora",
            "--lora-rank",
            "8",
            "--output-model-id",
            "merged",
        ],
    )

    cfg = merge_tool.parse_args()
    assert cfg.export_precision == "source"
    assert cfg.adapter == "accounts/a/models/lora"


def test_parse_args_accepts_adapter_gcs_alias(monkeypatch):
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

    assert merge_tool.parse_args().adapter == "gs://bucket/adapter"


@pytest.mark.parametrize(
    "precision", ["source", "bf16", "nvfp4", "mxfp8", "fp8_block128"]
)
def test_export_precision_does_not_force_full_model_dequantization(precision):
    assert merge_tool._training_quant_extra_args(precision) == []


@pytest.mark.parametrize(
    "job_id,keep,cleanup",
    [(None, False, True), (None, True, False), ("ci-policy", False, False)],
)
def test_run_loads_adapter_before_merge_and_preserves_attached_trainer(
    monkeypatch, job_id, keep, cleanup
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    events = []
    policy = Mock()
    policy.load_adapter.side_effect = lambda name: (
        events.append(("load", name)) or Mock()
    )
    policy.save_weights_for_sampler_ext.side_effect = lambda name, **kwargs: (
        events.append(("save", name, kwargs))
        or SimpleNamespace(path="sampler", snapshot_name=name)
    )
    service = Mock(trainer_job_id=job_id or "temporary")
    service.create_lora_training_client.return_value = policy
    build = Mock(return_value=service)
    model = {
        "name": "accounts/test/models/merged",
        "state": "READY",
        "kind": "HF_BASE_MODEL",
    }
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(
        merge_tool, "FireworksClient", Mock(return_value=Mock(account_id="test"))
    )
    monkeypatch.setattr(merge_tool, "TrainerJobManager", Mock())
    monkeypatch.setattr(merge_tool, "build_service_client", build)
    monkeypatch.setattr(
        merge_tool,
        "_resolve_merged_checkpoint",
        lambda *args: {"name": "exact-checkpoint"},
    )
    monkeypatch.setattr(merge_tool, "_poll_model_until_ready", lambda *args: model)
    cfg = merge_tool.MergeConfig(
        base_model="accounts/test/models/base",
        adapter="accounts/test/models/adapter",
        lora_rank=8,
        training_shape="",
        output_model_id="merged",
        export_precision="source",
        region=None,
        snapshot_name="ci-merged",
        keep_trainer=keep,
        trainer_timeout_s=1,
        op_timeout_s=1,
        checkpoint_poll_timeout_s=1,
        promote_poll_timeout_s=1,
        trainer_job_id=job_id,
    )
    assert merge_tool.run(cfg) == model
    assert events == [
        ("load", cfg.adapter),
        ("save", "ci-merged", {"checkpoint_type": "merged_base"}),
    ]
    assert build.call_args.kwargs["trainer"].job_id == job_id
    assert build.call_args.kwargs["cleanup_trainer_on_close"] is cleanup
    service.close.assert_called_once()
