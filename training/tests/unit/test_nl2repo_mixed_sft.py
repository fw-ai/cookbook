from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.examples.sft.nl2repo_mixed import train


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _make_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for split, rows in train.EXPECTED_SPLIT_ROWS.items():
        line = json.dumps({"messages": [{"role": "user", "content": split}]}) + "\n"
        (tmp_path / f"{split}.jsonl").write_text(
            line * rows,
            encoding="utf-8",
        )
    manifest_line = json.dumps({"split": "train", "row_index": 0}) + "\n"
    (tmp_path / "manifest.jsonl").write_text(
        manifest_line * sum(train.EXPECTED_SPLIT_ROWS.values()),
        encoding="utf-8",
    )

    target = {
        "model": train.BASE_MODEL,
        "renderer": train.RENDERER_NAME,
        "tokenizer_model": train.TOKENIZER_MODEL,
        "tokenizer_revision": train.TOKENIZER_REVISION,
        "max_seq_len": train.MAX_SEQ_LEN,
    }
    curation = {
        "schema": train.CURATION_SCHEMA,
        "profile": train.EXPECTED_PROFILE,
        "target": target,
        "selection": {"min_test_pass_rate": 0.8},
        "trial_quality_labels": train.EXPECTED_TRIAL_QUALITY,
        "trial_claude_code_versions": train.EXPECTED_VERSION_COUNTS,
        "selected_trials": 692,
        "selected_tasks": 268,
        "implementation": {
            "module": "experiments.nemotron_ultra_prd_to_repo.data.curation",
            "script_sha256": train.EXPECTED_CURATOR_SHA256,
        },
        "artifacts": {
            filename: {
                "path": str(tmp_path / filename),
                "sha256": _sha256(tmp_path / filename),
            }
            for filename in (
                "train.jsonl",
                "val.jsonl",
                "test.jsonl",
                "manifest.jsonl",
            )
        },
    }
    curation_path = tmp_path / "report.json"
    _write_json(curation_path, curation)

    renderer = {
        "schema": train.RENDER_REPORT_SCHEMA,
        "source_report_sha256": _sha256(curation_path),
        "implementation": {
            "module": ("experiments.nemotron_ultra_prd_to_repo.data.render_validate"),
            "script_sha256": train.EXPECTED_RENDER_VALIDATOR_SHA256,
        },
        "input_artifacts": {
            f"{split}.jsonl": _sha256(tmp_path / f"{split}.jsonl")
            for split in ("train", "val", "test")
        },
        "rows": train.EXPECTED_SPLIT_ROWS,
        "renderer": {
            "renderer_name": train.RENDERER_NAME,
            "tokenizer_model": train.TOKENIZER_MODEL,
            "tokenizer_revision": train.TOKENIZER_REVISION,
            "max_seq_len": train.MAX_SEQ_LEN,
            "rows_rendered": sum(train.EXPECTED_SPLIT_ROWS.values()),
            "datums_rendered": sum(train.EXPECTED_SPLIT_ROWS.values()),
            "oversize_count": 0,
            "empty_loss_count": 0,
            "think_boundary_weights": train.EXPECTED_THINK_BOUNDARY_COUNTS,
        },
        "checks": {
            "all_rows_rendered": True,
            "one_datum_per_row": True,
            "all_rows_within_context": True,
            "all_datums_have_loss": True,
            "prefilled_think_open_masked": True,
            "generated_think_close_weighted": True,
            "no_errors": True,
        },
        "errors": [],
        "ready_for_training": True,
    }
    _write_json(tmp_path / "renderer-report.json", renderer)
    monkeypatch.setattr(
        train,
        "EXPECTED_ARTIFACT_SHA256",
        {
            filename: _sha256(tmp_path / filename)
            for filename in (
                "train.jsonl",
                "val.jsonl",
                "test.jsonl",
                "manifest.jsonl",
            )
        },
    )
    return tmp_path


def test_load_dataset_bundle_accepts_bound_mixed_inputs(tmp_path, monkeypatch):
    run_dir = _make_bundle(tmp_path, monkeypatch)

    bundle = train.load_dataset_bundle(run_dir)

    assert bundle.train_path == run_dir / "train.jsonl"
    assert bundle.val_path == run_dir / "val.jsonl"
    assert bundle.manifest_path == run_dir / "manifest.jsonl"
    assert bundle.train_sha256 == _sha256(bundle.train_path)


def test_load_dataset_bundle_rejects_self_consistent_noncanonical_inputs(
    tmp_path, monkeypatch
):
    canonical_hashes = dict(train.EXPECTED_ARTIFACT_SHA256)
    run_dir = _make_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(train, "EXPECTED_ARTIFACT_SHA256", canonical_hashes)

    with pytest.raises(ValueError, match="canonical dataset"):
        train.load_dataset_bundle(run_dir)


def test_load_dataset_bundle_rejects_split_hash_drift(tmp_path, monkeypatch):
    run_dir = _make_bundle(tmp_path, monkeypatch)
    (run_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash differs from curation report"):
        train.load_dataset_bundle(run_dir)


def test_load_dataset_bundle_counts_physical_rows_after_rehash(tmp_path, monkeypatch):
    run_dir = _make_bundle(tmp_path, monkeypatch)
    train_path = run_dir / "train.jsonl"
    lines = train_path.read_text(encoding="utf-8").splitlines(keepends=True)
    train_path.write_text("".join(lines[:-1]), encoding="utf-8")

    curation_path = run_dir / "report.json"
    curation = json.loads(curation_path.read_text(encoding="utf-8"))
    curation["artifacts"]["train.jsonl"]["sha256"] = _sha256(train_path)
    _write_json(curation_path, curation)

    renderer_path = run_dir / "renderer-report.json"
    renderer = json.loads(renderer_path.read_text(encoding="utf-8"))
    renderer["source_report_sha256"] = _sha256(curation_path)
    renderer["input_artifacts"]["train.jsonl"] = _sha256(train_path)
    _write_json(renderer_path, renderer)
    expected_hashes = dict(train.EXPECTED_ARTIFACT_SHA256)
    expected_hashes["train.jsonl"] = _sha256(train_path)
    monkeypatch.setattr(train, "EXPECTED_ARTIFACT_SHA256", expected_hashes)

    with pytest.raises(ValueError, match="physical split rows"):
        train.load_dataset_bundle(run_dir)


def test_load_dataset_bundle_rejects_non_mixed_curation(tmp_path, monkeypatch):
    run_dir = _make_bundle(tmp_path, monkeypatch)
    curation_path = run_dir / "report.json"
    curation = json.loads(curation_path.read_text(encoding="utf-8"))
    curation["trial_quality_labels"]["partial_success"] = 0
    _write_json(curation_path, curation)
    renderer_path = run_dir / "renderer-report.json"
    renderer = json.loads(renderer_path.read_text(encoding="utf-8"))
    renderer["source_report_sha256"] = _sha256(curation_path)
    _write_json(renderer_path, renderer)

    with pytest.raises(ValueError, match="trial quality counts"):
        train.load_dataset_bundle(run_dir)


def test_load_dataset_bundle_rejects_weighted_think_open(tmp_path, monkeypatch):
    run_dir = _make_bundle(tmp_path, monkeypatch)
    renderer_path = run_dir / "renderer-report.json"
    renderer = json.loads(renderer_path.read_text(encoding="utf-8"))
    renderer["renderer"]["think_boundary_weights"]["weighted_open_count"] = 1
    _write_json(renderer_path, renderer)

    with pytest.raises(ValueError, match="think-boundary counts"):
        train.load_dataset_bundle(run_dir)


def test_load_dataset_bundle_rejects_unreviewed_validator(tmp_path, monkeypatch):
    run_dir = _make_bundle(tmp_path, monkeypatch)
    renderer_path = run_dir / "renderer-report.json"
    renderer = json.loads(renderer_path.read_text(encoding="utf-8"))
    renderer["implementation"]["script_sha256"] = "0" * 64
    _write_json(renderer_path, renderer)

    with pytest.raises(ValueError, match="reviewed validator"):
        train.load_dataset_bundle(run_dir)


def test_dry_run_writes_resolved_config_without_training(tmp_path, monkeypatch):
    run_dir = _make_bundle(tmp_path, monkeypatch)
    monkeypatch.setenv("FIREWORKS_SESSION_ID", "test-session")
    monkeypatch.setattr(
        train.sft_loop,
        "main",
        lambda config: pytest.fail("dry-run must not call sft_loop.main"),
    )

    result = train.main(
        ["--run-dir", str(run_dir), "--run-name", "dry-run-test", "--dry-run"]
    )

    assert result == 0
    resolved_path = run_dir / "training" / "dry-run-test" / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert resolved["model"]["renderer_name"] == "nemotron3_ultra"
    assert resolved["dataset"]["trial_quality_labels"] == train.EXPECTED_TRIAL_QUALITY
    assert resolved["training"]["output_model_id"] is None


def test_paid_run_requires_explicit_confirmation(tmp_path, monkeypatch):
    run_dir = _make_bundle(tmp_path, monkeypatch)
    monkeypatch.delenv("CONFIRM_FIREWORKS_TRAINING", raising=False)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")

    with pytest.raises(RuntimeError, match="CONFIRM_FIREWORKS_TRAINING=YES"):
        train.main(["--run-dir", str(run_dir)])


def test_confirmed_run_calls_sft_with_reviewed_defaults(tmp_path, monkeypatch):
    run_dir = _make_bundle(tmp_path, monkeypatch)
    seen = {}
    monkeypatch.setenv("CONFIRM_FIREWORKS_TRAINING", "YES")
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")

    def fake_main(config):
        seen["config"] = config
        return {"loss": 0.5}

    monkeypatch.setattr(train.sft_loop, "main", fake_main)

    assert train.main(["--run-dir", str(run_dir)]) == 0
    config = seen["config"]
    assert config.dataset == str(run_dir / "train.jsonl")
    assert config.evaluation_dataset == str(run_dir / "val.jsonl")
    assert config.renderer_name == "nemotron3_ultra"
    assert config.max_seq_len == 262_144
    assert config.learning_rate == pytest.approx(3e-7)
    assert config.lora_rank == 16
    assert config.lora_alpha == 32
    assert config.output_model_id is None
