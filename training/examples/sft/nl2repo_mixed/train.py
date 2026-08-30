#!/usr/bin/env python3
"""Train Nemotron Ultra on a validated mixed partial-success NL2Repo bundle.

The input contract is the fail-closed output of fw-ai/upheaval PR #148:
``report.json``, ``renderer-report.json``, and split JSONL files in one
directory. The launcher rechecks their hashes and rendering invariants before
it can create paid Fireworks resources.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import training.recipes.sft_loop as sft_loop
import training.renderer  # noqa: F401  (register local renderers)
from training.utils import TrainerConfig, WandBConfig

CURATION_SCHEMA = "upheaval-prd-to-repo-curation-v1"
RENDER_REPORT_SCHEMA = "upheaval-prd-to-repo-render-validation-v1"
EXPECTED_PROFILE = "wentao-v1"
BASE_MODEL = "accounts/fireworks/models/nemotron-3-ultra-bf16"
TOKENIZER_MODEL = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
TOKENIZER_REVISION = "624ba927cfbef0427354998700de3d51173c8c04"
RENDERER_NAME = "nemotron3_ultra"
MAX_SEQ_LEN = 262_144
LORA_SHAPE = "accounts/fireworks/trainingShapes/nemotron-3-ultra-550b-a55b-bf16-lora"
MASK_FIX_PROMOTION_COMMIT = "e11f1ba4260b5321e542377be2bc77f8a3123e7f"
MASK_FIX_SOURCE_PR = "https://github.com/fw-ai/fireworks/pull/44496"
CURATION_SOURCE_PR = "https://github.com/fw-ai/upheaval/pull/148"
SKILL_CLIENT_SOURCE = "fireworks-training-skill/2.0.0"
EXPECTED_TRIAL_QUALITY = {"full_success": 260, "partial_success": 432}
EXPECTED_VERSION_COUNTS = {"2.1.231": 448, "2.1.237": 244}
EXPECTED_SPLIT_ROWS = {"train": 10_215, "val": 168, "test": 145}
EXPECTED_ARTIFACT_SHA256 = {
    "train.jsonl": ("4940da51f3a8dbe756993468cec1a4bf6aa1c18a729fb63e6e36c4e5b8f4a299"),
    "val.jsonl": ("92ea8bd38474df14b92dd798888e3c1f9d9b3d2b3a7db90ba765f0d7e3cd8d8c"),
    "test.jsonl": ("929aa7dec19483019dcb5305d825593058f2dcb0398b0e686c0d582f3faaa1b5"),
    "manifest.jsonl": (
        "97e698e019fb8d1c251f7340df3d97d682628436f57ae387f6875e4be8ce7729"
    ),
}
EXPECTED_CURATOR_SHA256 = (
    "a3f47437c53a781dece2f5992611daea02f3609a9e7c4daed1093a865ed2fdec"
)
EXPECTED_RENDER_VALIDATOR_SHA256 = (
    "4fe62fd08cbd93662bf6cca8e63bce1bf6ee6aea5f9cf97c114a575461237c4c"
)
EXPECTED_RENDER_CHECKS = {
    "all_rows_rendered",
    "one_datum_per_row",
    "all_rows_within_context",
    "all_datums_have_loss",
    "prefilled_think_open_masked",
    "generated_think_close_weighted",
    "no_errors",
}
EXPECTED_TINKER_COOKBOOK_VERSION = "0.4.3"
EXPECTED_RENDERING_SOURCE_SHA256 = {
    "renderer/_think_prefill.py": (
        "c33d8d11b7a896f3f895ec3dce7e82c184ecfd235258d0b80cadf25fa2912c10"
    ),
    "renderer/_qwen3_split.py": (
        "1ef9c7b4332c33031fe7903bd5ef107e84989fc3f51895f3261b164a15b550b4"
    ),
    "renderer/_nemotron3_split.py": (
        "f1390ba1870d66a20308bde146102e63c6be21a6cf9b2e0ab1fb6dc92e0e01dc"
    ),
    "renderer/_disaggregate_mixin.py": (
        "5bae7e3225ba7d31d656fb32e4a69a0e46c1d4e903ca59d8cf47ad41dc9dad44"
    ),
    "renderer/message_weights.py": (
        "a69f32322fec39fcbd21d66d9c775dce7ccbd52e197f19f93abb300d4814fb9a"
    ),
    "renderer/__init__.py": (
        "8e7f44c014dfd77b3cd7e3c0db6a0dba8b5ded5d5d138962c50e7f58e651c3ad"
    ),
    "utils/supervised.py": (
        "f92fcd36a3cd40d97ebfca427eb7fed58c5f1ef58d27a2a0fa2d1b6da0671c06"
    ),
    "utils/tokenizers.py": (
        "76bf305c0788b108e994de707ccd0ea8d83a971a3f78387d9cd9e8072e962f99"
    ),
}
EXPECTED_THINK_BOUNDARY_COUNTS = {
    "weighted_open_count": 0,
    "masked_open_count": 607_144,
    "weighted_close_count": 81_763,
    "masked_close_count": 525_561,
    "datums_without_weighted_close_count": 0,
}


@dataclass(frozen=True)
class DatasetBundle:
    """Validated local inputs and their provenance."""

    root: Path
    curation_report: dict[str, Any]
    renderer_report: dict[str, Any]
    train_path: Path
    val_path: Path
    manifest_path: Path
    train_sha256: str
    val_sha256: str
    manifest_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_and_rows(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            digest.update(line)
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            rows += 1
    return digest.hexdigest(), rows


def _rendering_source_hashes() -> dict[str, str]:
    root = Path(__file__).parents[3]
    return {
        relative_path: sha256_file(root / relative_path)
        for relative_path in EXPECTED_RENDERING_SOURCE_SHA256
    }


def _require_reviewed_rendering_runtime() -> dict[str, str]:
    actual = _rendering_source_hashes()
    if actual != EXPECTED_RENDERING_SOURCE_SHA256:
        raise ValueError("rendering sources differ from the reviewed implementation")
    try:
        tinker_version = importlib.metadata.version("tinker-cookbook")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("tinker-cookbook is not installed") from error
    if tinker_version != EXPECTED_TINKER_COOKBOOK_VERSION:
        raise ValueError("tinker-cookbook differs from the reviewed version")
    return actual


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing required artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _curation_artifact(
    root: Path,
    curation_report: dict[str, Any],
    filename: str,
) -> tuple[Path, str, int]:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing required artifact: {path}")
    digest, rows = _sha256_and_rows(path)
    artifact = (curation_report.get("artifacts") or {}).get(filename) or {}
    if artifact.get("sha256") != digest:
        raise ValueError(f"{filename}: hash differs from curation report")
    if EXPECTED_ARTIFACT_SHA256.get(filename) != digest:
        raise ValueError(f"{filename}: hash differs from the canonical dataset")
    reported_path = artifact.get("path")
    if not isinstance(reported_path, str) or Path(reported_path).name != filename:
        raise ValueError(f"{filename}: malformed artifact path in curation report")
    return path, digest, rows


def _rendered_artifact(
    root: Path,
    curation_report: dict[str, Any],
    renderer_report: dict[str, Any],
    filename: str,
) -> tuple[Path, str, int]:
    path, digest, rows = _curation_artifact(root, curation_report, filename)
    if (renderer_report.get("input_artifacts") or {}).get(filename) != digest:
        raise ValueError(f"{filename}: hash differs from renderer report")
    return path, digest, rows


def load_dataset_bundle(run_dir: Path) -> DatasetBundle:
    """Load and fail closed on stale, non-mixed, or unsafe inputs."""

    _require_reviewed_rendering_runtime()
    root = run_dir.resolve()
    curation_path = root / "report.json"
    renderer_path = root / "renderer-report.json"
    curation = _load_json(curation_path)
    renderer = _load_json(renderer_path)

    if curation.get("schema") != CURATION_SCHEMA:
        raise ValueError("unsupported curation report schema")
    if curation.get("profile") != EXPECTED_PROFILE:
        raise ValueError(f"dataset must use the frozen {EXPECTED_PROFILE!r} profile")
    curation_implementation = curation.get("implementation") or {}
    if (
        curation_implementation.get("module")
        != "experiments.nemotron_ultra_prd_to_repo.data.curation"
        or curation_implementation.get("script_sha256") != EXPECTED_CURATOR_SHA256
    ):
        raise ValueError("curation report was not produced by the reviewed curator")
    if renderer.get("schema") != RENDER_REPORT_SCHEMA:
        raise ValueError("unsupported renderer report schema")
    renderer_implementation = renderer.get("implementation") or {}
    if (
        renderer_implementation.get("module")
        != "experiments.nemotron_ultra_prd_to_repo.data.render_validate"
        or renderer_implementation.get("script_sha256")
        != EXPECTED_RENDER_VALIDATOR_SHA256
    ):
        raise ValueError("renderer report was not produced by the reviewed validator")
    if renderer.get("source_report_sha256") != sha256_file(curation_path):
        raise ValueError("renderer report is not bound to the current curation report")
    if not renderer.get("ready_for_training"):
        raise ValueError("renderer report is not ready for training")
    if renderer.get("errors"):
        raise ValueError("renderer report contains rendering errors")
    checks = renderer.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != EXPECTED_RENDER_CHECKS
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("renderer report has failed or missing checks")

    target = curation.get("target")
    expected_target = {
        "model": BASE_MODEL,
        "renderer": RENDERER_NAME,
        "tokenizer_model": TOKENIZER_MODEL,
        "tokenizer_revision": TOKENIZER_REVISION,
        "max_seq_len": MAX_SEQ_LEN,
    }
    if target != expected_target:
        raise ValueError("curation target differs from the reviewed training target")

    renderer_target = renderer.get("renderer") or {}
    for curation_key, renderer_key in (
        ("renderer", "renderer_name"),
        ("tokenizer_model", "tokenizer_model"),
        ("tokenizer_revision", "tokenizer_revision"),
        ("max_seq_len", "max_seq_len"),
    ):
        if renderer_target.get(renderer_key) != expected_target[curation_key]:
            raise ValueError(
                f"renderer report {renderer_key} differs from curation target"
            )
    if renderer_target.get("oversize_count") != 0:
        raise ValueError("renderer report contains overlength datums")
    if renderer_target.get("empty_loss_count") != 0:
        raise ValueError("renderer report contains zero-loss datums")
    expected_rows = sum(EXPECTED_SPLIT_ROWS.values())
    if renderer_target.get("rows_rendered") != expected_rows:
        raise ValueError("renderer report has an unexpected rendered row count")
    if renderer_target.get("datums_rendered") != expected_rows:
        raise ValueError("renderer report has an unexpected rendered datum count")
    think_weights = renderer_target.get("think_boundary_weights") or {}
    actual_think_counts = {
        key: think_weights.get(key) for key in EXPECTED_THINK_BOUNDARY_COUNTS
    }
    if actual_think_counts != EXPECTED_THINK_BOUNDARY_COUNTS:
        raise ValueError("think-boundary counts differ from the reviewed render")

    selection = curation.get("selection") or {}
    if selection.get("min_test_pass_rate") != 0.8:
        raise ValueError("mixed dataset must use the reviewed 80% verifier threshold")
    quality_counts = curation.get("trial_quality_labels") or {}
    if quality_counts != EXPECTED_TRIAL_QUALITY:
        raise ValueError("trial quality counts differ from the frozen profile")
    if curation.get("trial_claude_code_versions") != EXPECTED_VERSION_COUNTS:
        raise ValueError("Claude Code version counts differ from the frozen profile")
    if curation.get("selected_trials") != sum(EXPECTED_TRIAL_QUALITY.values()):
        raise ValueError("selected trial count differs from the frozen profile")
    if curation.get("selected_tasks") != 268:
        raise ValueError("selected task count differs from the frozen profile")

    train_path, train_sha, train_rows = _rendered_artifact(
        root, curation, renderer, "train.jsonl"
    )
    val_path, val_sha, val_rows = _rendered_artifact(
        root, curation, renderer, "val.jsonl"
    )
    _, _, test_rows = _rendered_artifact(root, curation, renderer, "test.jsonl")
    manifest_path, manifest_sha, manifest_rows = _curation_artifact(
        root, curation, "manifest.jsonl"
    )
    rows = renderer.get("rows") or {}
    if rows != EXPECTED_SPLIT_ROWS:
        raise ValueError("renderer split rows differ from the frozen profile")
    if {"train": train_rows, "val": val_rows, "test": test_rows} != (
        EXPECTED_SPLIT_ROWS
    ):
        raise ValueError("physical split rows differ from the frozen profile")
    if manifest_rows != expected_rows:
        raise ValueError("physical manifest rows differ from the frozen profile")

    return DatasetBundle(
        root=root,
        curation_report=curation,
        renderer_report=renderer,
        train_path=train_path,
        val_path=val_path,
        manifest_path=manifest_path,
        train_sha256=train_sha,
        val_sha256=val_sha,
        manifest_sha256=manifest_sha,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="ultra-mixed-partial80")
    parser.add_argument("--training-shape", default=LORA_SHAPE)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--trainer-replicas", type=int, default=1)
    parser.add_argument(
        "--use-reservation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-eval-seqs", type=int, default=200)
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and write config without creating Fireworks resources.",
    )
    return parser.parse_args(argv)


def _validate_training_args(args: argparse.Namespace) -> None:
    if (
        not args.run_name
        or Path(args.run_name).name != args.run_name
        or args.run_name in {".", ".."}
    ):
        raise ValueError("--run-name must be one non-empty path segment")
    if not args.training_shape:
        raise ValueError("--training-shape must not be empty")
    positive = {
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--learning-rate": args.learning_rate,
        "--lora-rank": args.lora_rank,
        "--lora-alpha": args.lora_alpha,
        "--trainer-replicas": args.trainer_replicas,
        "--pipeline-depth": args.pipeline_depth,
        "--checkpoint-interval": args.checkpoint_interval,
        "--grad-clip-norm": args.grad_clip_norm,
        "--max-eval-seqs": args.max_eval_seqs,
    }
    for name, value in positive.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    if bool(args.wandb_entity) != bool(args.wandb_project):
        raise ValueError("--wandb-entity and --wandb-project must be set together")


def _resolved_config(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    session_id: str,
) -> dict[str, Any]:
    rows = bundle.renderer_report["rows"]
    quality = bundle.curation_report["trial_quality_labels"]
    return {
        "method": "Training API dedicated SFT",
        "experiment": "mixed full-success and >=80% verifier partial-success",
        "risk": "negative ablation; checkpoint promotion is intentionally unsupported",
        "run_name": args.run_name,
        "model": {
            "base_model": BASE_MODEL,
            "tokenizer_model": TOKENIZER_MODEL,
            "tokenizer_revision": TOKENIZER_REVISION,
            "renderer_name": RENDERER_NAME,
            "max_seq_len": MAX_SEQ_LEN,
        },
        "renderer_fix": {
            "source_pr": MASK_FIX_SOURCE_PR,
            "public_promotion_commit": MASK_FIX_PROMOTION_COMMIT,
            "policy": "mask_prefilled_open_train_generated_close",
            "source_sha256": _require_reviewed_rendering_runtime(),
            "tinker_cookbook_version": EXPECTED_TINKER_COOKBOOK_VERSION,
        },
        "dataset": {
            "curation_source_pr": CURATION_SOURCE_PR,
            "profile": bundle.curation_report["profile"],
            "curation_report": str(bundle.root / "report.json"),
            "curation_report_sha256": sha256_file(bundle.root / "report.json"),
            "renderer_report": str(bundle.root / "renderer-report.json"),
            "renderer_report_sha256": sha256_file(bundle.root / "renderer-report.json"),
            "provenance": {
                "path": str(bundle.manifest_path),
                "sha256": bundle.manifest_sha256,
            },
            "train": {
                "path": str(bundle.train_path),
                "sha256": bundle.train_sha256,
                "rows": rows["train"],
            },
            "validation": {
                "path": str(bundle.val_path),
                "sha256": bundle.val_sha256,
                "rows": rows["val"],
            },
            "trial_quality_labels": quality,
        },
        "training": {
            "training_shape": args.training_shape,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "trainer_replicas": args.trainer_replicas,
            "use_reservation": args.use_reservation,
            "seed": args.seed,
            "pipeline_depth": args.pipeline_depth,
            "checkpoint_interval": args.checkpoint_interval,
            "grad_clip_norm": args.grad_clip_norm,
            "lr_scheduler": {
                "type": "cosine",
                "warmup_ratio": args.warmup_ratio,
                "min_lr_ratio": args.min_lr_ratio,
            },
            "weight_decay": args.weight_decay,
            "max_eval_seqs": args.max_eval_seqs,
            "save_final_checkpoint": True,
            "output_model_id": None,
        },
        "wandb": {
            "entity": args.wandb_entity or None,
            "project": args.wandb_project or None,
            "run_name": args.wandb_run_name or None,
        },
        "client": {
            "session_id": session_id,
            "source": SKILL_CLIENT_SOURCE,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_training_args(args)
    bundle = load_dataset_bundle(args.run_dir)
    session_id = os.environ.setdefault("FIREWORKS_SESSION_ID", str(uuid.uuid4()))
    os.environ.setdefault("FIREWORKS_CLIENT_SOURCE", SKILL_CLIENT_SOURCE)

    log_path = bundle.root / "training" / args.run_name
    log_path.mkdir(parents=True, exist_ok=True)
    resolved = _resolved_config(args, bundle, session_id)
    resolved_path = log_path / "resolved_config.json"
    resolved_path.write_text(
        json.dumps(resolved, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(resolved, allow_nan=False, indent=2, sort_keys=True))
    print(f"Resolved config: {resolved_path}")

    if args.dry_run:
        return 0
    if os.environ.get("CONFIRM_FIREWORKS_TRAINING") != "YES":
        raise RuntimeError(
            "Refusing to create paid resources. Review resolved_config.json, "
            "then set CONFIRM_FIREWORKS_TRAINING=YES."
        )
    if not os.environ.get("FIREWORKS_API_KEY"):
        raise RuntimeError("FIREWORKS_API_KEY is required for paid training")

    config = sft_loop.Config(
        log_path=str(log_path),
        render_samples_file=str(log_path / "render_samples.jsonl"),
        render_samples_limit=100,
        base_model=BASE_MODEL,
        dataset=str(bundle.train_path),
        evaluation_dataset=str(bundle.val_path),
        eval_auto_carveout=False,
        max_eval_seqs=args.max_eval_seqs,
        tokenizer_model=TOKENIZER_MODEL,
        tokenizer_revision=TOKENIZER_REVISION,
        tokenizer_trust_remote_code=True,
        renderer_name=RENDERER_NAME,
        train_on_what="all_assistant_messages",
        learning_rate=args.learning_rate,
        lr_scheduler={
            "type": "cosine",
            "warmup_ratio": args.warmup_ratio,
            "min_lr_ratio": args.min_lr_ratio,
        },
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=MAX_SEQ_LEN,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        output_model_id=None,
        save_final_checkpoint=True,
        dcp_save_interval=args.checkpoint_interval,
        grad_clip_norm=args.grad_clip_norm,
        weight_decay=args.weight_decay,
        warmup_steps=0,
        seed=args.seed,
        group_by_length=True,
        pipeline_depth=args.pipeline_depth,
        trainer=TrainerConfig(
            training_shape_id=args.training_shape,
            replica_count=args.trainer_replicas,
            use_reservation=args.use_reservation,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity or None,
            project=args.wandb_project or None,
            run_name=args.wandb_run_name or None,
        ),
    )
    metrics = sft_loop.main(config)
    print(
        json.dumps({"final_metrics": metrics}, allow_nan=False, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
