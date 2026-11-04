#!/usr/bin/env python3
"""Plan a Humanizer lesson run locally; --execute opts in to paid training.

Uses the current shared SFT recipe. The two-epoch tutorial is a new run, not
an exact replay of the historical 100-step experiment displayed in the lesson.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def validate_dataset(path: Path) -> list[dict]:
    """Check reader-owned chat JSONL without importing SDKs or making requests."""
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {number}: invalid JSON ({exc.msg})") from exc
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(
                f"Line {number}: expected system, user, and assistant messages"
            )
        if not all(isinstance(m, dict) for m in messages):
            raise ValueError(f"Line {number}: each message must be an object")
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise ValueError(
                f"Line {number}: expected roles system, user, assistant in that order"
            )
        if any(
            not isinstance(m.get("content"), str) or not m["content"].strip()
            for m in messages
        ):
            raise ValueError(
                f"Line {number}: every message needs nonempty text content"
            )
        prompt = json.dumps(messages[:2], sort_keys=True)
        if prompt in seen:
            raise ValueError(
                f"Line {number}: repeated input; resolve duplicate targets before training"
            )
        seen.add(prompt)
        rows.append(row)
    if not rows:
        raise ValueError("Dataset has no training rows")
    return rows


def make_plan(
    dataset: Path, preset: str, output_model_id: str | None, run_dir: Path
) -> dict:
    rows = validate_dataset(dataset)
    count = min(8, len(rows)) if preset == "smoke" else len(rows)
    epochs = 1 if preset == "smoke" else 2
    return {
        "preset": preset,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "available_rows": len(rows),
        "training_rows": count,
        "epochs": epochs,
        "batch_size": 4,
        "estimated_steps": math.ceil(count / 4) * epochs,
        "base_model": "accounts/fireworks/models/glm-5p3-flash",
        "tokenizer_model": "zai-org/GLM-5.3-Flash",
        "lora_rank": 64,
        "lora_alpha": 128,
        "learning_rate": 1e-4,
        "seed": 29,
        "output_model_id": output_model_id,
        "run_dir": str(run_dir.resolve()),
        "purpose": (
            "mechanics only; no quality claim"
            if preset == "smoke"
            else "two-epoch tutorial; evaluate separately"
        ),
        "billing": "Paid API calls only with --execute. Check current token prices and account access first.",
    }


def execute(plan: dict) -> dict:
    if not os.environ.get("FIREWORKS_API_KEY"):
        raise ValueError(
            "Set FIREWORKS_API_KEY to your account-scoped key before --execute"
        )
    run_dir = Path(plan["run_dir"])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(
            "Run directory is not empty. Choose a new --run-dir; this lesson does not resume automatically."
        )
    # Keep paid dependencies out of the local inspection and validation path.
    from training.recipes import sft_loop
    from training.utils import RunnerConfig

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "lesson-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    cfg = sft_loop.Config(
        serverless=True,
        log_path=str(run_dir),
        dataset=plan["dataset"],
        base_model=plan["base_model"],
        tokenizer_model=plan["tokenizer_model"],
        lora_rank=plan["lora_rank"],
        lora_alpha=plan["lora_alpha"],
        learning_rate=plan["learning_rate"],
        seed=plan["seed"],
        epochs=plan["epochs"],
        batch_size=plan["batch_size"],
        max_examples=plan["training_rows"],
        max_seq_len=32768,
        dcp_save_interval=2 if plan["preset"] == "smoke" else 20,
        output_model_id=plan["output_model_id"],
        render_workers=1,
        runner=RunnerConfig(
            status_file=str(run_dir / "status.json"),
            metrics_file=str(run_dir / "metrics.jsonl"),
            output_model_path=str(run_dir / "output-model.json"),
        ),
    )
    result = sft_loop.main(cfg)
    (run_dir / "lesson-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=HERE / "pack/humanizer_train.jsonl"
    )
    parser.add_argument("--preset", choices=["smoke", "tutorial"], default="smoke")
    parser.add_argument(
        "--output-model-id",
        default=None,
        help="Optional new model ID; promotion does not deploy inference",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None, help="Default: results/lesson-<preset>"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Make paid calls; otherwise print a local plan only",
    )
    args = parser.parse_args(argv)
    try:
        plan = make_plan(
            args.dataset,
            args.preset,
            args.output_model_id,
            args.run_dir or HERE / "results" / f"lesson-{args.preset}",
        )
        print(json.dumps(plan, indent=2))
        if args.execute:
            print(json.dumps(execute(plan), indent=2))
        else:
            print(
                "LOCAL PLAN ONLY — no API calls or output files. Add --execute when ready."
            )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
