"""Opt-in evaluation with a fresh evidence directory for every invocation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent


def run_evaluation(
    adapter: str,
    run_dir: Path,
    *,
    split: str = "dev",
    max_inputs: int = 5,
    idempotence: bool = False,
) -> Path:
    if not adapter.startswith("accounts/"):
        raise ValueError(
            "Supply your serving-ready accounts/.../models/... inference name."
        )
    if split not in {"dev", "test"} or max_inputs < 0:
        raise ValueError(
            "Use split dev or test and a nonnegative max_inputs (0 means all)."
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(run_dir) / f"eval-{split}-{stamp}-{uuid4().hex[:8]}"
    # Exclusive creation protects existing evidence even if a generated name collides.
    output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(HERE / "evaluate_pack.py"),
        "--split",
        split,
        "--max-inputs",
        str(max_inputs),
        "--arm",
        "base_short=accounts/fireworks/models/glm-5p3-flash",
        "--arm",
        f"adapter={adapter}",
        "--out",
        str(output),
    ]
    if not idempotence:
        command.append("--no-idem")
    print(f"Evaluation evidence: {output}", flush=True)
    subprocess.run(command, check=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--max-inputs", type=int, default=5)
    parser.add_argument("--idempotence", action="store_true")
    parser.add_argument(
        "--execute", action="store_true", help="Opt in to paid inference calls"
    )
    args = parser.parse_args()
    if not args.execute:
        print(
            "Evaluation is off. Add --execute when your model is serving-ready; this makes paid inference calls."
        )
        return
    run_evaluation(
        args.adapter,
        args.run_dir,
        split=args.split,
        max_inputs=args.max_inputs,
        idempotence=args.idempotence,
    )


if __name__ == "__main__":
    main()
