from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_TARGETS = [
    "training/recipes/sft_loop.py",
    "training/recipes/dpo_loop.py",
    "training/recipes/orpo_loop.py",
    "training/recipes/rl_loop.py",
    "training/examples/rl/frozen_lake/train_frozen_lake.py",
    "training/examples/rl/deepmath/train_deepmath.py",
    "training/examples/sft/train_sft.py",
    "training/examples/rl/deepmath/prepare_data.py",
]


def _lookup_file(files: dict, target: str):
    if target in files:
        return target, files[target]
    relative = target.removeprefix("training/")
    if relative in files:
        return relative, files[relative]
    return None, None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize training-script coverage from coverage JSON.")
    parser.add_argument("coverage_json", help="Path to the coverage JSON report.")
    parser.add_argument("--fail-under", type=float, default=None, help="Fail if overall scoped coverage is below this percentage.")
    parser.add_argument(
        "--per-file-fail-under",
        type=float,
        default=None,
        help="Fail if any scoped file with coverage data is below this percentage.",
    )
    parser.add_argument("--fail-on-no-data", action="store_true", help="Fail if any scoped file is missing from coverage JSON.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data = json.loads(Path(args.coverage_json).read_text())

    files = data.get("files", {})
    rows = []
    total_statements = 0
    total_covered = 0
    missing_targets = []

    for target in SCRIPT_TARGETS:
        resolved_path, meta = _lookup_file(files, target)
        if meta is None:
            missing_targets.append(target)
            rows.append((target, 0.0, 0, 0, "NO_DATA"))
            continue

        summary = meta["summary"]
        covered = int(summary["covered_lines"])
        statements = int(summary["num_statements"])
        percent = float(summary["percent_covered"])
        total_covered += covered
        total_statements += statements
        rows.append((resolved_path or target, percent, covered, statements, ""))

    overall = 100.0 * total_covered / total_statements if total_statements else 0.0

    print("Training script coverage summary")
    print()
    print(f"{'Coverage':>8}  {'Lines':>11}  File")
    print(f"{'-' * 8}  {'-' * 11}  {'-' * 4}")
    for target, percent, covered, statements, note in rows:
        line = f"{percent:7.1f}%  {covered:4}/{statements:<6}  {target}"
        if note:
            line += f"  [{note}]"
        print(line)

    print()
    print(f"Overall scoped coverage: {overall:.1f}% ({total_covered}/{total_statements})")
    if missing_targets:
        print("Files with no coverage data:")
        for target in missing_targets:
            print(f"- {target}")

    low_coverage_targets: list[tuple[str, float]] = []
    if args.per_file_fail_under is not None:
        for target, percent, _covered, _statements, note in rows:
            if note == "NO_DATA":
                continue
            if percent < args.per_file_fail_under:
                low_coverage_targets.append((target, percent))
        if low_coverage_targets:
            print(
                f"Files below per-file coverage threshold ({args.per_file_fail_under:.1f}%):"
            )
            for target, percent in low_coverage_targets:
                print(f"- {target}: {percent:.1f}%")

    if args.fail_on_no_data and missing_targets:
        return 1
    if args.fail_under is not None and overall < args.fail_under:
        print(
            f"Coverage gate failed: overall scoped coverage {overall:.1f}% "
            f"is below {args.fail_under:.1f}%",
            file=sys.stderr,
        )
        return 1
    if low_coverage_targets:
        print(
            f"Coverage gate failed: one or more scoped files are below "
            f"{args.per_file_fail_under:.1f}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
