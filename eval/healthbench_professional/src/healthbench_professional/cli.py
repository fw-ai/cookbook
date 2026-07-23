"""Command-line entry points for the HealthBench Professional cookbook."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from . import adapter, aggregate, export_rl, run_job


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _prepare(args: argparse.Namespace) -> int:
    generator = adapter.HealthBenchProfessionalAdapter(
        args.output_dir,
        template_dir=args.template_dir,
    )
    generated = generator.generate_canonical_dataset(
        args.dataset_file,
        limit=args.limit,
        task_ids=args.task_ids,
        overwrite=args.overwrite,
    )
    print(
        f"Generated {len(generated)} HealthBench Professional tasks "
        f"in {args.output_dir}"
    )
    return 0


def _summarize(args: argparse.Namespace) -> int:
    summary = aggregate.summarize_job(args.job_dir)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(rendered, end="")
    return 0


def _export_rl(args: argparse.Namespace) -> int:
    summary = aggregate.summarize_job(args.job_dir)
    expected = summary["diagnostics"]["expected_trials"]
    trajectory_paths = export_rl.discover_trajectory_paths([args.job_dir])
    if len(trajectory_paths) != expected:
        raise ValueError(
            f"expected {expected} trajectories in {args.job_dir}, "
            f"found {len(trajectory_paths)}"
        )
    records = [export_rl.export_rl_record(path) for path in trajectory_paths]
    output_path = export_rl.write_rl_jsonl(records, args.output)
    print(f"Wrote {len(records)} validated RL records to {output_path}")
    return 0


def _job_kwargs(args: argparse.Namespace) -> dict:
    return {
        "model": args.model,
        "dataset_dir": args.dataset_dir,
        "jobs_dir": args.jobs_dir,
        "job_name": args.job_name or run_job.default_job_name(),
        "concurrency": args.concurrency,
        "limit": args.limit,
        "quiet": args.quiet,
        "template_path": args.job_template,
    }


def _resolve_job(args: argparse.Namespace) -> int:
    output = run_job.write_resolved_job(
        run_job.resolve_job_config(**_job_kwargs(args)), args.output
    )
    print(f"Wrote validated Harbor job config to {output}")
    return 0


def _run_job(args: argparse.Namespace) -> int:
    return run_job.resolve_and_run(**_job_kwargs(args))


def _add_job_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--model", default=run_job.DEFAULT_MODEL)
    command.add_argument("--dataset-dir", type=Path, default=adapter.DEFAULT_OUTPUT_DIR)
    command.add_argument("--jobs-dir", type=Path, default=Path("jobs"))
    command.add_argument("--job-name")
    command.add_argument(
        "--concurrency", type=_positive_int, default=run_job.DEFAULT_CONCURRENCY
    )
    command.add_argument("--limit", type=_positive_int)
    command.add_argument("--quiet", action="store_true")
    command.add_argument("--job-template", type=Path, default=run_job.DEFAULT_JOB_PATH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="prepare the pinned 525-case dataset")
    prepare.add_argument("--output-dir", type=Path, default=adapter.DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--dataset-file", type=Path)
    prepare.add_argument(
        "--template-dir", type=Path, default=adapter.DEFAULT_TEMPLATE_DIR
    )
    prepare.add_argument("--limit", type=_positive_int)
    prepare.add_argument("--task-id", action="append", dest="task_ids")
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(handler=_prepare)

    summarize = commands.add_parser(
        "summarize", help="summarize one completed Harbor job"
    )
    summarize.add_argument("job_dir", type=Path)
    summarize.add_argument("-o", "--output", type=Path)
    summarize.set_defaults(handler=_summarize)

    rl = commands.add_parser(
        "export-rl",
        help="export exact token-in/token-out trajectories and rewards",
    )
    rl.add_argument("job_dir", type=Path)
    rl.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("rl-output/healthbench-professional.jsonl"),
    )
    rl.set_defaults(handler=_export_rl)

    resolve = commands.add_parser(
        "resolve-job",
        help="write a validated Harbor config with safe model and limit overrides",
    )
    _add_job_arguments(resolve)
    resolve.add_argument("-o", "--output", type=Path, required=True)
    resolve.set_defaults(handler=_resolve_job)

    run = commands.add_parser(
        "run", help="resolve and run the pinned Harbor job without lossy CLI overrides"
    )
    _add_job_arguments(run)
    run.set_defaults(handler=_run_job)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
