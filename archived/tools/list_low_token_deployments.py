#!/usr/bin/env python3
"""List deployments whose daily token usage stayed below a threshold.

Uses the customer-facing firectl CLI (deployment-metrics + deployment list).
Per-deployment token counts come from Prometheus-backed deployment metrics,
not billing usage (dedicated deployment billing reports GPU-seconds only).

Download (no clone required):
    curl -fsSL -o list_low_token_deployments.py \\
      https://raw.githubusercontent.com/fw-ai/cookbook/main/archived/tools/list_low_token_deployments.py
    chmod +x list_low_token_deployments.py

Examples:
    export FIREWORKS_API_KEY=...
    python list_low_token_deployments.py --account my-account

    python list_low_token_deployments.py \\
      --account my-account \\
      --days 7 \\
      --threshold 10 \\
      --mode every-day \\
      --output json

    # Equivalent firectl calls (prompt + generated tokens, grouped by deployment):
    firectl deployment-metrics list -a my-account \\
      --metric tokens-prompt-per-second --group-by deployment \\
      --start 2026-07-02 --end 2026-07-09 --interval 24h -o json
    firectl deployment-metrics list -a my-account \\
      --metric tokens-per-second --group-by deployment \\
      --start 2026-07-02 --end 2026-07-09 --interval 24h -o json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class DeploymentInfo:
    deployment_id: str
    name: str
    base_model: str
    state: str
    replica_count: int


@dataclass(frozen=True)
class DeploymentUsageSummary:
    deployment_id: str
    name: str
    base_model: str
    state: str
    replica_count: int
    max_daily_tokens: float
    avg_daily_tokens: float
    total_tokens: float
    daily_tokens: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List deployments with low daily token usage over a recent window. "
            "Requires firectl and FIREWORKS_API_KEY (or --api-key)."
        ),
    )
    parser.add_argument("--account", required=True, help="Account ID, e.g. my-account.")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Lookback window in days. Default: 7.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Token threshold per day. Default: 10.",
    )
    parser.add_argument(
        "--mode",
        choices=("max-daily", "every-day", "avg-daily"),
        default="max-daily",
        help=(
            "How to compare against --threshold. "
            "max-daily: peak day in the window is below threshold (default). "
            "every-day: every day in the window is below threshold. "
            "avg-daily: average daily usage is below threshold."
        ),
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include deleted deployments from deployment list.",
    )
    parser.add_argument(
        "--min-replicas",
        type=int,
        default=0,
        help="Only consider deployments with at least this many replicas. Default: 0.",
    )
    parser.add_argument(
        "--firectl",
        default="firectl",
        help="Path to the firectl binary. Default: firectl.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("FIREWORKS_API_KEY", ""),
        help="Fireworks API key. Defaults to $FIREWORKS_API_KEY.",
    )
    parser.add_argument(
        "--server",
        default="",
        help='Optional gateway host, e.g. "gateway-dev.fireworks.ai:443".',
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format. Default: text.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the underlying firectl commands without executing them.",
    )
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def start_of_utc_day(timestamp: datetime) -> datetime:
    normalized = timestamp.astimezone(timezone.utc)
    return normalized.replace(hour=0, minute=0, second=0, microsecond=0)


def build_metrics_window(*, days: int, now: datetime | None = None) -> tuple[datetime, datetime, list[str]]:
    """Return UTC-aligned daily buckets covering the last `days` calendar days.

    Metrics `--end` is date-only and parsed as UTC midnight, so we pass
    tomorrow's date as the exclusive end to include all of today.
    """
    current_time = now or utc_now()
    metrics_end = start_of_utc_day(current_time) + timedelta(days=1)
    metrics_start = metrics_end - timedelta(days=days)
    expected_days = [(metrics_start + timedelta(days=offset)).date().isoformat() for offset in range(days)]
    return metrics_start, metrics_end, expected_days


def format_firectl_date(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d")


def deployment_id_from_label(*, label: str, account_id: str) -> str:
    prefix = f"accounts/{account_id}/deployments/"
    if label.startswith(prefix):
        return label[len(prefix) :]
    if label.startswith("accounts/") and "/deployments/" in label:
        return label.rsplit("/", 1)[-1]
    return label


def build_firectl_base_cmd(*, args: argparse.Namespace) -> list[str]:
    cmd = [args.firectl]
    if args.server:
        cmd.extend(["-s", args.server])
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    cmd.extend(["-a", args.account])
    return cmd


def format_dry_run_cmd(cmd: list[str]) -> str:
    """Render a firectl command for logging without exposing secrets."""
    redacted: list[str] = []
    skip_next = False
    for arg in cmd:
        if skip_next:
            redacted.append("<redacted>")
            skip_next = False
            continue
        if arg == "--api-key":
            redacted.append(arg)
            skip_next = True
            continue
        redacted.append(arg)
    return " ".join(redacted)


def run_firectl_json(*, args: argparse.Namespace, firectl_args: list[str]) -> Any:
    cmd = build_firectl_base_cmd(args=args) + firectl_args
    if args.dry_run:
        print(format_dry_run_cmd(cmd), file=sys.stderr)
        return None

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"{args.firectl!r} not found on PATH. Install firectl or pass --firectl.") from error
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError(f"firectl failed: {details}") from error

    stdout = completed.stdout.strip()
    if not stdout:
        return None
    return json.loads(stdout)


def list_deployments(*, args: argparse.Namespace) -> list[DeploymentInfo]:
    firectl_args = ["deployment", "list", "-o", "json", "--no-paginate"]
    if args.include_deleted:
        firectl_args.append("--show-deleted")

    payload = run_firectl_json(args=args, firectl_args=firectl_args)
    if payload is None:
        return []

    rows = payload if isinstance(payload, list) else payload.get("deployments", [])
    deployments: list[DeploymentInfo] = []
    for row in rows:
        name = str(row.get("name", ""))
        deployment_id = deployment_id_from_label(label=name, account_id=args.account)
        deployments.append(
            DeploymentInfo(
                deployment_id=deployment_id,
                name=name,
                base_model=str(row.get("base_model", "")),
                state=str(row.get("state", "")),
                replica_count=int(row.get("replica_count", 0) or 0),
            )
        )
    return deployments


def fetch_daily_token_rates(
    *,
    args: argparse.Namespace,
    metric: str,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, dict[str, float]]:
    """Return deployment_id -> {YYYY-MM-DD: tokens_per_day}."""
    payload = run_firectl_json(
        args=args,
        firectl_args=[
            "deployment-metrics",
            "list",
            "--metric",
            metric,
            "--group-by",
            "deployment",
            "--start",
            format_firectl_date(start_time),
            "--end",
            format_firectl_date(end_time),
            "--interval",
            "24h",
            "-o",
            "json",
        ],
    )
    if payload is None:
        return {}

    daily_by_deployment: dict[str, dict[str, float]] = defaultdict(dict)
    for series in payload.get("series", []):
        labels = series.get("labels", {})
        deployment_label = str(labels.get("deployment", ""))
        if not deployment_label:
            continue
        deployment_id = deployment_id_from_label(
            label=deployment_label,
            account_id=args.account,
        )
        for point in series.get("values", []):
            timestamp = int(point.get("timestamp", 0))
            if timestamp <= 0:
                continue
            day = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
            rate = float(point.get("value", "0") or 0)
            daily_tokens = rate * SECONDS_PER_DAY
            daily_by_deployment[deployment_id][day] = daily_by_deployment[deployment_id].get(day, 0.0) + daily_tokens
    return daily_by_deployment


def merge_prompt_and_generated(
    *,
    prompt_daily: dict[str, dict[str, float]],
    generated_daily: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = defaultdict(dict)
    deployment_ids = set(prompt_daily) | set(generated_daily)
    for deployment_id in deployment_ids:
        days = set(prompt_daily.get(deployment_id, {})) | set(generated_daily.get(deployment_id, {}))
        for day in days:
            merged[deployment_id][day] = prompt_daily.get(deployment_id, {}).get(day, 0.0) + generated_daily.get(
                deployment_id, {}
            ).get(day, 0.0)
    return merged


def summarize_usage(
    *,
    deployment: DeploymentInfo,
    daily_tokens: dict[str, float],
    expected_days: list[str],
) -> DeploymentUsageSummary:
    normalized_daily = {day: daily_tokens.get(day, 0.0) for day in expected_days}
    values = list(normalized_daily.values())
    total = sum(values)
    max_daily = max(values) if values else 0.0
    avg_daily = total / len(expected_days) if expected_days else 0.0
    return DeploymentUsageSummary(
        deployment_id=deployment.deployment_id,
        name=deployment.name,
        base_model=deployment.base_model,
        state=deployment.state,
        replica_count=deployment.replica_count,
        max_daily_tokens=max_daily,
        avg_daily_tokens=avg_daily,
        total_tokens=total,
        daily_tokens=normalized_daily,
    )


def matches_threshold(
    *,
    summary: DeploymentUsageSummary,
    threshold: float,
    mode: str,
    expected_days: list[str],
) -> bool:
    if mode == "max-daily":
        return summary.max_daily_tokens < threshold
    if mode == "avg-daily":
        return summary.avg_daily_tokens < threshold
    return all(summary.daily_tokens.get(day, 0.0) < threshold for day in expected_days)


def print_text_table(*, rows: list[DeploymentUsageSummary], threshold: float, mode: str) -> None:
    header = [
        "DEPLOYMENT",
        "STATE",
        "REPLICAS",
        "BASE_MODEL",
        "MAX_DAILY_TOKENS",
        "AVG_DAILY_TOKENS",
        "TOTAL_TOKENS",
    ]
    print("\t".join(header))
    for row in rows:
        print(
            "\t".join(
                [
                    row.deployment_id,
                    row.state,
                    str(row.replica_count),
                    row.base_model,
                    f"{row.max_daily_tokens:.2f}",
                    f"{row.avg_daily_tokens:.2f}",
                    f"{row.total_tokens:.2f}",
                ]
            )
        )
    print(
        f"\n{len(rows)} deployment(s) with {mode} usage below {threshold:g} tokens/day.",
        file=sys.stderr,
    )


def main() -> int:
    args = parse_args()
    if not args.api_key and not args.dry_run:
        print(
            "error: set FIREWORKS_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 1
    if args.days <= 0:
        print("error: --days must be positive.", file=sys.stderr)
        return 1

    end_time = utc_now()
    start_time, metrics_end, expected_days = build_metrics_window(days=args.days, now=end_time)

    deployments = list_deployments(args=args)
    if args.min_replicas > 0:
        deployments = [d for d in deployments if d.replica_count >= args.min_replicas]

    prompt_daily = fetch_daily_token_rates(
        args=args,
        metric="tokens-prompt-per-second",
        start_time=start_time,
        end_time=metrics_end,
    )
    generated_daily = fetch_daily_token_rates(
        args=args,
        metric="tokens-per-second",
        start_time=start_time,
        end_time=metrics_end,
    )
    if args.dry_run:
        return 0

    merged_daily = merge_prompt_and_generated(
        prompt_daily=prompt_daily,
        generated_daily=generated_daily,
    )

    low_usage: list[DeploymentUsageSummary] = []
    for deployment in deployments:
        summary = summarize_usage(
            deployment=deployment,
            daily_tokens=merged_daily.get(deployment.deployment_id, {}),
            expected_days=expected_days,
        )
        if matches_threshold(
            summary=summary,
            threshold=args.threshold,
            mode=args.mode,
            expected_days=expected_days,
        ):
            low_usage.append(summary)

    low_usage.sort(key=lambda row: (row.max_daily_tokens, row.deployment_id))

    result = {
        "account": args.account,
        "start_time": start_time.isoformat(),
        "end_time": metrics_end.isoformat(),
        "queried_at": end_time.isoformat(),
        "expected_days": expected_days,
        "days": args.days,
        "threshold": args.threshold,
        "mode": args.mode,
        "deployment_count": len(deployments),
        "low_usage_count": len(low_usage),
        "deployments": [asdict(row) for row in low_usage],
    }

    if args.output == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print_text_table(rows=low_usage, threshold=args.threshold, mode=args.mode)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
