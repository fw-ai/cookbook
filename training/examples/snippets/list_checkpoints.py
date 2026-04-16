#!/usr/bin/env python3
# ruff: noqa: E402
"""List checkpoints on an RLOR trainer job.

Calls the Fireworks control-plane list endpoint for a trainer's checkpoints.
Unlike ``checkpoints.jsonl`` (which only covers rows the cookbook recipe
wrote), this returns the authoritative list the server knows about —
including checkpoints inherited from a predecessor trainer via hotload.
No active trainer client is required; the job can be in any state.

Each row shows ``name``, ``createTime``, ``checkpointType``, and whether
the checkpoint is promotable. Pick the **latest ``createTime`` with
``promotable: true``** — step numbers mislead when a trainer inherits from
a predecessor.

Usage:
    export FIREWORKS_API_KEY=...

    # List all checkpoints on a trainer:
    python list_checkpoints.py --job-id <job-id>

    # Only show promotable ones, newest first:
    python list_checkpoints.py --job-id <job-id> --promotable-only

    # JSON output for piping into other tools:
    python list_checkpoints.py --job-id <job-id> --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from urllib.parse import urlencode

from dotenv import load_dotenv

_COOKBOOK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _COOKBOOK_ROOT not in sys.path:
    sys.path.insert(0, _COOKBOOK_ROOT)

from fireworks.training.sdk import FireworksClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List checkpoints for an RLOR trainer job.",
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="RLOR trainer job ID (the short ID, not the full resource name).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Max rows to fetch (default: 200).",
    )
    parser.add_argument(
        "--promotable-only",
        action="store_true",
        help="Filter to rows with promotable=true.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit raw JSON to stdout instead of a human table.",
    )
    return parser.parse_args()


def _fetch_checkpoints(
    client: FireworksClient,
    job_id: str,
    page_size: int,
) -> list[dict]:
    path = (
        f"/v1/accounts/{client.account_id}/rlorTrainerJobs/{job_id}"
        f"/checkpoints?{urlencode({'pageSize': page_size})}"
    )
    resp = client._get(path, timeout=30)  # noqa: SLF001
    if not resp.is_success:
        raise SystemExit(
            f"ERROR: list checkpoints failed (HTTP {resp.status_code}): {resp.text}"
        )
    body = resp.json()
    return body.get("checkpoints", []) or body.get("rlorTrainerJobCheckpoints", []) or []


def _sort_newest_first(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: r.get("createTime", ""), reverse=True)


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no checkpoints)")
        return
    header = f"{'name':<40}  {'createTime':<25}  {'type':<22}  promotable"
    print(header)
    print("-" * len(header))
    for r in rows:
        name = r.get("name", "").rsplit("/", 1)[-1]
        created = r.get("createTime", "")
        ckpt_type = r.get("checkpointType", "")
        promo = r.get("promotable", False)
        print(f"{name:<40}  {created:<25}  {ckpt_type:<22}  {promo}")


def main() -> None:
    args = parse_args()

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    client = FireworksClient(api_key=api_key, base_url=base_url)

    rows = _fetch_checkpoints(client, args.job_id, args.page_size)
    if args.promotable_only:
        rows = [r for r in rows if r.get("promotable")]
    rows = _sort_newest_first(rows)

    if args.as_json:
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_table(rows)


if __name__ == "__main__":
    main()
