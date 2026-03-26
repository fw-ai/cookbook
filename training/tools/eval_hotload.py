#!/usr/bin/env python3
"""Hotload training snapshots to an eval deployment, decoupled from the RL loop.

Polls a snapshot file written by the RL training loop and hotloads new
snapshots to a separate eval deployment.  Runs independently of training
so eval failures cannot affect the training loop.

The RL loop writes the latest snapshot identity to ``<log_path>/latest_snapshot.txt``
after each weight sync.  This script watches that file and hotloads to the
eval deployment whenever it changes.

Usage:
    export FIREWORKS_API_KEY=...

    # Basic: just hotload to eval deployment
    python -m tools.eval_hotload \
        --snapshot-file ./rl_logs/latest_snapshot.txt \
        --deployment-id my-eval-deployment \
        --base-model accounts/fireworks/models/qwen3-8b

    # With polling interval
    python -m tools.eval_hotload \
        --snapshot-file ./rl_logs/latest_snapshot.txt \
        --deployment-id my-eval-deployment \
        --base-model accounts/fireworks/models/qwen3-8b \
        --poll-interval 60
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    name = signal.Signals(signum).name
    logger.info("Received %s, shutting down...", name)
    _shutdown = True


def read_snapshot(path: str) -> str | None:
    """Read the latest snapshot identity from the file."""
    try:
        with open(path) as f:
            content = f.read().strip()
        return content if content else None
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Hotload training snapshots to an eval deployment")
    parser.add_argument("--snapshot-file", required=True, help="Path to latest_snapshot.txt written by the RL loop")
    parser.add_argument("--deployment-id", required=True, help="Eval deployment ID to hotload to")
    parser.add_argument("--base-model", required=True, help="Base model name (e.g. accounts/fireworks/models/qwen3-8b)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between polls (default 30)")
    parser.add_argument("--hotload-timeout", type=int, default=600, help="Hotload timeout in seconds (default 600)")
    parser.add_argument("--base-url", default=None, help="Fireworks API base URL (default: FIREWORKS_BASE_URL or https://api.fireworks.ai)")
    args = parser.parse_args()

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        logger.error("FIREWORKS_API_KEY environment variable is required")
        sys.exit(1)

    base_url = args.base_url or os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    # Import here to avoid import cost when showing --help
    from fireworks.training.sdk.deployment import DeploymentManager

    deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Track delta chain for incremental hotloads
    last_snapshot: str | None = None
    base_identity: str | None = None

    logger.info(
        "Watching %s for snapshots (deployment=%s, poll=%ds)",
        args.snapshot_file, args.deployment_id, args.poll_interval,
    )

    while not _shutdown:
        current = read_snapshot(args.snapshot_file)

        if current and current != last_snapshot:
            logger.info("New snapshot: %s", current)
            t0 = time.time()

            # Determine if we can do incremental hotload
            incremental = None
            if base_identity and base_identity != current:
                incremental = {
                    "previous_snapshot_identity": base_identity,
                    "compression_format": "arc_v2",
                    "checksum_format": "alder32",
                }
                logger.info("Incremental hotload from %s", base_identity)
            else:
                logger.info("Full hotload (no previous snapshot)")

            try:
                ok = deploy_mgr.hotload_and_wait(
                    deployment_id=args.deployment_id,
                    base_model=args.base_model,
                    snapshot_identity=current,
                    incremental_snapshot_metadata=incremental,
                    timeout_seconds=args.hotload_timeout,
                )
                if ok:
                    base_identity = current
                    last_snapshot = current
                    logger.info("Hotload complete: %s (%.1fs)", current, time.time() - t0)
                else:
                    logger.warning("Hotload failed for %s", current)
            except Exception as e:
                logger.error("Hotload error for %s: %s", current, e)

        # Sleep in small increments so we respond to signals quickly
        for _ in range(args.poll_interval):
            if _shutdown:
                break
            time.sleep(1)

    logger.info("Eval hotload watcher stopped")


if __name__ == "__main__":
    main()
