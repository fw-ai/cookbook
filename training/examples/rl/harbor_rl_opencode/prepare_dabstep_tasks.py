"""Prepare DABstep for OpenCode with sign-sensitive numeric scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.examples.rl.harbor import prepare_opencode_tasks
from training.examples.rl.harbor.trial import DEFAULT_OPENCODE_VERSION
from training.examples.rl.harbor_rl_opencode.dabstep import (
    make_numeric_scorer_sign_sensitive,
)


def prepare(
    source: Path,
    destination: Path,
    version: str,
    *,
    internal_network: str | None = None,
) -> list[Path]:
    prepared = prepare_opencode_tasks.prepare(
        source,
        destination,
        version,
        internal_network=internal_network,
    )
    for task in prepared:
        make_numeric_scorer_sign_sensitive(task)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--opencode-version", default=DEFAULT_OPENCODE_VERSION)
    parser.add_argument(
        "--internal-network",
        default=None,
        help="Optional Docker --internal network created by the generic preparer",
    )
    args = parser.parse_args()
    if args.internal_network:
        prepare_opencode_tasks.ensure_internal_network(args.internal_network)
    prepared = prepare(
        args.source,
        args.destination,
        args.opencode_version,
        internal_network=args.internal_network,
    )
    print(
        f"prepared {len(prepared)} sign-sensitive DABstep tasks in "
        f"{args.destination} with opencode-ai@{args.opencode_version}"
    )


if __name__ == "__main__":
    main()
