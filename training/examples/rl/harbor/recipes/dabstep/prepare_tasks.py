"""Prepare DABstep for OpenCode with sign-sensitive numeric scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.examples.rl.harbor.opencode import prepare_tasks as prepare_opencode_tasks
from training.examples.rl.harbor.opencode.constants import DEFAULT_OPENCODE_VERSION
from training.examples.rl.harbor.recipes.dabstep.manifest import (
    make_numeric_scorer_sign_sensitive,
)


def prepare(
    source: Path,
    destination: Path,
    version: str,
) -> list[Path]:
    prepared = prepare_opencode_tasks.prepare(
        source,
        destination,
        version,
    )
    for task in prepared:
        make_numeric_scorer_sign_sensitive(task)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--opencode-version", default=DEFAULT_OPENCODE_VERSION)
    args = parser.parse_args()
    prepared = prepare(
        args.source,
        args.destination,
        args.opencode_version,
    )
    print(
        f"prepared {len(prepared)} sign-sensitive DABstep tasks in "
        f"{args.destination} with opencode-ai@{args.opencode_version}"
    )


if __name__ == "__main__":
    main()
