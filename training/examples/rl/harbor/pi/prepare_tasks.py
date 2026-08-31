"""Copy Harbor task contexts and bake the pinned Pi CLI into each image."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from training.examples.rl.harbor.tito.prepare_tasks import (
    build_node_22_harness_suffix,
    prepare_with_installer,
)

from .constants import PINNED_PI_VERSION

_MARKER = "# Added by fireworks TITO harbor.pi.prepare_tasks"


def _suffix(version: str, restore_user: str | None) -> str:
    return build_node_22_harness_suffix(
        marker=_MARKER,
        package_install=(
            f"npm install -g --ignore-scripts @earendil-works/pi-coding-agent@{version}"
        ),
        version_check="pi --version",
        restore_user=restore_user,
    )


def prepare(
    source: Path,
    destination: Path,
    version: str = PINNED_PI_VERSION,
    *,
    base_image: str | None = None,
    task_names: Iterable[str] | None = None,
) -> list[Path]:
    return prepare_with_installer(
        source,
        destination,
        marker=_MARKER,
        suffix_builder=lambda restore_user: _suffix(version, restore_user),
        base_image=base_image,
        task_names=task_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--pi-version", default=PINNED_PI_VERSION)
    parser.add_argument(
        "--base-image",
        default=None,
        help="Optional immutable image@sha256 base for a single prepared task",
    )
    args = parser.parse_args()
    prepared = prepare(
        args.source,
        args.destination,
        args.pi_version,
        base_image=args.base_image,
    )
    print(
        f"prepared {len(prepared)} Harbor task image contexts in "
        f"{args.destination} with Pi {args.pi_version}"
    )


if __name__ == "__main__":
    main()
