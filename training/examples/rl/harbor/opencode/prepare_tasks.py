"""Copy Harbor tasks and bake a pinned OpenCode CLI into each image."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.examples.rl.harbor.tito.prepare_tasks import (
    build_node_22_harness_suffix,
    prepare_with_installer,
)

from .constants import DEFAULT_OPENCODE_VERSION

# Preserve both pre-refactor markers so prepared trees remain idempotent.
_MARKER = "# Added by harbor_rl_opencode.prepare_opencode_tasks"
_TRANSITIONAL_MARKER = "# Added by harbor.opencode.prepare_tasks"


def _suffix(version: str, restore_user: str | None) -> str:
    return build_node_22_harness_suffix(
        marker=_MARKER,
        package_install=f"npm install -g opencode-ai@{version}",
        version_check="opencode --version",
        restore_user=restore_user,
    )


def prepare(
    source: Path,
    destination: Path,
    version: str = DEFAULT_OPENCODE_VERSION,
    *,
    base_image: str | None = None,
) -> list[Path]:
    return prepare_with_installer(
        source,
        destination,
        marker=_MARKER,
        compatible_markers=(_TRANSITIONAL_MARKER,),
        suffix_builder=lambda restore_user: _suffix(version, restore_user),
        base_image=base_image,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--opencode-version", default=DEFAULT_OPENCODE_VERSION)
    parser.add_argument(
        "--base-image",
        default=None,
        help="Optional immutable image@sha256 base for a single prepared task",
    )
    args = parser.parse_args()
    prepared = prepare(
        args.source,
        args.destination,
        args.opencode_version,
        base_image=args.base_image,
    )
    print(
        f"prepared {len(prepared)} Harbor task image contexts in "
        f"{args.destination} with opencode-ai@{args.opencode_version}"
    )


if __name__ == "__main__":
    main()
