"""Copy Harbor tasks and install the TITO runtime used by Mini-SWE-Agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.examples.rl.harbor.tito.prepare_tasks import (
    build_python_sidecar_suffix,
    prepare_with_installer,
)

_MARKER = "# Added by fireworks TITO harbor.mini_swe.prepare_tasks"


def prepare(
    source: Path,
    destination: Path,
    *,
    base_image: str | None = None,
) -> list[Path]:
    return prepare_with_installer(
        source,
        destination,
        marker=_MARKER,
        suffix_builder=lambda restore_user: build_python_sidecar_suffix(
            marker=_MARKER,
            restore_user=restore_user,
        ),
        base_image=base_image,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--base-image",
        default=None,
        help="Optional immutable image@sha256 base for a single prepared task",
    )
    args = parser.parse_args()
    prepared = prepare(args.source, args.destination, base_image=args.base_image)
    print(
        f"prepared {len(prepared)} Harbor task image contexts in "
        f"{args.destination} with the TITO sidecar runtime"
    )


if __name__ == "__main__":
    main()
