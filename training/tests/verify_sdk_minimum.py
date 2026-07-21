#!/usr/bin/env python3
"""Resolve and verify the cookbook's declared minimum Fireworks SDK."""

from __future__ import annotations

import argparse
import re
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SDK_REQUIREMENT_RE = re.compile(
    r"^fireworks-ai(?:\[[^]]+\])?(?P<specifiers>[^;]*)(?:;.*)?$"
)
MINIMUM_RE = re.compile(r"(?:^|,)\s*>=\s*(?P<version>[^,\s]+)\s*(?=,|$)")


def declared_sdk_minimum(pyproject: Path) -> str:
    with pyproject.open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]

    sdk_requirements = []
    for dependency in dependencies:
        match = SDK_REQUIREMENT_RE.fullmatch(dependency.strip())
        if match:
            sdk_requirements.append(match.group("specifiers"))

    if len(sdk_requirements) != 1:
        raise ValueError(
            "project.dependencies must contain exactly one fireworks-ai requirement"
        )

    minimums = [
        match.group("version") for match in MINIMUM_RE.finditer(sdk_requirements[0])
    ]
    if len(minimums) != 1:
        raise ValueError(
            "the fireworks-ai requirement must contain exactly one explicit >= minimum"
        )
    return minimums[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--assert-installed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    minimum = declared_sdk_minimum(args.pyproject)
    if not args.assert_installed:
        print(minimum)
        return 0

    try:
        installed = version("fireworks-ai")
    except PackageNotFoundError:
        print("fireworks-ai is not installed")
        return 1
    if installed != minimum:
        print(
            f"installed fireworks-ai {installed} does not equal declared minimum {minimum}"
        )
        return 1
    print(f"verified declared minimum fireworks-ai=={minimum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
