#!/usr/bin/env python3
"""Helpers for validating cookbook release tags inside GitHub Actions."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tomllib


CANDIDATE_TAG_RE = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)-alpha\.(?P<iteration>\d+)$")
FINAL_TAG_RE = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)$")


class ReleaseTagError(ValueError):
    """Raised when a tag does not match the cookbook release contract."""


def parse_tag(tag: str, kind: str) -> dict[str, str]:
    if kind == "candidate":
        match = CANDIDATE_TAG_RE.fullmatch(tag)
        if not match:
            raise ReleaseTagError(
                f"Candidate tag {tag!r} must match vX.Y.Z-alpha.N."
            )
    elif kind == "final":
        match = FINAL_TAG_RE.fullmatch(tag)
        if not match:
            raise ReleaseTagError(f"Final tag {tag!r} must match vX.Y.Z.")
    else:
        raise ReleaseTagError(f"Unsupported tag kind: {kind}")

    version = match.group("version")
    metadata = {
        "tag": tag,
        "kind": kind,
        "version": version,
        "stable_tag": f"v{version}",
        "artifact_name": f"cookbook-release-{tag}",
    }

    if kind == "candidate":
        metadata["candidate_iteration"] = match.group("iteration")

    return metadata


def write_github_output(output_file: pathlib.Path, metadata: dict[str, str]) -> None:
    with output_file.open("a", encoding="utf-8") as handle:
        for key, value in metadata.items():
            handle.write(f"{key}={value}\n")


def load_pyproject_version(pyproject_path: pathlib.Path) -> str:
    with pyproject_path.open("rb") as handle:
        document = tomllib.load(handle)

    try:
        return document["project"]["version"]
    except KeyError as exc:
        raise ReleaseTagError(
            f"Unable to locate [project].version in {pyproject_path}."
        ) from exc


def assert_pyproject_version(tag: str, kind: str, pyproject_path: pathlib.Path) -> None:
    metadata = parse_tag(tag, kind)
    actual = load_pyproject_version(pyproject_path)
    expected = metadata["version"]
    if actual != expected:
        raise ReleaseTagError(
            f"{pyproject_path} version is {actual}, but {tag} requires {expected}."
        )


def candidate_sort_key(tag: str) -> tuple[int, str]:
    match = CANDIDATE_TAG_RE.fullmatch(tag)
    if not match:
        raise ReleaseTagError(f"Tag {tag!r} is not a valid candidate tag.")
    return (int(match.group("iteration")), tag)


def list_candidate_tags(version: str, sha: str) -> list[str]:
    result = subprocess.run(
        ["git", "tag", "--points-at", sha, "--list", f"v{version}-alpha.*"],
        check=True,
        capture_output=True,
        text=True,
    )
    tags = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return sorted(tags, key=candidate_sort_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    github_output = subparsers.add_parser(
        "github-output", help="Write parsed tag metadata to a GitHub Actions output file."
    )
    github_output.add_argument("--tag", required=True)
    github_output.add_argument("--kind", choices=("candidate", "final"), required=True)
    github_output.add_argument("--output-file", required=True, type=pathlib.Path)

    version_check = subparsers.add_parser(
        "assert-pyproject-version",
        help="Ensure training/pyproject.toml matches the version encoded in the tag.",
    )
    version_check.add_argument("--tag", required=True)
    version_check.add_argument("--kind", choices=("candidate", "final"), required=True)
    version_check.add_argument("--pyproject", required=True, type=pathlib.Path)

    candidates = subparsers.add_parser(
        "list-candidate-tags",
        help="List candidate tags for a version that point at one exact SHA.",
    )
    candidates.add_argument("--version", required=True)
    candidates.add_argument("--sha", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "github-output":
            metadata = parse_tag(args.tag, args.kind)
            write_github_output(args.output_file, metadata)
            return 0

        if args.command == "assert-pyproject-version":
            assert_pyproject_version(args.tag, args.kind, args.pyproject)
            return 0

        if args.command == "list-candidate-tags":
            for tag in list_candidate_tags(args.version, args.sha):
                print(tag)
            return 0
    except ReleaseTagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
