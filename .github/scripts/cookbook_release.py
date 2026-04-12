#!/usr/bin/env python3
"""Helpers for validating cookbook release tags inside GitHub Actions."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tomllib


CANDIDATE_TAG_RE = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)-alpha\.(?P<iteration>\d+)$")
FINAL_TAG_RE = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)$")
RELEASE_METADATA_RE = re.compile(
    r"<!--\s*cookbook-release-metadata\s*\n(?P<body>.*?)\n-->",
    re.DOTALL,
)
METADATA_LINE_RE = re.compile(r"^\s*(?P<key>[a-z0-9_-]+)\s*:\s*(?P<value>.*?)\s*$")


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


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def render_release_notes(version: str, candidate_tag: str) -> str:
    previous = previous_stable_tag(candidate_tag)
    revspec = f"{previous}..{candidate_tag}" if previous else candidate_tag
    result = subprocess.run(
        ["git", "log", "--pretty=format:%H%x09%s", revspec],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log_output = run_git("log", "--pretty=format:%H%x09%s", "HEAD~20..HEAD")
    else:
        log_output = result.stdout.strip()
    commits = [line.split("\t", 1) for line in log_output.splitlines() if line.strip()]

    lines = [
        f"# Draft release notes for v{version}",
        "",
        f"- Candidate tag: `{candidate_tag}`",
        f"- Previous stable tag: `{previous or 'none'}`",
        "",
        "## Changes since last stable release",
    ]
    if commits:
        for sha, subject in commits:
            short_sha = sha[:7]
            lines.append(f"- {subject} (`{short_sha}`)")
    else:
        lines.append("- No commits found between the previous stable tag and this candidate.")
    lines.extend(
        [
            "",
            "## Release checklist reminders",
            "",
            "- Link Fireworks internal test evidence",
            "- Link backend validation evidence when required",
            "- Confirm reviewer sign-off and rollback plan in the release record",
        ]
    )
    return "\n".join(lines) + "\n"


def previous_stable_tag(candidate_tag: str) -> str:
    result = subprocess.run(
        [
            "git",
            "describe",
            "--tags",
            "--abbrev=0",
            "--match",
            "v[0-9]*.[0-9]*.[0-9]*",
            f"{candidate_tag}^",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def parse_release_metadata(issue_body: str) -> dict[str, str]:
    match = RELEASE_METADATA_RE.search(issue_body)
    if not match:
        raise ReleaseTagError(
            "Release record is missing the cookbook-release-metadata block."
        )

    metadata: dict[str, str] = {}
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = METADATA_LINE_RE.fullmatch(line)
        if not parsed:
            raise ReleaseTagError(
                f"Malformed release metadata line: {raw_line!r}."
            )
        metadata[parsed.group("key")] = parsed.group("value")
    return metadata


def normalize_boolean_text(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "y"}:
        return "true"
    if lowered in {"false", "no", "n"}:
        return "false"
    raise ReleaseTagError(f"Expected boolean-like value, got {value!r}.")


def assert_release_record(
    *,
    issue_body_path: pathlib.Path,
    version: str,
    candidate_tag: str,
    sha: str,
) -> dict[str, str]:
    metadata = parse_release_metadata(issue_body_path.read_text(encoding="utf-8"))
    stable_tag = f"v{version}"
    expected = {
        "version": stable_tag,
        "candidate-tag": candidate_tag,
        "candidate-sha": sha,
        "go-approved": "true",
    }
    for key, value in expected.items():
        actual = metadata.get(key)
        if actual != value:
            raise ReleaseTagError(
                f"Release record metadata has {key}={actual!r}, expected {value!r}."
            )

    backend_required = normalize_boolean_text(
        metadata.get("backend-validation-required", "false")
    )
    return {
        "backend_validation_required": backend_required,
    }


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

    notes = subparsers.add_parser(
        "draft-release-notes",
        help="Generate draft release notes for one candidate tag.",
    )
    notes.add_argument("--version", required=True)
    notes.add_argument("--candidate-tag", required=True)
    notes.add_argument("--notes-file", required=True, type=pathlib.Path)
    notes.add_argument("--github-output", required=False, type=pathlib.Path)

    record = subparsers.add_parser(
        "assert-release-record",
        help="Verify that a release record issue matches the stable release inputs.",
    )
    record.add_argument("--issue-body-file", required=True, type=pathlib.Path)
    record.add_argument("--version", required=True)
    record.add_argument("--candidate-tag", required=True)
    record.add_argument("--candidate-sha", required=True)
    record.add_argument("--github-output", required=False, type=pathlib.Path)

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

        if args.command == "draft-release-notes":
            rendered = render_release_notes(args.version, args.candidate_tag)
            args.notes_file.write_text(rendered, encoding="utf-8")
            if args.github_output:
                write_github_output(
                    args.github_output,
                    {"previous_tag": previous_stable_tag(args.candidate_tag)},
                )
            return 0

        if args.command == "assert-release-record":
            metadata = assert_release_record(
                issue_body_path=args.issue_body_file,
                version=args.version,
                candidate_tag=args.candidate_tag,
                sha=args.candidate_sha,
            )
            if args.github_output:
                write_github_output(args.github_output, metadata)
            return 0
    except ReleaseTagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
