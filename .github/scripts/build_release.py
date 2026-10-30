"""Build PyPI distributions using the version of an existing cookbook release.

Promotion commits the package version before review. Release builds verify
that the tag agrees with that source version; they never rewrite metadata.
Run from the cookbook root with build, packaging, and twine installed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from datetime import date
from email.parser import BytesParser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def release_version(tag: str) -> str:
    match = re.fullmatch(r"cookbook-v(\d{4})\.(\d{1,2})\.(\d{1,2})\.([1-9]\d*)", tag)
    if match is None:
        raise ValueError(f"Expected cookbook-vYYYY.MM.DD.N release tag, got {tag!r}")
    year, month, day, _ = map(int, match.groups())
    date(year, month, day)  # Reject impossible dates.
    return str(Version(tag.removeprefix("cookbook-v")))


def verify_source_version(pyproject: Path, tag: str | None) -> str:
    version = tomllib.loads(pyproject.read_text())["project"]["version"]
    if tag and (release_version(tag) != version or tag != f"cookbook-v{version}"):
        raise ValueError("Release tag does not match the committed package version")
    return version


def verify_metadata(data: bytes, version: str) -> None:
    metadata = BytesParser().parsebytes(data)
    if (
        metadata["Name"] != "fireworks-training-cookbook"
        or metadata["Version"] != version
    ):
        raise ValueError(
            "Distribution name/version does not match the cookbook release"
        )
    for dependency in metadata.get_all("Requires-Dist", []):
        if Requirement(dependency).url:
            raise ValueError(
                f"PyPI does not support direct URL dependencies: {dependency}"
            )


def verify_distributions(dist: Path, version: str) -> None:
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one sdist")
    with zipfile.ZipFile(wheels[0]) as wheel:
        (metadata_path,) = [
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        ]
        verify_metadata(wheel.read(metadata_path), version)
        if "training/__init__.py" not in wheel.namelist():
            raise ValueError("Wheel is missing the training package")
    with tarfile.open(sdists[0]) as sdist:
        (metadata_path,) = [
            member
            for member in sdist.getmembers()
            if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")
        ]
        with sdist.extractfile(metadata_path) as metadata:
            verify_metadata(metadata.read(), version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", nargs="?")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    dist = root / "dist"
    if dist.exists() and any(dist.iterdir()):
        raise ValueError("dist must be empty before building a release")
    version = verify_source_version(root / "training" / "pyproject.toml", args.tag)
    # build's default builds the wheel FROM the sdist, testing source packaging.
    subprocess.run(
        [sys.executable, "-m", "build", "training", "--outdir", str(dist)],
        cwd=root,
        check=True,
    )
    verify_distributions(dist, version)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "twine",
            "check",
            "--strict",
            *map(str, sorted(dist.iterdir())),
        ],
        check=True,
    )
    print(f"Verified fireworks-training-cookbook=={version}")


if __name__ == "__main__":
    main()
