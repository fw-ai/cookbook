"""Capture a source patch without loading the agent's Git configuration."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


def collect(cwd: Path, base_commit: str, destination: Path) -> None:
    if not re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", base_commit):
        raise ValueError("Patch capture requires the pinned base commit")
    destination.unlink(missing_ok=True)
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_NO_REPLACE_OBJECTS="1",
    )
    objects = cwd / ".git/objects"
    if not objects.is_dir():
        raise ValueError("Patch capture cannot read the pinned repository objects")
    if (cwd / ".git/info/sparse-checkout").exists():
        raise ValueError(
            "Patch capture requires a complete checkout; materialize sparse checkouts first"
        )
    # Read base objects only; the fresh repository owns its config, index and writes.
    env["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = json.dumps(
        str(objects.resolve()), ensure_ascii=False
    )
    with tempfile.TemporaryDirectory(prefix="mimo-patch-") as temporary:
        git = [
            "git",
            "--git-dir",
            temporary,
            "--work-tree",
            str(cwd),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.excludesFile=/dev/null",
            "-c",
            "core.attributesFile=/dev/null",
        ]

        def run(args: list[str]) -> subprocess.CompletedProcess:
            return subprocess.run(
                args, cwd=cwd, env=env, check=True, capture_output=True
            )

        try:
            # SHA-1 is Git's default, including versions before --object-format.
            format_args = ["--object-format=sha256"] if len(base_commit) == 64 else []
            run(
                [
                    "git",
                    "-c",
                    "init.templateDir=",
                    "init",
                    "--bare",
                    "-q",
                    *format_args,
                    temporary,
                ]
            )
            # Repository attributes still apply with an empty global attributes
            # file. Override content conversion so capture preserves source bytes.
            attributes = Path(temporary) / "info/attributes"
            attributes.parent.mkdir(exist_ok=True)
            attributes.write_text(
                "* -text -filter -ident -working-tree-encoding -crlf\n"
            )
            run([*git, "read-tree", base_commit])
            run([*git, "add", "-A"])
            patch = run(
                [*git, "diff", "--cached", "--binary", "--no-renames", base_commit]
            ).stdout
        except subprocess.CalledProcessError:
            raise ValueError(
                "Cannot collect the candidate patch from its pinned base"
            ) from None
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(patch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    collect(args.cwd, args.base, args.destination)


if __name__ == "__main__":
    main()
