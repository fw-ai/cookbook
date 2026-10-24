"""Capture a source patch without loading the agent's Git configuration."""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import List, Sequence, Tuple


def validate_dependency_dirs(names: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise ValueError("dependency_dirs must be a sequence of relative directories")
    result = []
    for name in names:
        if not isinstance(name, str) or "\\" in name or "\x00" in name:
            raise ValueError("dependency_dirs must name dependency subdirectories")
        path = PurePosixPath(name)
        if path.is_absolute() or not path.parts or ".." in path.parts or ".git" in path.parts:
            raise ValueError("dependency_dirs must name dependency subdirectories")
        result.append(str(path))
    return tuple(result)


def collect(
    cwd: Path, base_commit: str, destination: Path, *, dependency_dirs: Sequence[str] = ()
) -> None:
    dependency_dirs = validate_dependency_dirs(dependency_dirs)
    if not re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", base_commit):
        raise ValueError("Patch capture requires the pinned base commit")
    try:
        destination.unlink()
    except FileNotFoundError:
        pass
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

        def run(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
            return subprocess.run(
                args, cwd=cwd, env=env, check=check,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
            if dependency_dirs:
                # Tracked changes always survive, including dependency paths.
                # Only untracked preparation-declared dependencies stay out.
                run([*git, "add", "-u"])
                excludes = []
                for name in dependency_dirs:
                    ignored = run([*git, "check-ignore", "--no-index", "--quiet", "--", name], check=False)
                    if ignored.returncode == 1:
                        excludes.append(":(top,exclude,literal)" + name)
                    elif ignored.returncode != 0:
                        ignored.check_returncode()
                # Git rejects an explicit negative pathspec for an ignored
                # directory. Its ignore rule already excludes untracked files;
                # tracked edits were staged separately above.
                run([*git, "add", "-A", "--", ".", *excludes])
            else:
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
    parser.add_argument("--dependency-dir", action="append", default=[])
    args = parser.parse_args()
    collect(args.cwd, args.base, args.destination, dependency_dirs=args.dependency_dir)


if __name__ == "__main__":
    main()
