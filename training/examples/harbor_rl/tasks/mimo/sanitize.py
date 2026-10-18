"""Rebuild coding root filesystems without the upstream repository object store.

Export an unstarted upstream container with ``docker export`` first. This tool
writes a new filesystem archive; deleting files in an image layer is insufficient.
The caller must separately validate the base and reference patches in a verifier.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import os
import posixpath
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

EPOCH = 946684800
SCRUB_ROOTS = ("tmp", "var/tmp", "var/log", "var/cache", "root", "home", "tests", "logs")


def _path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Rootfs archive contains an unsafe member path")
    return path


def _excluded(name: PurePosixPath, repo: PurePosixPath, dependencies: tuple[PurePosixPath, ...]) -> str | None:
    if ".git" in name.parts:
        return "original_git"
    if name.is_relative_to(repo) and not any(name.is_relative_to(dep) for dep in dependencies):
        return "replaced_source"
    if any(name.is_relative_to(root) for root in SCRUB_ROOTS) and not any(
        name.is_relative_to(dep) for dep in dependencies
    ):
        return "cache_log_temp"
    if name.parts and name.parts[0] in {"proc", "sys", "dev", "run"}:
        return "runtime"
    if {".cache", "__pycache__", ".pytest_cache", ".vitest"}.intersection(name.parts):
        return "compiled_cache"
    return None


def _git(git_dir: Path, *args: str, env: dict[str, str] | None = None, input_data: bytes | None = None) -> bytes:
    clean_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    clean_env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    clean_env.update(env or {})
    try:
        return subprocess.check_output(
            ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "--git-dir", str(git_dir), *args],
            env=clean_env,
            input=input_data,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        raise ValueError("Cannot rebuild the pinned repository base") from None


def _pinned_tree(source: tarfile.TarFile, repo: PurePosixPath, commit: str, temporary: Path) -> tuple[Path, str]:
    objects = temporary / "objects.git"
    objects.mkdir()
    (objects / "objects").mkdir()
    (objects / "refs").mkdir()
    (objects / "HEAD").write_text("ref: refs/heads/clean-base\n")
    (objects / "config").write_text("[core]\nrepositoryformatversion = 0\nbare = true\n")
    prefix = repo / ".git/objects"
    for member in source.getmembers():
        name = _path(member.name)
        if not name.is_relative_to(prefix):
            continue
        relative = name.relative_to(prefix).as_posix()
        if not re.fullmatch(r"(?:[0-9a-f]{2}/[0-9a-f]{38}|pack/pack-[0-9a-f]{40}\.(?:pack|idx))", relative):
            continue
        if not member.isfile():
            raise ValueError("Repository object files must be regular files")
        target = objects / "objects" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.extractfile(member).read())
    tree = _git(objects, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    checkout = temporary / "checkout"
    checkout.mkdir()
    # Read raw blobs: archive attributes and clean/smudge filters must not alter source.
    records = _git(objects, "ls-tree", "-rz", "--full-tree", commit).split(b"\0")
    entries = []
    for record in filter(None, records):
        metadata, filename = record.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        if kind != b"blob" or mode not in {b"100644", b"100755", b"120000"}:
            raise ValueError("Pinned base requires audited submodule materialization")
        relative = _path(os.fsdecode(filename))
        if any(part.lower() == ".git" for part in relative.parts):
            raise ValueError("Pinned source cannot supply Git configuration")
        entries.append((mode, oid, relative))
    blob_data = _git(objects, "cat-file", "--batch", input_data=b"".join(oid + b"\n" for _, oid, _ in entries))
    offset = 0
    symlinks: list[tuple[Path, str]] = []
    for mode, oid, relative in entries:
        header_end = blob_data.index(b"\n", offset)
        reported, kind, size = blob_data[offset:header_end].split()
        if reported != oid or kind != b"blob":
            raise ValueError("Pinned source contains an invalid blob")
        offset = header_end + 1
        content = blob_data[offset : offset + int(size)]
        offset += int(size) + 1
        target = checkout / relative
        try:
            if not target.parent.resolve().is_relative_to(checkout.resolve()):
                raise ValueError("Pinned source tree contains unsafe links")
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == b"120000":
                link = os.fsdecode(content)
                if PurePosixPath(link).is_absolute():
                    raise ValueError("Pinned source tree contains unsafe links")
                _path(posixpath.normpath(str(relative.parent / link)))
                symlinks.append((target, link))
            else:
                if target.is_symlink():
                    raise ValueError("Pinned source tree contains unsafe links")
                target.write_bytes(content)
                target.chmod(0o755 if mode == b"100755" else 0o644)
        except (OSError, RuntimeError):
            raise ValueError("Cannot materialize pinned source safely") from None
    # Materialize all regular files before links can redirect checkout writes.
    try:
        for target, link in symlinks:
            target.symlink_to(link)
        for target, _ in symlinks:
            if not target.resolve().is_relative_to(checkout.resolve()):
                raise ValueError("Pinned source tree contains unsafe links")
    except (OSError, RuntimeError):
        raise ValueError("Cannot materialize pinned source safely") from None
    # Construct Linux metadata explicitly: git init inherits host environment,
    # templates and filesystem probes that must not enter the rebuilt image.
    git_dir = checkout / ".git"
    (git_dir / "objects").mkdir(parents=True)
    (git_dir / "refs/heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/clean-base\n")
    (git_dir / "config").write_text(
        "[core]\nrepositoryformatversion = 0\nbare = false\n"
        "filemode = true\nignorecase = false\nsymlinks = true\nlogallrefupdates = true\n"
    )
    # Copy only base-tree objects, never original commits or unreachable answers.
    pack = _git(objects, "pack-objects", "--stdout", "--revs", "--threads=1", input_data=(tree + "\n").encode())
    _git(git_dir, "index-pack", "--stdin", input_data=pack)
    clean_commit = (
        _git(
            git_dir,
            "commit-tree",
            tree,
            input_data=b"clean base\n",
            env={
                "GIT_AUTHOR_NAME": "Experiment",
                "GIT_AUTHOR_EMAIL": "experiment@example.invalid",
                "GIT_COMMITTER_NAME": "Experiment",
                "GIT_COMMITTER_EMAIL": "experiment@example.invalid",
                "GIT_AUTHOR_DATE": f"{EPOCH} +0000",
                "GIT_COMMITTER_DATE": f"{EPOCH} +0000",
            },
        )
        .decode()
        .strip()
    )
    (git_dir / "refs/heads/clean-base").write_text(clean_commit + "\n")
    _git(git_dir, "read-tree", tree)
    if _git(git_dir, "rev-parse", "HEAD^{tree}").decode().strip() != tree:
        raise ValueError("Rebuilt source tree differs from the pinned base")
    return checkout, tree


def sanitize_rootfs(
    source_path: Path,
    output_path: Path,
    *,
    repo_path: str,
    base_commit: str,
    dependency_dirs: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Write a fresh tar.gz and return provenance; never modifies the input.

    Only tracked base source and explicitly retained dependency directories survive
    inside the checkout. Tracked paths always come from the pinned base, even in
    retained dependencies, which require a separate leak/lock audit.
    The output must be imported or unpacked on a clean OS base, not layered over
    the upstream task image. Reference-pass/base-fail QA remains mandatory.
    Ambiguous archives with duplicate normalized member paths are rejected.
    """
    repo = _path(repo_path.removeprefix("/"))
    if not repo.parts or repo == PurePosixPath("."):
        raise ValueError("repo_path must name a non-root checkout")
    if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        raise ValueError("base_commit must be a full SHA-1 commit")
    dependencies = tuple(repo / _path(name) for name in dependency_dirs)
    if any(path == repo or ".git" in path.parts for path in dependencies):
        raise ValueError("dependency_dirs must name dependency subdirectories")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Source and output archives must differ")
    excluded: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="mimo-clean-", dir=output_path.parent) as temporary:
        with tarfile.open(source_path) as source:
            members = source.getmembers()
            by_name = {}
            for member in members:
                name = _path(member.name)
                if name in by_name:
                    raise ValueError("Rootfs archive contains a duplicate member path")
                by_name[name] = member
            checkout, tree = _pinned_tree(source, repo, base_commit, Path(temporary))
            clean_commit = _git(checkout / ".git", "rev-parse", "HEAD").decode().strip()
            checkout_paths = [checkout, *sorted(checkout.rglob("*"))]
            rebuilt_paths = {repo / path.relative_to(checkout) for path in checkout_paths}
            rebuilt_leaves = {
                repo / path.relative_to(checkout)
                for path in checkout_paths
                if path.is_symlink() or not path.is_dir()
            }

            def exclusion_reason(name: PurePosixPath) -> str | None:
                reason = _excluded(name, repo, dependencies)
                if reason:
                    return reason
                # Never retain a dirty tracked dependency alongside the clean
                # member. Also drop descendants if an old directory became a file.
                if name in rebuilt_paths or any(parent in rebuilt_leaves for parent in name.parents):
                    return "replaced_source"
                return None

            hardlink_targets: dict[int, tarfile.TarInfo] = {}
            excluded_inodes: set[int] = set()
            for member in members:
                if not member.islnk():
                    continue
                link = member
                visited = set()
                excluded_alias = bool(exclusion_reason(_path(member.name)))
                while link.islnk() or link.issym():
                    target_name = (
                        link.linkname
                        if link.islnk()
                        else posixpath.normpath(str(_path(link.name).parent / link.linkname))
                    )
                    target = _path(target_name)
                    if target in visited or target not in by_name:
                        raise ValueError("Rootfs contains a cyclic or unresolved hardlink")
                    visited.add(target)
                    excluded_alias |= bool(exclusion_reason(target))
                    link = by_name[target]
                hardlink_targets[member.offset] = link
                if excluded_alias:
                    excluded_inodes.add(link.offset)
            with (
                output_path.open("wb") as raw,
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=3, mtime=EPOCH) as compressed,
                tarfile.open(fileobj=compressed, mode="w") as output,
            ):
                for original in members:
                    member = copy.copy(original)
                    name = _path(member.name)
                    reason = exclusion_reason(name)
                    if reason:
                        excluded[reason] = excluded.get(reason, 0) + 1
                        continue
                    if member.isdev() or member.isfifo():
                        continue
                    member.mtime = EPOCH
                    member.pax_headers = {}
                    resolved = hardlink_targets.get(member.offset, original)
                    if resolved.offset in excluded_inodes:
                        raise ValueError("Retained file links to excluded repository contents")
                    contents = source.extractfile(resolved) if member.isfile() or member.islnk() else None
                    if member.islnk():
                        member.type = tarfile.REGTYPE
                        member.linkname = ""
                        member.size = contents.seek(0, 2)
                        contents.seek(0)
                    if member.issym() and name.is_relative_to(repo):
                        target = _path(
                            posixpath.normpath(
                                member.linkname.lstrip("/")
                                if member.linkname.startswith("/")
                                else str(name.parent / member.linkname)
                            )
                        )
                        reason = exclusion_reason(target)
                        if reason and reason != "replaced_source":
                            raise ValueError("Dependency symlink targets excluded filesystem contents")
                    output.addfile(member, contents)
                for directory in ("tmp", "var/tmp"):
                    if PurePosixPath(directory) in rebuilt_paths:
                        continue
                    info = tarfile.TarInfo(directory)
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o1777
                    info.mtime = EPOCH
                    output.addfile(info)
                for path in checkout_paths:
                    info = output.gettarinfo(str(path), arcname=str(repo / path.relative_to(checkout)))
                    info.mtime = EPOCH
                    info.pax_headers = {}
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    if info.isdir():
                        info.mode = 0o1777 if info.name in {"tmp", "var/tmp"} else 0o755
                    elif info.isfile():
                        info.mode = 0o755 if info.mode & 0o111 else 0o644
                    if info.isfile():
                        with path.open("rb") as contents:
                            output.addfile(info, contents)
                    else:
                        output.addfile(info)
    with output_path.open("rb") as artifact:
        digest = hashlib.file_digest(artifact, "sha256").hexdigest()
    return {
        "base_commit": base_commit,
        "base_tree": tree,
        "clean_base_commit": clean_commit,
        "repo_path": str(repo),
        "dependency_dirs": list(dependency_dirs),
        "epoch": EPOCH,
        "excluded_members": excluded,
        "output_sha256": digest,
        "reference_verification": "required",
    }
