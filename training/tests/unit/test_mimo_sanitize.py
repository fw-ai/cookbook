"""Rebuilt filesystems must not carry future Git objects or source timestamps."""

import copy
import io
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from training.examples.harbor_rl.tasks.mimo.sanitize import EPOCH, sanitize_rootfs


def git(repo: Path, *args: str) -> str:
    return (
        subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-C",
                str(repo),
                *args,
            ],
            stderr=subprocess.PIPE,
        )
        .decode()
        .strip()
    )


@pytest.fixture
def upstream(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "raw"
    repo = root / "testbed"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "source.py").write_text("broken base\n")
    (repo / ".gitattributes").write_text(
        "kept.txt export-ignore\nsubstituted.txt export-subst\nsource.py filter=host-command\n"
    )
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "kept.txt").write_text("tracked internal source")
    (repo / "ignored.txt").write_text("tracked despite ignore rule")
    (repo / "substituted.txt").write_text("$Format:%H$")
    git(repo, "add", "--all")
    git(repo, "add", "--force", "ignored.txt")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")
    (repo / "source.py").write_text("SECRET_REFERENCE_FIX\n")
    git(repo, "commit", "-qam", "future fix")
    future = git(repo, "rev-parse", "HEAD")
    git(repo, "reset", "--hard", base)
    git(repo, "remote", "add", "origin", "https://example.invalid/future")
    git(repo, "gc", "--prune=never")
    (repo / "node_modules/dep").mkdir(parents=True)
    (repo / "node_modules/dep/runtime.js").write_text("dependency")
    (repo / "node_modules/.cache").mkdir()
    (repo / "node_modules/.cache/answer").write_text("SECRET_REFERENCE_FIX")
    (repo / "untracked.patch").write_text("SECRET_REFERENCE_FIX")
    (root / "root/.cache").mkdir(parents=True)
    (root / "root/.cache/answer").write_text("SECRET_REFERENCE_FIX")
    source = tmp_path / "upstream.tar"
    with tarfile.open(source, "w") as archive:
        for item in root.iterdir():
            archive.add(item, arcname=item.name)
    return source, base, future


def test_rebuilt_archive_has_only_clean_base_history_and_dependencies(tmp_path: Path, upstream) -> None:
    source, base, future = upstream
    output = tmp_path / "clean.tar.gz"
    manifest = sanitize_rootfs(
        source, output, repo_path="/testbed", base_commit=base, dependency_dirs=("node_modules",)
    )
    extracted = tmp_path / "rebuilt"
    with tarfile.open(output) as archive:
        assert all(member.mtime == EPOCH for member in archive)
        assert archive.getmember("tmp").mode == 0o1777
        assert archive.getmember("var/tmp").mode == 0o1777
        archive.extractall(extracted, filter="data")
    repo = extracted / "testbed"
    assert (repo / "source.py").read_text() == "broken base\n"
    assert (repo / "kept.txt").read_text() == "tracked internal source"
    assert (repo / "ignored.txt").read_text() == "tracked despite ignore rule"
    assert (repo / "substituted.txt").read_text() == "$Format:%H$"
    assert (extracted / "tmp").is_dir()
    assert (extracted / "var/tmp").is_dir()
    assert (repo / "node_modules/dep/runtime.js").read_text() == "dependency"
    assert not (repo / "node_modules/.cache/answer").exists()
    assert not (repo / "untracked.patch").exists()
    assert not (extracted / "root/.cache/answer").exists()
    assert git(repo, "rev-list", "--all", "--count") == "1"
    assert git(repo, "remote", "-v") == ""
    assert git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert git(repo, "fsck", "--full", "--no-reflogs", "--unreachable") == ""
    with pytest.raises(subprocess.CalledProcessError):
        git(repo, "cat-file", "-e", future)
    assert git(repo, "rev-parse", "HEAD") == manifest["clean_base_commit"]
    assert manifest["reference_verification"] == "required"
    second = tmp_path / "clean-again.tar.gz"
    again = sanitize_rootfs(source, second, repo_path="/testbed", base_commit=base, dependency_dirs=("node_modules",))
    assert again["output_sha256"] == manifest["output_sha256"]


def test_archive_path_escape_rejected(tmp_path: Path, upstream) -> None:
    source, base, _ = upstream
    with tarfile.open(source, "a") as archive:
        member = tarfile.TarInfo("../escaped")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe member path"):
        sanitize_rootfs(source, tmp_path / "clean.tar.gz", repo_path="testbed", base_commit=base)
    assert not (tmp_path / "escaped").exists()


def test_hardlink_cannot_restore_excluded_cache(tmp_path: Path, upstream) -> None:
    source, base, _ = upstream
    with tarfile.open(source, "a") as archive:
        member = tarfile.TarInfo("usr/share/answer")
        member.type = tarfile.LNKTYPE
        member.linkname = "root/.cache/answer"
        archive.addfile(member)
    with pytest.raises(ValueError, match="excluded repository contents"):
        sanitize_rootfs(source, tmp_path / "clean.tar.gz", repo_path="testbed", base_commit=base)


def test_retained_hardlink_chains_preserve_contents(tmp_path: Path, upstream) -> None:
    source, base, _ = upstream
    with tarfile.open(source, "a") as archive:
        first = tarfile.TarInfo("usr/share/a")
        first.size = 5
        archive.addfile(first, io.BytesIO(b"HELLO"))
        for name, target in (("b", "a"), ("c", "b")):
            link = tarfile.TarInfo("usr/share/" + name)
            link.type = tarfile.LNKTYPE
            link.linkname = "usr/share/" + target
            archive.addfile(link)
    output = tmp_path / "clean.tar.gz"
    sanitize_rootfs(source, output, repo_path="testbed", base_commit=base)
    with tarfile.open(output) as archive:
        assert [archive.extractfile("usr/share/" + name).read() for name in "abc"] == [b"HELLO"] * 3


def test_dependency_symlink_cannot_access_removed_home(tmp_path: Path, upstream) -> None:
    source, base, _ = upstream
    with tarfile.open(source, "a") as archive:
        link = tarfile.TarInfo("testbed/node_modules/leak")
        link.type = tarfile.SYMTYPE
        link.linkname = "/root/.cache/answer"
        archive.addfile(link)
    with pytest.raises(ValueError, match="symlink targets excluded"):
        sanitize_rootfs(
            source, tmp_path / "clean.tar.gz", repo_path="testbed", base_commit=base, dependency_dirs=("node_modules",)
        )


def test_clean_filters_are_not_executed(tmp_path: Path, upstream, monkeypatch: pytest.MonkeyPatch) -> None:
    source, base, _ = upstream
    marker = tmp_path / "filter-executed"
    config = tmp_path / "global.gitconfig"
    config.write_text(f'[filter "host-command"]\nclean = "touch {marker}; cat"\nrequired = true\n')
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    sanitize_rootfs(source, tmp_path / "clean.tar.gz", repo_path="testbed", base_commit=base)
    assert not marker.exists()


def test_legacy_git_ignores_system_pack_configuration(tmp_path: Path, upstream, monkeypatch: pytest.MonkeyPatch) -> None:
    source, base, _ = upstream
    system_config = tmp_path / "system.gitconfig"
    system_config.write_text("[pack]\nwindowMemory = invalid-host-setting\n")
    original = subprocess.check_output

    def legacy_git(args: list[str], **kwargs) -> bytes:
        # Older Git does not honor GIT_CONFIG_SYSTEM=/dev/null.
        env = dict(kwargs["env"])
        env["GIT_CONFIG_SYSTEM"] = str(system_config)
        return original(args, **{**kwargs, "env": env})

    monkeypatch.setattr(subprocess, "check_output", legacy_git)
    sanitize_rootfs(source, tmp_path / "clean.tar.gz", repo_path="testbed", base_commit=base)


def test_retained_inode_cannot_precede_its_excluded_hardlink(tmp_path: Path, upstream) -> None:
    source, base, _ = upstream
    with tarfile.open(source, "a") as archive:
        member = tarfile.TarInfo("usr/share/answer")
        member.size = 6
        archive.addfile(member, io.BytesIO(b"SECRET"))
        link = tarfile.TarInfo("root/.cache/alias")
        link.type = tarfile.LNKTYPE
        link.linkname = member.name
        archive.addfile(link)
    with pytest.raises(ValueError, match="excluded repository contents"):
        sanitize_rootfs(source, tmp_path / "clean.tar.gz", repo_path="testbed", base_commit=base)


def test_normalized_hardlink_target_keeps_original_tar_spelling(tmp_path: Path, upstream) -> None:
    source, base, _ = upstream
    with tarfile.open(source, "a") as archive:
        first = tarfile.TarInfo("./usr/share/a")
        first.size = 5
        archive.addfile(first, io.BytesIO(b"HELLO"))
        link = tarfile.TarInfo("usr/share/b")
        link.type = tarfile.LNKTYPE
        link.linkname = first.name
        archive.addfile(link)
    output = tmp_path / "clean.tar.gz"
    sanitize_rootfs(source, output, repo_path="testbed", base_commit=base)
    with tarfile.open(output) as archive:
        assert archive.extractfile("usr/share/b").read() == b"HELLO"


def test_explicit_dependencies_survive_a_checkout_under_home(tmp_path: Path, upstream) -> None:
    source, base, _ = upstream
    moved = tmp_path / "home-checkout.tar"
    with tarfile.open(source) as original, tarfile.open(moved, "w") as output:
        for member in original:
            replacement = copy.copy(member)
            if member.name == "testbed" or member.name.startswith("testbed/"):
                replacement.name = "home/agent/" + member.name
            output.addfile(replacement, original.extractfile(member) if member.isfile() else None)
    rebuilt = tmp_path / "clean.tar.gz"
    sanitize_rootfs(
        moved, rebuilt, repo_path="home/agent/testbed", base_commit=base, dependency_dirs=("node_modules",)
    )
    with tarfile.open(rebuilt) as archive:
        assert archive.extractfile("home/agent/testbed/node_modules/dep/runtime.js").read() == b"dependency"
        assert "home/agent/testbed/node_modules/.cache/answer" not in archive.getnames()


def test_archive_digest_does_not_depend_on_host_umask(tmp_path: Path, upstream) -> None:
    source, base, _ = upstream
    digests = []
    for umask in (0o002, 0o022, 0o077):
        previous = os.umask(umask)
        try:
            output = tmp_path / f"clean-{umask}.tar.gz"
            result = sanitize_rootfs(source, output, repo_path="testbed", base_commit=base)
        finally:
            os.umask(previous)
        digests.append(result["output_sha256"])
        with tarfile.open(output) as archive:
            assert archive.getmember("testbed").mode == 0o755
            assert archive.getmember("testbed/.git/objects").mode == 0o755
    assert len(set(digests)) == 1


def test_composed_source_symlinks_cannot_escape_after_later_links_exist(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    repo = raw / "testbed"
    (repo / "d").mkdir(parents=True)
    (repo / "d/a").symlink_to("b/../../outside")
    (repo / "d/b").symlink_to("..")
    git(repo, "init", "-q")
    git(repo, "add", "d")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")
    source = tmp_path / "source.tar"
    with tarfile.open(source, "w") as archive:
        archive.add(repo, arcname="testbed")
    with pytest.raises(ValueError, match="unsafe links"):
        sanitize_rootfs(source, tmp_path / "clean.tar.gz", repo_path="testbed", base_commit=base)
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("excluded_alias", [False, True])
def test_duplicate_archive_paths_are_rejected_before_link_resolution(tmp_path: Path, upstream, excluded_alias) -> None:
    source, base, _ = upstream
    with tarfile.open(source, "a") as archive:
        first = tarfile.TarInfo("usr/share/a")
        first.size = 6
        archive.addfile(first, io.BytesIO(b"SECRET"))
        link = tarfile.TarInfo("root/.cache/alias" if excluded_alias else "usr/share/b")
        link.type = tarfile.LNKTYPE
        link.linkname = first.name
        archive.addfile(link)
        replacement = tarfile.TarInfo("./usr/share/a")
        replacement.size = 5
        archive.addfile(replacement, io.BytesIO(b"CLEAN"))
    with pytest.raises(ValueError, match="duplicate member path"):
        sanitize_rootfs(source, tmp_path / "clean.tar.gz", repo_path="testbed", base_commit=base)


def test_git_initialization_ignores_host_environment(tmp_path: Path, upstream, monkeypatch) -> None:
    source, base, _ = upstream
    clean = sanitize_rootfs(source, tmp_path / "baseline.tar.gz", repo_path="testbed", base_commit=base)
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "HOST_ONLY").write_text("must not enter the image")
    redirected = tmp_path / "redirected.git"
    monkeypatch.setenv("GIT_TEMPLATE_DIR", str(templates))
    monkeypatch.setenv("GIT_DIR", str(redirected))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "redirected-objects"))
    rebuilt = sanitize_rootfs(source, tmp_path / "clean.tar.gz", repo_path="testbed", base_commit=base)
    assert rebuilt["output_sha256"] == clean["output_sha256"]
    assert not redirected.exists()
    with tarfile.open(tmp_path / "clean.tar.gz") as archive:
        config = archive.extractfile("testbed/.git/config").read().decode()
        assert "ignorecase = false" in config
        assert "filemode = true" in config
        assert "symlinks = true" in config
        assert not any("HOST_ONLY" in name for name in archive.getnames())


def test_home_checkout_dependency_links_can_target_rebuilt_source(tmp_path: Path, upstream) -> None:
    source, base, _ = upstream
    moved = tmp_path / "home-checkout.tar"
    with tarfile.open(source) as original, tarfile.open(moved, "w") as output:
        for member in original:
            replacement = copy.copy(member)
            if member.name == "testbed" or member.name.startswith("testbed/"):
                replacement.name = "home/agent/" + member.name
            output.addfile(replacement, original.extractfile(member) if member.isfile() else None)
        link = tarfile.TarInfo("home/agent/testbed/node_modules/workspace-source")
        link.type = tarfile.SYMTYPE
        link.linkname = "../source.py"
        output.addfile(link)
    result = tmp_path / "clean.tar.gz"
    sanitize_rootfs(moved, result, repo_path="home/agent/testbed", base_commit=base, dependency_dirs=("node_modules",))
    with tarfile.open(result) as archive:
        assert archive.getmember("home/agent/testbed/node_modules/workspace-source").linkname == "../source.py"


def test_agent_can_recover_its_own_post_sanitization_commit(tmp_path: Path, upstream) -> None:
    source, base, upstream_fix = upstream
    result = tmp_path / "clean.tar.gz"
    manifest = sanitize_rootfs(source, result, repo_path="testbed", base_commit=base)
    extracted = tmp_path / "extracted"
    with tarfile.open(result) as archive:
        archive.extractall(extracted, filter="data")
    repo = extracted / "testbed"
    (repo / "source.py").write_text("agent candidate\n")
    git(repo, "commit", "-qam", "candidate")
    candidate = git(repo, "rev-parse", "HEAD")
    git(repo, "reset", "--hard", manifest["clean_base_commit"])
    assert git(repo, "rev-parse", "HEAD@{1}") == candidate
    with pytest.raises(subprocess.CalledProcessError):
        git(repo, "cat-file", "-e", upstream_fix)


def test_root_verifier_artifacts_are_removed(tmp_path: Path, upstream) -> None:
    source, base, _ = upstream
    payload = b"SECRET_REFERENCE_FIX"
    with tarfile.open(source, "a") as archive:
        for name in ("tests/hidden.patch", "tests/verify.py", "logs/verifier/reward.json", "logs/agent/solution.log"):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    output = tmp_path / "clean.tar.gz"
    sanitize_rootfs(source, output, repo_path="/testbed", base_commit=base)
    with tarfile.open(output) as archive:
        assert not any(Path(member.name).parts[0] in {"tests", "logs"} for member in archive)


def test_retained_dependencies_cannot_keep_test_runner_caches(tmp_path: Path, upstream) -> None:
    source, base, _ = upstream
    with tarfile.open(source, "a") as archive:
        for cache in (".vitest", ".pytest_cache"):
            member = tarfile.TarInfo("testbed/node_modules/" + cache + "/answer")
            payload = b"SECRET_REFERENCE_FIX"
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    output = tmp_path / "clean.tar.gz"
    sanitize_rootfs(source, output, repo_path="/testbed", base_commit=base, dependency_dirs=("node_modules",))
    with tarfile.open(output) as archive:
        assert not any({".vitest", ".pytest_cache"}.intersection(Path(member.name).parts) for member in archive)


def test_rebuilt_object_store_cannot_recover_reference_source(tmp_path: Path, upstream) -> None:
    source, base, _ = upstream
    output = tmp_path / "clean.tar.gz"
    sanitize_rootfs(source, output, repo_path="/testbed", base_commit=base)
    extracted = tmp_path / "objects-audit"
    with tarfile.open(output) as archive:
        archive.extractall(extracted, filter="data")
    repo = extracted / "testbed"
    # Enumerate physical objects, including unreachable blobs and packed objects.
    inventory = git(repo, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)")
    for oid in inventory.splitlines():
        data = subprocess.check_output(["git", "-C", str(repo), "cat-file", "-p", oid])
        assert b"SECRET_REFERENCE_FIX" not in data


@pytest.mark.parametrize("dirty_kind", ["file", "directory", "symlink", "hardlink-alias"])
def test_tracked_dependencies_cannot_retain_dirty_archive_members(
    tmp_path: Path, upstream, dirty_kind: str
) -> None:
    source, _, _ = upstream
    repo = source.parent / "raw/testbed"
    tracked = repo / "node_modules/dep/tracked.js"
    tracked.write_text("clean tracked dependency\n")
    git(repo, "add", "node_modules/dep/tracked.js")
    git(repo, "commit", "-qm", "base with tracked dependency")
    base = git(repo, "rev-parse", "HEAD")
    tracked.unlink()
    payload = b"DIRTY_TRACKED_DEPENDENCY_ANSWER"
    if dirty_kind == "directory":
        tracked.mkdir()
        (tracked / "answer").write_bytes(payload)
    elif dirty_kind == "symlink":
        tracked.symlink_to(payload.decode())
    else:
        tracked.write_bytes(payload)
    dirty = tmp_path / "dirty-dependency.tar"
    with tarfile.open(dirty, "w") as archive:
        archive.add(repo, arcname="testbed")
        if dirty_kind == "hardlink-alias":
            alias = tarfile.TarInfo("usr/share/dirty-alias")
            alias.type = tarfile.LNKTYPE
            alias.linkname = "testbed/node_modules/dep/tracked.js"
            archive.addfile(alias)
    output = tmp_path / "clean-dependency.tar.gz"
    if dirty_kind == "hardlink-alias":
        with pytest.raises(ValueError, match="excluded repository contents"):
            sanitize_rootfs(dirty, output, repo_path="testbed", base_commit=base, dependency_dirs=("node_modules",))
        return
    sanitize_rootfs(dirty, output, repo_path="testbed", base_commit=base, dependency_dirs=("node_modules",))
    with tarfile.open(output) as archive:
        names = [str(Path(member.name)) for member in archive]
        assert len(names) == len(set(names))
        for member in archive:
            assert payload.decode() not in member.linkname
            if member.isfile():
                assert payload not in archive.extractfile(member).read()
        assert archive.extractfile("testbed/node_modules/dep/tracked.js").read() == b"clean tracked dependency\n"
        assert archive.extractfile("testbed/node_modules/dep/runtime.js").read() == b"dependency"
        assert "testbed/node_modules/dep/tracked.js/answer" not in names
        archive.extractall(tmp_path / "dependency-extracted", filter="data")
