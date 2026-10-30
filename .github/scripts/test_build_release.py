"""Release versions and metadata must be safe to publish as PyPI dependencies."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("build_release.py")
spec = importlib.util.spec_from_file_location("build_release", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.mark.parametrize(
    ("tag", "version"),
    [
        ("cookbook-v2026.10.26.1", "2026.10.26.1"),
        ("cookbook-v2026.01.02.12", "2026.1.2.12"),
    ],
)
def test_release_version(tag, version):
    assert release.release_version(tag) == version


@pytest.mark.parametrize(
    "tag",
    [
        "v0.1.0",
        "main",
        "cookbook-v2026.02.30.1",
        "cookbook-v2026.10.26.0",
        "cookbook-v2026.10.26.1/other",
    ],
)
def test_invalid_release_tag(tag):
    with pytest.raises(ValueError):
        release.release_version(tag)


def test_release_requires_committed_version(tmp_path):
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nversion = "2026.10.26.1"\n')
    assert (
        release.verify_source_version(project, "cookbook-v2026.10.26.1")
        == "2026.10.26.1"
    )
    with pytest.raises(ValueError, match="committed package version"):
        release.verify_source_version(project, "cookbook-v2026.10.26.2")


def test_distribution_metadata():
    metadata = b"Name: fireworks-training-cookbook\nVersion: 2026.10.26.1\nRequires-Dist: fireworks-ai>=1.2.30,<2\n"
    release.verify_metadata(metadata, "2026.10.26.1")
    with pytest.raises(ValueError, match="name/version"):
        release.verify_metadata(metadata, "0.1.0")
    with pytest.raises(ValueError, match="direct URL"):
        release.verify_metadata(
            metadata
            + b'Requires-Dist: other @ git+https://example.com/repo.git ; extra == "dev"\n',
            "2026.10.26.1",
        )


@pytest.mark.parametrize("bad_artifact", [None, "wheel", "sdist"])
def test_distribution_versions_agree(tmp_path, bad_artifact):
    import io
    import tarfile
    import zipfile

    version = "2026.10.26.1"

    def metadata(kind):
        artifact_version = "0.1.0" if kind == bad_artifact else version
        return (
            f"Name: fireworks-training-cookbook\nVersion: {artifact_version}\n".encode()
        )

    with zipfile.ZipFile(tmp_path / "cookbook.whl", "w") as wheel:
        wheel.writestr("cookbook.dist-info/METADATA", metadata("wheel"))
        wheel.writestr("training/__init__.py", "")
    with tarfile.open(tmp_path / "cookbook.tar.gz", "w:gz") as sdist:
        data = metadata("sdist")
        info = tarfile.TarInfo("cookbook/PKG-INFO")
        info.size = len(data)
        sdist.addfile(info, io.BytesIO(data))
    if bad_artifact:
        with pytest.raises(ValueError, match="name/version"):
            release.verify_distributions(tmp_path, version)
    else:
        release.verify_distributions(tmp_path, version)
