from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor.environments.e2b")
pytest.importorskip("e2b")
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from training.examples.harbor_rl.tito import e2b_frozen


@pytest.fixture
def frozen_context(tmp_path, monkeypatch):
    directory = tmp_path / "environment"
    directory.mkdir()
    (directory / "Dockerfile").write_text("FROM example.invalid/clean\nWORKDIR /testbed\n")
    archive = directory / "build-input.bin"
    archive.write_bytes(b"build bytes")
    tags = []

    async def get_tags(_alias):
        return tags

    async def assign_tags(_alias, tag):
        value = SimpleNamespace(tag=tag, build_id="qualified-build")
        tags.append(value)
        return value

    monkeypatch.setattr(e2b_frozen.AsyncTemplate, "get_tags", get_tags)
    monkeypatch.setattr(e2b_frozen.AsyncTemplate, "assign_tags", assign_tags)
    pin = asyncio.run(
        e2b_frozen.pin_e2b_context(directory, "sample", cpus=4, memory_mb=8192, build_assets=[archive.name])
    )
    document = {"format_version": 1, "contexts": {"sample:environment": pin}}
    manifest = tmp_path / "manifest.json"
    paths = TrialPaths(trial_dir=tmp_path / "trial")
    paths.mkdir()

    def create(*, cpus=4):
        manifest.write_text(json.dumps(document))
        return e2b_frozen.FrozenE2BEnvironment(
            directory,
            "sample",
            session_id="unit",
            trial_paths=paths,
            task_env_config=EnvironmentConfig(cpus=cpus, memory_mb=8192),
            frozen_manifest=manifest,
        )

    return directory, archive, document, pin, tags, create


def test_runtime_uses_pinned_build_without_reading_archives(frozen_context, monkeypatch):
    _, archive, _, pin, _, create = frozen_context
    original = Path.read_bytes

    def read_bytes(path):
        assert path != archive, "runtime must not rehash the build archive"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    environment = create()
    assert environment.environment_id == pin["content_hash"]
    assert environment._template_name == pin["template_ref"]
    assert asyncio.run(environment._does_template_exist())
    with pytest.raises(RuntimeError, match="cannot build"):
        asyncio.run(environment._create_template())
    with pytest.raises(ValueError, match="force_build"):
        asyncio.run(environment.start(force_build=True))


def test_runtime_rejects_changed_template_build(frozen_context):
    _, _, _, _, tags, create = frozen_context
    environment = create()
    tags[0].build_id = "different-build"
    with pytest.raises(RuntimeError, match="missing or changed"):
        asyncio.run(environment._does_template_exist())


@pytest.mark.parametrize("drift", ["source", "role", "resources"])
def test_runtime_rejects_context_or_resource_drift(frozen_context, drift):
    directory, _, _, pin, _, create = frozen_context
    if drift == "source":
        (directory / "Dockerfile").write_text("FROM changed\n")
    elif drift == "role":
        pin["role"] = "tests"
    with pytest.raises(ValueError, match="changed|role mismatch|resources"):
        create(cpus=8 if drift == "resources" else 4)


def test_frozen_runtime_allows_removed_build_only_archives(frozen_context):
    _, archive, _, _, _, create = frozen_context
    archive.unlink()
    assert asyncio.run(create()._does_template_exist())


def test_frozen_runtime_rejects_changed_archive_size(frozen_context):
    _, archive, _, _, _, create = frozen_context
    archive.write_bytes(b"different build input size")
    with pytest.raises(ValueError, match="size changed"):
        create()


def test_replaced_manifest_is_observed_in_the_same_process(frozen_context):
    _, _, _, pin, tags, create = frozen_context
    first = create()
    original = first._template_name
    pin.update(tag="second-tag", build_id="second-build")
    pin["template_ref"] = pin["template_alias"] + ":second-tag"
    tags.append(SimpleNamespace(tag="second-tag", build_id="second-build"))
    second = create()
    assert second._template_name != original
    assert second._template_name == pin["template_ref"]
    assert asyncio.run(second._does_template_exist())


def test_frozen_context_rejects_symlink_inputs(frozen_context):
    directory, _, _, _, _, create = frozen_context
    (directory / "linked-input").symlink_to("Dockerfile")
    with pytest.raises(ValueError, match="symlinks"):
        create()
