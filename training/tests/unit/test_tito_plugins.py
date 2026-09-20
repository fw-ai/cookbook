"""Tests for TITO renderer certifications supplied by installed distributions."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.renderer.tito import plugins as tito_plugins
from training.renderer.tito import registry as tito_registry
from training.renderer.tito import shared as tito_shared
from training.renderer.tito.plugins import (
    TITORendererExtension,
    register_tito_extension,
    registered_tito_extensions,
)


@pytest.fixture(autouse=True)
def _isolated_extensions():
    saved = registered_tito_extensions()
    tito_plugins.reset_tito_renderer_plugins_for_tests()
    yield
    tito_plugins.reset_tito_renderer_plugins_for_tests(saved)


@pytest.fixture
def package_root(tmp_path: Path) -> Path:
    root = tmp_path / "acme_private_tito"
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    return root


def _certification(*names: str, certification_id: str = "acme@v1"):
    return tito_shared.TITORendererCertification(
        certification_id=certification_id,
        renderer_names=frozenset(names),
        tokenizer_fingerprint="a" * 64,
        renderer_factory=lambda tokenizer, certification: ("acme", tokenizer),
    )


def _extension(package_root: Path, *names: str, name: str = "acme", **kwargs):
    return TITORendererExtension(
        name=name,
        certifications=(kwargs.pop("certification", None) or _certification(*names),),
        sidecar_source_root=package_root,
        sidecar_entry_point=kwargs.pop(
            "sidecar_entry_point", "acme_private_tito:register"
        ),
        **kwargs,
    )


def test_extension_certification_resolves_and_builds(package_root, monkeypatch) -> None:
    register_tito_extension(_extension(package_root, "acme_model"))
    monkeypatch.setattr(
        tito_shared, "_tokenizer_fingerprint", lambda _tokenizer: "a" * 64
    )
    tokenizer = object()

    built = tito_registry.build_sidecar_tito_renderer(tokenizer, "acme_model")

    assert built == ("acme", tokenizer)


def test_builtin_certifications_still_resolve(package_root, monkeypatch) -> None:
    register_tito_extension(_extension(package_root, "acme_model"))
    builtin_name = next(iter(tito_registry._TITO_CERTIFICATION_BY_RENDERER))
    expected = tito_registry._TITO_CERTIFICATION_BY_RENDERER[builtin_name]
    monkeypatch.setattr(
        tito_shared,
        "_tokenizer_fingerprint",
        lambda _tokenizer: expected.tokenizer_fingerprint,
    )

    resolved = tito_registry.get_tito_renderer_certification(builtin_name, object())

    assert resolved is expected


def test_unknown_renderer_still_fails_closed(package_root) -> None:
    register_tito_extension(_extension(package_root, "acme_model"))

    with pytest.raises(ValueError, match="no production TITO certification"):
        tito_registry.get_tito_renderer_certification("other_model", object())


def test_extension_may_not_shadow_a_builtin(package_root) -> None:
    builtin_name = next(iter(tito_registry._TITO_CERTIFICATION_BY_RENDERER))
    register_tito_extension(_extension(package_root, builtin_name))

    with pytest.raises(ValueError, match="may not redefine built-in"):
        tito_registry.get_tito_renderer_certification("other_model", object())


def test_conflicting_extensions_are_rejected(package_root, tmp_path) -> None:
    other_root = tmp_path / "other_private_tito"
    other_root.mkdir()
    (other_root / "__init__.py").write_text("", encoding="utf-8")
    register_tito_extension(_extension(package_root, "acme_model"))

    with pytest.raises(ValueError, match="claim the same renderer names"):
        register_tito_extension(
            TITORendererExtension(
                name="other",
                certifications=(_certification("acme_model", certification_id="o@v1"),),
                sidecar_source_root=other_root,
                sidecar_entry_point="other_private_tito:register",
            )
        )


def test_reregistering_an_identical_extension_is_idempotent(package_root) -> None:
    certification = _certification("acme_model")
    register_tito_extension(_extension(package_root, certification=certification))
    register_tito_extension(_extension(package_root, certification=certification))

    assert len(registered_tito_extensions()) == 1


def test_redefining_a_registered_extension_is_rejected(package_root) -> None:
    register_tito_extension(_extension(package_root, "acme_model"))

    with pytest.raises(ValueError, match="already registered with a different"):
        register_tito_extension(_extension(package_root, "acme_other"))


def test_sidecar_entry_point_must_match_the_shipped_package(package_root) -> None:
    with pytest.raises(ValueError, match="not importable from its source directory"):
        _extension(package_root, "acme_model", sidecar_entry_point="elsewhere:register")


def test_sidecar_entry_point_must_name_a_callable(package_root) -> None:
    with pytest.raises(ValueError, match="must be 'module:callable'"):
        _extension(package_root, "acme_model", sidecar_entry_point="acme_private_tito")


def test_sidecar_source_root_must_be_a_package(tmp_path) -> None:
    bare = tmp_path / "acme_private_tito"
    bare.mkdir()

    with pytest.raises(ValueError, match="is not a package"):
        _extension(bare, "acme_model")
