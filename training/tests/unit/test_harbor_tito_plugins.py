"""Shipping TITO renderer extensions into the Harbor agent sidecar bundle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.examples.rl.harbor.tito import sidecar as sidecar_runtime
from training.renderer.tito import plugins as tito_plugins
from training.renderer.tito.plugins import TITORendererExtension
from training.renderer.tito.shared import TITORendererCertification

_FINGERPRINT = "a" * 64

# Registered inside the sandbox from bundled sources rather than by discovering
# an installed distribution, so it must not import the training stack.
_PLUGIN_SOURCE = '''"""Synthetic TITO renderer extension."""

from pathlib import Path

_registered = False


def register():
    global _registered
    if _registered:
        return
    from training.renderer.tito.plugins import (
        TITORendererExtension,
        register_tito_extension,
    )
    from training.renderer.tito.shared import TITORendererCertification

    register_tito_extension(
        TITORendererExtension(
            name="acme",
            certifications=(
                TITORendererCertification(
                    certification_id="acme@v1",
                    renderer_names=frozenset({"acme_model"}),
                    tokenizer_fingerprint="a" * 64,
                    renderer_factory=lambda tokenizer, certification: "acme-renderer",
                ),
            ),
            sidecar_source_root=Path(__file__).resolve().parent,
            sidecar_entry_point="acme_private_tito:register",
        )
    )
    _registered = True
'''

_SANDBOX_PROBE = (
    "import json; from pathlib import Path; "
    "from training.examples.rl.harbor.tito import sidecar; "
    "from training.renderer.tito import build_sidecar_tito_renderer; "
    "from training.renderer.tito import shared; "
    "manifest = json.loads(Path('manifest.json').read_text()); "
    # Extracted here rather than at the sandbox's absolute bundle root.
    "sidecar._register_bundled_tito_plugins("
    "manifest['tito_plugin_entry_points'], Path('plugins')); "
    "shared._tokenizer_fingerprint = lambda _tokenizer: 'a' * 64; "
    "assert build_sidecar_tito_renderer(object(), 'acme_model') == 'acme-renderer'"
)


class _Backend:
    @staticmethod
    def to_str() -> str:
        return '{"model":{"type":"WordLevel","vocab":{"x":0}}}'


class _Tokenizer:
    backend_tokenizer = _Backend()
    chat_template = "{{ messages }}"
    special_tokens_map = {"eos_token": "</s>"}

    @staticmethod
    def save_pretrained(path: Path) -> None:
        path.mkdir(parents=True)
        (path / "tokenizer.json").write_text(_Backend.to_str())
        (path / "chat_template.jinja").write_text(_Tokenizer.chat_template)


@pytest.fixture(autouse=True)
def _isolated_extensions():
    saved = tito_plugins.registered_tito_extensions()
    tito_plugins.reset_tito_renderer_plugins_for_tests()
    yield
    tito_plugins.reset_tito_renderer_plugins_for_tests(saved)


def _setup(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer=_Tokenizer(),
        tokenizer_id="tokenizer",
        sample_kwargs={"max_tokens": 128, "max_seq_len": 4096},
        extras={
            "renderer_name": "acme_model",
            "tito_sidecar_bundle_root": str(tmp_path / "bundles"),
        },
        inference_base_url="http://deployment",
        api_key="deployment-key",
        model="deployment",
    )


def _build_bundle(tmp_path: Path, body: str = _PLUGIN_SOURCE):
    package = tmp_path / "src" / "acme_private_tito"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(body, encoding="utf-8")
    tito_plugins.register_tito_extension(
        TITORendererExtension(
            name="acme",
            certifications=(
                TITORendererCertification(
                    certification_id="acme@v1",
                    renderer_names=frozenset({"acme_model"}),
                    tokenizer_fingerprint=_FINGERPRINT,
                    renderer_factory=lambda tokenizer, certification: "acme-renderer",
                ),
            ),
            sidecar_source_root=package,
            sidecar_entry_point="acme_private_tito:register",
        )
    )
    return sidecar_runtime.build_sidecar_bundle(_setup(tmp_path))


def test_bundle_ships_and_rebinds_renderer_extensions(tmp_path) -> None:
    bundle = _build_bundle(tmp_path)

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(bundle.path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        archive.extractall(extracted)

    assert "plugins/acme_private_tito/__init__.py" in names
    assert manifest["tito_plugin_entry_points"] == ["acme_private_tito:register"]
    subprocess.run(
        [sys.executable, "-c", _SANDBOX_PROBE],
        check=True,
        cwd=extracted,
        env={
            **os.environ,
            "PYTHONPATH": (
                f"{extracted / 'python-sdk'}:{extracted / 'cookbook'}:"
                f"{extracted / 'plugins'}"
            ),
        },
    )


def test_bundle_digest_covers_extension_sources(tmp_path) -> None:
    first = _build_bundle(tmp_path / "a")
    tito_plugins.reset_tito_renderer_plugins_for_tests()
    second = _build_bundle(
        tmp_path / "b",
        body=_PLUGIN_SOURCE + "\n# a reviewed protocol change\n",
    )

    assert first.digest != second.digest


def test_bundle_without_extensions_records_none(tmp_path) -> None:
    bundle = sidecar_runtime.build_sidecar_bundle(_setup(tmp_path))

    with zipfile.ZipFile(bundle.path) as archive:
        manifest = json.loads(archive.read("manifest.json"))

    assert manifest["tito_plugin_entry_points"] == []
    sidecar_runtime._register_bundled_tito_plugins(())


def test_malformed_bundled_entry_point_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid bundled TITO plugin entry point"):
        sidecar_runtime._register_bundled_tito_plugins(("acme_private_tito",))


def test_entry_point_outside_the_shipped_plugins_is_rejected(tmp_path) -> None:
    """The manifest is not integrity-checked at serve time, so treat it as input.

    Without this bound, a rewritten manifest could name any importable module
    on the sidecar's path and have ``serve`` call an attacker-chosen callable.
    """
    plugins_root = tmp_path / "plugins"
    plugins_root.mkdir()

    with pytest.raises(ValueError, match="is not a package shipped under"):
        sidecar_runtime._register_bundled_tito_plugins(
            ("os:system",),
            plugins_root,
        )


@pytest.mark.parametrize(
    "entry_point",
    ["../evil:register", "/abs/evil:register", "evil/../os:system"],
)
def test_traversal_style_entry_points_are_rejected(tmp_path, entry_point) -> None:
    plugins_root = tmp_path / "plugins"
    plugins_root.mkdir()

    with pytest.raises(ValueError, match="is not a package shipped under"):
        sidecar_runtime._register_bundled_tito_plugins((entry_point,), plugins_root)


def test_shipped_plugin_package_is_imported(tmp_path) -> None:
    plugins_root = tmp_path / "plugins"
    package = plugins_root / "acme_shipped_tito"
    package.mkdir(parents=True)
    marker = tmp_path / "registered.txt"
    (package / "__init__.py").write_text(
        "from pathlib import Path\n\n\n"
        f"def register():\n    Path({str(marker)!r}).write_text('yes')\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(plugins_root))
    try:
        sidecar_runtime._register_bundled_tito_plugins(
            ("acme_shipped_tito:register",),
            plugins_root,
        )
    finally:
        sys.path.remove(str(plugins_root))
        sys.modules.pop("acme_shipped_tito", None)

    assert marker.read_text() == "yes"
