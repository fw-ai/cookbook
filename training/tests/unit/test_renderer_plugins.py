"""Generic extension discovery tests for the public renderer API."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from training.renderer import plugins


@dataclass(frozen=True)
class _EntryPoint:
    name: str
    value: str
    registrar: object

    def load(self) -> object:
        return self.registrar


def test_load_renderer_plugins_invokes_discovered_registrars(monkeypatch) -> None:
    observed: list[str] = []
    monkeypatch.setattr(plugins, "_plugins_loaded", False)
    monkeypatch.setattr(plugins, "_plugins_loading", False)
    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda *, group: [
            _EntryPoint("second", "pkg:second", lambda: observed.append("second")),
            _EntryPoint("first", "pkg:first", lambda: observed.append("first")),
        ],
    )

    plugins.load_renderer_plugins()
    plugins.load_renderer_plugins()

    assert observed == ["first", "second"]


def test_plugin_renderer_name_resolution_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        plugins,
        "_renderer_name_resolvers",
        [
            lambda model: "extension_a" if model == "vendor/model" else None,
            lambda model: "extension_b" if model == "vendor/model" else None,
        ],
    )

    with pytest.raises(ValueError, match="conflicting renderer names"):
        plugins.resolve_renderer_name_from_plugins("vendor/model")


def test_plugin_renderer_name_resolution_accepts_one_match(monkeypatch) -> None:
    monkeypatch.setattr(
        plugins,
        "_renderer_name_resolvers",
        [
            lambda model: "extension" if model == "vendor/model" else None,
            lambda _model: None,
        ],
    )

    assert plugins.resolve_renderer_name_from_plugins("vendor/model") == "extension"
    assert plugins.resolve_renderer_name_from_plugins("vendor/other") is None
