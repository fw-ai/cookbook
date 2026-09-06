"""Optional renderer extensions supplied by installed Python distributions.

Extensions are discovered through the ``fireworks.training.renderer_plugins``
entry-point group. Each entry point must resolve to a zero-argument callable;
the callable may register renderer factories through :func:`register_renderer`
and model-name resolvers through :func:`register_renderer_name_resolver`.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

RendererNameResolver = Callable[[str], str | None]
RendererImageProcessorLoader = Callable[[str], Any]

_ENTRY_POINT_GROUP = "fireworks.training.renderer_plugins"
_renderer_name_resolvers: list[RendererNameResolver] = []
_renderer_image_processor_loaders: dict[str, RendererImageProcessorLoader] = {}
_renderer_image_support: dict[str, bool] = {}
_renderer_tool_image_support: dict[str, bool] = {}
_plugins_loaded = False
_plugins_loading = False


def register_renderer_name_resolver(resolver: RendererNameResolver) -> None:
    """Register a model-name resolver supplied by an installed extension."""

    if not callable(resolver):
        raise TypeError("renderer name resolver must be callable")
    if resolver not in _renderer_name_resolvers:
        _renderer_name_resolvers.append(resolver)


def resolve_renderer_name_from_plugins(tokenizer_model: str) -> str | None:
    """Return one extension-provided renderer name, detecting conflicts."""

    matches: set[str] = set()
    for resolver in tuple(_renderer_name_resolvers):
        renderer_name = resolver(tokenizer_model)
        if renderer_name is None:
            continue
        renderer_name = renderer_name.strip()
        if not renderer_name:
            raise ValueError("renderer plugin returned an empty renderer name")
        matches.add(renderer_name)
    if len(matches) > 1:
        raise ValueError(
            "renderer plugins returned conflicting renderer names: "
            + ", ".join(sorted(matches))
        )
    return next(iter(matches), None)


def register_renderer_image_capability(
    renderer_name: str,
    image_processor_loader: RendererImageProcessorLoader,
    *,
    supports_images: bool = False,
    supports_tool_images: bool = False,
) -> None:
    """Register image wiring and production readiness for an extension renderer.

    Supplying a loader lets the common builder construct image chunks. The
    support flags are deliberately separate: managed dataset validation must
    stay closed until the selected training backend can consume those chunks.
    """

    if not renderer_name:
        raise ValueError("renderer name must be non-empty")
    if not callable(image_processor_loader):
        raise TypeError("renderer image processor loader must be callable")
    existing = _renderer_image_processor_loaders.get(renderer_name)
    if existing is not None and existing is not image_processor_loader:
        raise ValueError(
            f"renderer {renderer_name!r} already has an image processor loader"
        )
    if supports_tool_images and not supports_images:
        raise ValueError("tool-image support requires general image support")
    _renderer_image_processor_loaders[renderer_name] = image_processor_loader
    _renderer_image_support[renderer_name] = bool(supports_images)
    _renderer_tool_image_support[renderer_name] = bool(supports_tool_images)


def renderer_image_processor_loader(
    renderer_name: str,
) -> RendererImageProcessorLoader | None:
    """Return the extension-owned image processor loader, when registered."""

    return _renderer_image_processor_loaders.get(renderer_name)


def renderer_plugin_supports_images(renderer_name: str) -> bool | None:
    """Return an extension renderer's end-to-end image readiness, if known."""

    return _renderer_image_support.get(renderer_name)


def renderer_plugin_supports_tool_images(renderer_name: str) -> bool | None:
    """Return an extension renderer's tool-image capability, if known."""

    return _renderer_tool_image_support.get(renderer_name)


def load_renderer_plugins() -> None:
    """Load installed renderer extensions once per interpreter."""

    global _plugins_loaded, _plugins_loading
    if _plugins_loaded or _plugins_loading:
        return
    _plugins_loading = True
    try:
        discovered = sorted(
            entry_points(group=_ENTRY_POINT_GROUP),
            key=lambda entry_point: (entry_point.name, entry_point.value),
        )
        for entry_point in discovered:
            register = entry_point.load()
            if not callable(register):
                raise TypeError(
                    f"renderer plugin {entry_point.name!r} must resolve to a callable"
                )
            register()
        _plugins_loaded = True
    finally:
        _plugins_loading = False


__all__ = [
    "RendererImageProcessorLoader",
    "RendererNameResolver",
    "load_renderer_plugins",
    "register_renderer_image_capability",
    "register_renderer_name_resolver",
    "renderer_image_processor_loader",
    "renderer_plugin_supports_images",
    "renderer_plugin_supports_tool_images",
    "resolve_renderer_name_from_plugins",
]
