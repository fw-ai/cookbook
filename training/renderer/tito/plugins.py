"""TITO renderer certifications supplied by installed Python distributions.

Extensions are discovered through the ``fireworks.training.tito_renderer_plugins``
entry-point group. Each entry point must resolve to a zero-argument callable that
registers one :class:`TITORendererExtension`.

This is deliberately separate from ``training.renderer.plugins``: that group
registers training renderers and runs inside the trainer, where Tinker and Torch
are already loaded. TITO renderers also run inside the agent sidecar sandbox,
which ships a stub ``training.renderer`` package and must not import the training
stack. Importing this module must stay as cheap as importing ``tito.shared``.

An extension also declares how it reaches that sandbox. The sidecar bundle is a
content-addressed source zip rather than an installed environment, so each
extension names the package directory to copy into the bundle and the
``module:callable`` entry point the sandbox calls to re-register itself. That
directory must be a self-contained top-level package whose import does not pull
in the training stack.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path

from training.renderer.tito.shared import TITORendererCertification

_ENTRY_POINT_GROUP = "fireworks.training.tito_renderer_plugins"


@dataclass(frozen=True)
class TITORendererExtension:
    """One installed distribution's TITO certifications and sandbox source."""

    name: str
    certifications: tuple[TITORendererCertification, ...]
    sidecar_source_root: Path
    sidecar_entry_point: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TITO extension name must be non-empty")
        if not self.certifications:
            raise ValueError(
                f"TITO extension {self.name!r} registered no certifications"
            )
        object.__setattr__(
            self,
            "certifications",
            tuple(self.certifications),
        )
        seen: set[str] = set()
        for certification in self.certifications:
            if not isinstance(certification, TITORendererCertification):
                raise TypeError(
                    f"TITO extension {self.name!r} registered a non-certification"
                )
            collisions = seen.intersection(certification.renderer_names)
            if collisions:
                raise ValueError(
                    f"TITO extension {self.name!r} registered conflicting renderer "
                    "names: " + ", ".join(sorted(collisions))
                )
            seen.update(certification.renderer_names)
        source_root = Path(self.sidecar_source_root)
        object.__setattr__(self, "sidecar_source_root", source_root)
        if not source_root.is_dir():
            raise ValueError(
                f"TITO extension {self.name!r} sidecar source root is not a "
                f"directory: {source_root}"
            )
        if not (source_root / "__init__.py").is_file():
            raise ValueError(
                f"TITO extension {self.name!r} sidecar source root is not a "
                f"package: {source_root}"
            )
        module, separator, attribute = self.sidecar_entry_point.partition(":")
        if not separator or not module or not attribute:
            raise ValueError(
                f"TITO extension {self.name!r} sidecar entry point must be "
                f"'module:callable', got {self.sidecar_entry_point!r}"
            )
        if module.partition(".")[0] != source_root.name:
            # The bundle copies the directory verbatim onto the sandbox path, so
            # the top-level module name and the directory name are one contract.
            raise ValueError(
                f"TITO extension {self.name!r} sidecar entry point {module!r} is "
                f"not importable from its source directory {source_root.name!r}"
            )

    def renderer_names(self) -> frozenset[str]:
        return frozenset(
            name
            for certification in self.certifications
            for name in certification.renderer_names
        )


_extensions: dict[str, TITORendererExtension] = {}
_plugins_loaded = False
_plugins_loading = False


def register_tito_extension(extension: TITORendererExtension) -> None:
    """Register one installed distribution's TITO renderer certifications."""

    if not isinstance(extension, TITORendererExtension):
        raise TypeError("TITO extension must be a TITORendererExtension")
    existing = _extensions.get(extension.name)
    if existing is not None:
        if existing == extension:
            return
        raise ValueError(
            f"TITO extension {extension.name!r} is already registered with a "
            "different definition"
        )
    claimed = extension.renderer_names()
    for other in _extensions.values():
        collisions = claimed.intersection(other.renderer_names())
        if collisions:
            raise ValueError(
                f"TITO extensions {extension.name!r} and {other.name!r} claim "
                "the same renderer names: " + ", ".join(sorted(collisions))
            )
    _extensions[extension.name] = extension


def load_tito_renderer_plugins() -> None:
    """Load installed TITO renderer extensions once per interpreter."""

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
                    f"TITO renderer plugin {entry_point.name!r} must resolve to a "
                    "callable"
                )
            register()
        _plugins_loaded = True
    finally:
        _plugins_loading = False


def registered_tito_extensions() -> tuple[TITORendererExtension, ...]:
    """Return every registered extension in a deterministic order."""

    return tuple(_extensions[name] for name in sorted(_extensions))


def extension_certifications() -> dict[str, TITORendererCertification]:
    """Map every extension-provided renderer name to its certification."""

    resolved: dict[str, TITORendererCertification] = {}
    for extension in registered_tito_extensions():
        for certification in extension.certifications:
            for renderer_name in certification.renderer_names:
                resolved[renderer_name] = certification
    return resolved


def reset_tito_renderer_plugins_for_tests(
    extensions: Iterable[TITORendererExtension] = (),
) -> None:
    """Replace the process-wide extension state; tests only."""

    global _plugins_loaded, _plugins_loading
    _extensions.clear()
    _plugins_loaded = True
    _plugins_loading = False
    for extension in extensions:
        register_tito_extension(extension)


__all__ = [
    "TITORendererExtension",
    "extension_certifications",
    "load_tito_renderer_plugins",
    "register_tito_extension",
    "registered_tito_extensions",
    "reset_tito_renderer_plugins_for_tests",
]
