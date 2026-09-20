"""Model rendering and parsing for TITO, one module per certified model.

Prompt construction delegates to the pinned tokenizer's authoritative chat
template. This package owns only protocol normalization plus per-model output
parsers; importing it must not load Tinker or Torch. Full-history rendering is
the production default. Incremental rendering is experimental and requires a
model/template-specific suffix-and-junction implementation.

Layout: ``shared`` holds the certification dataclass, tokenizer fingerprint,
and template normalization; ``<model>.py`` holds each certified renderer;
``registry`` owns the production certification table; ``plugins`` accepts
certifications from installed distributions whose protocol cannot ship here.
"""

from training.renderer.tito.glm52 import GLM52TITORenderer
from training.renderer.tito.glm53 import GLM53TITORenderer
from training.renderer.tito.muse_glimmer import MuseGlimmerTITORenderer
from training.renderer.tito.plugins import (
    TITORendererExtension,
    load_tito_renderer_plugins,
    register_tito_extension,
    registered_tito_extensions,
)
from training.renderer.tito.qwen38 import Qwen38TITORenderer
from training.renderer.tito.registry import (
    build_sidecar_tito_renderer,
    get_tito_renderer_certification,
)
from training.renderer.tito.shared import (
    TITORendererCertification,
    load_sidecar_tokenizer,
)

__all__ = [
    "GLM52TITORenderer",
    "GLM53TITORenderer",
    "MuseGlimmerTITORenderer",
    "Qwen38TITORenderer",
    "TITORendererCertification",
    "TITORendererExtension",
    "build_sidecar_tito_renderer",
    "get_tito_renderer_certification",
    "load_sidecar_tokenizer",
    "load_tito_renderer_plugins",
    "register_tito_extension",
    "registered_tito_extensions",
]
