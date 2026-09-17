"""Production TITO renderer certification registry.

Each entry pins one renderer implementation to one tokenizer contract
(backend + chat template + special tokens fingerprint). Adding a model means
adding a per-model module next to this file plus one entry here.
"""

from __future__ import annotations

from typing import Any

from fireworks.training.sdk import TITORenderer

from training.renderer.tito import shared
from training.renderer.tito.glm52 import GLM52_RENDERER_NAME, GLM52TITORenderer
from training.renderer.tito.qwen38 import QWEN38_RENDERER_NAME, Qwen38TITORenderer
from training.renderer.tito.shared import TITORendererCertification


def _build_glm52_tito_renderer(
    tokenizer: Any,
    certification: TITORendererCertification,
) -> TITORenderer:
    return GLM52TITORenderer(tokenizer, certification=certification)


def _build_qwen38_tito_renderer(
    tokenizer: Any,
    certification: TITORendererCertification,
) -> TITORenderer:
    return Qwen38TITORenderer(tokenizer, certification=certification)


_TITO_RENDERER_CERTIFICATIONS = (
    TITORendererCertification(
        certification_id="glm-5.2-preserved@b4734de4-v7",
        renderer_names=frozenset({GLM52_RENDERER_NAME}),
        tokenizer_fingerprint=(
            "5591741bd28d5acb92d4b7d735e0084d4d76d9ce50e2afe99aec6b01e1ef3ef0"
        ),
        renderer_factory=_build_glm52_tito_renderer,
    ),
    TITORendererCertification(
        certification_id="qwen3.8-27b-preserved@1d4bf0f2-v1",
        renderer_names=frozenset({QWEN38_RENDERER_NAME}),
        tokenizer_fingerprint=(
            "90e8dd75a5fa5c8009f981975336de7177bb5ab04d41940e49c9a6e2d37325c7"
        ),
        renderer_factory=_build_qwen38_tito_renderer,
    ),
)
_TITO_CERTIFICATION_BY_RENDERER = {
    renderer_name: certification
    for certification in _TITO_RENDERER_CERTIFICATIONS
    for renderer_name in certification.renderer_names
}


def get_tito_renderer_certification(
    renderer_name: str,
    tokenizer: Any,
) -> TITORendererCertification:
    """Resolve and verify the source-controlled production artifact."""
    certification = _TITO_CERTIFICATION_BY_RENDERER.get(renderer_name)
    if certification is None:
        raise ValueError(
            f"renderer {renderer_name!r} has no production TITO certification"
        )
    actual = shared._tokenizer_fingerprint(tokenizer)
    if actual != certification.tokenizer_fingerprint:
        raise ValueError(
            "tokenizer does not match TITO certification "
            f"{certification.certification_id!r}"
        )
    return certification


def build_sidecar_tito_renderer(
    tokenizer: Any,
    renderer_name: str,
) -> TITORenderer:
    """Build a renderer admitted to the lightweight agent-sidecar runtime."""
    certification = get_tito_renderer_certification(renderer_name, tokenizer)
    return certification.renderer_factory(tokenizer, certification)
