"""Production TITO renderer certification registry.

Each entry pins one renderer implementation to one tokenizer contract
(backend + chat template + special tokens fingerprint). Adding a model means
adding a per-model module next to this file plus one entry here.

Additional model certifications can be registered by an installed distribution
through ``training.renderer.tito.plugins``. Built-in
certifications always win, and an extension that claims a built-in renderer name
is rejected rather than silently ignored.
"""

from __future__ import annotations

from typing import Any

from fireworks.training.sdk import TITORenderer

from training.renderer.tito import shared
from training.renderer.tito.deepseek_v4 import (
    DEEPSEEK_V4_ENCODER_FILE,
    DEEPSEEK_V4_ENCODER_SHA256,
    DEEPSEEK_V4_PRO_RENDERER_NAME,
    DEEPSEEK_V4_PRO_REPOSITORY,
    DEEPSEEK_V4_PRO_REVISION,
    DEEPSEEK_V4_RENDERER_NAME,
    DEEPSEEK_V4_REPOSITORY,
    DEEPSEEK_V4_REVISION,
    DEEPSEEK_V4_TOKENIZER_FINGERPRINT,
    DeepSeekV4TITORenderer,
)
from training.renderer.tito.deepseek_v41 import (
    DEEPSEEK_V41_ENCODER_FILE,
    DEEPSEEK_V41_ENCODER_SHA256,
    DEEPSEEK_V41_RENDERER_NAME,
    DEEPSEEK_V41_REPOSITORY,
    DEEPSEEK_V41_REVISION,
    DEEPSEEK_V41_TOKENIZER_FINGERPRINT,
    DeepSeekV41TITORenderer,
)
from training.renderer.tito.plugins import (
    extension_certifications,
    load_tito_renderer_plugins,
)
from training.renderer.tito.glm52 import GLM52_RENDERER_NAME, GLM52TITORenderer
from training.renderer.tito.glm53 import GLM53_RENDERER_NAME, GLM53TITORenderer
from training.renderer.tito.glm53_flash import (
    GLM53_FLASH_RENDERER_NAME,
    GLM53_FLASH_TOKENIZER_FINGERPRINT,
    GLM53FlashTITORenderer,
)
from training.renderer.tito.muse_glimmer import (
    MUSE_GLIMMER_RENDERER_NAME,
    MuseGlimmerTITORenderer,
)
from training.renderer.tito.qwen35 import (
    QWEN35_RENDERER_NAME,
    QWEN35_TOKENIZER_FINGERPRINT,
    Qwen35TITORenderer,
)
from training.renderer.tito.qwen36 import (
    QWEN36_RENDERER_NAME,
    QWEN36_TOKENIZER_FINGERPRINT,
    Qwen36TITORenderer,
)
from training.renderer.tito.qwen38 import QWEN38_RENDERER_NAME, Qwen38TITORenderer
from training.renderer.tito.shared import TITORendererCertification


def _build_glm52_tito_renderer(
    tokenizer: Any,
    certification: TITORendererCertification,
) -> TITORenderer:
    return GLM52TITORenderer(tokenizer, certification=certification)


def _build_glm53_tito_renderer(
    tokenizer: Any,
    certification: TITORendererCertification,
) -> TITORenderer:
    return GLM53TITORenderer(tokenizer, certification=certification)


def _build_glm53_flash_tito_renderer(
    tokenizer: Any,
    certification: TITORendererCertification,
) -> TITORenderer:
    return GLM53FlashTITORenderer(tokenizer, certification=certification)


def _build_muse_glimmer_tito_renderer(
    tokenizer: Any,
    certification: TITORendererCertification,
) -> TITORenderer:
    return MuseGlimmerTITORenderer(tokenizer, certification=certification)


def _build_qwen35_tito_renderer(
    tokenizer: Any,
    certification: TITORendererCertification,
) -> TITORenderer:
    return Qwen35TITORenderer(tokenizer, certification=certification)


def _build_qwen36_tito_renderer(
    tokenizer: Any,
    certification: TITORendererCertification,
) -> TITORenderer:
    return Qwen36TITORenderer(tokenizer, certification=certification)


def _build_qwen38_tito_renderer(
    tokenizer: Any,
    certification: TITORendererCertification,
) -> TITORenderer:
    return Qwen38TITORenderer(tokenizer, certification=certification)


def _build_deepseek_v4_tito_renderer(tokenizer: Any, certification: TITORendererCertification) -> TITORenderer:
    return DeepSeekV4TITORenderer(tokenizer, certification=certification)


def _build_deepseek_v41_tito_renderer(tokenizer: Any, certification: TITORendererCertification) -> TITORenderer:
    return DeepSeekV41TITORenderer(tokenizer, certification=certification)


_TITO_RENDERER_CERTIFICATIONS = (
    TITORendererCertification(
        certification_id="deepseek-v4.1-flash@2cba9e42-full-history-v1",
        renderer_names=frozenset({DEEPSEEK_V41_RENDERER_NAME}),
        tokenizer_fingerprint=DEEPSEEK_V41_TOKENIZER_FINGERPRINT,
        renderer_factory=_build_deepseek_v41_tito_renderer,
        artifact_repository=DEEPSEEK_V41_REPOSITORY,
        artifact_revision=DEEPSEEK_V41_REVISION,
        artifact_files=((DEEPSEEK_V41_ENCODER_FILE, DEEPSEEK_V41_ENCODER_SHA256),),
    ),
    TITORendererCertification(
        certification_id="deepseek-v4-pro-0813@72e1d323-full-history-v1",
        renderer_names=frozenset({DEEPSEEK_V4_PRO_RENDERER_NAME}),
        tokenizer_fingerprint=DEEPSEEK_V4_TOKENIZER_FINGERPRINT,
        renderer_factory=_build_deepseek_v4_tito_renderer,
        artifact_repository=DEEPSEEK_V4_PRO_REPOSITORY,
        artifact_revision=DEEPSEEK_V4_PRO_REVISION,
        artifact_files=((DEEPSEEK_V4_ENCODER_FILE, DEEPSEEK_V4_ENCODER_SHA256),),
    ),
    TITORendererCertification(
        certification_id="glm-5.3-flash-preserved@690b7052-full-history-v1",
        renderer_names=frozenset({GLM53_FLASH_RENDERER_NAME}),
        tokenizer_fingerprint=GLM53_FLASH_TOKENIZER_FINGERPRINT,
        renderer_factory=_build_glm53_flash_tito_renderer,
    ),
    TITORendererCertification(
        certification_id="deepseek-v4-flash-0731@7872f01b-full-history-v1",
        renderer_names=frozenset({DEEPSEEK_V4_RENDERER_NAME}),
        tokenizer_fingerprint=DEEPSEEK_V4_TOKENIZER_FINGERPRINT,
        renderer_factory=_build_deepseek_v4_tito_renderer,
        artifact_repository=DEEPSEEK_V4_REPOSITORY,
        artifact_revision=DEEPSEEK_V4_REVISION,
        artifact_files=((DEEPSEEK_V4_ENCODER_FILE, DEEPSEEK_V4_ENCODER_SHA256),),
    ),
    TITORendererCertification(
        certification_id="glm-5.3-preserved@935644c0-experimental-r5",
        renderer_names=frozenset({GLM53_RENDERER_NAME}),
        tokenizer_fingerprint=(
            "add5eacab6a81583dd16eb18ea02ff43c4342aa32917a60a6f8fb55ad813c2cf"
        ),
        renderer_factory=_build_glm53_tito_renderer,
    ),
    TITORendererCertification(
        certification_id="glm-5.2-preserved@b4734de4-v8",
        renderer_names=frozenset({GLM52_RENDERER_NAME}),
        tokenizer_fingerprint=(
            "5591741bd28d5acb92d4b7d735e0084d4d76d9ce50e2afe99aec6b01e1ef3ef0"
        ),
        renderer_factory=_build_glm52_tito_renderer,
    ),
    TITORendererCertification(
        certification_id="muse-glimmer-30b-preserved@a4e59da5-v1",
        renderer_names=frozenset({MUSE_GLIMMER_RENDERER_NAME}),
        tokenizer_fingerprint=(
            "2fec80a849b8cb52120e297e001ce675c0ee1b0de067aed449d7f1e756b04e3a"
        ),
        renderer_factory=_build_muse_glimmer_tito_renderer,
    ),
    TITORendererCertification(
        certification_id="qwen3.5-35b-a3b-fp8@5a3b7d66-v1",
        renderer_names=frozenset({QWEN35_RENDERER_NAME}),
        tokenizer_fingerprint=QWEN35_TOKENIZER_FINGERPRINT,
        renderer_factory=_build_qwen35_tito_renderer,
    ),
    TITORendererCertification(
        certification_id="qwen3.6-35b-a3b-preserved@995ad96e-candidate",
        renderer_names=frozenset({QWEN36_RENDERER_NAME}),
        tokenizer_fingerprint=QWEN36_TOKENIZER_FINGERPRINT,
        renderer_factory=_build_qwen36_tito_renderer,
    ),
    TITORendererCertification(
        certification_id="qwen3.8-27b-preserved@1d4bf0f2-v2",
        renderer_names=frozenset({QWEN38_RENDERER_NAME}),
        # v2: same model revision and identical tokens/offsets, re-pinned
        # because load_tokenizer now honors the declared add_bos_token like
        # serving does, which rebuilds the (behaviorally equivalent)
        # post-processor and therefore the serialized backend hash.
        tokenizer_fingerprint=(
            "572e8b1a43a756b093105d29a62ebbc4cda02aae8870b5009089430a8405de49"
        ),
        renderer_factory=_build_qwen38_tito_renderer,
    ),
)
_TITO_CERTIFICATION_BY_RENDERER = {
    renderer_name: certification
    for certification in _TITO_RENDERER_CERTIFICATIONS
    for renderer_name in certification.renderer_names
}


def _resolve_certification(renderer_name: str) -> TITORendererCertification | None:
    """Resolve a built-in certification, then installed extensions."""
    certification = _TITO_CERTIFICATION_BY_RENDERER.get(renderer_name)
    if certification is not None:
        return certification
    load_tito_renderer_plugins()
    from_extensions = extension_certifications()
    shadowed = sorted(from_extensions.keys() & _TITO_CERTIFICATION_BY_RENDERER.keys())
    if shadowed:
        raise ValueError(
            "TITO extensions may not redefine built-in certifications: "
            + ", ".join(shadowed)
        )
    return from_extensions.get(renderer_name)


def get_tito_renderer_certification(
    renderer_name: str,
    tokenizer: Any,
) -> TITORendererCertification:
    """Resolve and verify the source-controlled production artifact."""
    certification = _resolve_certification(renderer_name)
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
