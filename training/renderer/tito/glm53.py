"""GLM-5.3 full-history TITO rendering through its pinned chat template."""

from __future__ import annotations

from typing import Any

from training.renderer.tito.glm52 import GLM52TITORenderer
from training.renderer.tito.shared import TITORendererCertification

GLM53_RENDERER_NAME = "glm53_preserve_thinking"


class GLM53TITORenderer(GLM52TITORenderer):
    def __init__(
        self, tokenizer: Any, *, certification: TITORendererCertification
    ) -> None:
        super().__init__(tokenizer, certification=certification)
        self.renderer_id = GLM53_RENDERER_NAME

    def prepare_incremental_prompt(self, *args: Any, **kwargs: Any) -> None:
        # GLM-5.3 can reorder tool-result blocks; certify full-history only.
        return None
