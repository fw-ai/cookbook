"""Qwen3.6-35B-A3B full-history, preserved-thinking TITO rendering."""

from __future__ import annotations

from typing import Any

from training.renderer.tito.qwen38 import Qwen38TITORenderer
from training.renderer.tito.shared import TITORendererCertification

QWEN36_RENDERER_NAME = "qwen3_6_preserved"
QWEN36_REPOSITORY = "Qwen/Qwen3.6-35B-A3B"
QWEN36_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
QWEN36_TOKENIZER_FINGERPRINT = "67d2bb2f9adc67c31beffe22b3b3f4bad530d5a1b22e6ab9e15fcb6e357d20a7"


class Qwen36TITORenderer(Qwen38TITORenderer):
    """Reuse Qwen XML parsing with Qwen3.6's explicitly preserved history."""

    def __init__(self, tokenizer: Any, *, certification: TITORendererCertification) -> None:
        super().__init__(tokenizer, certification=certification)
        self.renderer_id = QWEN36_RENDERER_NAME
        self._label = "Qwen3.6"

    def _render_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any:
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=True,
            preserve_thinking=True,
        )
