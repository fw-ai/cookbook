"""Pinned Qwen3.5-35B-A3B preserved-thinking TITO renderer.

The chat template uses the same generation prompt (``<think>\\n``) and the same
``<tool_call><function=name><parameter=key>`` wire format as Qwen3.8, so parsing
is the Qwen3.8 renderer. This module only pins the Qwen3.5 tokenizer contract.
Full-history rendering only; incremental joins are not certified.
"""

from __future__ import annotations

from typing import Any

from training.renderer.tito.qwen38 import Qwen38TITORenderer
from training.renderer.tito.shared import TITORendererCertification

QWEN35_RENDERER_NAME = "qwen3_5"
QWEN35_TOKENIZER_FINGERPRINT = (
    "5a3b7d661540e4208c135cce5d34737c845d1e2d26458c9544d1a5c40bce92ad"
)


class Qwen35TITORenderer(Qwen38TITORenderer):
    """Qwen3.5-35B-A3B full-history TITO renderer."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        certification: TITORendererCertification,
    ) -> None:
        super().__init__(tokenizer, certification=certification)
        self.renderer_id = QWEN35_RENDERER_NAME
        self._label = "Qwen3.5"
