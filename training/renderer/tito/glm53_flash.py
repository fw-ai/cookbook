from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fireworks.training.sdk import TITOChatRequest, TITOError

from training.renderer.tito.glm52 import GLM52TITORenderer
from training.renderer.tito.shared import TITORendererCertification

GLM53_FLASH_RENDERER_NAME = "glm53_flash_preserve_thinking"
GLM53_FLASH_REPOSITORY = "zai-org/GLM-5.3-Flash"
GLM53_FLASH_REVISION = "690b705278a3a58e538fcb37c2ca8b5f9511213c"
GLM53_FLASH_TOKENIZER_FINGERPRINT = "3ca8f70498142219fa2bf835e20440c32a58d63624a96e05024ce6881ab3b6bf"


class GLM53FlashTITORenderer(GLM52TITORenderer):
    def __init__(self, tokenizer: Any, *, certification: TITORendererCertification) -> None:
        super().__init__(tokenizer, certification=certification)
        self.renderer_id = GLM53_FLASH_RENDERER_NAME

    def prepare_incremental_prompt(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _template_messages(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        normalized = super()._template_messages(messages)
        for message in normalized:
            for call in message.get("tool_calls") or ():
                function = call.get("function") or {}
                if not isinstance(function.get("arguments"), Mapping):
                    raise TITOError(
                        "tito_invalid_request", 400, "historical assistant tool arguments must be JSON objects"
                    )
        return normalized

    def _validate_request(self, request: TITOChatRequest) -> None:
        super()._validate_request(request)
        wire = request.wire_value()
        messages = wire.get("messages", ()) if wire is not None else request.messages
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise TITOError("tito_invalid_request", 400, "unsupported GLM-5.3 Flash role")
            content = message.get("content")
            if content is None or isinstance(content, str):
                continue
            if isinstance(content, Sequence) and not isinstance(content, bytes):
                if all(
                    isinstance(part, Mapping) and part.get("type") == "text" and isinstance(part.get("text"), str)
                    for part in content
                ):
                    continue
            raise TITOError("tito_invalid_request", 400, "GLM-5.3 Flash TITO supports text only")
