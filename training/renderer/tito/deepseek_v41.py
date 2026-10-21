"""Text-only V4.1 prompts from the pinned template, output from its native parser."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from fireworks.training.sdk import TITOChatRequest, TITOError

from training.renderer.tito.deepseek_v4 import DeepSeekV4TITORenderer

DEEPSEEK_V41_RENDERER_NAME = "deepseek_v41_flash"
DEEPSEEK_V41_REPOSITORY = "deepseek-ai/DeepSeek-V4.1-Flash"
DEEPSEEK_V41_REVISION = "2cba9e42aa026125f3ed06c6d98c1db82f7ca027"
# Include the declared BOS/EOS post-processor used by serving and recipe loading.
DEEPSEEK_V41_TOKENIZER_FINGERPRINT = "e3736d43b24b355ca5552875d19cf04524d5ea6b424a362a112692e61189ce75"
DEEPSEEK_V41_ENCODER_FILE = "encoding/encoding.py"
DEEPSEEK_V41_ENCODER_SHA256 = "502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1"


class DeepSeekV41TITORenderer(DeepSeekV4TITORenderer):
    """Fixed thinking/high-effort (75), preserved-history function-tool contract."""

    encoder_file = DEEPSEEK_V41_ENCODER_FILE
    reasoning_effort = "high"

    def _template_messages(self, request: TITOChatRequest) -> list[dict[str, Any]]:
        wire = request.wire_value()
        tools = (wire.get("tools") or ()) if wire is not None else request.tools
        if any((tool.get("function") or {}).get("namespace") for tool in tools):
            raise TITOError("tito_invalid_request", 400, "DeepSeek V4.1 TITO does not support tool namespaces")
        messages = super()._template_messages(request)
        for message in messages:
            for field in ("content", "reasoning_content"):
                text = message.get(field) or ""
                if any(marker in text for marker in (self._encoder.IMAGE_PLACEHOLDER, "<image>", "</image>")):
                    raise TITOError("tito_invalid_request", 400, "DeepSeek V4.1 TITO supports text only")
            for item in (*message.get("tools", ()), *message.get("tool_calls", ())):
                function = item.get("function") or {}
                if item.get("namespace") or function.get("namespace") or "::" in function.get("name", ""):
                    raise TITOError("tito_invalid_request", 400, "DeepSeek V4.1 TITO does not support tool namespaces")
        return messages

    def render_conversation_tokens(self, request: TITOChatRequest) -> Sequence[int]:
        messages = self._template_messages(request)
        # The Jinja contract accepts mappings, while the native parser and
        # OpenAI wire contract use JSON strings. Only the template copy changes.
        for message in messages:
            for call in message.get("tool_calls") or ():
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        return tuple(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                thinking_mode="thinking",
                reasoning_effort=self.reasoning_effort,
                drop_thinking=False,
            )
        )
