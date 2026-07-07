"""Renderer for Moonshot AI's Kimi K3 chat format.

Kimi K3 uses a structured, self-delimiting chat format that is unrelated to
the K2.x ``<|im_*|>`` / ``<think>`` families. Every span is opened with
``<|open|>``, separated from its body by ``<|sep|>``, and closed with
``<|close|><name>``; each top-level message ends with ``<|end_of_msg|>``.

Text roles render (validated byte-for-byte against the tokenizer's own
``apply_chat_template``)::

    system / user:
        <|open|>message role="<role>"<|sep|><content><|close|>message<|sep|><|end_of_msg|>

    assistant (history + SFT target; thinking is always stripped from
    history, so only the ``response`` section is emitted):
        <|open|>message role="assistant"<|sep|><|open|>response<|sep|><content>
        <|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>

The generation suffix that primes sampling differs by thinking mode:

    thinking (default):
        <|open|>message role="assistant"<|sep|><|open|>think<|sep|>
    disable-thinking (``kimi_k3_disable_thinking``, HF ``thinking=False``):
        <|open|>message role="assistant"<|sep|><|open|>response<|sep|>

Because thinking is *uniformly* stripped from history (unlike Qwen3/Kimi-K2.5
which conditionally keep the terminal turn's reasoning), every message renders
identically regardless of position and each conversation prefix is a strict
token prefix of the next. The default concatenating ``build_supervised_example``
/ ``build_generation_prompt`` are therefore exact, and multi-turn
``ALL_ASSISTANT_MESSAGES`` SFT is well defined.

Out of scope for this renderer (the coding/text SFT path does not use them and
the K3 tool/vision encodings are substantially more involved — see
``encoding_k3.py`` / ``kimi_k3_processor.py``):

- tool declarations (``role="system" type="tool-declare"``) and assistant
  ``tool_calls`` / ``role="tool"`` results (the nested
  ``<|open|>tools ... call ... argument ...`` structure).
- multimodal / image content parts.

Callers that need those should render via the upstream tokenizer directly until
this renderer is extended.
"""

from __future__ import annotations

import tinker

from tinker_cookbook.image_processing_utils import ImageProcessor
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    get_text_content,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

_OPEN = "<|open|>"
_SEP = "<|sep|>"
_CLOSE = "<|close|>"
_EOM = "<|end_of_msg|>"


class KimiK3Renderer(Renderer):
    """Kimi K3 renderer with thinking enabled (HF default).

    The generation suffix opens a ``think`` section so the model reasons before
    responding. Historical assistant turns are always response-only.
    """

    # Sampling primes a ``think`` section; ``response`` for the disable variant.
    _assistant_gen_section: str = "think"

    def __init__(self, tokenizer: Tokenizer, image_processor: ImageProcessor | None = None):
        super().__init__(tokenizer)
        self.image_processor = image_processor

    @property
    def has_extension_property(self) -> bool:
        # Thinking is uniformly stripped from history, so each conversation
        # prefix's tokens are a strict prefix of the next (empirically verified).
        return True

    @property
    def _bos_tokens(self) -> list[int]:
        # K3's apply_chat_template emits no leading BOS.
        return []

    def _encode(self, text: str) -> list[int]:
        # ``allowed_special="all"`` makes the structural markers (<|open|> etc.)
        # tokenize to their single special-token ids, matching apply_chat_template.
        try:
            return self.tokenizer.encode(text, add_special_tokens=False, allowed_special="all")
        except TypeError:
            # HF fast/slow tokenizers don't take ``allowed_special`` but already
            # treat registered added-special tokens atomically.
            return self.tokenizer.encode(text, add_special_tokens=False)

    def get_stop_sequences(self) -> list[str]:
        return [_EOM]

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]
        content = get_text_content(message)
        if role == "assistant":
            header = f'{_OPEN}message role="assistant"{_SEP}{_OPEN}response{_SEP}'
            output = f"{content}{_CLOSE}response{_SEP}{_CLOSE}message{_SEP}{_EOM}"
        else:
            header = f'{_OPEN}message role="{role}"{_SEP}'
            output = f"{content}{_CLOSE}message{_SEP}{_EOM}"
        return RenderedMessage(
            header=tinker.types.EncodedTextChunk(tokens=self._encode(header)),
            output=[tinker.types.EncodedTextChunk(tokens=self._encode(output))],
        )

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        if role == "assistant":
            return self._encode(
                f'{_OPEN}message role="assistant"{_SEP}{_OPEN}{self._assistant_gen_section}{_SEP}'
            )
        return self._encode(f'{_OPEN}message role="{role}"{_SEP}')

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        text = self.tokenizer.decode(self._normalize_response_tokens(response))
        termination = ParseTermination.EOS
        for marker in (f"{_CLOSE}response", _EOM, f"{_CLOSE}message"):
            if marker in text:
                text = text.split(marker, 1)[0]
                termination = ParseTermination.STOP_SEQUENCE
                break
        # Drop a leading think and/or response section opener if present so the
        # returned content is just the assistant's response text.
        for opener in (f"{_OPEN}think{_SEP}", f"{_OPEN}response{_SEP}"):
            if opener in text:
                text = text.split(opener, 1)[1]
        return Message(role="assistant", content=text.strip()), termination


class KimiK3DisableThinkingRenderer(KimiK3Renderer):
    """Kimi K3 renderer with thinking disabled (HF ``thinking=False``).

    Identical history rendering; the only difference is that the generation
    suffix primes a ``response`` section directly instead of ``think``.
    """

    _assistant_gen_section: str = "response"


def _kimi_k3_factory(
    tokenizer: Tokenizer, image_processor: ImageProcessor | None = None
) -> KimiK3Renderer:
    return KimiK3Renderer(tokenizer, image_processor=image_processor)


def _kimi_k3_disable_thinking_factory(
    tokenizer: Tokenizer, image_processor: ImageProcessor | None = None
) -> KimiK3DisableThinkingRenderer:
    return KimiK3DisableThinkingRenderer(tokenizer, image_processor=image_processor)


register_renderer("kimi_k3", _kimi_k3_factory)
register_renderer("kimi_k3_disable_thinking", _kimi_k3_disable_thinking_factory)
