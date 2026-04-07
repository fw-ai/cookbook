"""Renderer for Google Gemma 4 instruct models.

Built against the official ``chat_template.jinja`` shipped with
``google/gemma-4-E2B-it`` (and the other gemma-4 *-it variants on the
HuggingFace Hub). The earlier version of this file was built against an
unofficial dev fixture and was wrong on multiple fundamental dimensions
(role name, generation suffix, content trimming, tool format) — see
``training/renderer/README.md`` for the postmortem.

Format produced for a basic conversation::

    <bos><|turn>system
    {trimmed system content}<turn|>
    <|turn>user
    {trimmed user content}<turn|>
    <|turn>model
    {trimmed model content}<turn|>
    <|turn>user
    {trimmed user content}<turn|>
    <|turn>model    <- generation prompt

Key facts (all derived from the official Jinja template, not from the
model card or the older dev fixture):

* The assistant role is rendered as ``model``, not ``assistant``. Other
  roles (``system``, ``user``, ``tool``, ``developer``) are emitted
  verbatim.
* Every message's text content is run through Jinja's ``| trim`` filter,
  i.e. ``str.strip()``. Leading and trailing whitespace never reach the
  tokenizer.
* Each turn produces ``<|turn>{role}\\n{trimmed_content}<turn|>\\n``. The
  trailing ``\\n`` after ``<turn|>`` is the inter-turn separator and is
  part of every closed turn.
* The generation prompt appended at the end is ``<|turn>model\\n``.
* The model role's content is additionally passed through the template's
  ``strip_thinking`` macro, which removes ``<|channel>...<channel|>``
  blocks from history. We replicate this for parity.
* This renderer covers the base text-only path. Tool definitions
  (``tools=`` argument), ``message['tool_calls']`` /
  ``message['tool_responses']`` fields, ``enable_thinking=True``, and
  multimodal (image/audio/video) parts are NOT yet implemented; see the
  README for the explicit gap list.

Special-token IDs (Gemma 4 tokenizer)::

    <bos>     = 2
    <|turn>   = 105      <turn|>   = 106
    \\n        = 107
    model token (string ``model``) varies by tokenizer
"""

from __future__ import annotations

import re

import tinker
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    TextPart,
    ThinkingPart,
    parse_response_for_stop_token,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

_BOS_TOKEN = "<bos>"
_TURN_OPEN = "<|turn>"
_TURN_CLOSE = "<turn|>"
_CHANNEL_OPEN = "<|channel>"
_CHANNEL_CLOSE = "<channel|>"
_MODEL_ROLE = "model"

# Pattern matching the template's strip_thinking macro: remove every
# `<|channel>...<channel|>` block from a string and `.strip()` what's left.
_THINKING_BLOCK_RE = re.compile(
    re.escape(_CHANNEL_OPEN) + r".*?" + re.escape(_CHANNEL_CLOSE),
    re.DOTALL,
)


def _strip_thinking(text: str) -> str:
    """Match the template's ``strip_thinking`` macro: drop ``<|channel>...<channel|>``
    blocks and ``.strip()`` the result."""
    return _THINKING_BLOCK_RE.sub("", text).strip()


def _render_content(message: Message) -> str:
    """Flatten message content (string or content-parts) and apply the
    template's per-role text transformation.

    For ``role=model`` (== assistant) every text chunk is passed through
    ``strip_thinking``. For all other roles every text chunk is plain
    ``str.strip()``-ed. The result is the joined text.

    Tool calls / tool responses / image / audio / video parts are NOT
    yet handled — they fall through silently. See README for the gap.
    """
    is_model = message["role"] == "assistant"
    transform = _strip_thinking if is_model else str.strip
    content = message["content"]
    if isinstance(content, str):
        return transform(content)
    parts: list[str] = []
    for part in content:
        if part["type"] == "text":
            parts.append(transform(part["text"]))
        # Other part types intentionally not handled in this base version.
    return "".join(parts)


class Gemma4Renderer(Renderer):
    """Renderer matching the official Gemma 4 chat template byte-for-byte
    for the base text-only path.

    Thread safety: instances are immutable after construction and safe to
    share across threads. The underlying tokenizer must also be thread-safe.
    """

    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer)
        self._bos_token_id = self._encode_single_special(_BOS_TOKEN)
        self._turn_close_id = self._encode_single_special(_TURN_CLOSE)
        # Validated for fail-fast even though the IDs aren't read elsewhere:
        # an unexpected multi-token encoding signals the wrong tokenizer.
        self._encode_single_special(_TURN_OPEN)

    def _encode_single_special(self, token_str: str) -> int:
        token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
        assert len(token_ids) == 1, (
            f"Expected single token for {token_str!r}, got {token_ids}. "
            "This tokenizer does not look like a Gemma 4 tokenizer."
        )
        return token_ids[0]

    @property
    def has_extension_property(self) -> bool:
        # No history rewriting beyond the per-message strip_thinking, which
        # only ever shortens content, so the prefix-extension property holds
        # at the conversation-prefix level used by build_supervised_example.
        return True

    @property
    def _bos_tokens(self) -> list[int]:
        return [self._bos_token_id]

    @property
    def _end_message_token(self) -> int:
        return self._turn_close_id

    @staticmethod
    def _gemma_role_for(message: Message) -> str:
        """Map the message role to the literal string emitted in the turn header."""
        return _MODEL_ROLE if message["role"] == "assistant" else message["role"]

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = self._gemma_role_for(message)
        text = _render_content(message)
        header_str = f"{_TURN_OPEN}{role}\n"
        body_str = f"{text}{_TURN_CLOSE}\n"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False),
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(body_str, add_special_tokens=False),
            ),
        ]
        return RenderedMessage(header=header, output=output)

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        # The official template emits exactly `<|turn>model\n` when
        # add_generation_prompt is true (and prev_message_type is not
        # tool_response, an edge case we don't handle yet).
        suffix_str = f"{_TURN_OPEN}{_MODEL_ROLE}\n"
        return self.tokenizer.encode(suffix_str, add_special_tokens=False)

    def get_stop_sequences(self) -> list[int]:
        return [self._turn_close_id]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """Parse sampled tokens into an assistant Message.

        Decodes up to the ``<turn|>`` stop token, then splits any
        ``<|channel>...<channel|>`` thinking blocks out of the body into
        separate ``ThinkingPart`` content parts so the round-trip
        (render → sample → parse) preserves structured content.
        """
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._turn_close_id
        )
        if not parse_success:
            return assistant_message, False

        assert isinstance(assistant_message["content"], str)
        text = assistant_message["content"]
        if _CHANNEL_OPEN not in text:
            assistant_message["content"] = text
            return assistant_message, True

        # Split into alternating text / thinking parts in order.
        parts: list[ThinkingPart | TextPart] = []
        cursor = 0
        for match in _THINKING_BLOCK_RE.finditer(text):
            before = text[cursor : match.start()]
            if before:
                parts.append(TextPart(type="text", text=before))
            inner = match.group(0)[len(_CHANNEL_OPEN) : -len(_CHANNEL_CLOSE)]
            parts.append(ThinkingPart(type="thinking", thinking=inner))
            cursor = match.end()
        tail = text[cursor:]
        if tail:
            parts.append(TextPart(type="text", text=tail))
        assistant_message["content"] = parts
        return assistant_message, True


def _gemma4_factory(tokenizer: Tokenizer, image_processor=None) -> Gemma4Renderer:
    # ``image_processor`` is part of the register_renderer factory contract
    # but Gemma 4 text-only does not use it.
    return Gemma4Renderer(tokenizer)


register_renderer("gemma4", _gemma4_factory)
