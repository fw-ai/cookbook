"""Shared native encoding adapter for the Kimi K3/K3.1 renderers.

The reviewed code-free tokenizer is the fast bundle built by
``training.utils.kimi_native_tokenizer``; this module reproduces the
checkpoint ``TikTokenTokenizer`` encode contract (``_encode_text_piece`` /
``_encode_chat_segments``) on that bundle, including its chunking, so renderer
token ids are identical whether the supplied tokenizer is the native bundle
or the legacy remote-code checkpoint tokenizer.

Two private backend copies are made per adapter: one matches special tokens
in the input text, one encodes them as ordinary text. Separate instances keep
``encode_special_tokens`` from being toggled on a tokenizer that other
threads may share.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

# ``TikTokenTokenizer._encode_text_piece`` chunking, reproduced so long inputs
# split at the same boundaries as the checkpoint tokenizer.
_TIKTOKEN_MAX_ENCODE_CHARS = 400_000
_MAX_NO_WHITESPACES_CHARS = 25_000


def _split_whitespaces_or_nonwhitespaces(text: str, limit: int) -> Iterator[str]:
    """Same slicing as ``TikTokenTokenizer._split_whitespaces_or_nonwhitespaces``."""
    current_len = 0
    current_is_space = text[0].isspace() if text else False
    start = 0
    for index, char in enumerate(text):
        is_space = char.isspace()
        if current_is_space ^ is_space:
            current_len = 1
            current_is_space = is_space
        else:
            current_len += 1
            if current_len > limit:
                yield text[start:index]
                start = index
                current_len = 1
    yield text[start:]


class KimiEncodingAdapter:
    """The checkpoint tokenizer's encode contract on reviewed installed code.

    A tokenizer with a ``backend_tokenizer`` (the native fast bundle) encodes
    through the private backend copies; a legacy checkpoint tokenizer keeps
    its own ``_encode_text_piece`` so deliberately loaded remote-code
    tokenizers behave exactly as before. Anything else is rejected.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None and hasattr(backend, "to_str"):
            # lazy: the tokenizers import is needed only on the native path.
            from tokenizers import Tokenizer

            serialized = backend.to_str()
            self._special_backend: Any = Tokenizer.from_str(serialized)
            self._special_backend.encode_special_tokens = False
            self._text_backend: Any = Tokenizer.from_str(serialized)
            self._text_backend.encode_special_tokens = True
            # Saved fast tokenizers may retain a caller's backend padding or
            # truncation settings; neither may apply to individual segments.
            for private_backend in (self._special_backend, self._text_backend):
                private_backend.no_padding()
                private_backend.no_truncation()
        elif callable(getattr(tokenizer, "_encode_text_piece", None)):
            self._special_backend = None
            self._text_backend = None
        else:
            raise TypeError(
                "Kimi K3/K3.1 rendering needs the native fast tokenizer bundle "
                "(training.utils.kimi_native_tokenizer) or the checkpoint "
                "TikTokenTokenizer; got " + type(tokenizer).__name__
            )

    @property
    def is_native(self) -> bool:
        """Whether encoding goes through the reviewed fast backend copies."""
        return self._special_backend is not None

    def encode_piece(self, text: str, *, allow_special_tokens: bool) -> list[int]:
        """``TikTokenTokenizer._encode_text_piece`` with identical chunking."""
        if not self.is_native:
            return list(self._tokenizer._encode_text_piece(text, allow_special_tokens=allow_special_tokens))
        backend = self._special_backend if allow_special_tokens else self._text_backend
        tokens: list[int] = []
        for offset in range(0, len(text), _TIKTOKEN_MAX_ENCODE_CHARS):
            for piece in _split_whitespaces_or_nonwhitespaces(
                text[offset : offset + _TIKTOKEN_MAX_ENCODE_CHARS],
                _MAX_NO_WHITESPACES_CHARS,
            ):
                if piece:
                    tokens.extend(backend.encode(piece, add_special_tokens=False).ids)
        return [int(token) for token in tokens]

    def encode_chat_segments(self, segments: list[Any]) -> list[int]:
        """``TikTokenTokenizer._encode_chat_segments``: per-segment encode."""
        tokens: list[int] = []
        for segment in segments:
            tokens.extend(
                self.encode_piece(
                    segment.text,
                    allow_special_tokens=bool(segment.allow_special),
                )
            )
        return tokens


__all__ = ["KimiEncodingAdapter"]
