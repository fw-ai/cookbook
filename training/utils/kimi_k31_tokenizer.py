"""Build the fast tokenizer that the ``kimi_k31`` TITO renderer certifies.

Kimi K3.1 ships a tiktoken vocabulary (``tiktoken.model``) behind a remote-code
``TikTokenTokenizer`` and no ``tokenizer.json``. TITO needs a serializable fast
tokenizer, so this module converts the pinned tiktoken files into an
id-identical ``tokenizer.json``. Pass the output directory as
``--tokenizer-model`` together with ``--renderer-name kimi_k31``.

Only data files are read. The checkpoint's Python (``tokenization_kimi.py``) is
hash-checked, because the token pattern and reserved-token naming below are
copied from it, but never imported.

Usage::

    python -m training.utils.kimi_k31_tokenizer <out-dir> [--source <checkpoint-dir>]

Without ``--source`` the files are downloaded from the pinned Hugging Face
revision. The command verifies the source hashes, writes the bundle, and prints
the tokenizer fingerprint that ``KIMI_K31_TOKENIZER_FINGERPRINT`` pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from training.renderer.tito.kimi_k31 import KIMI_K31_REPOSITORY, KIMI_K31_REVISION

# Tokenizer source files at ``KIMI_K31_REPOSITORY@KIMI_K31_REVISION``.
KIMI_K31_SOURCE_SHA256 = {
    "tiktoken.model": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    "tokenizer_config.json": "9fc322181b8ec44236e148916aea4b6c2467b66ded7bfaff0efe623ad750b852",
    "tokenization_kimi.py": "204ee39fbf13a6d6608b0d3ca1b4b04a14ac5f0a062a7fdee02e61f5d936f9c6",
}

# ``TikTokenTokenizer.pat_str`` and ``num_reserved_special_tokens`` from the
# pinned ``tokenization_kimi.py``.
KIMI_K31_PAT_STR = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)
KIMI_K31_NUM_RESERVED_SPECIAL_TOKENS = 256
KIMI_K31_BASE_VOCAB_SIZE = 163_584

_BOS, _EOS, _UNK, _PAD = "[BOS]", "[EOS]", "[UNK]", "[PAD]"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_source() -> Path:
    """Fetch the pinned tokenizer source files and return their directory."""
    # lazy: Hugging Face access is needed only when no local checkpoint is given.
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=KIMI_K31_REPOSITORY,
            revision=KIMI_K31_REVISION,
            allow_patterns=sorted(KIMI_K31_SOURCE_SHA256),
        )
    )


def verify_source_dir(source_dir: str | Path) -> Path:
    """Fail closed unless every pinned source file is present and unchanged."""
    root = Path(source_dir)
    mismatched = []
    for filename, expected in sorted(KIMI_K31_SOURCE_SHA256.items()):
        path = root / filename
        if not path.is_file():
            mismatched.append(f"{filename}: missing")
        elif _sha256(path) != expected:
            mismatched.append(f"{filename}: sha256 {_sha256(path)} != {expected}")
    if mismatched:
        raise ValueError(
            f"Kimi K3.1 tokenizer sources do not match {KIMI_K31_REPOSITORY}@{KIMI_K31_REVISION}:\n  "
            + "\n  ".join(mismatched)
        )
    return root


def special_tokens(source_dir: str | Path) -> list[tuple[str, int]]:
    """Every special token in id order, named exactly as ``TikTokenTokenizer`` names them."""
    root = Path(source_dir)
    config = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
    named = {int(token_id): entry["content"] for token_id, entry in config.get("added_tokens_decoder", {}).items()}
    first = KIMI_K31_BASE_VOCAB_SIZE
    return [
        (named.get(token_id, f"<|reserved_token_{token_id}|>"), token_id)
        for token_id in range(first, first + KIMI_K31_NUM_RESERVED_SPECIAL_TOKENS)
    ]


def build_fast_tokenizer(source_dir: str | Path) -> Any:
    """Convert the pinned tiktoken vocabulary into an id-identical fast tokenizer.

    ``TikTokenConverter`` appends special tokens after the base vocabulary in
    the order given, so every one of the 256 reserved ids is added explicitly in
    id order; passing only the named tokens would shift their ids.
    """
    # lazy: conversion needs the optional tokenizers/transformers stack.
    from tokenizers import AddedToken
    from transformers.convert_slow_tokenizer import TikTokenConverter
    from transformers.tokenization_utils_tokenizers import TokenizersBackend

    root = verify_source_dir(source_dir)
    backend = TikTokenConverter(
        vocab_file=str(root / "tiktoken.model"),
        pattern=KIMI_K31_PAT_STR,
        additional_special_tokens=[],
    ).converted()
    if backend.get_vocab_size() != KIMI_K31_BASE_VOCAB_SIZE:
        raise ValueError(
            f"Kimi K3.1 base vocabulary has {backend.get_vocab_size()} tokens, expected {KIMI_K31_BASE_VOCAB_SIZE}"
        )
    ordered = special_tokens(root)
    backend.add_special_tokens([AddedToken(name, special=True, normalized=False) for name, _ in ordered])
    for name, token_id in ordered:
        if backend.token_to_id(name) != token_id:
            raise ValueError(f"special token {name!r} landed at {backend.token_to_id(name)}, expected {token_id}")
    return TokenizersBackend(
        tokenizer_object=backend,
        bos_token=_BOS,
        eos_token=_EOS,
        unk_token=_UNK,
        pad_token=_PAD,
        clean_up_tokenization_spaces=False,
    )


def write_bundle(out_dir: str | Path, *, source_dir: str | Path | None = None) -> str:
    """Write the fast tokenizer to ``out_dir`` and return its fingerprint."""
    # lazy: tokenizer loading pulls in transformers, which this module defers.
    from training.renderer.tito.shared import _tokenizer_fingerprint
    from training.utils.tokenizers import load_tokenizer

    root = verify_source_dir(source_dir if source_dir is not None else download_source())
    destination = Path(out_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    build_fast_tokenizer(root).save_pretrained(str(destination))
    # Fingerprint the bundle exactly as a recipe loads ``--tokenizer-model``.
    return _tokenizer_fingerprint(load_tokenizer(str(destination)))


def _main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Build the Kimi K3.1 fast tokenizer for TITO.")
    parser.add_argument("out_dir", help="empty directory to write tokenizer.json into")
    parser.add_argument(
        "--source",
        default=None,
        help=f"local checkpoint directory; default downloads {KIMI_K31_REPOSITORY}@{KIMI_K31_REVISION[:8]}",
    )
    args = parser.parse_args(argv)
    print(write_bundle(args.out_dir, source_dir=args.source))
    return 0


__all__ = [
    "KIMI_K31_PAT_STR",
    "KIMI_K31_SOURCE_SHA256",
    "build_fast_tokenizer",
    "download_source",
    "special_tokens",
    "verify_source_dir",
    "write_bundle",
]


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(_main(sys.argv[1:]))
