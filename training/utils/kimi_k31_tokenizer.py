"""Build the fast tokenizer that the ``kimi_k31`` TITO renderer certifies.

Kimi K3.1 ships a tiktoken vocabulary (``tiktoken.model``) behind a remote-code
``TikTokenTokenizer`` and no ``tokenizer.json``. TITO needs a serializable fast
tokenizer, so this module converts the pinned tiktoken files into an
id-identical ``tokenizer.json``. Pass the output directory as
``--tokenizer-model`` together with ``--renderer-name kimi_k31``.

Only data files are read. The checkpoint's Python (``tokenization_kimi.py``) is
hash-checked, because the token pattern and reserved-token naming below are
copied from it, but never imported.

The conversion itself lives in the reviewed shared implementation behind
``training.utils.kimi_native_tokenizer`` (SDK-first with a generated snapshot
fallback), which covers the byte-identical Kimi K3 vocabulary as well; this
module keeps the K3.1 profile pins, the CLI, and the certified bundle
fingerprint.

Usage::

    python -m training.utils.kimi_k31_tokenizer <out-dir> [--source <checkpoint-dir>]

Without ``--source`` the files are downloaded from the pinned Hugging Face
revision. The command verifies the source hashes, writes the bundle, and prints
the tokenizer fingerprint that ``KIMI_K31_TOKENIZER_FINGERPRINT`` pins.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from training.renderer.tito.kimi_k31 import KIMI_K31_REPOSITORY, KIMI_K31_REVISION
from training.utils import kimi_native_tokenizer as _native

# Tokenizer source files at ``KIMI_K31_REPOSITORY@KIMI_K31_REVISION``.
KIMI_K31_SOURCE_SHA256 = _native.KIMI_K31_SOURCE_SHA256

# ``TikTokenTokenizer.pat_str`` and ``num_reserved_special_tokens`` from the
# pinned ``tokenization_kimi.py`` (identical for Kimi K3).
KIMI_K31_PAT_STR = _native.KIMI_K3_K31_PAT_STR
KIMI_K31_NUM_RESERVED_SPECIAL_TOKENS = _native.KIMI_NUM_RESERVED_SPECIAL_TOKENS
KIMI_K31_BASE_VOCAB_SIZE = _native.KIMI_BASE_VOCAB_SIZE


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
    _native.verify_source_dir(root, profile=_native.KIMI_K31_PROFILE)
    return root


def special_tokens(source_dir: str | Path) -> list[tuple[str, int]]:
    """Every special token in id order, named exactly as ``TikTokenTokenizer`` names them."""
    return _native.special_tokens(verify_source_dir(source_dir))


def build_fast_tokenizer(source_dir: str | Path) -> Any:
    """Convert the pinned tiktoken vocabulary into an id-identical fast tokenizer.

    ``TikTokenConverter`` appends special tokens after the base vocabulary in
    the order given, so every one of the 256 reserved ids is added explicitly in
    id order; passing only the named tokens would shift their ids.
    """
    return _native.build_fast_tokenizer(verify_source_dir(source_dir), profile=_native.KIMI_K31_PROFILE)


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
