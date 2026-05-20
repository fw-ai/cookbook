"""Tokenizer loading with HF_TOKEN passthrough and friendly errors.

The verifier needs HuggingFace tokenizers for two surfaces — the live
probe and the CPU HF parity test. Both go through this helper so they
share auth handling, caching behaviour, and the same error guidance
when something goes wrong.

``transformers`` caches downloads under ``~/.cache/huggingface/`` by
default, so the *second* call for a given ``model_id`` is fast and
offline-friendly. The first call needs network access (and, for gated
repos, a token).
"""

from __future__ import annotations

import os
from typing import Any


def load_tokenizer(model_id: str) -> Any:
    """Load an HF tokenizer, forwarding ``HF_TOKEN`` if set."""
    import transformers  # noqa: PLC0415 — heavy optional dep

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not load tokenizer {model_id!r}: {exc}\n\n"
            "Common causes:\n"
            "  • Typo in the model id — check capitalization and hyphens "
            "(e.g. 'zai-org/GLM-5.1', not 'zai-org/GLM5.1').\n"
            "  • Gated / private repo — run `hf auth login` or "
            "`export HF_TOKEN=hf_...` and retry.\n"
            "  • No network access on first load — once cached under "
            "~/.cache/huggingface/ the tokenizer loads offline."
        ) from exc
