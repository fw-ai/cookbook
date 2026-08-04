"""HuggingFace asset loading for live verifier probes.

Live probe surfaces use this helper for token passthrough and friendly
errors. CPU HF parity owns its loading kwargs because it also preserves
tokenizer revisions and explicit remote-code policy.

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
    # lazy: keep verifier utilities importable without the heavy optional Transformers dependency.
    import transformers  # noqa: PLC0415 — heavy optional dep
    from training.utils.tokenizers import needs_mistral_regex_fix  # noqa: PLC0415

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "token": hf_token,
    }
    if needs_mistral_regex_fix(model_id):
        kwargs["fix_mistral_regex"] = True
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model_id,
            **kwargs,
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


def load_image_processor(model_id: str) -> Any:
    """Load an image processor from a HuggingFace id or local directory."""
    from transformers.models.auto.image_processing_auto import (  # noqa: PLC0415
        AutoImageProcessor,
    )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        return AutoImageProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not load image processor {model_id!r}: {exc}\n\n"
            "Pass the HuggingFace id or local directory containing the "
            "processor config and any trusted custom processor code."
        ) from exc
