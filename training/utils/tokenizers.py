"""Shared HuggingFace tokenizer loading helpers."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import transformers
from huggingface_hub import errors as hf_errors

logger = logging.getLogger(__name__)


_HTTP_STATUS_PATTERN = re.compile(r"\b([45]\d\d)\b")
_MISTRAL_TOKENIZER_NAME_PARTS = ("mistral", "ministral")

# A repo file that simply is not there answers the backend question ("no
# declaration, use AutoTokenizer"). Any other failure does not: it means we
# could not read the declaration, and guessing would silently change token
# identity. Those propagate instead.
_MISSING_REPO_FILE_ERRORS: tuple[type[BaseException], ...] = (
    hf_errors.EntryNotFoundError,  # covers the local (cache-miss) subclass too
    hf_errors.RepositoryNotFoundError,  # covers GatedRepoError
    hf_errors.RevisionNotFoundError,
    hf_errors.DisabledRepoError,
    hf_errors.HFValidationError,
    hf_errors.OfflineModeIsEnabled,
)


def patch_kimi_tokenizer_bytes_to_unicode() -> None:
    """Re-export ``bytes_to_unicode`` at its legacy gpt2 location.

    Model-shipped ``tokenization_kimi.py`` (Kimi K2.5 and related) does
    ``from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode``.
    Transformers 5.x moved that helper to ``transformers.convert_slow_tokenizer``,
    so ``AutoTokenizer.from_pretrained(..., trust_remote_code=True)`` crashes
    before the dataset is opened.
    """
    try:
        # lazy: optional compat imports; older transformers still export the symbol
        import transformers.models.gpt2.tokenization_gpt2 as gpt2_module
        from transformers.convert_slow_tokenizer import bytes_to_unicode
    except ImportError:
        return
    if not hasattr(gpt2_module, "bytes_to_unicode"):
        gpt2_module.bytes_to_unicode = bytes_to_unicode


# Eager so DataLoader workers and other AutoTokenizer callers are covered even
# if they never go through load_tokenizer().
patch_kimi_tokenizer_bytes_to_unicode()


def _is_generic_config_missing_max_position_embeddings(exc: AttributeError) -> bool:
    """Whether Transformers failed while validating an unrecognized model config."""
    config = getattr(exc, "obj", None)
    return (
        getattr(exc, "name", None) == "max_position_embeddings"
        and type(config) is transformers.PreTrainedConfig
    )


def needs_mistral_regex_fix(tokenizer_model: str | None) -> bool:
    if tokenizer_model is None:
        return False
    normalized_model = tokenizer_model.casefold()
    return any(part in normalized_model for part in _MISTRAL_TOKENIZER_NAME_PARTS)


def _read_repo_json(
    tokenizer_model: str,
    filename: str,
    *,
    revision: str | None,
    local_files_only: bool,
) -> dict[str, Any] | None:
    """Read one JSON file from a local tokenizer dir or the HuggingFace Hub.

    ``None`` means the file is absent. A file that exists but cannot be
    fetched or parsed raises: backend selection changes token identity, so an
    unreadable declaration must not be answered by falling through to the
    default path.
    """
    local_dir = Path(tokenizer_model)
    if local_dir.is_dir():
        path = local_dir / filename
        if not path.is_file():
            return None
    else:
        # lazy: hub import only on the hub path
        from huggingface_hub import hf_hub_download

        try:
            path = Path(
                hf_hub_download(
                    tokenizer_model,
                    filename,
                    revision=revision,
                    local_files_only=local_files_only,
                )
            )
        except _MISSING_REPO_FILE_ERRORS:
            return None
        except Exception:
            # Surface the Hub failure unchanged so existing callers and the
            # managed-adapter status mapping keep classifying it, rather than
            # guessing a backend and silently changing token identity.
            logger.error(
                "Could not fetch %s for tokenizer %r while selecting the "
                "tokenizer backend; refusing to guess.",
                filename,
                tokenizer_model,
            )
            raise

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"could not parse {filename} for tokenizer {tokenizer_model!r} "
            "while selecting the tokenizer backend"
        ) from exc
    return parsed if isinstance(parsed, dict) else None


def _declared_tokenizer_class(tokenizer_config: dict[str, Any]) -> str | None:
    """The declared ``tokenizer_class`` that needs a model-type check."""
    if tokenizer_config.get("auto_map") is not None:
        # Remote-code tokenizers own their own loading path.
        return None
    declared = tokenizer_config.get("tokenizer_class")
    if not isinstance(declared, str) or not declared:
        return None
    return declared


def _mismatches_model_type(declared: str, model_type: str) -> bool:
    from transformers.models.auto import tokenization_auto

    mapping_name = tokenization_auto.TOKENIZER_MAPPING_NAMES.get(model_type)
    return (mapping_name or "").replace("Fast", "") != declared.replace("Fast", "")


def _tokenizer_selection_is_ambiguous(
    tokenizer_model: str,
    tokenizer_config: dict[str, Any],
    *,
    revision: str | None,
    local_files_only: bool,
) -> bool:
    """Serving's tokenizer-backend rule, ported from ``py/fireworks/text/tokenizer.py``.

    Transformers >= 5.9 honors ``tokenizer_config.json``'s declared
    ``tokenizer_class`` and rebuilds the tokenizer from that class's hardcoded
    pre-tokenizer instead of reading ``tokenizer.json`` verbatim (<= 5.5.4
    ignored the declaration). When the declared class does not match the
    model-type mapping -- Qwen3.8 declares ``Qwen2Tokenizer`` while its
    ``qwen3_5`` model type maps to ``Qwen3_5Tokenizer`` -- serving keeps
    loading ``tokenizer.json`` verbatim via ``TokenizersBackend`` on every
    Transformers version. Training must resolve to the same artifact so its
    tokens stay byte-identical to inference.

    Deviation from serving: serving reads ``model_type`` through
    ``AutoConfig``, which raises for model types its Transformers pin does not
    register. The cookbook reads the declared ``model_type`` directly, which
    returns the same answer for every model serving can actually serve without
    depending on the cookbook's older Transformers pin.
    """
    declared = _declared_tokenizer_class(tokenizer_config)
    if declared is None:
        return False

    model_config = _read_repo_json(
        tokenizer_model,
        "config.json",
        revision=revision,
        local_files_only=local_files_only,
    )
    model_type = (model_config or {}).get("model_type")
    if not isinstance(model_type, str) or not model_type:
        # Serving's fallback for local tokenizer-only dirs whose tokenizer
        # config implies a fast tokenizer but that carry no model config.
        return Path(tokenizer_model).is_dir() and declared.endswith("Fast")
    return _mismatches_model_type(declared, model_type)


def _declared_post_processor_kwargs(tokenizer_config: dict[str, Any]) -> dict[str, Any]:
    """``add_bos_token``/``add_eos_token`` exactly as serving forwards them.

    Transformers only rebuilds the post-processor when these arrive as
    explicit kwargs (``TokenizersBackend.__init__`` gates on
    ``"add_bos_token" in kwargs``); the values in ``tokenizer_config.json`` are
    otherwise ignored. Dropping them diverges from serving in both directions:
    DeepSeek-V3.1 (``add_bos_token: true``) loses its BOS, and
    Nemotron-Nano-9B-v2 (``add_bos_token: false``) gains one. Presence, not
    truthiness, is the trigger -- same as serving.
    """
    return {
        key: tokenizer_config[key]
        for key in ("add_bos_token", "add_eos_token")
        if key in tokenizer_config
    }


def _load_tokenizer_json_verbatim(
    tokenizer_model: str,
    kwargs: dict[str, Any],
) -> Any | None:
    """Load ``tokenizer.json`` verbatim via ``TokenizersBackend``.

    ``None`` means the caller should fall through to ``AutoTokenizer``, which
    is what serving does when the artifact cannot be loaded verbatim -- an
    ambiguous sentencepiece-only checkpoint such as
    ``mistralai/Ministral-8B-Instruct-2410`` ships no ``tokenizer.json`` at
    all. Falling back is logged at warning level because it restores the
    divergent path; TITO certification catches it downstream through the
    tokenizer fingerprint.

    A Hub transport failure is not an artifact failure, so it propagates.
    Serving only ever loads local directories and cannot hit this; falling
    back on it would let a rate-limited download silently pick a different
    tokenizer than inference uses.
    """
    from transformers.tokenization_utils_tokenizers import TokenizersBackend

    try:
        return TokenizersBackend.from_pretrained(tokenizer_model, **kwargs)
    except Exception as exc:
        if _huggingface_http_status_code(exc) is not None:
            raise
        logger.warning(
            "Failed to load tokenizer.json verbatim for %r; falling back to "
            "AutoTokenizer, which may not match inference tokens.",
            tokenizer_model,
            exc_info=True,
        )
        return None


def _huggingface_http_status_code(exc: BaseException) -> int | None:
    """Find a Hugging Face HTTP status in a wrapped exception graph."""
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            try:
                parsed_status_code = int(status_code)
            except (TypeError, ValueError):
                pass
            else:
                if 400 <= parsed_status_code <= 599:
                    return parsed_status_code

        message = str(current)
        normalized_message = message.lower()
        if (
            "huggingface" in normalized_message
            or "huggingface.co" in normalized_message
            or "hf hub" in normalized_message
        ):
            match = _HTTP_STATUS_PATTERN.search(message)
            if match is not None:
                return int(match.group(1))

        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return None


_TokenizerLoader = Callable[..., Any]


def _propagate_huggingface_http_status(loader: _TokenizerLoader) -> _TokenizerLoader:
    """Preserve a wrapped Hugging Face HTTP status at the tokenizer boundary."""

    @wraps(loader)
    def wrapped(
        tokenizer_model: str | None,
        tokenizer_revision: str | None = None,
        trust_remote_code: bool | None = None,
        *,
        local_files_only: bool = False,
    ) -> Any:
        try:
            return loader(
                tokenizer_model,
                tokenizer_revision,
                trust_remote_code,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            status_code = _huggingface_http_status_code(exc)
            if status_code is None:
                raise
            raise RuntimeError(
                "Hugging Face Hub request failed while loading tokenizer "
                f"{tokenizer_model!r} (HTTP {status_code})."
            ) from exc

    return wrapped


@_propagate_huggingface_http_status
def load_tokenizer(
    tokenizer_model: str | None,
    tokenizer_revision: str | None = None,
    trust_remote_code: bool | None = None,
    *,
    local_files_only: bool = False,
) -> Any:
    """Load a tokenizer with cookbook defaults.

    ``tokenizer_revision`` is optional; empty strings are treated as unset so
    existing configs keep resolving HuggingFace ``main``. ``None`` preserves
    the legacy remote-code policy (enabled), while a reviewed tokenizer plan
    can explicitly enable or disable it. ``local_files_only`` supports callers
    that attempt a cache-only load before allowing network access.
    """
    kwargs: dict[str, Any] = {
        "revision": tokenizer_revision or None,
        "trust_remote_code": True if trust_remote_code is None else trust_remote_code,
    }
    if local_files_only:
        kwargs["local_files_only"] = True
    if needs_mistral_regex_fix(tokenizer_model):
        # Use Transformers' upstream Mistral pre-tokenizer repair. The corrected
        # implementation is available in the pinned Transformers 5.10.4 release.
        kwargs["fix_mistral_regex"] = True

    patch_kimi_tokenizer_bytes_to_unicode()

    if tokenizer_model is not None:
        tokenizer_config = _read_repo_json(
            tokenizer_model,
            "tokenizer_config.json",
            revision=kwargs["revision"],
            local_files_only=local_files_only,
        )
        if tokenizer_config is not None and _tokenizer_selection_is_ambiguous(
            tokenizer_model,
            tokenizer_config,
            revision=kwargs["revision"],
            local_files_only=local_files_only,
        ):
            tokenizer = _load_tokenizer_json_verbatim(
                tokenizer_model,
                {**kwargs, **_declared_post_processor_kwargs(tokenizer_config)},
            )
            if tokenizer is not None:
                return tokenizer

    try:
        return transformers.AutoTokenizer.from_pretrained(tokenizer_model, **kwargs)
    except AttributeError as exc:
        if not _is_generic_config_missing_max_position_embeddings(exc):
            raise

    # Transformers 5.5 validates RoPE while loading a fallback PreTrainedConfig.
    # Unknown model types can therefore fail before their otherwise standard
    # tokenizer is inspected. Supplying a config skips that model-only parsing
    # while preserving AutoTokenizer's tokenizer_config and remote-code routing.
    return transformers.AutoTokenizer.from_pretrained(
        tokenizer_model,
        config=transformers.PreTrainedConfig(),
        **kwargs,
    )


def load_deployment_tokenizer(deployment: Any) -> Any:
    """Load the tokenizer configured on a deployment config-like object."""
    return load_tokenizer(
        getattr(deployment, "tokenizer_model", None),
        getattr(deployment, "tokenizer_revision", None),
        getattr(deployment, "tokenizer_trust_remote_code", None),
    )
