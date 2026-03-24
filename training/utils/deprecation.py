"""Shared deprecation utilities for cookbook parameters.

All deprecated parameters should use :func:`warn_deprecated_param` or
:func:`warn_ignored_param` so that warnings are visually consistent
and impossible to miss in logs.

# TODO: remove in 5 releases:
#   - base_model  (auto-resolved from training shape)
#   - grad_accum  (replaced by batch_size)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 70


def warn_deprecated_param(
    old_name: str,
    new_name: str,
    *,
    extra: str = "",
) -> None:
    """Emit a highly visible deprecation warning for a renamed parameter.

    Example output::

        ======================================================================
          DEPRECATED: 'tokenizer_model' is deprecated and will be removed
          in a future release. Use 'hf_tokenizer_name' instead.
        ======================================================================
    """
    msg = (
        f"'{old_name}' is deprecated and will be removed in a future release. "
        f"Use '{new_name}' instead."
    )
    if extra:
        msg = f"{msg} {extra}"
    logger.warning("\n%s\n  DEPRECATED: %s\n%s", _SEPARATOR, msg, _SEPARATOR)


def warn_ignored_param(
    name: str,
    reason: str,
) -> None:
    """Emit a highly visible warning for a parameter that is accepted but ignored.

    Example output::

        ======================================================================
          IGNORED: 'base_model' is ignored when a training shape is set.
          The training shape already specifies the base model. This
          parameter will be removed in a future release.
        ======================================================================
    """
    msg = (
        f"'{name}' is ignored — {reason} "
        "This parameter will be removed in a future release."
    )
    logger.warning("\n%s\n  IGNORED: %s\n%s", _SEPARATOR, msg, _SEPARATOR)
