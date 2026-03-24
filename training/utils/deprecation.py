"""Shared deprecation utilities for cookbook parameters.

All deprecated parameters should use :func:`warn_deprecated_param` so
that warnings are visually consistent and impossible to miss in logs.
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
    """Emit a highly visible deprecation warning for a renamed/removed parameter.

    Example output::

        ======================================================================
          DEPRECATED: 'base_model' is deprecated and will be removed in a
          future release. Use 'profile.base_model' instead.
        ======================================================================
    """
    msg = (
        f"'{old_name}' is deprecated and will be removed in a future release. "
        f"Use '{new_name}' instead."
    )
    if extra:
        msg = f"{msg} {extra}"
    logger.warning("\n%s\n  DEPRECATED: %s\n%s", _SEPARATOR, msg, _SEPARATOR)
