"""Environment helpers for Fireworks API requests.

Trainer and deployment provisioning now live behind SDK-managed helpers. The
legacy cookbook ``setup_infra`` orchestration
and the ``create_trainer_job`` / ``setup_deployment`` / ``ResourceCleanup``
helpers have been removed; recipes and examples provision through the SDK and
tear down via ``FiretitanServiceClient.close()``. This module retains only the
small environment-parsing helper shared by every recipe.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_FIREWORKS_API_EXTRA_HEADERS_ENV = "FIREWORKS_API_EXTRA_HEADERS"


def read_api_extra_headers_env() -> dict[str, str] | None:
    """Parse ``FIREWORKS_API_EXTRA_HEADERS`` into a header dict.

    The env var, when set, must be a JSON object whose values are strings.
    Used to pass additional HTTP headers (e.g. routing, auth, correlation IDs)
    to every Fireworks API request. Returns ``None`` when the var is unset or
    empty.
    """
    raw = os.environ.get(_FIREWORKS_API_EXTRA_HEADERS_ENV, "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Invalid %s (not valid JSON); ignoring: %s",
            _FIREWORKS_API_EXTRA_HEADERS_ENV,
            exc,
        )
        return None
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        logger.warning(
            "%s must be a JSON object of string->string; ignoring",
            _FIREWORKS_API_EXTRA_HEADERS_ENV,
        )
        return None
    return parsed or None
