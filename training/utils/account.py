"""Fail-closed Fireworks account provenance checks.

Resource-owning account identity must come from an explicit run decision, not
from a credential name, model resource name, or the credential itself.  This
module resolves the credential's account through a read-only control-plane
client and compares it with that independent expectation before any managed
training service can create or reattach resources.
"""

from __future__ import annotations

import os
import re

from fireworks.training.sdk import FireworksClient

EXPECTED_ACCOUNT_ENV = "FIREWORKS_EXPECTED_ACCOUNT_ID"
_ACCOUNT_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,62}$")


class FireworksAccountProvenanceError(RuntimeError):
    """The authenticated Fireworks account does not match the run contract."""


def resolve_expected_account_id(explicit: str | None = None) -> str:
    """Return and validate an explicitly configured resource-owning account."""

    expected = explicit if explicit is not None else os.environ.get(EXPECTED_ACCOUNT_ENV)
    if not expected:
        raise FireworksAccountProvenanceError(
            "expected Fireworks account is required; pass expected_account_id "
            f"or set {EXPECTED_ACCOUNT_ENV}"
        )
    if not _ACCOUNT_ID_RE.fullmatch(expected):
        raise FireworksAccountProvenanceError(
            f"expected Fireworks account ID is invalid: {expected!r}"
        )
    return expected


def assert_expected_fireworks_account(
    *,
    api_key: str,
    base_url: str,
    additional_headers: dict[str, str] | None,
    expected_account_id: str | None = None,
) -> str:
    """Resolve credential ownership without creating resources and assert it.

    The caller must supply the expected account directly or through
    ``FIREWORKS_EXPECTED_ACCOUNT_ID``.  The returned value is the authenticated
    account ID and is safe to persist as provenance evidence.
    """

    expected = resolve_expected_account_id(expected_account_id)
    with FireworksClient(
        api_key=api_key,
        base_url=base_url,
        additional_headers=additional_headers,
    ) as client:
        actual = client.account_id
    if not isinstance(actual, str) or not _ACCOUNT_ID_RE.fullmatch(actual):
        raise FireworksAccountProvenanceError(
            "authenticated Fireworks account ID is absent or invalid"
        )
    if actual != expected:
        raise FireworksAccountProvenanceError(
            "authenticated Fireworks account differs from expected account: "
            f"authenticated={actual!r}, expected={expected!r}"
        )
    return actual
