"""Tests for explicit, non-creating Fireworks account provenance checks."""

from __future__ import annotations

import pytest

from training.utils import account


class _FakeClient:
    account_id = "research-train"
    calls: list[dict] = []
    exited = False

    def __init__(self, **kwargs):
        self.calls.append(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        type(self).exited = True


def test_account_guard_matches_explicit_account_without_resource_creation(monkeypatch):
    _FakeClient.calls = []
    _FakeClient.exited = False
    monkeypatch.setattr(account, "FireworksClient", _FakeClient)
    monkeypatch.setenv(account.EXPECTED_ACCOUNT_ENV, "wrong-env-account")

    actual = account.assert_expected_fireworks_account(
        api_key="secret",
        base_url="https://api.example.com",
        additional_headers={"X-Test": "1"},
        expected_account_id="research-train",
    )

    assert actual == "research-train"
    assert _FakeClient.calls == [
        {
            "api_key": "secret",
            "base_url": "https://api.example.com",
            "additional_headers": {"X-Test": "1"},
        }
    ]
    assert _FakeClient.exited is True


def test_account_guard_accepts_expected_account_from_environment(monkeypatch):
    monkeypatch.setattr(account, "FireworksClient", _FakeClient)
    monkeypatch.setenv(account.EXPECTED_ACCOUNT_ENV, "research-train")

    assert (
        account.assert_expected_fireworks_account(
            api_key="secret",
            base_url="https://api.example.com",
            additional_headers=None,
        )
        == "research-train"
    )


def test_account_guard_fails_closed_without_expected_account(monkeypatch):
    monkeypatch.delenv(account.EXPECTED_ACCOUNT_ENV, raising=False)
    monkeypatch.setattr(
        account,
        "FireworksClient",
        lambda **_kwargs: pytest.fail("resolver must not run without expectation"),
    )

    with pytest.raises(
        account.FireworksAccountProvenanceError,
        match="expected Fireworks account is required",
    ):
        account.assert_expected_fireworks_account(
            api_key="secret",
            base_url="https://api.example.com",
            additional_headers=None,
        )


@pytest.mark.parametrize("expected", ["accounts/research-train", "Research", "bad account", "-bad"])
def test_account_guard_rejects_invalid_expected_account_before_resolving(
    monkeypatch, expected
):
    monkeypatch.setattr(
        account,
        "FireworksClient",
        lambda **_kwargs: pytest.fail("resolver must not run for invalid expectation"),
    )

    with pytest.raises(
        account.FireworksAccountProvenanceError,
        match="expected Fireworks account ID is invalid",
    ):
        account.assert_expected_fireworks_account(
            api_key="secret",
            base_url="https://api.example.com",
            additional_headers=None,
            expected_account_id=expected,
        )


def test_account_guard_rejects_authenticated_mismatch(monkeypatch):
    class WrongAccountClient(_FakeClient):
        account_id = "other-account"

    monkeypatch.setattr(account, "FireworksClient", WrongAccountClient)

    with pytest.raises(
        account.FireworksAccountProvenanceError,
        match="authenticated Fireworks account differs from expected account",
    ):
        account.assert_expected_fireworks_account(
            api_key="secret",
            base_url="https://api.example.com",
            additional_headers=None,
            expected_account_id="research-train",
        )


@pytest.mark.parametrize("actual", [None, "", "accounts/research-train", "Bad Account"])
def test_account_guard_rejects_absent_or_invalid_authenticated_account(
    monkeypatch, actual
):
    class InvalidAccountClient(_FakeClient):
        account_id = actual

    monkeypatch.setattr(account, "FireworksClient", InvalidAccountClient)

    with pytest.raises(
        account.FireworksAccountProvenanceError,
        match="authenticated Fireworks account ID is absent or invalid",
    ):
        account.assert_expected_fireworks_account(
            api_key="secret",
            base_url="https://api.example.com",
            additional_headers=None,
            expected_account_id="research-train",
        )
