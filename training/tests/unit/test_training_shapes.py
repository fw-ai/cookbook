"""Tests for training_shapes auto-selection and error handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.utils.training_shapes import auto_select_training_shape


class _FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None):
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300
        self._body = body or {}

    def json(self):
        return self._body


class _FakeMgr:
    """Minimal stand-in for TrainerJobManager used by shape selection."""

    def __init__(self, *, list_response: dict | None = None, model_response: _FakeResponse | None = None):
        self._list_response = list_response or {"trainingShapeVersions": []}
        self._model_response = model_response or _FakeResponse(200, {})

    def _get(self, url: str, **kwargs) -> _FakeResponse:
        if "/versions" in url:
            return _FakeResponse(200, self._list_response)
        return self._model_response


class TestModelFetch403FallsThrough:
    """When the model GET returns 403, auto_select should skip the
    parameter-count fallback gracefully and raise ValueError (not RuntimeError).
    """

    def test_403_gives_actionable_value_error(self):
        mgr = _FakeMgr(
            list_response={"trainingShapeVersions": []},
            model_response=_FakeResponse(403),
        )

        with pytest.raises(ValueError, match="Provide an explicit training_shape_id"):
            auto_select_training_shape(
                mgr,
                base_model="accounts/fireworks/models/qwen3p5-397b-a17b",
                max_seq_len=32768,
            )

    def test_404_gives_actionable_value_error(self):
        mgr = _FakeMgr(
            list_response={"trainingShapeVersions": []},
            model_response=_FakeResponse(404),
        )

        with pytest.raises(ValueError, match="Provide an explicit training_shape_id"):
            auto_select_training_shape(
                mgr,
                base_model="accounts/fireworks/models/nonexistent-model",
                max_seq_len=32768,
            )


class TestExactMatchSkipsFetch:
    """When an exact base_model match is found, _fetch_model_context is never
    called, so a 403 on the model endpoint should not matter."""

    def test_exact_match_returns_without_model_fetch(self):
        mgr = _FakeMgr(
            list_response={
                "trainingShapeVersions": [
                    {
                        "name": "accounts/fw/trainingShapes/ts-1/versions/v1",
                        "snapshot": {
                            "trainerMode": "POLICY_TRAINER",
                            "maxSupportedContextLength": 65536,
                        },
                    }
                ]
            },
            model_response=_FakeResponse(403),
        )

        result = auto_select_training_shape(
            mgr,
            base_model="accounts/fireworks/models/qwen3p5-397b-a17b",
            max_seq_len=32768,
        )
        assert result == "accounts/fw/trainingShapes/ts-1"
