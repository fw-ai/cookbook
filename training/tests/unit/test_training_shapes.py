"""Tests for training shape auto-selection logic.

Covers the multi-pass fallback in ``auto_select_training_shape``:
  Pass 1  — exact base_model + trainer_mode server-side filter
  Pass 1b — relaxed server-side filter (base_model only, mode checked client-side)
  Pass 1c — account-scoped parent derived from base_model
  Pass 2  — model_type + parameter_count fallback
  Pass 2b — relaxed parameter filter (no trainer_mode server-side)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from training.utils.training_shapes import (
    auto_select_training_shape,
    _account_from_base_model,
    _expected_trainer_mode,
    _normalize_trainer_mode,
    _param_count_bounds,
)

BASE_MODEL = "accounts/fireworks/models/qwen3p5-397b-a17b"
SHAPE_NAME = "accounts/fireworks/trainingShapes/qwen3p5-397b-a17b-262k-b200"
SHAPE_VERSION = f"{SHAPE_NAME}/versions/k7sw735k"


def _make_shape_version(
    name: str = SHAPE_VERSION,
    base_model: str = BASE_MODEL,
    trainer_mode: str = "POLICY_TRAINER",
    ctx_len: int = 262144,
    **extra_snap: Any,
) -> dict:
    snap = {
        "baseModel": base_model,
        "trainerMode": trainer_mode,
        "maxSupportedContextLength": ctx_len,
        **extra_snap,
    }
    return {"name": name, "snapshot": snap}


class FakeResponse:
    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300

    def json(self):
        return self._data


class FakeTrainerMgr:
    """Stub TrainerJobManager that returns canned responses."""

    def __init__(self, responses: dict[str, FakeResponse] | None = None):
        self._responses = responses or {}
        self.get_calls: list[str] = []

    def _get(self, path: str, timeout: int = 30) -> FakeResponse:
        self.get_calls.append(path)
        for prefix, resp in self._responses.items():
            if prefix in path:
                return resp
        return FakeResponse({"trainingShapeVersions": []})


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestAccountFromBaseModel:
    def test_standard_model(self):
        assert _account_from_base_model("accounts/fireworks/models/qwen3-8b") == "fireworks"

    def test_custom_account(self):
        assert _account_from_base_model("accounts/my-org/models/custom-7b") == "my-org"

    def test_no_match(self):
        assert _account_from_base_model("random-string") is None


class TestExpectedTrainerMode:
    def test_lora(self):
        assert _expected_trainer_mode("policy", lora_rank=16) == "LORA_TRAINER"

    def test_reference(self):
        assert _expected_trainer_mode("reference", lora_rank=0) == "FORWARD_ONLY"

    def test_policy(self):
        assert _expected_trainer_mode("policy", lora_rank=0) == "POLICY_TRAINER"


class TestNormalizeTrainerMode:
    def test_string_passthrough(self):
        assert _normalize_trainer_mode("POLICY_TRAINER") == "POLICY_TRAINER"

    def test_int_code(self):
        assert _normalize_trainer_mode(1) == "POLICY_TRAINER"
        assert _normalize_trainer_mode(2) == "FORWARD_ONLY"
        assert _normalize_trainer_mode(3) == "LORA_TRAINER"

    def test_none(self):
        assert _normalize_trainer_mode(None) is None

    def test_empty_string(self):
        assert _normalize_trainer_mode("") is None


class TestParamCountBounds:
    def test_small_model(self):
        lo, hi = _param_count_bounds(7_000_000_000)
        assert lo == 7_000_000_000
        assert hi == 8_000_000_000

    def test_large_model(self):
        lo, hi = _param_count_bounds(403_000_000_000)
        assert lo == 400_000_000_000
        assert hi == 410_000_000_000


# ---------------------------------------------------------------------------
# Integration tests for auto_select_training_shape
# ---------------------------------------------------------------------------


class TestPass1ExactMatch:
    """Pass 1: server-side filter returns an exact base_model match."""

    def test_single_candidate(self):
        mgr = FakeTrainerMgr({
            "versions": FakeResponse({
                "trainingShapeVersions": [_make_shape_version()],
            }),
        })
        result = auto_select_training_shape(
            mgr, base_model=BASE_MODEL, max_seq_len=32768,
        )
        assert result == SHAPE_NAME

    def test_picks_smallest_sufficient_ctx(self):
        v32k = _make_shape_version(
            name="accounts/fireworks/trainingShapes/small-32k/versions/v1",
            ctx_len=32768,
        )
        v262k = _make_shape_version(
            name="accounts/fireworks/trainingShapes/large-262k/versions/v2",
            ctx_len=262144,
        )
        mgr = FakeTrainerMgr({
            "versions": FakeResponse({
                "trainingShapeVersions": [v262k, v32k],
            }),
        })
        result = auto_select_training_shape(
            mgr, base_model=BASE_MODEL, max_seq_len=32768,
        )
        assert result == "accounts/fireworks/trainingShapes/small-32k"


class TestPass1bRelaxedFilter:
    """Pass 1b: full filter returns nothing, relaxed filter finds shapes."""

    def test_mode_filter_stripped_from_server_side(self):
        call_count = {"n": 0}
        shape = _make_shape_version()

        class _Mgr(FakeTrainerMgr):
            def _get(self, path: str, timeout: int = 30) -> FakeResponse:
                self.get_calls.append(path)
                if "versions" in path:
                    call_count["n"] += 1
                    if call_count["n"] == 1:
                        return FakeResponse({"trainingShapeVersions": []})
                    return FakeResponse({"trainingShapeVersions": [shape]})
                return FakeResponse({})

        mgr = _Mgr()
        result = auto_select_training_shape(
            mgr, base_model=BASE_MODEL, max_seq_len=32768,
        )
        assert result == SHAPE_NAME
        assert call_count["n"] >= 2


class TestPass1cAccountScoped:
    """Pass 1c: wildcard parent returns nothing, account-scoped parent works."""

    def test_derives_account_from_base_model(self):
        call_count = {"n": 0}
        shape = _make_shape_version()

        class _Mgr(FakeTrainerMgr):
            def _get(self, path: str, timeout: int = 30) -> FakeResponse:
                self.get_calls.append(path)
                if "versions" in path:
                    call_count["n"] += 1
                    if "accounts/fireworks/trainingShapes/-" in path:
                        return FakeResponse({"trainingShapeVersions": [shape]})
                    return FakeResponse({"trainingShapeVersions": []})
                return FakeResponse({})

        mgr = _Mgr()
        result = auto_select_training_shape(
            mgr, base_model=BASE_MODEL, max_seq_len=32768,
        )
        assert result == SHAPE_NAME

    def test_skips_account_scoped_when_shape_account_given(self):
        """When caller provides shape_account, Pass 1c is skipped."""
        call_count = {"n": 0}

        class _Mgr(FakeTrainerMgr):
            def _get(self, path: str, timeout: int = 30) -> FakeResponse:
                self.get_calls.append(path)
                if "versions" in path:
                    call_count["n"] += 1
                return FakeResponse({"trainingShapeVersions": []})

        mgr = _Mgr()
        with pytest.raises(ValueError, match="Provide an explicit training_shape_id"):
            auto_select_training_shape(
                mgr, base_model=BASE_MODEL,
                shape_account="explicit-account",
            )


class TestPass2ParameterCountFallback:
    """Pass 2: model_type + param_count bucket fallback."""

    def test_uses_model_context_for_fallback(self):
        shape = _make_shape_version(
            name="accounts/fireworks/trainingShapes/qwen3p5-moe-b200/versions/v1",
            base_model="accounts/fireworks/models/qwen3p5-other",
            trainer_mode="POLICY_TRAINER",
            modelType="qwen3_5_moe",
            parameterCount=403_000_000_000,
        )
        call_count = {"n": 0}

        class _Mgr(FakeTrainerMgr):
            def _get(self, path: str, timeout: int = 30) -> FakeResponse:
                self.get_calls.append(path)
                if "versions" in path:
                    call_count["n"] += 1
                    if "parameter_count" in path:
                        return FakeResponse({"trainingShapeVersions": [shape]})
                    return FakeResponse({"trainingShapeVersions": []})
                if f"/v1/{BASE_MODEL}" in path:
                    return FakeResponse({
                        "baseModelDetails": {
                            "modelType": "qwen3_5_moe",
                            "parameterCount": 403_000_000_000,
                        },
                    })
                return FakeResponse({})

        mgr = _Mgr()
        result = auto_select_training_shape(
            mgr, base_model=BASE_MODEL, max_seq_len=32768,
        )
        assert result == "accounts/fireworks/trainingShapes/qwen3p5-moe-b200"


class TestModelFetch403:
    """Handles HTTP 403 from model details API gracefully."""

    def test_403_skips_param_fallback(self):
        class _Mgr(FakeTrainerMgr):
            def _get(self, path: str, timeout: int = 30) -> FakeResponse:
                self.get_calls.append(path)
                if "versions" in path:
                    return FakeResponse({"trainingShapeVersions": []})
                if f"/v1/{BASE_MODEL}" in path:
                    return FakeResponse({}, status_code=403)
                return FakeResponse({})

        mgr = _Mgr()
        with pytest.raises(ValueError, match="Provide an explicit training_shape_id"):
            auto_select_training_shape(
                mgr, base_model=BASE_MODEL, max_seq_len=32768,
            )


class TestClientSideModeFilter:
    """Verifies client-side mode filtering catches mismatched modes."""

    def test_filters_wrong_mode(self):
        lora_shape = _make_shape_version(trainer_mode="LORA_TRAINER")

        mgr = FakeTrainerMgr({
            "versions": FakeResponse({
                "trainingShapeVersions": [lora_shape],
            }),
        })
        with pytest.raises(ValueError, match="Provide an explicit training_shape_id"):
            auto_select_training_shape(
                mgr, base_model=BASE_MODEL, trainer_role="policy", lora_rank=0,
            )

    def test_filters_insufficient_ctx(self):
        small_shape = _make_shape_version(ctx_len=8192)

        mgr = FakeTrainerMgr({
            "versions": FakeResponse({
                "trainingShapeVersions": [small_shape],
            }),
        })
        with pytest.raises(ValueError, match="Provide an explicit training_shape_id"):
            auto_select_training_shape(
                mgr, base_model=BASE_MODEL, max_seq_len=32768,
            )

    def test_numeric_mode_normalized(self):
        """Trainer mode returned as proto enum int (1=POLICY_TRAINER)."""
        shape = _make_shape_version()
        shape["snapshot"]["trainerMode"] = 1
        del shape["snapshot"]["trainerMode"]
        shape["snapshot"]["trainer_mode"] = 1

        mgr = FakeTrainerMgr({
            "versions": FakeResponse({
                "trainingShapeVersions": [shape],
            }),
        })
        result = auto_select_training_shape(
            mgr, base_model=BASE_MODEL, max_seq_len=32768,
        )
        assert result == SHAPE_NAME


class TestNoMaxSeqLen:
    """When max_seq_len is None, picks first candidate (newest)."""

    def test_picks_first(self):
        v1 = _make_shape_version(
            name="accounts/fireworks/trainingShapes/newest/versions/v1",
            ctx_len=262144,
        )
        v2 = _make_shape_version(
            name="accounts/fireworks/trainingShapes/older/versions/v2",
            ctx_len=32768,
        )
        mgr = FakeTrainerMgr({
            "versions": FakeResponse({
                "trainingShapeVersions": [v1, v2],
            }),
        })
        result = auto_select_training_shape(
            mgr, base_model=BASE_MODEL,
        )
        assert result == "accounts/fireworks/trainingShapes/newest"
