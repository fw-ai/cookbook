"""Tests for rollout-local request gates."""

from __future__ import annotations

import asyncio

import pytest

from training.utils.rl.rollout import DEFAULT_REQUEST_GATE_CONCURRENCY, FixedRequestGate


def test_acquire_release_basic():
    gate = FixedRequestGate(max_concurrency=2)

    async def _run():
        await gate.acquire()
        await gate.acquire()
        assert gate._semaphore._value == 0
        gate.release()
        gate.release()
        assert gate._semaphore._value == 2

    asyncio.run(_run())


def test_default_width_is_32():
    gate = FixedRequestGate()
    assert DEFAULT_REQUEST_GATE_CONCURRENCY == 32
    assert gate.max_concurrency == 32


def test_rejects_non_positive_width():
    with pytest.raises(ValueError, match="max_concurrency"):
        FixedRequestGate(max_concurrency=0)
