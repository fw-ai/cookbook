"""Unit tests for training.utils.rl.rollout.turn_matching.

Covers the strategy-independent classifier (new/append/wipe), the longest
common-prefix helper, both fingerprinter strategies (message-hash vs
token-prefix), and the make_fingerprinter switch.  Pure CPU; no SDK/renderer.
"""

from __future__ import annotations

import pytest

from training.utils.rl.rollout.turn_matching import (
    DEFAULT_TURN_MATCHING,
    MessageHashFingerprinter,
    TokenPrefixFingerprinter,
    TurnKind,
    TurnRequest,
    classify,
    common_prefix_len,
    make_fingerprinter,
)


class TestCommonPrefixLen:
    def test_full_prefix(self):
        assert common_prefix_len([1, 2, 3], [1, 2, 3, 4]) == 3

    def test_diverges_midway(self):
        assert common_prefix_len([1, 2, 3], [1, 9, 3]) == 1

    def test_empty_stored(self):
        assert common_prefix_len([], [1, 2]) == 0


class TestClassify:
    def test_empty_chain_is_new(self):
        decision = classify([], ["a"])
        assert decision.kind is TurnKind.NEW

    def test_extends_prefix_is_append(self):
        decision = classify(["a", "b"], ["a", "b", "c"])
        assert decision.kind is TurnKind.APPEND

    def test_exact_match_is_append(self):
        decision = classify(["a", "b"], ["a", "b"])
        assert decision.kind is TurnKind.APPEND

    def test_divergence_is_wipe(self):
        decision = classify(["a", "b"], ["a", "x"])
        assert decision.kind is TurnKind.WIPE

    def test_wipe_reports_matched_prefix_len(self):
        decision = classify(["a", "b", "c"], ["a", "b", "x"])
        assert decision.matched_prefix_len == 2


class TestMessageHashFingerprinter:
    def test_append_detected_through_message_hashes(self):
        fp = MessageHashFingerprinter()
        stored = fp.units(TurnRequest(system="S", messages=[{"role": "user", "content": "a"}]))
        incoming = fp.units(TurnRequest(
            system="S", messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        ))
        assert classify(stored, incoming).kind is TurnKind.APPEND

    def test_system_change_is_divergence(self):
        fp = MessageHashFingerprinter()
        stored = fp.units(TurnRequest(system="S", messages=[{"role": "user", "content": "a"}]))
        incoming = fp.units(TurnRequest(system="OTHER", messages=[{"role": "user", "content": "a"}]))
        assert classify(stored, incoming).kind is TurnKind.WIPE

    def test_tolerates_token_drift(self):
        """Message-hash ignores rendered token ids entirely -- identical
        messages with different prompt_ids still append."""
        fp = MessageHashFingerprinter()
        messages = [{"role": "user", "content": "a"}]
        stored = fp.units(TurnRequest(messages=messages, prompt_ids=[1, 2, 3]))
        incoming = fp.units(TurnRequest(messages=messages, prompt_ids=[9, 9, 9]))
        assert stored == incoming


class TestTokenPrefixFingerprinter:
    def test_append_detected_through_token_prefix(self):
        fp = TokenPrefixFingerprinter()
        stored = fp.units(TurnRequest(prompt_ids=[1, 2, 3]))
        incoming = fp.units(TurnRequest(prompt_ids=[1, 2, 3, 4, 5]))
        assert classify(stored, incoming).kind is TurnKind.APPEND

    def test_token_shift_is_divergence(self):
        """Strict: a token shift the message-hash strategy would tolerate is a
        divergence here."""
        fp = TokenPrefixFingerprinter()
        stored = fp.units(TurnRequest(prompt_ids=[1, 2, 3]))
        incoming = fp.units(TurnRequest(prompt_ids=[1, 9, 3, 4]))
        assert classify(stored, incoming).kind is TurnKind.WIPE


class TestMakeFingerprinter:
    def test_default_is_message_hash(self):
        assert make_fingerprinter(DEFAULT_TURN_MATCHING).name == "message_hash"

    def test_builds_token_prefix(self):
        assert make_fingerprinter("token_prefix").name == "token_prefix"

    def test_unknown_strategy_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown turn-matching strategy"):
            make_fingerprinter("bogus")
