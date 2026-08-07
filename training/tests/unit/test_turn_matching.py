"""Unit tests for structured-message turn matching."""

from __future__ import annotations

from training.utils.rl.agent.turn_matching import (
    MessageHashFingerprinter,
    TurnKind,
    TurnRequest,
    classify,
    common_prefix_len,
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

    def test_message_change_is_divergence(self):
        fp = MessageHashFingerprinter()
        stored = fp.units(TurnRequest(messages=[{"role": "user", "content": "a"}]))
        incoming = fp.units(TurnRequest(messages=[{"role": "user", "content": "b"}]))
        assert classify(stored, incoming).kind is TurnKind.WIPE
