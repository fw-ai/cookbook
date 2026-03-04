"""Tests for PromptGroup data container."""

from __future__ import annotations

from unittest.mock import MagicMock

from training_cookbook.utils.rl.losses import PromptGroup


class TestPromptGroup:
    def test_default_inf_logprobs(self):
        pg = PromptGroup(
            data=[MagicMock()],
            advantages=[1.0],
            ref_logprobs=[[0.1]],
            prompt_len=5,
            rewards=[1.0],
        )
        assert pg.inf_logprobs == []

    def test_fields(self):
        data = [MagicMock(), MagicMock()]
        pg = PromptGroup(
            data=data,
            advantages=[1.0, -0.5],
            ref_logprobs=[[0.1], [0.2]],
            prompt_len=10,
            rewards=[1.0, 0.0],
            inf_logprobs=[[0.3], [0.4]],
        )
        assert len(pg.data) == 2
        assert pg.prompt_len == 10
        assert pg.advantages == [1.0, -0.5]
