"""Tests for the protocol-neutral deployment training-session tree."""

from __future__ import annotations

import asyncio

import pytest

from training.utils.rl.agent.session import DeploymentTrainingSession
from training.utils.rl.agent.trajectory import (
    SelectedLeaf,
    TrainingSessionTree,
    TurnRecord,
)


def _turn(prompt: list[int], output: list[int], logprob: float) -> TurnRecord:
    return TurnRecord(
        prompt_ids=prompt,
        output_ids=output,
        output_log_probs=[logprob] * len(output),
        finish_reason="stop",
    )


def test_materialize_masks_shared_generated_prefix_after_first_leaf():
    tree = TrainingSessionTree()
    root = tree.add_turn(
        _turn([1], [10], -0.1),
        response_id="response-root",
    )
    left = tree.add_turn(
        _turn([1, 10, 2], [20], -0.2),
        parent_id=root.node_id,
        response_id="response-left",
    )
    right = tree.add_turn(
        _turn([1, 10, 3], [30], -0.3),
        parent_id=root.node_id,
        response_id="response-right",
    )

    segments = tree.materialize(
        [
            SelectedLeaf(left.node_id, {"branch": "left"}),
            SelectedLeaf(right.node_id, {"branch": "right"}),
        ]
    )

    assert [segment.response_ids for segment in segments] == [
        [10, 2, 20],
        [10, 3, 30],
    ]
    assert [segment.loss_mask for segment in segments] == [
        [1, 0, 1],
        [0, 0, 1],
    ]
    assert segments[1].rollout_log_probs == [0.0, 0.0, -0.3]
    assert segments[0].metadata["tree_response_ids"] == [
        "response-root",
        "response-left",
    ]
    assert segments[1].metadata["branch"] == "right"


def test_materialize_does_not_claim_dropped_branch_outputs():
    tree = TrainingSessionTree()
    root = tree.add_turn(_turn([1], [10], -0.1))
    shared = tree.add_turn(
        _turn([9], [20], -0.2),
        parent_id=root.node_id,
    )
    oversized = tree.add_turn(
        _turn([9, 20, 2, 2, 2], [30], -0.3),
        parent_id=shared.node_id,
    )
    sibling = tree.add_turn(
        _turn([9, 20, 4], [40], -0.4),
        parent_id=shared.node_id,
    )

    segments = tree.materialize(
        [SelectedLeaf(oversized.node_id), SelectedLeaf(sibling.node_id)],
        max_context_tokens=5,
    )

    assert [segment.response_ids for segment in segments] == [
        [10],
        [20, 4, 40],
    ]
    assert [segment.loss_mask for segment in segments] == [
        [1],
        [1, 0, 1],
    ]


def test_materialize_forks_token_mismatch_without_losing_output():
    tree = TrainingSessionTree()
    root = tree.add_turn(_turn([1, 2], [10, 11], -0.1))
    leaf = tree.add_turn(
        # The re-rendered prompt diverges inside the sampled root output.
        _turn([1, 2, 10, 99], [30], -0.2),
        parent_id=root.node_id,
    )

    segments = tree.materialize([SelectedLeaf(leaf.node_id)])

    assert [segment.prompt_ids for segment in segments] == [[1, 2], [1, 2, 10, 99]]
    assert [segment.response_ids for segment in segments] == [[10, 11], [30]]
    assert [segment.loss_mask for segment in segments] == [[1, 1], [1]]
    assert [segment.trainable_turn_indices for segment in segments] == [[0], [1]]
    assert [segment.metadata["append_token_mismatches"] for segment in segments] == [
        0,
        1,
    ]


def test_materialize_rejects_nonleaf_and_duplicate_selections():
    tree = TrainingSessionTree()
    root = tree.add_turn(_turn([1], [10], -0.1))
    leaf = tree.add_turn(_turn([1, 10, 2], [20], -0.2), parent_id=root.node_id)

    with pytest.raises(ValueError, match="is not a leaf"):
        tree.materialize([SelectedLeaf(root.node_id)])
    with pytest.raises(ValueError, match="selected twice"):
        tree.materialize([SelectedLeaf(leaf.node_id), SelectedLeaf(leaf.node_id)])


def test_add_turn_rejects_unknown_parent():
    tree = TrainingSessionTree()

    with pytest.raises(ValueError, match="unknown training-session node"):
        tree.add_turn(_turn([1], [10], -0.1), parent_id=3)


def test_deployment_session_uses_opaque_affinity_for_every_call():
    class Sampler:
        def __init__(self) -> None:
            self.users: list[str] = []

        async def sample_with_prompt_tokens(self, prompt_ids, **kwargs):
            del prompt_ids
            self.users.append(kwargs["user"])
            return []

    session = DeploymentTrainingSession(affinity_key="opaque-affinity")
    sampler = Sampler()

    asyncio.run(session.sample_with_prompt_tokens(sampler, [1]))
    asyncio.run(session.sample_with_prompt_tokens(sampler, [2]))

    assert sampler.users == ["opaque-affinity", "opaque-affinity"]


def test_deployment_session_rejects_conflicting_user_override():
    session = DeploymentTrainingSession(affinity_key="opaque-affinity")

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            session.sample_with_prompt_tokens(
                object(),
                [1],
                user="logical-rollout-id",
            )
        )
