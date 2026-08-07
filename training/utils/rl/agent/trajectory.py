"""Token provenance and trajectory assembly for agent rollouts.

Examples own conversation semantics: messages, tools, compaction, retries, and
the parent of each sampled turn. This module owns only sampled-token ancestry:

* exact token ancestry appends to the current training segment;
* any token mismatch starts a new segment without discarding sampled output;
* shared tree prefixes contribute loss to only one selected leaf;
* every segment remains part of the same logical rollout and GRPO group member.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """Exact token snapshot for one assistant generation."""

    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = dataclasses.field(default_factory=list)
    output_raw_log_probs: list[float] | None = None
    output_routing_matrices: list[str] | None = None
    text: str = ""
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class TokenSegment:
    """One trainer-ready segment of a logical agent rollout."""

    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]
    rollout_log_probs: list[float] = dataclasses.field(default_factory=list)
    rollout_raw_log_probs: list[float] | None = None
    routing_matrices: list[str] | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    trainable_turn_indices: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class TurnSegment:
    """One selected turn path before token-level assembly."""

    turns: list[TurnRecord]
    train_outputs: list[bool] | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


def make_turn_segment(
    turns: list[TurnRecord],
    *,
    kind: str = "",
    metadata: dict[str, Any] | None = None,
    train_outputs: list[bool] | None = None,
) -> TurnSegment:
    """Create a selected turn path and tag its example-owned boundary kind."""
    frozen_turns = list(turns)
    if train_outputs is not None and len(train_outputs) != len(frozen_turns):
        raise ValueError("train_outputs must align with turns")
    segment_metadata = dict(metadata or {})
    if kind:
        segment_metadata.setdefault("segment_kind", kind)
    segment_metadata.setdefault(
        "finish_reason",
        frozen_turns[-1].finish_reason if frozen_turns else "",
    )
    segment_metadata.setdefault("num_turns", len(frozen_turns))
    return TurnSegment(
        turns=frozen_turns,
        train_outputs=list(train_outputs) if train_outputs is not None else None,
        metadata=segment_metadata,
    )


@dataclasses.dataclass(frozen=True)
class TrainingSessionNode:
    """One immutable sampled turn in a training session tree."""

    node_id: int
    parent_id: int | None
    turn: TurnRecord
    response_id: str = ""
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class SelectedLeaf:
    """A caller-selected terminal path and its example-owned metadata."""

    node_id: int
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class TrainingSessionTree:
    """Immutable sampled-token ancestry for one logical rollout.

    The caller resolves protocol history and supplies ``parent_id``. Recording
    a turn never rewrites existing nodes, so branches can share a sampled prefix
    without duplicating its training loss.
    """

    def __init__(self) -> None:
        self._nodes: list[TrainingSessionNode] = []
        self._children: dict[int, list[int]] = {}

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def recorded_turns(self) -> list[TurnRecord]:
        """Turns in commit order, for example-local rendering support."""
        return [node.turn for node in self._nodes]

    @property
    def leaf_ids(self) -> list[int]:
        return [
            node.node_id for node in self._nodes if not self._children.get(node.node_id)
        ]

    def node(self, node_id: int) -> TrainingSessionNode:
        if node_id < 0 or node_id >= len(self._nodes):
            raise ValueError(f"unknown training-session node {node_id}")
        return self._nodes[node_id]

    def add_turn(
        self,
        turn: TurnRecord,
        *,
        parent_id: int | None = None,
        response_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TrainingSessionNode:
        if parent_id is not None:
            self.node(parent_id)
        node = TrainingSessionNode(
            node_id=len(self._nodes),
            parent_id=parent_id,
            turn=turn,
            response_id=response_id,
            metadata=dict(metadata or {}),
        )
        self._nodes.append(node)
        if parent_id is not None:
            self._children.setdefault(parent_id, []).append(node.node_id)
        return node

    def path(self, node_id: int) -> list[TrainingSessionNode]:
        path: list[TrainingSessionNode] = []
        node = self.node(node_id)
        while True:
            path.append(node)
            if node.parent_id is None:
                break
            node = self.node(node.parent_id)
        path.reverse()
        return path

    def materialize(
        self,
        selected_leaves: Iterable[SelectedLeaf] | None = None,
        *,
        max_context_tokens: int = 0,
    ) -> list[TokenSegment]:
        """Materialize selected root-to-leaf paths in caller-provided order.

        The first selected path containing a generated node owns that node's
        loss. Later paths retain shared generated tokens as masked context.
        """
        selections = list(selected_leaves or ())
        leaf_ids = self.leaf_ids
        if not selections:
            selections = [SelectedLeaf(node_id) for node_id in leaf_ids]
        leaf_id_set = set(leaf_ids)

        seen_leaves: set[int] = set()
        trained_nodes: set[int] = set()
        output: list[TokenSegment] = []
        for selection in selections:
            if selection.node_id in seen_leaves:
                raise ValueError(
                    f"training-session leaf {selection.node_id} selected twice"
                )
            seen_leaves.add(selection.node_id)
            if selection.node_id not in leaf_id_set:
                raise ValueError(
                    f"training-session node {selection.node_id} is not a leaf"
                )

            nodes = self.path(selection.node_id)
            train_outputs = [node.node_id not in trained_nodes for node in nodes]
            metadata = dict(selection.metadata)
            metadata.setdefault("tree_leaf_id", selection.node_id)
            metadata.setdefault("tree_path_node_ids", [node.node_id for node in nodes])
            response_ids = [node.response_id for node in nodes if node.response_id]
            if response_ids:
                metadata.setdefault("tree_response_ids", response_ids)
            segments = merge_turn_segments(
                [
                    make_turn_segment(
                        [node.turn for node in nodes],
                        metadata=metadata,
                        train_outputs=train_outputs,
                    )
                ],
                max_context_tokens=max_context_tokens,
            )
            segments = [segment for segment in segments if any(segment.loss_mask)]
            if not segments:
                continue
            output.extend(segments)
            trained_nodes.update(
                nodes[index].node_id
                for segment in segments
                for index in segment.trainable_turn_indices
            )
        return output


def _required_output_values(
    turn: TurnRecord,
    attribute: str,
) -> list[float] | list[str]:
    values = getattr(turn, attribute)
    if values is None:
        raise ValueError(f"{attribute} is required for every output token")
    if len(values) != len(turn.output_ids):
        raise ValueError(
            f"{attribute} has length {len(values)} for "
            f"{len(turn.output_ids)} output tokens"
        )
    return list(values)


@dataclasses.dataclass
class _MergeState:
    """Aligned token arrays for one exact-ancestry training segment."""

    prompt_ids: list[int]
    keep_raw_log_probs: bool
    keep_routing: bool
    split_reason: str | None = None
    response_ids: list[int] = dataclasses.field(default_factory=list)
    loss_mask: list[int] = dataclasses.field(default_factory=list)
    rollout_log_probs: list[float] = dataclasses.field(default_factory=list)
    rollout_raw_log_probs: list[float] | None = None
    routing_matrices: list[str] | None = None
    output_spans: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    num_turns: int = 0
    finish_reason: str = ""

    @classmethod
    def create(
        cls,
        prompt_ids: list[int],
        *,
        keep_raw_log_probs: bool,
        keep_routing: bool,
        split_reason: str | None = None,
    ) -> _MergeState:
        return cls(
            prompt_ids=list(prompt_ids),
            keep_raw_log_probs=keep_raw_log_probs,
            keep_routing=keep_routing,
            split_reason=split_reason,
            rollout_raw_log_probs=[] if keep_raw_log_probs else None,
            routing_matrices=[] if keep_routing else None,
        )

    def prompt_mismatch(self, prompt_ids: list[int]) -> str | None:
        """Return the exact-ancestry mismatch kind, if any."""
        if prompt_ids[: len(self.prompt_ids)] != self.prompt_ids:
            return "base"
        suffix = prompt_ids[len(self.prompt_ids) :]
        if suffix[: len(self.response_ids)] != self.response_ids:
            return "suffix"
        return None

    def append_prompt(self, prompt_ids: list[int]) -> None:
        """Append caller-validated exact ancestry as masked context."""
        suffix = prompt_ids[len(self.prompt_ids) :]
        self._append_context(suffix[len(self.response_ids) :])

    def _append_context(self, token_ids: list[int]) -> None:
        self.response_ids.extend(token_ids)
        self.loss_mask.extend([0] * len(token_ids))
        self.rollout_log_probs.extend([0.0] * len(token_ids))
        if self.rollout_raw_log_probs is not None:
            self.rollout_raw_log_probs.extend([0.0] * len(token_ids))
        if self.routing_matrices is not None:
            self.routing_matrices.extend([""] * len(token_ids))

    def append_output(
        self,
        turn: TurnRecord,
        *,
        train_output: bool,
        turn_index: int,
    ) -> None:
        output_start = len(self.response_ids)
        self.response_ids.extend(turn.output_ids)
        mask = int(train_output)
        self.loss_mask.extend([mask] * len(turn.output_ids))
        output_log_probs = _required_output_values(turn, "output_log_probs")
        self.rollout_log_probs.extend(
            output_log_probs if train_output else [0.0] * len(turn.output_ids)
        )
        if self.rollout_raw_log_probs is not None:
            raw_log_probs = _required_output_values(turn, "output_raw_log_probs")
            self.rollout_raw_log_probs.extend(
                raw_log_probs if train_output else [0.0] * len(turn.output_ids)
            )
        if self.routing_matrices is not None:
            routing_matrices = _required_output_values(
                turn,
                "output_routing_matrices",
            )
            self.routing_matrices.extend(
                routing_matrices if train_output else [""] * len(turn.output_ids)
            )
        self.output_spans.append((turn_index, output_start, len(self.response_ids)))
        self.num_turns += 1
        self.finish_reason = turn.finish_reason

    def to_segment(
        self,
        metadata: dict[str, Any] | None,
        *,
        split_index: int,
        split_count: int,
        source_num_turns: int,
    ) -> TokenSegment:
        segment_metadata = dict(metadata or {})
        base_mismatches = int(self.split_reason == "base")
        suffix_mismatches = int(self.split_reason == "suffix")
        segment_metadata.update(
            num_turns=self.num_turns,
            finish_reason=self.finish_reason,
            append_token_base_mismatches=base_mismatches,
            append_token_suffix_mismatches=suffix_mismatches,
            append_token_mismatches=base_mismatches + suffix_mismatches,
        )
        if split_count > 1:
            segment_metadata.update(
                token_split_index=split_index,
                token_split_count=split_count,
                source_num_turns=source_num_turns,
            )
            if self.split_reason is not None:
                segment_metadata["token_split_reason"] = self.split_reason
        return TokenSegment(
            prompt_ids=self.prompt_ids,
            response_ids=self.response_ids,
            loss_mask=self.loss_mask,
            rollout_log_probs=[
                value if mask else 0.0
                for value, mask in zip(
                    self.rollout_log_probs,
                    self.loss_mask,
                    strict=True,
                )
            ],
            rollout_raw_log_probs=(
                [
                    value if mask else 0.0
                    for value, mask in zip(
                        self.rollout_raw_log_probs,
                        self.loss_mask,
                        strict=True,
                    )
                ]
                if self.rollout_raw_log_probs is not None
                else None
            ),
            routing_matrices=(
                [
                    value if mask else ""
                    for value, mask in zip(
                        self.routing_matrices,
                        self.loss_mask,
                        strict=True,
                    )
                ]
                if self.routing_matrices is not None
                else None
            ),
            metadata=segment_metadata,
            trainable_turn_indices=sorted(
                turn_index
                for turn_index, start, end in self.output_spans
                if any(self.loss_mask[start:end])
            ),
        )


def _assemble_turns(
    turns: list[TurnRecord],
    train_outputs: list[bool],
    turn_indices: list[int],
) -> list[_MergeState]:
    if not turns:
        return []
    keep_raw_log_probs = all(turn.output_raw_log_probs is not None for turn in turns)
    keep_routing = all(turn.output_routing_matrices is not None for turn in turns)
    state = _MergeState.create(
        turns[0].prompt_ids,
        keep_raw_log_probs=keep_raw_log_probs,
        keep_routing=keep_routing,
    )
    states: list[_MergeState] = []
    for index, (turn, train_output, turn_index) in enumerate(
        zip(turns, train_outputs, turn_indices, strict=True)
    ):
        if index > 0:
            mismatch = state.prompt_mismatch(turn.prompt_ids)
            if mismatch is None:
                state.append_prompt(turn.prompt_ids)
            else:
                states.append(state)
                state = _MergeState.create(
                    turn.prompt_ids,
                    keep_raw_log_probs=keep_raw_log_probs,
                    keep_routing=keep_routing,
                    split_reason=mismatch,
                )
        state.append_output(
            turn,
            train_output=train_output,
            turn_index=turn_index,
        )
    states.append(state)
    return states


def merge_turn_segments(
    segments: list[TurnSegment],
    *,
    max_context_tokens: int = 0,
) -> list[TokenSegment]:
    """Materialize paths, splitting every non-append token boundary.

    A split preserves both sides as physical training segments of the same
    logical rollout. It never realigns, truncates, or discards sampled output.
    """
    output: list[TokenSegment] = []
    for turn_segment in segments:
        train_outputs = (
            [True] * len(turn_segment.turns)
            if turn_segment.train_outputs is None
            else turn_segment.train_outputs
        )
        if len(train_outputs) != len(turn_segment.turns):
            raise ValueError("train_outputs must align with turns")
        states = _assemble_turns(
            turn_segment.turns,
            train_outputs,
            list(range(len(turn_segment.turns))),
        )
        if len(states) > 1:
            logger.info(
                "[trajectory] token ancestry split %d turns into %d segments",
                len(turn_segment.turns),
                len(states),
            )

        for index, state in enumerate(states):
            token_segment = state.to_segment(
                turn_segment.metadata,
                split_index=index,
                split_count=len(states),
                source_num_turns=len(turn_segment.turns),
            )
            total_tokens = len(token_segment.prompt_ids) + len(
                token_segment.response_ids
            )
            if token_segment.response_ids and (
                max_context_tokens <= 0 or total_tokens <= max_context_tokens
            ):
                output.append(token_segment)
    return output


__all__ = [
    "SelectedLeaf",
    "TokenSegment",
    "TrainingSessionNode",
    "TrainingSessionTree",
    "TurnRecord",
    "TurnSegment",
    "make_turn_segment",
    "merge_turn_segments",
]
