"""Native rollout trajectory analysis.

Rollout correctness is a trajectory property, not a renderer artifact
property.  This module keeps the primary shape close to the RL data model:
turns, token rows, and issues derived from token-native rollout data.

The verifier UI can visualize this shape directly and reuse its rule engine
over ``trajectory.tokens``.  No live probe or renderer round trip is needed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass
class TrajectoryIssue:
    code: str
    severity: str
    message: str
    turn_idx: int | None = None
    token_idx: int | None = None


@dataclass
class TrajectoryToken:
    idx: int
    token_id: int
    decoded: str = ""
    role: str = ""
    turn_idx: int | None = None
    source: str = ""
    train_weight: float = 0.0
    provenance: str = "prompt_hard_append"
    issues: list[str] = field(default_factory=list)

    def to_viewer_row(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "token_id": self.token_id,
            "decoded": self.decoded,
            "chunk_source": self.source,
            "msg_idx": None,
            "turn_idx": self.turn_idx,
            "role": self.role,
            "renderer_claim_weight": self.train_weight,
            "provenance": self.provenance,
            "issues": list(self.issues),
        }


@dataclass
class RolloutTrajectory:
    source: str
    tokens: list[TrajectoryToken]
    issues: list[TrajectoryIssue] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "turn_count": len(self.turns),
            "token_count": len(self.tokens),
            "generated_token_count": sum(1 for t in self.tokens if t.train_weight > 0.0),
            "issue_count": len(self.issues),
            "prefix_mismatch_count": sum(1 for i in self.issues if i.code == "prefix_mismatch"),
            "logprob_mismatch_count": sum(1 for i in self.issues if i.code == "logprob_mismatch"),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "rollout_trajectory",
            "source": self.source,
            "metadata": dict(self.metadata),
            "summary": self.summary(),
            "turns": list(self.turns),
            "tokens": [t.to_viewer_row() for t in self.tokens],
            "issues": [asdict(i) for i in self.issues],
        }


def analyze_token_turn_traces(
    token_turn_traces: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any | None = None,
    source: str = "token_turn_traces",
    metadata: Mapping[str, Any] | None = None,
) -> RolloutTrajectory:
    """Analyze per-engine-call traces with the prefix-extension invariant."""
    tokens: list[TrajectoryToken] = []
    issues: list[TrajectoryIssue] = []
    turns: list[dict[str, Any]] = []
    seq: list[int] = []

    def add_token_rows(
        ids: Sequence[int],
        *,
        turn_idx: int,
        role: str,
        source_name: str,
        train_weight: float,
        provenance: str,
        row_issues: list[str] | None = None,
    ) -> None:
        for tok_id in ids:
            tokens.append(
                TrajectoryToken(
                    idx=len(tokens),
                    token_id=int(tok_id),
                    decoded=_decode_one(tokenizer, int(tok_id)),
                    role=role,
                    turn_idx=turn_idx,
                    source=source_name,
                    train_weight=train_weight,
                    provenance=provenance,
                    issues=list(row_issues or []),
                )
            )

    for turn_idx, trace in enumerate(token_turn_traces):
        prompt_ids = [int(x) for x in trace.get("prompt_ids") or []]
        completion_ids = [int(x) for x in trace.get("completion_ids") or []]
        completion_logprobs = trace.get("completion_logprobs")
        turns.append(
            {
                "turn_idx": turn_idx,
                "prompt_len": len(prompt_ids),
                "completion_len": len(completion_ids),
                "finish_reason": trace.get("finish_reason"),
            }
        )

        if completion_logprobs is not None and len(list(completion_logprobs)) != len(completion_ids):
            issues.append(
                TrajectoryIssue(
                    code="logprob_mismatch",
                    severity="error",
                    message=(
                        f"turn {turn_idx} has {len(completion_ids)} completion tokens "
                        f"but {len(list(completion_logprobs))} logprobs"
                    ),
                    turn_idx=turn_idx,
                )
            )

        if len(prompt_ids) >= len(seq) and prompt_ids[: len(seq)] == seq:
            gap = prompt_ids[len(seq) :]
            add_token_rows(
                gap,
                turn_idx=turn_idx,
                role="user",
                source_name="prompt_delta",
                train_weight=0.0,
                provenance="prompt_hard_append",
            )
            seq.extend(gap)
        else:
            idx = _first_divergence(seq, prompt_ids)
            issues.append(
                TrajectoryIssue(
                    code="prefix_mismatch",
                    severity="error",
                    message=(
                        f"turn {turn_idx} prompt does not extend accumulated trajectory "
                        f"at index {idx}"
                    ),
                    turn_idx=turn_idx,
                    token_idx=idx,
                )
            )
            add_token_rows(
                prompt_ids,
                turn_idx=turn_idx,
                role="user",
                source_name="prompt_full_after_mismatch",
                train_weight=0.0,
                provenance="tokenization_diverged",
                row_issues=["prefix_mismatch"],
            )
            seq = list(prompt_ids)

        add_token_rows(
            completion_ids,
            turn_idx=turn_idx,
            role="assistant",
            source_name="completion",
            train_weight=1.0,
            provenance="native_generated",
        )
        seq.extend(completion_ids)

    return RolloutTrajectory(
        source=source,
        tokens=tokens,
        issues=issues,
        turns=turns,
        metadata=dict(metadata or {}),
    )


def analyze_turns(
    turns: Iterable[Any],
    *,
    tokenizer: Any | None = None,
    source: str = "rollout_payload",
    metadata: Mapping[str, Any] | None = None,
) -> RolloutTrajectory:
    """Analyze ``RolloutPayload.turns`` or equivalent dictionaries."""
    out_tokens: list[TrajectoryToken] = []
    issues: list[TrajectoryIssue] = []
    out_turns: list[dict[str, Any]] = []
    for turn_idx, turn in enumerate(turns):
        role = str(_get(turn, "role") or "")
        token_ids = [int(x) for x in (_get(turn, "token_ids") or [])]
        logprobs = _get(turn, "logprobs")
        out_turns.append({"turn_idx": turn_idx, "role": role, "token_len": len(token_ids)})
        if role == "assistant" and (logprobs is None or len(list(logprobs)) != len(token_ids)):
            issues.append(
                TrajectoryIssue(
                    code="logprob_mismatch",
                    severity="error",
                    message=(
                        f"assistant turn {turn_idx} has {len(token_ids)} tokens "
                        f"but {0 if logprobs is None else len(list(logprobs))} logprobs"
                    ),
                    turn_idx=turn_idx,
                )
            )
        train_weight = 1.0 if role == "assistant" else 0.0
        for tok_id in token_ids:
            out_tokens.append(
                TrajectoryToken(
                    idx=len(out_tokens),
                    token_id=tok_id,
                    decoded=_decode_one(tokenizer, tok_id),
                    role=role,
                    turn_idx=turn_idx,
                    source="turn",
                    train_weight=train_weight,
                    provenance="native_generated" if train_weight else "prompt_hard_append",
                )
            )

    if out_turns and out_turns[-1]["role"] != "assistant":
        issues.append(
            TrajectoryIssue(
                code="terminal_non_assistant",
                severity="warning",
                message=f"trajectory ends with role={out_turns[-1]['role']!r}",
                turn_idx=int(out_turns[-1]["turn_idx"]),
            )
        )

    return RolloutTrajectory(
        source=source,
        tokens=out_tokens,
        issues=issues,
        turns=out_turns,
        metadata=dict(metadata or {}),
    )


def analyze_flat_sample(
    sample: Any,
    *,
    tokenizer: Any | None = None,
    source: str = "rollout_sample",
    metadata: Mapping[str, Any] | None = None,
) -> RolloutTrajectory:
    """Analyze a flat ``RolloutSample`` or equivalent dictionary."""
    ids = [int(x) for x in (_get(sample, "tokens") or [])]
    mask = [float(x) for x in (_get(sample, "loss_mask") or [])]
    issues: list[TrajectoryIssue] = []
    if len(ids) != len(mask):
        issues.append(
            TrajectoryIssue(
                code="length_mismatch",
                severity="error",
                message=f"tokens/loss_mask length mismatch: {len(ids)} != {len(mask)}",
            )
        )

    tokens: list[TrajectoryToken] = []
    for idx, tok_id in enumerate(ids):
        train_weight = mask[idx] if idx < len(mask) else 0.0
        tokens.append(
            TrajectoryToken(
                idx=idx,
                token_id=tok_id,
                decoded=_decode_one(tokenizer, tok_id),
                role="assistant" if train_weight > 0.0 else "context",
                turn_idx=None,
                source="loss_mask",
                train_weight=train_weight,
                provenance="native_generated" if train_weight > 0.0 else "prompt_hard_append",
            )
        )

    return RolloutTrajectory(
        source=source,
        tokens=tokens,
        issues=issues,
        turns=[],
        metadata=dict(metadata or {}),
    )


def _decode_one(tokenizer: Any | None, tok_id: int) -> str:
    if tokenizer is None:
        return ""
    try:
        return tokenizer.decode([tok_id], skip_special_tokens=False)
    except Exception:  # noqa: BLE001
        return ""


def _first_divergence(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    for idx in range(n):
        if a[idx] != b[idx]:
            return idx
    return n


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)
