"""Materialize SDK-neutral TITO results into the existing rollout contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from training.utils.rl.rollout.types import RolloutRun, RolloutSample

if TYPE_CHECKING:
    from fireworks.training.sdk import TITOTrajectoryArtifact, TITOTurn


def _assistant_text(turn: TITOTurn) -> str:
    content = turn.assistant.message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    output: list[str] = []
    for part in content:
        if isinstance(part, str):
            output.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            output.append(str(part.get("text", "")))
    return "".join(output)


def _required_sampling_logprobs(turn: TITOTurn) -> list[float]:
    values = turn.sampling_logprobs
    if values is None or len(values) != len(turn.exact_completion_ids):
        raise ValueError(
            f"turn {turn.turn_id} has no completion-aligned sampling logprobs"
        )
    if any(value is None for value in values):
        raise ValueError(f"turn {turn.turn_id} has missing sampling logprob values")
    return [float(value) for value in values if value is not None]


def _request_semantically_continues_turn(
    prior: TITOTurn,
    later: TITOTurn,
) -> bool:
    """Return whether a later accepted request contains the prior action."""
    expected = (*prior.request.messages, prior.assistant.message)
    messages = later.request.messages
    return len(messages) >= len(expected) and all(
        actual == prior_message for actual, prior_message in zip(messages, expected)
    )


def _turn_exactly_continues_prior(prior: TITOTurn, turn: TITOTurn) -> bool:
    if turn.prompt_disposition == "append":
        checkpoint = prior.exact_checkpoint_ids
        trim_tokens = turn.incremental_checkpoint_trim_tokens
        retained = len(checkpoint) - trim_tokens
        return (
            0 <= retained
            and turn.prefix_match_tokens is not None
            and turn.prefix_match_tokens >= retained
            and turn.exact_prompt_ids[:retained] == checkpoint[:retained]
        )
    if turn.prompt_disposition != "realign":
        return False
    start = turn.realign_from_token
    return (
        start is not None
        and 0
        <= start
        <= min(len(prior.exact_checkpoint_ids), len(turn.exact_prompt_ids))
        and prior.exact_checkpoint_ids[:start] == turn.exact_prompt_ids[:start]
        and turn.prefix_match_tokens is not None
        and turn.prefix_match_tokens >= start
        and turn.realigned_masked_tokens == len(turn.exact_prompt_ids) - start
    )


@dataclass
class _SampleBuilder:
    prompt_ids: list[int]
    tokens: list[int]
    logprobs: list[float]
    raw_logprobs: list[float] | None
    loss_mask: list[int]
    response_routes: list[str] | None
    turns: list[TITOTurn] = field(default_factory=list)
    masked_fail_closed_turns: int = 0
    realigned_masked_tokens: int = 0
    incremental_checkpoint_trimmed_tokens: int = 0

    @classmethod
    def from_turn(cls, turn: TITOTurn) -> "_SampleBuilder":
        prompt = list(turn.exact_prompt_ids)
        return cls(
            prompt_ids=prompt,
            tokens=list(prompt),
            logprobs=[0.0] * len(prompt),
            raw_logprobs=(
                [0.0] * len(prompt) if turn.inference_logprobs is not None else None
            ),
            loss_mask=[0] * len(prompt),
            response_routes=[] if turn.routing_matrices is not None else None,
        )

    def can_append(self, turn: TITOTurn) -> bool:
        prompt = list(turn.exact_prompt_ids)
        trim_tokens = turn.incremental_checkpoint_trim_tokens
        retained = len(self.tokens) - trim_tokens
        return (
            0 <= retained
            and len(prompt) >= retained
            and prompt[:retained] == self.tokens[:retained]
        )

    def can_realign(self, turn: TITOTurn) -> bool:
        start = turn.realign_from_token
        prompt = turn.exact_prompt_ids
        return (
            turn.prompt_disposition == "realign"
            and start is not None
            and 0 <= start <= min(len(self.tokens), len(prompt))
            and tuple(self.tokens[:start]) == prompt[:start]
            and turn.prefix_match_tokens is not None
            and turn.prefix_match_tokens >= start
            and turn.realigned_masked_tokens == len(prompt) - start
        )

    def realign(self, turn: TITOTurn, *, trainable: bool = True) -> None:
        if not self.can_realign(turn):
            raise ValueError("turn has invalid bounded realignment evidence")
        start = turn.realign_from_token
        assert start is not None
        prompt = list(turn.exact_prompt_ids)
        replacement = prompt[start:]
        self.tokens[start:] = replacement
        self.logprobs[start:] = [0.0] * len(replacement)
        self.loss_mask[start:] = [0] * len(replacement)
        if self.raw_logprobs is not None:
            self.raw_logprobs[start:] = [0.0] * len(replacement)
        if self.response_routes is not None:
            route_start = start - len(self.prompt_ids)
            if route_start < 0:
                raise ValueError("realignment cannot replace the initial prompt")
            self.response_routes[route_start:] = [""] * len(replacement)
        self.realigned_masked_tokens += turn.realigned_masked_tokens
        self._append_completion(turn, trainable=trainable)

    def append(self, turn: TITOTurn, *, trainable: bool = True) -> None:
        prompt = list(turn.exact_prompt_ids)
        if not self.can_append(turn):
            raise ValueError("turn does not extend this materialization segment")
        trim_tokens = turn.incremental_checkpoint_trim_tokens
        if prompt[: len(self.tokens)] != self.tokens:
            retained = len(self.tokens) - trim_tokens
            if trim_tokens <= 0 or retained < len(self.prompt_ids):
                raise ValueError(
                    "incremental checkpoint trim is not safely materializable"
                )
            del self.tokens[retained:]
            del self.logprobs[retained:]
            del self.loss_mask[retained:]
            if self.raw_logprobs is not None:
                del self.raw_logprobs[retained:]
            if self.response_routes is not None:
                if trim_tokens > len(self.response_routes):
                    raise ValueError(
                        "incremental checkpoint trim exceeds response routes"
                    )
                del self.response_routes[-trim_tokens:]
            self.incremental_checkpoint_trimmed_tokens += trim_tokens
        suffix = prompt[len(self.tokens) :]
        self.tokens.extend(suffix)
        self.logprobs.extend([0.0] * len(suffix))
        self.loss_mask.extend([0] * len(suffix))
        if self.raw_logprobs is not None:
            if turn.inference_logprobs is None:
                self.raw_logprobs = None
            else:
                self.raw_logprobs.extend([0.0] * len(suffix))
        if self.response_routes is not None:
            if turn.routing_matrices is None:
                raise ValueError(
                    "R3 must be present for every turn in one TITO segment"
                )
            self.response_routes.extend([""] * len(suffix))
        elif turn.routing_matrices is not None:
            raise ValueError("R3 cannot begin partway through one TITO segment")

        self._append_completion(turn, trainable=trainable)

    def _append_completion(self, turn: TITOTurn, *, trainable: bool) -> None:
        completion = list(turn.exact_completion_ids)
        self.tokens.extend(completion)
        self.logprobs.extend(_required_sampling_logprobs(turn))
        self.loss_mask.extend([int(trainable)] * len(completion))
        if not trainable:
            self.masked_fail_closed_turns += 1
        if self.raw_logprobs is not None:
            if turn.inference_logprobs is None or len(turn.inference_logprobs) != len(
                completion
            ):
                self.raw_logprobs = None
            else:
                self.raw_logprobs.extend(
                    float(value) for value in turn.inference_logprobs
                )
        if self.response_routes is not None:
            if turn.routing_matrices is None or len(turn.routing_matrices) != len(
                completion
            ):
                raise ValueError(
                    f"turn {turn.turn_id} has completion-misaligned R3 matrices"
                )
            self.response_routes.extend(turn.routing_matrices)
        self.turns.append(turn)

    def build(self, reward: float) -> RolloutSample:
        routing = None
        if self.response_routes is not None:
            # Trainer model_input has len(tokens)-1 positions. Completion-only
            # R3 is padded here for the first prompt and for every later
            # external/tool suffix while retaining one route per sampled token.
            routing = [""] * (len(self.prompt_ids) - 1) + self.response_routes
            if len(routing) != len(self.tokens) - 1:
                raise ValueError(
                    "materialized R3 does not align with model-input positions"
                )
        return RolloutSample(
            tokens=list(self.tokens),
            logprobs=list(self.logprobs),
            raw_logprobs=(
                list(self.raw_logprobs) if self.raw_logprobs is not None else None
            ),
            loss_mask=list(self.loss_mask),
            routing_matrices=routing,
            reward=reward,
            finish_reason=self.turns[-1].finish_reason,
            text="\n".join(_assistant_text(turn) for turn in self.turns),
        )


def materialize_tito_trajectory(
    result: TITOTrajectoryArtifact,
    *,
    reward: float,
    harness_visible_turn_ids: set[str] | None = None,
    harness_abandoned_turn_ids: set[str] | None = None,
    max_context_tokens: int | None = None,
    debug_enabled: bool = False,
) -> RolloutRun:
    """Convert every retained exact segment without decoding/re-tokenizing.

    A turn is retained when a response attempt completed, the next accepted
    policy request semantically contains that assistant response (including
    across a segment boundary), or the harness supplies an equivalent
    visibility fact. An explicitly abandoned length attempt is omitted even
    after completed transport emission. A terminal action whose every response
    attempt is ambiguous is omitted by default. A turn admitted only by the SDK
    classifier's fail-closed fallback remains exact context but its completion
    is masked from training; a physical sample containing no trainable
    completion is omitted. ``max_context_tokens`` is the same total
    prompt-plus-output limit used during inference: over-limit turns are
    omitted at exact turn boundaries, never truncated or re-tokenized.
    """
    if max_context_tokens is not None and max_context_tokens < 1:
        raise ValueError("max_context_tokens must be positive")
    visible = set(harness_visible_turn_ids or ())
    abandoned = set(harness_abandoned_turn_ids or ())
    turns_by_id = {
        turn.turn_id: turn for segment in result.segments for turn in segment.turns
    }
    unknown_abandoned = abandoned - turns_by_id.keys()
    if unknown_abandoned:
        raise ValueError(
            "harness abandoned unknown TITO turns: "
            + ", ".join(sorted(unknown_abandoned))
        )
    invalid_abandoned = {
        turn_id
        for turn_id in abandoned
        if turns_by_id[turn_id].finish_reason != "length"
    }
    if invalid_abandoned:
        raise ValueError(
            "only length-truncated TITO turns may be abandoned by the harness: "
            + ", ".join(sorted(invalid_abandoned))
        )
    visible.update(
        attempt.turn_id
        for attempt in result.response_attempts
        if attempt.emission == "completed"
    )
    chronological_turns = [
        turn for segment in result.segments for turn in segment.turns
    ]
    for prior, later in zip(chronological_turns, chronological_turns[1:]):
        if _request_semantically_continues_turn(prior, later):
            visible.add(prior.turn_id)
    visible.difference_update(abandoned)
    fail_closed_turn_ids = {
        call.turn_id
        for call in result.calls
        if call.kind == "policy"
        and call.outcome == "succeeded"
        and call.classification_source == "fail_closed"
        and call.turn_id is not None
    }
    samples: list[RolloutSample] = []
    materialized_segments: list[dict[str, Any]] = []
    masked_fail_closed_turns = 0
    retention_dropped_turns = 0
    retention_dropped_trainable_tokens = 0
    for segment_index, segment in enumerate(result.segments):
        retained: list[TITOTurn] = []
        for turn in segment.turns:
            if turn.turn_id in abandoned:
                break
            if turn.turn_id not in visible:
                break
            retained.append(turn)
        if not retained:
            continue

        builders: list[_SampleBuilder] = []
        builder: _SampleBuilder | None = None
        prior_turn: TITOTurn | None = None
        for turn_index, turn in enumerate(retained):
            if turn_index == 0:
                if turn.prompt_disposition != "new_segment":
                    raise ValueError(
                        "the first turn in a TITO segment must start a new segment"
                    )
            elif prior_turn is None or not _turn_exactly_continues_prior(
                prior_turn, turn
            ):
                raise ValueError(
                    "turn disposition does not continue its exact prior checkpoint"
                )
            prior_turn = turn

            if (
                max_context_tokens is not None
                and len(turn.exact_prompt_ids) + len(turn.exact_completion_ids)
                > max_context_tokens
            ):
                if builder is not None and builder.turns:
                    builders.append(builder)
                builder = None
                retention_dropped_turns += 1
                if turn.turn_id not in fail_closed_turn_ids:
                    retention_dropped_trainable_tokens += len(turn.exact_completion_ids)
                continue

            trainable = turn.turn_id not in fail_closed_turn_ids
            if builder is None:
                # A post-hoc retention split may begin at any later exact
                # prompt after an over-limit turn.  The complete prompt becomes
                # masked context; no sampled token or aligned array is rebuilt.
                builder = _SampleBuilder.from_turn(turn)
                if turn.prompt_disposition == "realign":
                    builder.realigned_masked_tokens += turn.realigned_masked_tokens
                builder.append(turn, trainable=trainable)
                continue
            if turn.prompt_disposition == "append" and not builder.can_append(turn):
                raise ValueError(
                    "append disposition does not extend the exact checkpoint"
                )
            elif turn.prompt_disposition == "realign" and not builder.can_realign(turn):
                raise ValueError(
                    "realign disposition does not match its exact evidence"
                )
            elif turn.prompt_disposition not in {"append", "realign"}:
                raise ValueError(
                    f"unknown TITO prompt disposition: {turn.prompt_disposition!r}"
                )
            if turn.prompt_disposition == "realign":
                builder.realign(turn, trainable=trainable)
            else:
                builder.append(turn, trainable=trainable)
        if builder is not None:
            builders.append(builder)

        for physical_index, item in enumerate(builders):
            masked_fail_closed_turns += item.masked_fail_closed_turns
            if not any(item.loss_mask):
                continue
            samples.append(item.build(reward))
            materialized_segments.append(
                {
                    "tito_segment_id": segment.segment_id,
                    "tito_segment_index": segment_index,
                    "tito_physical_split_index": physical_index,
                    "tito_start_reason": segment.start_reason,
                    "tito_closed_reason": segment.closed_reason,
                    "tito_masked_fail_closed_turn_count": item.masked_fail_closed_turns,
                    "tito_realigned_masked_tokens": item.realigned_masked_tokens,
                    "tito_incremental_checkpoint_trimmed_tokens": (
                        item.incremental_checkpoint_trimmed_tokens
                    ),
                }
            )
    tito_metrics = result.metrics.flattened(root="tito")
    model_malformed_calls = int(
        result.metrics.counters.get("parser/model_malformed", 0)
    )
    if model_malformed_calls:
        # This is an association, not a causal failure label: the harness may
        # recover on a later request, or may terminate for an unrelated reason.
        tito_metrics["tito/trajectory/model_malformed_observed"] = 1.0
        terminal_suffix = (
            "followed_by_completed"
            if result.status == "completed"
            else "followed_by_terminal"
        )
        tito_metrics[f"tito/trajectory/model_malformed_{terminal_suffix}"] = 1.0

    return RolloutRun(
        segments=samples,
        run_id=result.trajectory_id,
        metadata={
            **dict(result.metadata),
            "tito_metrics": tito_metrics,
            "tito_debug_enabled": bool(debug_enabled),
            "tito_segment_count": len(result.segments),
            "tito_response_attempt_count": len(result.response_attempts),
            "tito_abandoned_turn_count": len(abandoned),
            "tito_masked_fail_closed_turn_count": masked_fail_closed_turns,
            "tito_max_context_tokens": max_context_tokens,
            "tito_retention_dropped_turn_count": retention_dropped_turns,
            "tito_retention_dropped_trainable_tokens": (
                retention_dropped_trainable_tokens
            ),
            "tito_materialized_segments": materialized_segments,
        },
    )


__all__ = ["materialize_tito_trajectory"]
