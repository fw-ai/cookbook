"""Multi-turn trajectory assembly with prefix-equality invariant.

The cookbook's :class:`RolloutPayload` is token-native: every turn carries
``token_ids`` straight from the inference call, and assistant turns carry
per-token ``logprobs`` aligned 1:1 with those token IDs.  When a rollout
function makes more than one engine call (multi-turn agents, tool-using
loops, retry-with-feedback patterns), it has to stitch the per-call
results into a single payload *without* re-tokenizing any text.  Re-
tokenization silently drifts the loss mask off the BPE boundary the
engine actually generated and misaligns the inference logprobs with the
tokens the trainer sees.

This module provides :class:`TrajectoryAssembler` -- a thin helper that
carries the AReaL ``MultiTurnWorkflow`` invariant
(``areal/workflow/multi_turn.py``): each engine call's ``input_tokens``
must start with the already-accumulated sequence.  When the invariant
holds, the assembler folds the call into the running trajectory
(non-assistant gap + assistant output).  When it breaks -- engine saw
something the assembler didn't record, usually because the rollout
function re-rendered messages or skipped a turn -- the assembler raises
:class:`PrefixMismatch` with the first divergence index instead of
training on misaligned tokens.

Slime relies on author discipline (no assert), AReaL has the assert in
each workflow's ``arun_episode``, tinker-cookbook silently splits into
extra datums.  We pick AReaL's "loud crash" mode and centralize it in
one helper so every rollout function gets the same guarantee for free.

Usage::

    assembler = TrajectoryAssembler(tokenizer_id=ctx.tokenizer_id)

    # First engine call -- the input is the initial prompt.
    call = extract_completion(response.choices[0])
    assembler.add_call(call)

    # ... feed tool result, build next prompt by concatenating engine
    #     tokens (NOT re-rendering text), call engine again ...

    call2 = extract_completion(response2.choices[0])
    assembler.add_call(call2, role_before="tool")  # gap was a tool reply

    return assembler.to_payload(total_reward=reward)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from training.utils.rl.rollout_service import Role, RolloutPayload, TurnRecord


__all__ = [
    "InferenceCall",
    "PrefixMismatch",
    "TrajectoryAssembler",
]


@dataclass
class InferenceCall:
    """One engine round trip, captured token-natively.

    ``input_tokens`` is the full prompt the engine saw (including any
    chat template suffix appended after the prior turn's assistant
    message).  ``output_tokens`` and ``output_logprobs`` come straight
    from the engine response and must align 1:1.
    """

    input_tokens: List[int]
    output_tokens: List[int]
    output_logprobs: List[float]
    finish_reason: str = "stop"
    output_versions: Optional[List[int]] = None
    """Optional per-token deployment version, aligned with
    ``output_tokens``.  Threaded through :meth:`TrajectoryAssembler.to_flat`
    for off-policy correction; not yet carried through ``RolloutPayload``."""


class PrefixMismatch(RuntimeError):
    """Raised when an engine call's ``input_tokens`` doesn't extend the
    accumulated trajectory.

    Almost always caused by re-tokenizing text between turns instead of
    concatenating the engine's output tokens verbatim.  The exception
    message includes the first divergence index and the conflicting
    token IDs.
    """


def _first_divergence(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _is_strict_prefix(prior: List[int], full: List[int]) -> bool:
    if len(full) < len(prior):
        return False
    for i, t in enumerate(prior):
        if full[i] != t:
            return False
    return True


@dataclass
class TrajectoryAssembler:
    """Stitch multi-turn engine calls into a token-native trajectory.

    Each :meth:`add_call` records one engine round trip.  Between calls,
    any tokens injected by the environment (tool replies, user
    follow-ups, generation-prompt suffixes) are derived from the next
    call's ``input_tokens`` -- the assembler asserts those gap tokens
    sit cleanly after the previously accumulated sequence.

    Attributes:
        tokenizer_id: Identifier for the tokenizer that produced the
            token IDs.  Propagated to :class:`RolloutPayload` so the
            packer can fail loud on a mismatched-vocab integration.
        role_for_input: Default role assigned to non-assistant gap
            tokens.  Override per call with ``add_call(role_before=...)``.
            ``"user"`` is the right default for chat-template gaps;
            pass ``"tool"`` when the gap is a tool reply.
    """

    tokenizer_id: Optional[str] = None
    role_for_input: Role = "user"
    _turns: List[TurnRecord] = field(default_factory=list, init=False, repr=False)
    _seq: List[int] = field(default_factory=list, init=False, repr=False)
    _flat_versions: List[int] = field(default_factory=list, init=False, repr=False)

    def add_call(self, call: InferenceCall, *, role_before: Optional[Role] = None) -> None:
        """Record one engine call and advance the trajectory.

        Raises:
            PrefixMismatch: if ``call.input_tokens`` doesn't start with
                the already-accumulated sequence.
            ValueError: if ``output_logprobs`` length doesn't match
                ``output_tokens``, or if ``output_versions`` is set and
                its length doesn't match.
        """
        if len(call.output_logprobs) != len(call.output_tokens):
            raise ValueError(
                f"output_logprobs length ({len(call.output_logprobs)}) "
                f"!= output_tokens length ({len(call.output_tokens)})",
            )
        if call.output_versions is not None and len(call.output_versions) != len(call.output_tokens):
            raise ValueError(
                f"output_versions length ({len(call.output_versions)}) "
                f"!= output_tokens length ({len(call.output_tokens)})",
            )

        prior_len = len(self._seq)
        if prior_len:
            if not _is_strict_prefix(self._seq, call.input_tokens):
                idx = _first_divergence(self._seq, call.input_tokens)
                prior_tok = self._seq[idx] if idx < prior_len else None
                input_tok = call.input_tokens[idx] if idx < len(call.input_tokens) else None
                raise PrefixMismatch(
                    f"engine input_tokens diverges from accumulated sequence "
                    f"at index {idx} (prior_len={prior_len}, "
                    f"input_len={len(call.input_tokens)}, "
                    f"prior_token={prior_tok}, input_token={input_tok}). "
                    f"This usually means the rollout function re-tokenized text "
                    f"between turns instead of concatenating engine output tokens "
                    f"verbatim.  Build the next prompt by appending engine tokens + "
                    f"a precomputed chat-template suffix (see "
                    f"training.utils.rl.rollout_helpers.precompute_chat_suffix).",
                )
            gap_tokens = list(call.input_tokens[prior_len:])
        else:
            gap_tokens = list(call.input_tokens)

        gap_role: Role = role_before or self.role_for_input
        if gap_tokens:
            self._turns.append(TurnRecord(role=gap_role, token_ids=gap_tokens))
            self._seq.extend(gap_tokens)
            self._flat_versions.extend([-1] * len(gap_tokens))

        self._turns.append(
            TurnRecord(
                role="assistant",
                token_ids=list(call.output_tokens),
                logprobs=list(call.output_logprobs),
                finish_reason=call.finish_reason,
            ),
        )
        self._seq.extend(call.output_tokens)
        if call.output_versions is not None:
            self._flat_versions.extend(call.output_versions)
        else:
            self._flat_versions.extend([-1] * len(call.output_tokens))

    def add_environment_tokens(
        self,
        tokens: List[int],
        *,
        role: Role = "tool",
    ) -> None:
        """Record non-assistant tokens that won't appear in the next call's
        ``input_tokens``.

        Most users don't need this -- :meth:`add_call` derives gap tokens
        from the prefix delta, which works whenever the engine receives
        the full conversation each turn.  Use this only when your engine
        API takes incremental prompts and the trajectory has to record
        tokens the engine never sees as input.
        """
        if not tokens:
            return
        self._turns.append(TurnRecord(role=role, token_ids=list(tokens)))
        self._seq.extend(tokens)
        self._flat_versions.extend([-1] * len(tokens))

    def to_payload(self, *, total_reward: Optional[float] = None) -> RolloutPayload:
        """Emit a :class:`RolloutPayload` ready for the trainer packer.

        Sets the ``_assembled`` flag so the packer skips its defensive
        prefix-consistency check.
        """
        if not self._turns:
            raise RuntimeError("assembler is empty; call add_call first")
        last_assistant = next(
            (t for t in reversed(self._turns) if t.role == "assistant"),
            None,
        )
        if last_assistant is None:
            raise RuntimeError("assembler has no assistant turn; nothing to train on")
        payload = RolloutPayload(
            turns=list(self._turns),
            total_reward=total_reward,
            tokenizer_id=self.tokenizer_id,
            finish_reason=last_assistant.finish_reason or "stop",
        )
        payload._assembled = True  # type: ignore[attr-defined]
        return payload

    def to_flat(self) -> tuple[List[int], List[float], List[int], List[int]]:
        """Return ``(tokens, logprobs, loss_mask, versions)`` directly.

        Useful when feeding a custom packer that doesn't take
        :class:`RolloutPayload`.  ``versions`` is filled with ``-1`` for
        non-assistant tokens and for assistant tokens whose call didn't
        provide ``output_versions``.
        """
        tokens: List[int] = []
        logprobs: List[float] = []
        loss_mask: List[int] = []
        for t in self._turns:
            assert t.token_ids is not None
            n = len(t.token_ids)
            tokens.extend(t.token_ids)
            if t.role == "assistant":
                assert t.logprobs is not None and len(t.logprobs) == n
                logprobs.extend(t.logprobs)
                loss_mask.extend([1] * n)
            else:
                logprobs.extend([0.0] * n)
                loss_mask.extend([0] * n)
        assert len(self._flat_versions) == len(tokens)
        return tokens, logprobs, loss_mask, list(self._flat_versions)

    @property
    def accumulated_tokens(self) -> List[int]:
        """The current engine-visible token sequence (read-only view)."""
        return list(self._seq)
