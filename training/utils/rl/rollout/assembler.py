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
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from training.utils.rl.rollout.service import Role, RolloutPayload, TurnRecord


__all__ = [
    "InferenceCall",
    "PrefixMismatch",
    "TrajectoryAssembler",
    "extract_completion",
    "precompute_chat_suffix",
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

    def add_call(
        self,
        call: InferenceCall,
        *,
        role_before: Optional[Role] = None,
        max_trim_tokens: int = 0,
    ) -> None:
        """Record one engine call and advance the trajectory.

        Raises:
            PrefixMismatch: if ``call.input_tokens`` doesn't start with
                the already-accumulated sequence, after applying the explicit
                tokenizer-boundary trim allowed by ``max_trim_tokens``.
            ValueError: if ``output_logprobs`` length doesn't match
                ``output_tokens``.
        """
        if len(call.output_logprobs) != len(call.output_tokens):
            raise ValueError(
                f"output_logprobs length ({len(call.output_logprobs)}) "
                f"!= output_tokens length ({len(call.output_tokens)})",
            )

        if max_trim_tokens < 0:
            raise ValueError("max_trim_tokens must be >= 0")

        prior_len = len(self._seq)
        if prior_len:
            if not _is_strict_prefix(self._seq, call.input_tokens):
                keep_len = prior_len - max_trim_tokens
                if keep_len < 0 or not _is_strict_prefix(self._seq[:keep_len], call.input_tokens):
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
                        f":func:`precompute_chat_suffix` in this module).",
                    )
                self._trim_engine_visible_tail(max_trim_tokens)
                prior_len = len(self._seq)
            gap_tokens = list(call.input_tokens[prior_len:])
        else:
            gap_tokens = list(call.input_tokens)

        gap_role: Role = role_before or self.role_for_input
        if gap_tokens:
            self._turns.append(TurnRecord(role=gap_role, token_ids=gap_tokens))
            self._seq.extend(gap_tokens)

        self._turns.append(
            TurnRecord(
                role="assistant",
                token_ids=list(call.output_tokens),
                logprobs=list(call.output_logprobs),
                finish_reason=call.finish_reason,
            ),
        )
        self._seq.extend(call.output_tokens)

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

        Important: these tokens are NOT added to ``_seq`` (the
        engine-visible accumulated sequence) because they will not appear
        in the next ``call.input_tokens``.  Adding them would make
        :meth:`add_call`'s strict-prefix invariant fail on the very next
        engine call: ``input_tokens`` would not start with ``_seq +
        env_tokens``.  We still record them in ``_turns`` so they show up
        in the flat trajectory the trainer consumes.
        """
        if not tokens:
            return
        self._turns.append(TurnRecord(role=role, token_ids=list(tokens)))

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

    def to_flat(self) -> tuple[List[int], List[float], List[int]]:
        """Return ``(tokens, logprobs, loss_mask)`` directly.

        Useful when feeding a custom packer that doesn't take
        :class:`RolloutPayload`.
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
        return tokens, logprobs, loss_mask

    def to_trajectory(
        self,
        *,
        tokenizer: Any | None = None,
        source: str = "trajectory_assembler",
    ):
        """Return a native trajectory analysis for verifier visualization."""
        from training.utils.rl.rollout.trace import analyze_turns

        return analyze_turns(self._turns, tokenizer=tokenizer, source=source)

    @property
    def accumulated_tokens(self) -> List[int]:
        """The current engine-visible token sequence (read-only view)."""
        return list(self._seq)

    def _trim_engine_visible_tail(self, n: int) -> None:
        """Drop an explicit tokenizer-boundary tail from the flat trajectory."""
        if n == 0:
            return
        if n > len(self._seq):
            raise PrefixMismatch(f"cannot trim {n} tokens from accumulated sequence of length {len(self._seq)}")
        self._seq = self._seq[:-n]
        remaining = n
        while remaining and self._turns:
            turn = self._turns[-1]
            assert turn.token_ids is not None
            if len(turn.token_ids) > remaining:
                del turn.token_ids[-remaining:]
                if turn.role == "assistant" and turn.logprobs is not None:
                    del turn.logprobs[-remaining:]
                remaining = 0
            else:
                remaining -= len(turn.token_ids)
                self._turns.pop()
        if remaining:
            raise PrefixMismatch(f"cannot trim {n} tokens from trajectory turns")


# ---------------------------------------------------------------------------
# Token-native helpers used alongside TrajectoryAssembler
# ---------------------------------------------------------------------------


def extract_completion(
    choice: Mapping[str, Any],
    *,
    input_tokens: Sequence[int],
) -> InferenceCall:
    """Build an :class:`InferenceCall` from a completions-API choice dict.

    Expects Fireworks/OpenAI-shaped fields:

    * ``token_ids``: list of int, the engine's output token IDs.
    * ``logprobs.token_logprobs``: list of float aligned 1:1 with
      ``token_ids``.  ``None`` entries are coerced to ``0.0`` (some
      providers omit the logprob for the first token).
    * ``finish_reason``: str (default ``"stop"``).

    Raises:
        ValueError: if ``token_ids`` is missing/empty or ``logprobs``
            doesn't align with ``token_ids``.  Both are required for
            token-native training.
    """
    raw_token_ids = choice.get("token_ids")
    if not raw_token_ids:
        raise ValueError(
            "completion choice is missing 'token_ids'; this is required for "
            "token-native rollout assembly.  Enable token-id passthrough on "
            "the inference call (e.g. set ``logprobs=True`` and request "
            "``token_ids`` from the Fireworks Completions API).",
        )

    raw_logprobs = choice.get("logprobs")
    if isinstance(raw_logprobs, Mapping):
        token_logprobs: Iterable[Any] = raw_logprobs.get("token_logprobs") or []
    else:
        token_logprobs = []
    raw_logprobs_list: List[Any] = list(token_logprobs)

    # Filter token_ids/token_logprobs in lockstep so a None placeholder
    # in tokens drops its paired logprob; otherwise the remaining logprobs
    # would shift onto the wrong tokens and corrupt the PPO/GRPO ratio.
    output_tokens: List[int] = []
    output_logprobs: List[float] = []
    if len(raw_logprobs_list) == len(raw_token_ids):
        for tok, lp in zip(raw_token_ids, raw_logprobs_list):
            if tok is None:
                continue
            output_tokens.append(int(tok))
            output_logprobs.append(float(lp) if lp is not None else 0.0)
    else:
        output_tokens = [int(t) for t in raw_token_ids if t is not None]
        output_logprobs = [
            float(lp) if lp is not None else 0.0 for lp in raw_logprobs_list
        ]
        if len(output_logprobs) > len(output_tokens):
            output_logprobs = output_logprobs[: len(output_tokens)]

    if len(output_logprobs) != len(output_tokens):
        raise ValueError(
            f"completion has {len(output_tokens)} token_ids but "
            f"{len(output_logprobs)} logprobs; the inference call must "
            f"return per-token logprobs aligned with token_ids "
            f"(slime/AReaL convention).",
        )

    finish_reason = choice.get("finish_reason") or "stop"

    return InferenceCall(
        input_tokens=list(input_tokens),
        output_tokens=output_tokens,
        output_logprobs=output_logprobs,
        finish_reason=str(finish_reason),
    )


def precompute_chat_suffix(
    tokenizer: Any,
    *,
    follow_up_content: str,
    follow_up_role: str = "user",
    add_generation_prompt: bool = True,
) -> List[int]:
    """Tokenize the chat-template scaffolding that follows an assistant turn.

    Returns the token IDs for ``[end-of-assistant-turn + follow_up_role
    wrapper + follow_up_content + generation prompt]``, computed by
    diffing two ``apply_chat_template`` outputs.  Append this to the
    engine's assistant output tokens to build the next prompt without
    re-tokenizing the assistant content.

    AReaL reference: ``MultiTurnWorkflow.__init__``
    (``areal/workflow/multi_turn.py:41-57``).

    Raises:
        RuntimeError: if the tokenizer's chat template doesn't extend
            cleanly when a follow-up message is appended -- usually
            because the renderer mutates earlier turns (e.g. strips
            ``<think>`` blocks).  In that case there is no stable
            suffix and you need a renderer-aware assembly path.
    """
    base = [{"role": "assistant", "content": "x"}]
    s1 = list(tokenizer.apply_chat_template(base, tokenize=True))

    extended = base + [{"role": follow_up_role, "content": follow_up_content}]
    s2 = list(
        tokenizer.apply_chat_template(
            extended,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        ),
    )

    if len(s2) < len(s1) or s2[: len(s1)] != s1:
        raise RuntimeError(
            "tokenizer's chat template does not extend cleanly when adding a "
            "follow-up message: the prefix re-tokenizes differently after the "
            "second turn is appended.  This usually means the renderer mutates "
            "earlier turns (e.g. Qwen3 strips <think> blocks, KimiK2 injects "
            "default system prompts).  No stable suffix exists for this model "
            "-- use renderer-aware assembly that re-renders each turn through "
            "the renderer's own helpers.",
        )
    return s2[len(s1) :]
