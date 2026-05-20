"""Message-in multi-turn assembly with TITO-style token preservation.

This module bridges OpenAI-style message loops to the token-native rollout
contract.  Prior assistant outputs stay as engine token IDs; only newly
appended non-assistant messages are tokenized before the next model call.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, List, Optional

from training.utils.rl.rollout.assembler import InferenceCall, TrajectoryAssembler
from training.utils.rl.rollout.service import RolloutPayload


class MessageTrajectoryError(ValueError):
    """Base error for message trajectory validation failures."""


class MessageValidationError(MessageTrajectoryError):
    """Raised when messages are not append-only or violate role policy."""


class TokenizationError(MessageTrajectoryError):
    """Raised when incremental message tokenization cannot be made safe."""


def _assert_append_only_with_allowed_roles(
    old_messages: List[dict[str, Any]],
    new_messages: List[dict[str, Any]],
    allowed_roles: List[str],
) -> None:
    if len(new_messages) < len(old_messages):
        raise MessageValidationError(
            f"new messages shorter than stored prefix: {len(new_messages)} < {len(old_messages)}"
        )
    for idx, old_msg in enumerate(old_messages):
        if old_msg != new_messages[idx]:
            raise MessageValidationError(f"message at index {idx} modifies stored prefix")
    for idx, msg in enumerate(new_messages[len(old_messages) :], start=len(old_messages)):
        role = msg.get("role")
        if role not in allowed_roles:
            raise MessageValidationError(
                f"appended role={role!r} at index {idx} is not allowed; allowed={allowed_roles}"
            )


class TITOTokenizer:
    """Incrementally tokenize appended non-assistant turns.

    ``merge_tokens`` preserves the pretokenized assistant prefix exactly and
    appends tokenized user/tool/system deltas plus the next assistant opener.
    """

    max_trim_tokens: int = 0

    def __init__(
        self,
        tokenizer: Any,
        *,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        allowed_append_roles: Optional[List[str]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.allowed_append_roles = list(allowed_append_roles or ["tool", "user", "system"])

    def render_initial_prompt(
        self,
        messages: List[dict[str, Any]],
        *,
        tools: Optional[List[dict[str, Any]]] = None,
    ) -> List[int]:
        rendered = self._render_messages(messages, add_generation_prompt=True, tools=tools)
        return self._encode_text(rendered)

    def merge_tokens(
        self,
        *,
        old_messages: List[dict[str, Any]],
        new_messages: List[dict[str, Any]],
        pretokenized_token_ids: List[int],
        tools: Optional[List[dict[str, Any]]] = None,
    ) -> List[int]:
        return list(pretokenized_token_ids) + self.tokenize_additional_non_assistant(
            old_messages,
            new_messages,
            tools=tools,
        )

    def tokenize_additional_non_assistant(
        self,
        old_messages: List[dict[str, Any]],
        new_messages: List[dict[str, Any]],
        *,
        tools: Optional[List[dict[str, Any]]] = None,
    ) -> List[int]:
        _assert_append_only_with_allowed_roles(old_messages, new_messages, self.allowed_append_roles)
        appended_messages = new_messages[len(old_messages) :]
        incremental: List[int] = []
        for segment in self._split_appended_segments(appended_messages):
            role = segment[0].get("role")
            if role == "tool":
                incremental.extend(self._tokenize_tool_segment(segment, tools=tools))
            elif role in {"user", "system"}:
                incremental.extend(self._tokenize_user_or_system_segment(segment[0], tools=tools))
            else:
                raise TokenizationError(f"unsupported appended role for TITO tokenization: {role!r}")
        incremental.extend(self._tokenize_rendered_suffix(new_messages, [], add_generation_prompt=True, tools=tools))
        return incremental

    def _split_appended_segments(self, appended_messages: List[dict[str, Any]]) -> List[List[dict[str, Any]]]:
        segments: List[List[dict[str, Any]]] = []
        i = 0
        while i < len(appended_messages):
            role = appended_messages[i].get("role")
            if role == "tool":
                j = i + 1
                while j < len(appended_messages) and appended_messages[j].get("role") == "tool":
                    j += 1
                segments.append(appended_messages[i:j])
                i = j
            elif role in {"user", "system"}:
                segments.append([appended_messages[i]])
                i += 1
            else:
                raise TokenizationError(f"unsupported appended role for TITO segmentation: {role!r}")
        return segments

    def _tokenize_tool_segment(
        self,
        appended_messages: List[dict[str, Any]],
        *,
        tools: Optional[List[dict[str, Any]]] = None,
    ) -> List[int]:
        dummy_assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": msg.get("tool_call_id") or f"call_{idx}",
                    "type": "function",
                    "function": {"name": msg.get("name") or "tool", "arguments": "{}"},
                }
                for idx, msg in enumerate(appended_messages)
            ],
        }
        return self._tokenize_rendered_suffix(
            [
                {"role": "system", "content": "dummy system"},
                {"role": "user", "content": "dummy user"},
                dummy_assistant,
            ],
            appended_messages,
            tools=tools,
        )

    def _tokenize_user_or_system_segment(
        self,
        appended_message: dict[str, Any],
        *,
        tools: Optional[List[dict[str, Any]]] = None,
    ) -> List[int]:
        return self._tokenize_rendered_suffix(
            [
                {"role": "system", "content": "dummy system"},
                {"role": "user", "content": "dummy user"},
            ],
            [appended_message],
            tools=tools,
        )

    def _tokenize_rendered_suffix(
        self,
        base_messages: List[dict[str, Any]],
        appended_messages: List[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        tools: Optional[List[dict[str, Any]]] = None,
    ) -> List[int]:
        without = self._render_messages(base_messages, add_generation_prompt=False, tools=tools)
        with_appended = self._render_messages(
            base_messages + appended_messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        if not with_appended.startswith(without):
            roles = [msg.get("role") for msg in appended_messages] if appended_messages else ["generation_prompt"]
            raise TokenizationError(f"rendered suffix diff failed for roles={roles}")
        return self._encode_text(with_appended[len(without) :])

    def _render_messages(
        self,
        messages: List[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: Optional[List[dict[str, Any]]] = None,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **self.chat_template_kwargs,
        )

    def _encode_text(self, text: str) -> List[int]:
        if hasattr(self.tokenizer, "encode"):
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        encoded = self.tokenizer(text, add_special_tokens=False)
        return list(encoded["input_ids"])


def get_tito_tokenizer(
    tokenizer: Any,
    **kwargs: Any,
) -> TITOTokenizer:
    return TITOTokenizer(tokenizer, **kwargs)


@dataclass
class _Checkpoint:
    messages: List[dict[str, Any]]
    seq: List[int]
    turns: list


@dataclass
class MessageTrajectoryAssembler:
    """Message-level adapter over the token-native :class:`TrajectoryAssembler`."""

    tito_tokenizer: TITOTokenizer
    trajectory: TrajectoryAssembler = field(default_factory=TrajectoryAssembler)
    max_assistant_rollback_steps: int = 1
    messages: List[dict[str, Any]] = field(default_factory=list)
    checkpoints: List[_Checkpoint] = field(default_factory=list)

    @property
    def token_ids(self) -> List[int]:
        return self.trajectory.accumulated_tokens

    def prepare_next_input(
        self,
        request_messages: List[dict[str, Any]],
        *,
        tools: Optional[List[dict[str, Any]]] = None,
    ) -> List[int]:
        if not self.checkpoints:
            return self.tito_tokenizer.render_initial_prompt(request_messages, tools=tools)

        self._try_detect_and_rollback_to_assistant_checkpoint(request_messages)
        _assert_append_only_with_allowed_roles(
            self.messages,
            request_messages,
            self.tito_tokenizer.allowed_append_roles,
        )
        return self.tito_tokenizer.merge_tokens(
            old_messages=self.messages,
            new_messages=request_messages,
            pretokenized_token_ids=self.token_ids,
            tools=tools,
        )

    def add_assistant_response(
        self,
        *,
        request_messages: List[dict[str, Any]],
        assistant_message: dict[str, Any],
        prompt_token_ids: List[int],
        completion_token_ids: List[int],
        completion_logprobs: List[float],
        finish_reason: str = "stop",
    ) -> None:
        self.trajectory.add_call(
            InferenceCall(
                input_tokens=list(prompt_token_ids),
                output_tokens=list(completion_token_ids),
                output_logprobs=list(completion_logprobs),
                finish_reason=finish_reason,
            ),
            max_trim_tokens=self.tito_tokenizer.max_trim_tokens,
        )
        self.messages = list(request_messages) + [assistant_message]
        self.checkpoints.append(self._make_checkpoint())

    def to_payload(self, *, total_reward: Optional[float] = None) -> RolloutPayload:
        return self.trajectory.to_payload(total_reward=total_reward)

    def to_trajectory(
        self,
        *,
        tokenizer: Any | None = None,
        source: str = "message_trajectory_assembler",
    ):
        """Return a native trajectory analysis for verifier visualization."""
        return self.trajectory.to_trajectory(tokenizer=tokenizer, source=source)

    def _make_checkpoint(self) -> _Checkpoint:
        return _Checkpoint(
            messages=deepcopy(self.messages),
            seq=list(self.trajectory._seq),
            turns=deepcopy(self.trajectory._turns),
        )

    def _restore_checkpoint(self, checkpoint: _Checkpoint) -> None:
        self.messages = deepcopy(checkpoint.messages)
        self.trajectory._seq = list(checkpoint.seq)
        self.trajectory._turns = deepcopy(checkpoint.turns)

    def _try_detect_and_rollback_to_assistant_checkpoint(self, request_messages: List[dict[str, Any]]) -> None:
        stored = self.messages
        if not stored or not self.checkpoints:
            return

        match_len = 0
        for idx in range(min(len(request_messages), len(stored))):
            if stored[idx] == request_messages[idx]:
                match_len = idx + 1
            else:
                break
        if match_len >= len(stored):
            return

        checkpoint_index = -1
        assistant_count = 0
        for idx in range(match_len):
            if stored[idx].get("role") == "assistant":
                checkpoint_index = assistant_count
                assistant_count += 1
        if checkpoint_index < 0:
            raise MessageValidationError(f"rollback failed: no assistant message found in first {match_len} messages")

        discard_count = len(self.checkpoints) - (checkpoint_index + 1)
        if discard_count > self.max_assistant_rollback_steps:
            raise MessageValidationError(
                f"rollback failed: discard_count={discard_count} exceeds "
                f"max_assistant_rollback_steps={self.max_assistant_rollback_steps}"
            )
        self.checkpoints = self.checkpoints[: checkpoint_index + 1]
        self._restore_checkpoint(self.checkpoints[-1])
