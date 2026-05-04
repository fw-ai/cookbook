"""Rollout primitives for the RL recipes.

The trainer's user-facing contract is per-sample (matches AReaL/slime)::

    async def rollout_fn(sample_prompt) -> RolloutSample | None: ...

``sample_prompt`` is a dataset row's dict, renamed at the recipe seam
to mark that it is now per-sample input rather than dataset-cursor
state.  The recipe fans each dataset row out to
``completions_per_prompt`` parallel calls and joins them by row id via
:class:`GroupAssembler` before handing the assembled
:class:`PromptGroup` to the trainer.

This package layers the supporting types and helpers:

* :mod:`.types` — :class:`Rollout` (group of samples) and
  :class:`RolloutSample` (single trajectory), plus the
  :func:`rollout_to_prompt_group` adapter that packs a group into the
  trainer's :class:`PromptGroup`.
* :mod:`.group_assembler` — :class:`GroupAssembler` joins per-sample
  rollouts into PromptGroups by row id once all samples for a row have
  settled.
* :mod:`.service` — service-agnostic Protocol + payload types
  (:class:`RolloutService`, :class:`RolloutPayload`, :class:`TurnRecord`).
* :mod:`.remote` — drives a :class:`RolloutService` and packs payloads
  (:func:`make_remote_rollout_fn`, :func:`pack_payload_to_sample`).
* :mod:`.renderer` — renderer-backed single-turn helper
  (:func:`single_turn_renderer_rollout`, :func:`model_input_to_token_ids`).
* :mod:`.assembler` — multi-turn token-native stitching
  (:class:`TrajectoryAssembler`, :func:`extract_completion`,
  :func:`precompute_chat_suffix`).
* :mod:`.message` — message-in multi-turn assembly that preserves prior
  assistant tokens with TITO-style incremental tokenization.
* :mod:`.trace` — native rollout trajectory analysis for visualization and
  diagnostics without a live verifier probe.
* :mod:`.eval_protocol` — adapter from ``@evaluation_test`` metadata to the
  per-sample rollout function contract.
"""

from training.utils.rl.rollout.assembler import (
    InferenceCall,
    PrefixMismatch,
    TrajectoryAssembler,
    extract_completion,
    precompute_chat_suffix,
)
from training.utils.rl.rollout.eval_protocol import (
    default_completion_params_factory,
    default_eval_row_factory,
    get_eval_protocol_params,
    load_eval_protocol_input_rows,
    make_eval_protocol_rollout_fn_factory,
)
from training.utils.rl.rollout.group_assembler import (
    GroupAssembler,
    PendingGroup,
)
from training.utils.rl.rollout.message import (
    MessageTrajectoryAssembler,
    MessageTrajectoryError,
    MessageValidationError,
    TITOTokenizer,
    TokenizationError,
    get_tito_tokenizer,
)
from training.utils.rl.rollout.remote import (
    make_remote_rollout_fn,
    pack_payload_to_sample,
)
from training.utils.rl.rollout.renderer import (
    MultimodalRenderingNotSupported,
    model_input_to_token_ids,
    single_turn_renderer_rollout,
)
from training.utils.rl.rollout.service import (
    RolloutPayload,
    RolloutService,
    TurnRecord,
)
from training.utils.rl.rollout.trace import (
    RolloutTrajectory,
    TrajectoryIssue,
    TrajectoryToken,
    analyze_flat_sample,
    analyze_token_turn_traces,
    analyze_turns,
)
from training.utils.rl.rollout.types import (
    Rollout,
    RolloutSample,
    rollout_to_prompt_group,
)


__all__ = [
    "InferenceCall",
    "GroupAssembler",
    "MessageTrajectoryAssembler",
    "MessageTrajectoryError",
    "MessageValidationError",
    "MultimodalRenderingNotSupported",
    "PendingGroup",
    "PrefixMismatch",
    "Rollout",
    "RolloutPayload",
    "RolloutSample",
    "RolloutService",
    "RolloutTrajectory",
    "TITOTokenizer",
    "TrajectoryAssembler",
    "TrajectoryIssue",
    "TrajectoryToken",
    "TurnRecord",
    "TokenizationError",
    "analyze_flat_sample",
    "analyze_token_turn_traces",
    "analyze_turns",
    "default_completion_params_factory",
    "default_eval_row_factory",
    "extract_completion",
    "get_eval_protocol_params",
    "get_tito_tokenizer",
    "load_eval_protocol_input_rows",
    "make_eval_protocol_rollout_fn_factory",
    "make_remote_rollout_fn",
    "model_input_to_token_ids",
    "pack_payload_to_sample",
    "precompute_chat_suffix",
    "rollout_to_prompt_group",
    "single_turn_renderer_rollout",
]
