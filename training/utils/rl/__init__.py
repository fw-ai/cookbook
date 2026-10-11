"""RL utilities: batches, rollouts and router replay.

Objectives live in :mod:`training.utils.rl.algorithm`, one module each.
"""

__all__ = [
    # Losses & algorithms
    "PromptGroup",
    "has_reward_variance",
    "build_r3_routing_matrices",
    # Metrics helpers
    "add_train_perf_metrics",
    "total_target_tokens",
    # Rollout contract
    "Rollout",
    "RolloutRun",
    "RolloutSample",
    "GroupAssembler",
    "rollout_to_prompt_group",
    # Service-agnostic rollout adapter
    "RolloutPayload",
    "RolloutService",
    "TurnRecord",
    "make_remote_rollout_fn",
    # Multi-turn assembly
    "InferenceCall",
    "MessageTrajectoryAssembler",
    "MessageTrajectoryError",
    "MessageValidationError",
    "PrefixMismatch",
    "TrajectoryAssembler",
    "TITOTokenizer",
    "extract_completion",
    "get_tito_tokenizer",
    "precompute_chat_suffix",
    "TokenizationError",
]

from training.utils.rl.losses import PromptGroup, has_reward_variance
from training.utils.rl.metrics import (
    total_target_tokens,
    add_train_perf_metrics,
)
from training.utils.rl.router_replay import build_r3_routing_matrices
from training.utils.rl.rollout import (
    GroupAssembler,
    InferenceCall,
    MessageTrajectoryAssembler,
    MessageTrajectoryError,
    MessageValidationError,
    PrefixMismatch,
    Rollout,
    RolloutRun,
    RolloutPayload,
    RolloutSample,
    RolloutService,
    TITOTokenizer,
    TrajectoryAssembler,
    TurnRecord,
    TokenizationError,
    extract_completion,
    get_tito_tokenizer,
    make_remote_rollout_fn,
    precompute_chat_suffix,
    rollout_to_prompt_group,
)
