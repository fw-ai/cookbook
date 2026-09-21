"""RL utilities: losses, training loop, TIS, router replay."""

__all__ = [
    # Losses & algorithms
    "CISPOConfig",
    "DAPOConfig",
    "DPPOConfig",
    "DROConfig",
    "ScoreCenteringConfig",
    "TISConfig",
    "GSPOConfig",
    "PromptGroup",
    "build_r3_routing_matrices",
    "make_cispo_loss_fn",
    "make_dapo_loss_fn",
    "make_dppo_loss_fn",
    "make_dro_loss_fn",
    "make_score_centering_loss_fn",
    "build_score_centering_datums",
    "make_grpo_loss_fn",
    "make_gspo_loss_fn",
    # Training loop
    "DynamicFilterFn",
    "TrainStepFns",
    "run_rl_loop",
    # Metrics helpers
    "add_train_perf_metrics",
    "total_target_tokens",
    # IGPO
    "IGPOTurnScorer",
    "compute_turn_advantages",
    "expand_turn_advantages",
    "expand_turn_advantages_from_spans",
    "make_igpo_loss_fn",
    "score_prefix",
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

from training.utils.rl.dapo import DAPOConfig, make_dapo_loss_fn
from training.utils.rl.dppo import DPPOConfig, make_dppo_loss_fn
from training.utils.rl.dro import DROConfig, make_dro_loss_fn
from training.utils.rl.score_centering import (
    ScoreCenteringConfig,
    build_score_centering_datums,
    make_score_centering_loss_fn,
)
from training.utils.rl.grpo import make_grpo_loss_fn
from training.utils.rl.gspo import GSPOConfig, make_gspo_loss_fn
from training.utils.rl.cispo import CISPOConfig, make_cispo_loss_fn
from training.utils.rl.train import (
    DynamicFilterFn,
    TrainStepFns,
    run_rl_loop,
)
from training.utils.rl.losses import PromptGroup
from training.utils.rl.metrics import (
    total_target_tokens,
    add_train_perf_metrics,
)
from training.utils.rl.router_replay import build_r3_routing_matrices
from training.utils.rl.tis import TISConfig
from training.utils.rl.igpo import (
    IGPOTurnScorer,
    compute_turn_advantages,
    expand_turn_advantages,
    expand_turn_advantages_from_spans,
    make_igpo_loss_fn,
    score_prefix,
)
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
