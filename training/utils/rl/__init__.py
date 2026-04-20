"""RL utilities: losses, training loop, PP recommendation, TIS, router replay."""

__all__ = [
    # Losses & algorithms
    "CISPOConfig",
    "DAPOConfig",
    "TISConfig",
    "GSPOConfig",
    "PPBatchRecommendation",
    "PromptGroup",
    "build_r3_routing_matrices",
    "compute_pp_recommendation",
    "make_cispo_loss_fn",
    "make_dapo_loss_fn",
    "make_grpo_loss_fn",
    "make_gspo_loss_fn",
    # Training loop
    "DynamicFilterFn",
    "TrainStepFns",
    "run_rl_loop",
    # Datum construction (low-level plumbing)
    "align_inference_logprobs",
    "make_policy_datum",
    "make_reference_datum",
    # Async rollout scheduler (opt-in)
    "AsyncRolloutScheduler",
    "RolloutStats",
    # Infra setup (heavy lifting)
    "Infra",
    "setup_infra",
    # Metrics helpers
    "add_response_length_stats",
    "add_train_perf_metrics",
    "build_loop_metrics",
    "total_target_tokens",
    # IGPO
    "IGPOTurnScorer",
    "compute_turn_advantages",
    "expand_turn_advantages",
    "expand_turn_advantages_from_spans",
    "make_igpo_loss_fn",
    "score_prefix",
]

from training.utils.rl.pp import PPBatchRecommendation, compute_pp_recommendation
from training.utils.rl.dapo import DAPOConfig, make_dapo_loss_fn
from training.utils.rl.grpo import make_grpo_loss_fn
from training.utils.rl.gspo import GSPOConfig, make_gspo_loss_fn
from training.utils.rl.cispo import CISPOConfig, make_cispo_loss_fn
from training.utils.rl.train import (
    DynamicFilterFn,
    TrainStepFns,
    run_rl_loop,
)
from training.utils.rl.losses import PromptGroup
from training.utils.rl.datum import (
    align_inference_logprobs,
    make_policy_datum,
    make_reference_datum,
)
from training.utils.rl.scheduler import AsyncRolloutScheduler, RolloutStats
from training.utils.rl.infra_setup import Infra, setup_infra
from training.utils.rl.metrics import (
    build_loop_metrics,
    total_target_tokens,
    add_train_perf_metrics,
    add_response_length_stats,
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
