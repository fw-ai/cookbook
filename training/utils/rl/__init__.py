"""RL utilities: losses, training loop, PP recommendation, TIS, router replay."""

__all__ = [
    # Config
    "Config",
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
    "TrainContext",
    "TrainStepFns",
    "train_one_step",
    "run_rl_loop",
    # Rollout scheduling
    "AsyncRolloutScheduler",
    "RolloutStats",
    # Datum building
    "build_prompt_group",
    # Rewards
    "default_math_reward",
    "default_variance_filter",
    # Metrics helpers
    "add_response_length_stats",
    "add_train_perf_metrics",
    "build_loop_metrics",
    "total_target_tokens",
]

from training.utils.rl.config import Config
from training.utils.rl.pp import PPBatchRecommendation, compute_pp_recommendation
from training.utils.rl.dapo import DAPOConfig, make_dapo_loss_fn
from training.utils.rl.grpo import make_grpo_loss_fn
from training.utils.rl.gspo import GSPOConfig, make_gspo_loss_fn
from training.utils.rl.cispo import CISPOConfig, make_cispo_loss_fn
from training.utils.rl.train import (
    DynamicFilterFn,
    TrainContext,
    TrainStepFns,
    train_one_step,
    run_rl_loop,
)
from training.utils.rl.rollout import (
    AsyncRolloutScheduler,
    RolloutStats,
)
from training.utils.rl.losses import PromptGroup
from training.utils.rl.datum import build_prompt_group
from training.utils.rl.rewards import default_math_reward, default_variance_filter
from training.utils.rl.metrics import (
    build_loop_metrics,
    total_target_tokens,
    add_train_perf_metrics,
    add_response_length_stats,
)
from training.utils.rl.router_replay import build_r3_routing_matrices
from training.utils.rl.tis import TISConfig
