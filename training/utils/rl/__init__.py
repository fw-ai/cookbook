"""RL utilities: losses, training loop, PP recommendation, importance sampling, router replay."""

__all__ = [
    # Losses & algorithms
    "CISPOConfig",
    "DAPOConfig",
    "DecoupledConfig",
    "GSPOConfig",
    "ISConfig",
    "PPBatchRecommendation",
    "PromptGroup",
    "build_r3_routing_matrices",
    "compute_pp_recommendation",
    "make_cispo_loss_fn",
    "make_dapo_loss_fn",
    "make_grpo_loss_fn",
    "make_gspo_loss_fn",
    "make_tis_weights_fn",
    # Training loop
    "AsyncConfig",
    "DynamicFilterFn",
    "TrainStepFns",
    "run_rl_loop",
    "run_rl_loop_async",
    # Metrics helpers
    "add_response_length_stats",
    "add_train_perf_metrics",
    "build_loop_metrics",
    "total_target_tokens",
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
from training.utils.rl.train_async import AsyncConfig, run_rl_loop_async
from training.utils.rl.losses import PromptGroup
from training.utils.rl.metrics import (
    build_loop_metrics,
    total_target_tokens,
    add_train_perf_metrics,
    add_response_length_stats,
)
from training.utils.rl.router_replay import build_r3_routing_matrices
from training.utils.rl.importance_sampling import DecoupledConfig, ISConfig, make_tis_weights_fn
