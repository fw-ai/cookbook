"""RL utilities: losses, training loop, PP recommendation, importance sampling, router replay."""

__all__ = [
    # Losses & algorithms
    "CISPOConfig",
    "DAPOConfig",
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
    "DynamicFilterFn",
    "MinibatchTrainFns",
    "run_rl_loop",
    # Metrics helpers
    "add_response_length_stats",
    "add_train_perf_metrics",
    "build_loop_metrics",
    "total_target_tokens",
]

from training_cookbook.utils.rl.pp import PPBatchRecommendation, compute_pp_recommendation
from training_cookbook.utils.rl.dapo import DAPOConfig, make_dapo_loss_fn
from training_cookbook.utils.rl.grpo import make_grpo_loss_fn
from training_cookbook.utils.rl.gspo import GSPOConfig, make_gspo_loss_fn
from training_cookbook.utils.rl.cispo import CISPOConfig, make_cispo_loss_fn
from training_cookbook.utils.rl.train import (
    DynamicFilterFn,
    MinibatchTrainFns,
    run_rl_loop,
)
from training_cookbook.utils.rl.losses import PromptGroup
from training_cookbook.utils.rl.metrics import (
    build_loop_metrics,
    total_target_tokens,
    add_train_perf_metrics,
    add_response_length_stats,
)
from training_cookbook.utils.rl.router_replay import build_r3_routing_matrices
from training_cookbook.utils.rl.importance_sampling import ISConfig, make_tis_weights_fn
