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
    # Bundled infra setup
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
    # Async loop + rollout contract
    "run_async_rl_loop",
    "Rollout",
    "RolloutSample",
    "rollout_to_prompt_group",
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
from training.utils.infra import Infra, setup_infra
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
from training.utils.rl.async_train import run_async_rl_loop
from training.utils.rl.rollout import Rollout, RolloutSample, rollout_to_prompt_group
