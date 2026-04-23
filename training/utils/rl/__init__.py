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
    # Async training loop + env-driven sampling
    "AsyncConfig",
    "run_rl_loop_async",
    "run_env_group_to_trajectories",
    "run_env_to_trajectory",
    "trajectories_to_prompt_group",
    "build_sample_fn",
    "RewardFn",
    "EnvBuilder",
    "RolloutSource",
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
    # Env-based rollout abstraction
    "Message",
    "MessageEnv",
    "MessageStepResult",
    "Transition",
    "Trajectory",
    "SingleTurnEnv",
    "wrap_reward_fn",
    "tokenize_chat_turn",
    "get_prefill_logprobs",
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
from training.utils.rl.env import (
    Message,
    MessageEnv,
    MessageStepResult,
    Trajectory,
    Transition,
)
from training.utils.rl.env_adapters import SingleTurnEnv, wrap_reward_fn
from training.utils.rl.tokenize import get_prefill_logprobs, tokenize_chat_turn
from training.utils.rl.train_async import AsyncConfig, run_rl_loop_async
from training.utils.rl.rollout_runner import (
    run_env_group_to_trajectories,
    run_env_to_trajectory,
)
from training.utils.rl.rollout_builder import trajectories_to_prompt_group
from training.utils.rl.sample_fn_factory import (
    EnvBuilder,
    RewardFn,
    RolloutSource,
    build_sample_fn,
)
