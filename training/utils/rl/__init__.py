"""RL utilities: pure functions, dataclasses, and one-shot SDK wrappers.

Library-style. Everything here is a function the user calls or a data
type the user passes around. No runners, no callback registries, no
context managers, no protocols.

If you want a complete training loop, copy the body of
``recipes/rl_loop.py`` (sync) or ``recipes/async_rl_loop.py`` (async
streaming).
"""

__all__ = [
    # Loss configs & builders
    "CISPOConfig",
    "DAPOConfig",
    "TISConfig",
    "GSPOConfig",
    "make_cispo_loss_fn",
    "make_dapo_loss_fn",
    "make_grpo_loss_fn",
    "make_gspo_loss_fn",
    # Data shapes
    "PromptGroup",
    # PP recommendation
    "PPBatchRecommendation",
    "compute_pp_recommendation",
    # Datum construction (low-level plumbing, pure)
    "align_inference_logprobs",
    "make_policy_datum",
    "make_reference_datum",
    # Composable infra builders (one-shot, no lifecycle hidden)
    "make_concurrency_controller",
    "provision_trainer_pair",
    "resolve_policy_profile",
    "resolve_reference_profile",
    # Per-step training primitives
    "TrainContext",
    "dump_trajectory_jsonl",
    "finish_step",
    "ref_fwd_bwd",
    # Router replay (R3) helper (pure)
    "build_r3_routing_matrices",
    # Metrics helpers (pure)
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
    TrainContext,
    dump_trajectory_jsonl,
    finish_step,
    ref_fwd_bwd,
)
from training.utils.rl.losses import PromptGroup
from training.utils.rl.datum import (
    align_inference_logprobs,
    make_policy_datum,
    make_reference_datum,
)
from training.utils.rl.infra_setup import (
    make_concurrency_controller,
    provision_trainer_pair,
    resolve_policy_profile,
    resolve_reference_profile,
)
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
