"""Cookbook utilities -- infrastructure, losses, data, logging, and more.

RL-specific utilities (losses, training loop, PP recommendation,
importance sampling, router replay) live in
``training.utils.rl``.
"""

__all__ = [
    "DEFAULT_ADAM",
    "DeployConfig",
    "EvalFn",
    "HotloadConfig",
    "InfraConfig",
    "ReconnectableClient",
    "ResumeConfig",
    "RewardFn",
    "StepCallback",
    "WandBConfig",
    "compute_advantages",
    "compute_pass_at_k",
    "create_trainer_job",
    "encode_text",
    "extract_text",
    "find_common_prefix_length",
    "load_jsonl_dataset",
    "load_preference_dataset",
    "log_metrics_json",
    "make_dpo_loss_fn",
    "make_orpo_loss_fn",
    "make_batch_sft_loss_fn",
    "make_sft_loss_fn",
    "setup_deployment",
    "setup_resume",
    "setup_training_client",
    "setup_wandb",
    "flush_timing",
    "timed",
    "timer",
    "validate_config",
    "validate_preflight",
    "validate_streaming_config",
    "wandb_finish",
    "wandb_log",
]

from training.utils.data import (
    encode_text,
    extract_text,
    compute_advantages,
    load_jsonl_dataset,
    load_preference_dataset,
    find_common_prefix_length,
)
from training.utils.infra import (
    setup_deployment,
    create_trainer_job,
    setup_training_client,
)
from training.utils.timer import timed, timer, flush_timing
from training.utils.client import ReconnectableClient
from training.utils.config import (
    DEFAULT_ADAM,
    EvalFn,
    RewardFn,
    InfraConfig,
    WandBConfig,
    DeployConfig,
    ResumeConfig,
    StepCallback,
    HotloadConfig,
)
from training.utils.losses import (
    make_dpo_loss_fn,
    make_sft_loss_fn,
    make_orpo_loss_fn,
    make_batch_sft_loss_fn,
)
from training.utils.resume import setup_resume
from training.utils.logging import (
    wandb_log,
    setup_wandb,
    wandb_finish,
    log_metrics_json,
    compute_pass_at_k,
)
from training.utils.validation import validate_config, validate_preflight, validate_streaming_config
