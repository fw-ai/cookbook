"""Cookbook utilities -- infrastructure, losses, data, logging, and more.

RL-specific utilities (losses, training loop, PP recommendation,
importance sampling, router replay) live in
``training.utils.rl``.
"""

# ---------------------------------------------------------------------------
# Suppress noisy third-party warnings that flood training output
# ---------------------------------------------------------------------------
import warnings as _warnings

# Pydantic union discriminator warnings (~45 lines per datum per step) from
# ModelInput serialization.  Harmless but produce thousands of lines that
# bury actual training metrics.
_warnings.filterwarnings(
    "ignore",
    message=r".*PydanticSerializationUnexpectedValue.*",
)

# Renderer emits a ``train_on_what`` / extension-property compatibility
# warning *per example* instead of once.  Downgrade to once-per-session so
# operators see it but aren't flooded.
_warnings.filterwarnings(
    "once",
    message=r".*train_on_what.*",
)
_warnings.filterwarnings(
    "once",
    message=r".*extension.prop.*",
)

del _warnings

__all__ = [
    "DEFAULT_ADAM",
    "DeployConfig",
    "EvalFn",
    "WeightSyncConfig",
    "InfraConfig",
    "ReconnectableClient",
    "ResourceCleanup",
    "RewardFn",
    "RLPromptDataset",
    "RunnerConfig",
    "RunnerIO",
    "RunStatus",
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
    "make_orpo_loss_fn",
    "make_batch_orpo_loss_fn",
    "make_batch_dpo_loss_fn",
    "make_batch_sft_loss_fn",
    "make_batch_weighted_sft_loss_fn",
    "make_sft_loss_fn",
    "RenderedSupervisedDatum",
    "RenderedPreferencePair",
    "build_next_token_datum",
    "build_datum_from_token_mask",
    "build_datum_from_tokens_and_weights",
    "build_renderer",
    "normalize_messages",
    "parse_train_on_what",
    "render_preference_pair",
    "render_messages_to_datum",
    "resolve_renderer_name",
    "prepare_sampling_messages",
    "setup_deployment",
    "setup_training_client",
    "setup_wandb",
    "flush_timing",
    "timed",
    "timer",
    "canonical_base_model",
    "materialize_profile_infra",
    "prepare_training_shape_launch",
    "ShapeSelectionRequest",
    "ShapeSelectionResult",
    "select_validated_launch_shapes",
    "TRAINING_SHAPES_DOCS_URL",
    "validate_config",
    "validate_preflight",
    "wandb_finish",
    "wandb_log",
    "WeightSyncer",
]

from training.utils.data import (
    RLPromptDataset,
    encode_text,
    extract_text,
    compute_advantages,
    load_jsonl_dataset,
    load_preference_dataset,
    find_common_prefix_length,
    prepare_sampling_messages,
)
from training.utils.infra import (
    ResourceCleanup,
    get_deployment_gpu_count,
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
    ConcurrencyConfig,
    InfraConfig,
    WandBConfig,
    DeployConfig,
    StepCallback,
    WeightSyncConfig,
)
from training.utils.weight_sync import WeightSyncer
from training.utils.training_shapes import (
    TRAINING_SHAPES_DOCS_URL,
    canonical_base_model,
    materialize_profile_infra,
    prepare_training_shape_launch,
    ShapeSelectionRequest,
    ShapeSelectionResult,
    select_validated_launch_shapes,
)
from training.utils.losses import (
    make_sft_loss_fn,
    make_orpo_loss_fn,
    make_batch_orpo_loss_fn,
    make_batch_dpo_loss_fn,
    make_batch_sft_loss_fn,
    make_batch_weighted_sft_loss_fn,
)
from training.utils.supervised import (
    RenderedPreferencePair,
    RenderedSupervisedDatum,
    build_datum_from_token_mask,
    build_next_token_datum,
    build_datum_from_tokens_and_weights,
    build_renderer,
    normalize_messages,
    parse_train_on_what,
    render_preference_pair,
    render_messages_to_datum,
    resolve_renderer_name,
)
from training.utils.logging import (
    wandb_log,
    setup_wandb,
    wandb_finish,
    log_metrics_json,
    compute_pass_at_k,
)
from training.utils.runner import RunnerConfig, RunnerIO, RunStatus
from training.utils.validation import validate_config, validate_preflight
