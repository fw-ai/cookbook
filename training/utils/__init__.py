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
    "AppendOnlyPickleLog",
    "DEFAULT_ADAM",
    "DEFAULT_PREFETCH_FACTOR",
    "DEFAULT_RENDER_WORKERS",
    "DeployConfig",
    "EvalFn",
    "WeightSyncConfig",
    "WeightSyncScope",
    "InfraConfig",
    "JsonlRenderDataset",
    "MemTracer",
    "ReconnectableClient",
    "ResourceCleanup",
    "RewardFn",
    "RawRowCursor",
    "RLPromptDataset",
    "RunnerConfig",
    "RunnerIO",
    "RunStatus",
    "StepCallback",
    "WandBConfig",
    "compute_advantages",
    "compute_pass_at_k",
    "CursorDataLoader",
    "CursorItem",
    "create_trainer_job",
    "request_trainer_job",
    "wait_trainer_job",
    "request_deployment",
    "wait_deployment",
    "read_api_extra_headers_env",
    "encode_text",
    "extract_text",
    "find_common_prefix_length",
    "iter_preference_examples",
    "load_jsonl_dataset",
    "load_deployment_tokenizer",
    "load_preference_dataset",
    "load_tokenizer",
    "replicate_rows_for_epochs",
    "log_metrics_json",
    "make_render_dataloader",
    "make_orpo_loss_fn",
    "make_batch_orpo_loss_fn",
    "make_batch_dpo_loss_fn",
    "make_batch_sft_loss_fn",
    "make_batch_weighted_sft_loss_fn",
    "make_sft_loss_fn",
    "normalize_preference_row",
    "RenderedSupervisedDatum",
    "RenderedPreferencePair",
    "build_next_token_datum",
    "build_datum_from_token_mask",
    "build_datum_from_tokens_and_weights",
    "build_renderer",
    "normalize_messages",
    "parse_train_on_what",
    "populate_render_worker_state",
    "render_preference_pair",
    "render_messages_to_datum",
    "render_messages_to_datums",
    "resolve_renderer_name",
    "prepare_sampling_messages",
    "setup_deployment",
    "setup_or_reattach_deployment",
    "setup_training_client",
    "setup_wandb",
    "flush_timing",
    "timed",
    "timer",
    "auto_select_training_shape",
    "validate_config",
    "validate_preflight",
    "wandb_finish",
    "wandb_log",
]

from training.utils.client import ReconnectableClient
from training.utils.config import (
    DEFAULT_ADAM,
    ConcurrencyConfig,
    DeployConfig,
    EvalFn,
    InfraConfig,
    RewardFn,
    StepCallback,
    WandBConfig,
    WeightSyncConfig,
    WeightSyncScope,
)
from training.utils.data import (
    RLPromptDataset,
    compute_advantages,
    encode_text,
    extract_text,
    find_common_prefix_length,
    iter_preference_examples,
    load_jsonl_dataset,
    load_preference_dataset,
    normalize_preference_row,
    prepare_sampling_messages,
    replicate_rows_for_epochs,
)
from training.utils.dataloader import CursorDataLoader, CursorItem
from training.utils.dataloader_cursor import RawRowCursor
from training.utils.infra import (
    Infra,
    ResourceCleanup,
    create_trainer_job,
    get_deployment_gpu_count,
    read_api_extra_headers_env,
    request_deployment,
    request_trainer_job,
    setup_deployment,
    setup_infra,
    setup_or_reattach_deployment,
    setup_training_client,
    wait_deployment,
    wait_trainer_job,
)
from training.utils.logging import (
    compute_pass_at_k,
    log_metrics_json,
    setup_wandb,
    wandb_finish,
    wandb_log,
)
from training.utils.losses import (
    make_batch_dpo_loss_fn,
    make_batch_orpo_loss_fn,
    make_batch_sft_loss_fn,
    make_batch_weighted_sft_loss_fn,
    make_orpo_loss_fn,
    make_sft_loss_fn,
)
from training.utils.memlog import MemTracer
from training.utils.runner import RunnerConfig, RunnerIO, RunStatus
from training.utils.streaming import (
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_RENDER_WORKERS,
    AppendOnlyPickleLog,
    JsonlRenderDataset,
    make_render_dataloader,
)
from training.utils.supervised import (
    RenderedPreferencePair,
    RenderedSupervisedDatum,
    build_datum_from_token_mask,
    build_datum_from_tokens_and_weights,
    build_next_token_datum,
    build_renderer,
    normalize_messages,
    parse_train_on_what,
    populate_render_worker_state,
    render_messages_to_datum,
    render_messages_to_datums,
    render_preference_pair,
    resolve_renderer_name,
)
from training.utils.timer import flush_timing, timed, timer
from training.utils.tokenizers import load_deployment_tokenizer, load_tokenizer
from training.utils.training_shapes import (
    auto_select_training_shape,
)
from training.utils.validation import validate_config, validate_preflight
