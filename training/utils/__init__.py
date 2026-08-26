# ruff: noqa: E402
# Warning filters must be installed before importing modules that emit noisy
# import-time warnings.
"""Cookbook utilities -- infrastructure, losses, data, logging, and more.

RL-specific utilities (losses, training loop, importance sampling,
router replay) live in ``training.utils.rl``.
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
    "AdaptiveConcurrencyController",
    "CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO",
    "GradAccNormalization",
    "AppendOnlyPickleLog",
    "DEFAULT_ADAM",
    "DEFAULT_PREFETCH_FACTOR",
    "DEFAULT_RENDER_WORKERS",
    "JSONL_ROW_INDEX_KEY",
    "DeployConfig",
    "EvalFn",
    "WeightSyncConfig",
    "WeightSyncScope",
    "ConcurrencyConfig",
    "InfraConfig",
    "TrainerConfig",
    "JsonlRenderDataset",
    "MemTracer",
    "ReconnectableClient",
    "RewardFn",
    "RawRowCursor",
    "RLPromptDataset",
    "RunnerConfig",
    "RunnerIO",
    "RunStatus",
    "DatasetError",
    "NO_VALID_PREFERENCE_PAIRS_MESSAGE",
    "NO_VALID_TRAINING_EXAMPLES_MESSAGE",
    "UserConfigError",
    "StepCallback",
    "WandBConfig",
    "compute_advantages",
    "CursorDataLoader",
    "CursorItem",
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
    "log_metrics",
    "make_render_dataloader",
    "make_orpo_loss_fn",
    "make_batch_orpo_loss_fn",
    "make_batch_dpo_loss_fn",
    "make_batch_sft_loss_fn",
    "make_batch_weighted_sft_loss_fn",
    "make_sft_loss_fn",
    "normalize_preference_row",
    "RenderedSupervisedDatum",
    "RenderedChunkSpan",
    "RenderedPreferencePair",
    "build_next_token_datum",
    "build_datum_from_token_mask",
    "build_datum_from_tokens_and_weights",
    "build_renderer",
    "build_renderer_from_resolved_name",
    "normalize_messages",
    "parse_train_on_what",
    "populate_render_worker_state",
    "render_preference_pair",
    "render_messages_to_datum",
    "render_messages_to_datums",
    "rendered_chunk_spans",
    "renderer_supports_images",
    "renderer_supports_tool_images",
    "resolve_renderer_name",
    "resolve_renderer_plan",
    "resolve_renderer_snapshot",
    "prepare_sampling_messages",
    "build_service_client",
    "resolve_router_replay_enabled",
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

from fireworks.training.sdk import CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO
from fireworks.training.sdk.deployment import AdaptiveConcurrencyController

from training.utils.client import GradAccNormalization, ReconnectableClient
from training.utils.config import (
    DEFAULT_ADAM,
    ConcurrencyConfig,
    DeployConfig,
    EvalFn,
    InfraConfig,
    TrainerConfig,
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
from training.utils.infra import read_api_extra_headers_env
from training.utils.logging import (
    log_metrics,
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
from training.utils.runner import (
    DatasetError,
    NO_VALID_PREFERENCE_PAIRS_MESSAGE,
    NO_VALID_TRAINING_EXAMPLES_MESSAGE,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    UserConfigError,
)
from training.utils.service import build_service_client, resolve_router_replay_enabled
from training.utils.streaming import (
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_RENDER_WORKERS,
    JSONL_ROW_INDEX_KEY,
    AppendOnlyPickleLog,
    JsonlRenderDataset,
    make_render_dataloader,
)
from training.utils.supervised import (
    RenderedChunkSpan,
    RenderedPreferencePair,
    RenderedSupervisedDatum,
    build_datum_from_token_mask,
    build_datum_from_tokens_and_weights,
    build_next_token_datum,
    build_renderer,
    build_renderer_from_resolved_name,
    normalize_messages,
    parse_train_on_what,
    populate_render_worker_state,
    render_messages_to_datum,
    render_messages_to_datums,
    rendered_chunk_spans,
    render_preference_pair,
    renderer_supports_images,
    renderer_supports_tool_images,
    resolve_renderer_name,
    resolve_renderer_plan,
    resolve_renderer_snapshot,
)
from training.utils.timer import flush_timing, timed, timer
from training.utils.tokenizers import load_deployment_tokenizer, load_tokenizer
from training.utils.training_shapes import (
    auto_select_training_shape,
)
from training.utils.validation import validate_config, validate_preflight
