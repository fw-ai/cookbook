#!/usr/bin/env python3
"""Minimal SFT (Supervised Fine-Tuning) training loop.

A readable, modifiable fine-tuning loop using the Fireworks RLOR API.
Uses a single RLOR trainer job with cross-entropy loss on response tokens.

Dataset format (JSONL, OpenAI chat format):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.sft_loop


Maximize GPU Utilization
===========================================================================

The cookbook overlaps GPU/server work along two dimensions:

1. Intra-step overlap (always on). Each training step submits both
   forward_backward and optim_step as futures, then awaits them together.
   This hides the round-trip between fwdbwd and optim within a single step.

2. Inter-step pipelining (Config.pipeline_depth >= 2). Keep N (fwdbwd, optim)
   pairs in flight on the server's continuous-batching coalescer, so step
   N+1's server-side prep overlaps with step N's GPU compute. Increase this
   number (i.e. set to 4) to improve throughput.

"""

from __future__ import annotations

import functools
import json
import logging
import os
import random
import signal
import tempfile
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List

import tinker
import torch
from dotenv import load_dotenv
from fireworks.training.sdk.training_spec import (
    LRSchedulerSpec,
    compute_lr,
    default_constant_schedule,
    normalize_lr_scheduler_spec,
)

from training.utils import fileio
from training.utils import (
    DEFAULT_ADAM,
    DEFAULT_RENDER_WORKERS,
    TrainerConfig,
    JSONL_ROW_INDEX_KEY,
    JsonlRenderDataset,
    RawRowCursor,
    ReconnectableClient,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    build_service_client,
    log_metrics_json,
    make_render_dataloader,
    parse_train_on_what,
    populate_render_worker_state,
    read_api_extra_headers_env,
    render_messages_to_datums,
    resolve_renderer_name,
    setup_wandb,
    validate_config,
    wandb_finish,
    wandb_log,
)
from training.utils.checkpoints import TrainingCheckpoints, validate_warm_start_config
from training.utils.client import DEFAULT_TIMEOUT_S
from training.utils.serverless import setup_serverless_training
from training.utils.losses import make_batch_weighted_sft_loss_fn
from training.utils.runner_state import start_running, write_completed, write_running_step
from training.utils.timer import flush_timing, timer

load_dotenv()
logger = logging.getLogger(__name__)

# Module-level so DataLoader worker_init_fn can populate it once per
# worker (avoids re-pickling the tokenizer on every batch). The parent
# process also initialises this dict to support eval-set rendering and
# auto carve-out, which run in-process with the same render_fn.
_worker_state: dict = {}

RENDER_SAMPLE_LIMIT_ENV = "FIRETITAN_SFT_RENDER_SAMPLES_LIMIT"
DEFAULT_RENDER_SAMPLE_LIMIT = 20


def _parse_render_samples_limit(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"all", "full", "unlimited", "-1"}:
        return None
    limit = int(normalized)
    if limit < 0:
        return None
    return limit


def _resolve_render_samples_limit(config_limit: int | None) -> int | None:
    env_limit = os.environ.get(RENDER_SAMPLE_LIMIT_ENV)
    if env_limit is not None:
        try:
            return _parse_render_samples_limit(env_limit)
        except ValueError:
            logger.warning(
                "Invalid %s=%r; using default render sample limit %d",
                RENDER_SAMPLE_LIMIT_ENV,
                env_limit,
                DEFAULT_RENDER_SAMPLE_LIMIT,
            )
            return DEFAULT_RENDER_SAMPLE_LIMIT
    if config_limit is None:
        return DEFAULT_RENDER_SAMPLE_LIMIT
    if config_limit < 0:
        return None
    return int(config_limit)


def _decode_tokens(token_ids: list[int]) -> list[str]:
    tokenizer = _worker_state.get("tokenizer")
    if tokenizer is None:
        return ["" for _ in token_ids]
    decoded: list[str] = []
    for token_id in token_ids:
        try:
            text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        except TypeError:
            text = tokenizer.decode([int(token_id)])
        except Exception:  # noqa: BLE001 - token decode is best effort for samples
            text = ""
        decoded.append(str(text))
    return decoded


def _render_sample_worker_path() -> str:
    local_dir = _worker_state.get("render_samples_local_dir") or ""
    worker_id = _worker_state.get("worker_id")
    worker_label = "main" if worker_id is None else str(worker_id)
    return os.path.join(local_dir, f"render_samples.worker-{worker_label}.jsonl")


def _remaining_render_sample_capacity() -> int | None:
    limit = _worker_state.get("render_samples_limit")
    if limit is None:
        return None
    written = int(_worker_state.get("render_samples_written", 0))
    return max(0, int(limit) - written)


def _write_render_samples(row: dict, rendered_examples: list[Any]) -> None:
    local_dir = _worker_state.get("render_samples_local_dir") or ""
    if not local_dir:
        return
    remaining = _remaining_render_sample_capacity()
    if remaining == 0:
        return

    selected = rendered_examples if remaining is None else rendered_examples[:remaining]
    if not selected:
        return

    row_index = row.get(JSONL_ROW_INDEX_KEY)
    try:
        row_index_int = int(row_index) if row_index is not None else None
    except (TypeError, ValueError):
        row_index_int = None

    path = _render_sample_worker_path()
    try:
        with open(path, "a", encoding="utf-8") as handle:
            for split_index, rendered in enumerate(selected):
                token_ids = [int(x) for x in rendered.token_ids]
                token_weights = [float(x) for x in rendered.token_weights]
                datum = rendered.datum
                target_tokens = [int(x) for x in datum.loss_fn_inputs["target_tokens"].data]
                training_weights = [float(x) for x in datum.loss_fn_inputs["weights"].data]
                record = {
                    # source_jsonl_row_index is 0-based; source_jsonl_line_number
                    # is 1-based to match editor / CLI line-number conventions.
                    "source_jsonl_row_index": row_index_int,
                    "source_jsonl_line_number": row_index_int + 1 if row_index_int is not None else None,
                    "split_index": split_index,
                    "worker_id": _worker_state.get("worker_id"),
                    "renderer": _worker_state.get("resolved_renderer_name", ""),
                    "train_on_what": _worker_state.get("train_on_what_str", ""),
                    "token_ids": token_ids,
                    "decoded_tokens": _decode_tokens(token_ids),
                    "token_weights": token_weights,
                    "training_target_token_ids": target_tokens,
                    "training_loss_weights": training_weights,
                }
                handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
                _worker_state["render_samples_written"] = int(
                    _worker_state.get("render_samples_written", 0)
                ) + 1
    except Exception:  # noqa: BLE001 - render sample write must not fail training
        if not _worker_state.get("render_samples_error_logged"):
            logger.warning("Failed to write SFT render sample", exc_info=True)
            _worker_state["render_samples_error_logged"] = True


def _round_robin_render_sample_lines(files: list[Path]) -> list[str]:
    handles = []
    try:
        for path in files:
            handles.append(path.open(encoding="utf-8"))
        lines: list[str] = []
        while handles:
            next_handles = []
            for handle in handles:
                line = handle.readline()
                if line:
                    lines.append(line)
                    next_handles.append(handle)
                else:
                    handle.close()
            handles = next_handles
        return lines
    finally:
        for handle in handles:
            if not handle.closed:
                handle.close()


def _finalize_render_samples(
    local_dir: str,
    output_path: str,
    render_samples_limit: int | None,
) -> None:
    if not local_dir or not output_path:
        return
    try:
        files = sorted(Path(local_dir).glob("render_samples.worker-*.jsonl"))
        all_lines = _round_robin_render_sample_lines(files)
        content_parts = all_lines
        truncated_count = 0
        if render_samples_limit is not None:
            content_parts = all_lines[:render_samples_limit]
            truncated_count = max(0, len(all_lines) - render_samples_limit)
        if not content_parts:
            logger.info("No SFT render samples were captured; skipping %s", output_path)
            return
        content = "".join(content_parts)
        content_bytes = content.encode("utf-8")
        fileio.write_bytes(output_path, content_bytes)
        logger.info(
            "Uploaded %d SFT render sample records (%d bytes) to %s",
            len(content_parts),
            len(content_bytes),
            output_path,
        )
        if truncated_count:
            logger.info(
                "Skipped %d extra SFT render sample records after global limit=%d",
                truncated_count,
                render_samples_limit,
            )
    except Exception:  # noqa: BLE001 - render sample upload must not fail training
        logger.warning("Failed to finalize SFT render samples to %s", output_path, exc_info=True)


def _configure_render_sample_state(
    render_samples_local_dir: str = "",
    render_samples_limit: int | None = 0,
    _worker_id: int | None = None,
) -> None:
    _worker_state.update(
        worker_id=_worker_id,
        render_samples_local_dir=render_samples_local_dir,
        render_samples_limit=render_samples_limit,
        render_samples_written=0,
        render_samples_error_logged=False,
    )


def _init_render_worker(
    tokenizer_model: str,
    renderer_name: str,
    train_on_what_str: str,
    max_seq_len: int,
    tokenizer_revision: str = "",
    render_samples_local_dir: str = "",
    render_samples_limit: int | None = 0,
    _worker_id: int | None = None,
) -> None:
    """DataLoader ``worker_init_fn`` for SFT chat-row rendering.

    Module-level (so spawn workers can pickle it) and accepts
    ``_worker_id`` so it can be used as a DataLoader ``worker_init_fn``.
    """
    populate_render_worker_state(
        _worker_state,
        tokenizer_model=tokenizer_model,
        tokenizer_revision=tokenizer_revision,
        renderer_name=renderer_name,
        max_seq_len=max_seq_len,
        train_on_what=parse_train_on_what(train_on_what_str),
    )
    _worker_state.update(
        resolved_renderer_name=resolve_renderer_name(tokenizer_model, renderer_name),
        train_on_what_str=train_on_what_str,
    )
    _configure_render_sample_state(
        render_samples_local_dir=render_samples_local_dir,
        render_samples_limit=render_samples_limit,
        _worker_id=_worker_id,
    )


def _render_one_worker(row: dict) -> tinker.Datum | list[tinker.Datum] | None:
    """Render a chat row to one or more Datums, dropping empty / long sequences.

    Reads renderer / train_on_what / max_seq_len from the per-process
    ``_worker_state`` populated by ``_init_render_worker``. Top-level
    so spawn workers can pickle it as the DataLoader's render_fn.
    """
    messages = row.get("messages", [])
    if not messages:
        return None
    tools = row.get("tools")
    rendered_examples = render_messages_to_datums(
        messages,
        renderer=_worker_state["renderer"],
        train_on_what=_worker_state["train_on_what"],
        tools=tools,
    )
    if not isinstance(rendered_examples, list):
        rendered_examples = [rendered_examples]
    valid_rendered_examples = [
        rendered
        for rendered in rendered_examples
        if 2 <= len(rendered.token_ids) <= _worker_state["max_seq_len"]
        and any(w > 0 for w in rendered.token_weights)
    ]
    _write_render_samples(row, valid_rendered_examples)
    datums = [rendered.datum for rendered in valid_rendered_examples]
    if not datums:
        return None
    if len(datums) == 1:
        return datums[0]
    return datums


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

DEFAULT_EVAL_CARVE_RATIO = 0.1
DEFAULT_MAX_EVAL_SEQS = 100


def compute_eval_carveout(
    total_samples: int,
    max_ratio: float = DEFAULT_EVAL_CARVE_RATIO,
    max_seqs: int = DEFAULT_MAX_EVAL_SEQS,
) -> int:
    """Number of head samples to carve out as eval (0 if dataset too small)."""
    if total_samples <= 1:
        return 0
    carveout = min(int(total_samples * max_ratio), max_seqs)
    return 0 if carveout >= total_samples else carveout


def _render_eagerly(ds: JsonlRenderDataset, n: int) -> List[tinker.Datum]:
    """Render the first ``n`` rows of ``ds`` in-process, dropping Nones."""
    datums: List[tinker.Datum] = []
    for item in (ds[i] for i in range(n)):
        if item is None:
            continue
        if isinstance(item, list):
            datums.extend(item)
        else:
            datums.append(item)
    return datums


def _flatten_rendered_batch(
    batch: list[tinker.Datum | list[tinker.Datum]],
) -> list[tinker.Datum]:
    datums: list[tinker.Datum] = []
    for item in batch:
        if isinstance(item, list):
            datums.extend(item)
        else:
            datums.append(item)
    return datums


def _prepare_datasets(
    cfg: "Config",
) -> tuple[JsonlRenderDataset, List[tinker.Datum]]:
    """Build the training dataset and (optional) eval set.

    Eval can come from an explicit ``cfg.evaluation_dataset`` or be
    carved out from a seeded random subset of the training dataset. In
    the carve-out case the returned training dataset excludes those eval
    rows but otherwise preserves raw-file order; the training loader
    still does its own per-epoch shuffling.
    """
    training_ds = JsonlRenderDataset(
        cfg.dataset,
        _render_one_worker,
        max_examples=cfg.max_examples,
        row_index_key=JSONL_ROW_INDEX_KEY,
    )
    if len(training_ds) == 0:
        raise RuntimeError(f"No examples found in {cfg.dataset}")

    if cfg.evaluation_dataset:
        eval_ds = JsonlRenderDataset(
            cfg.evaluation_dataset,
            _render_one_worker,
            row_index_key=JSONL_ROW_INDEX_KEY,
        )
        eval_data = _render_eagerly(eval_ds, len(eval_ds))
        logger.info(
            "Loaded %d eval examples from %s",
            len(eval_data), cfg.evaluation_dataset,
        )
        return training_ds, eval_data

    if cfg.eval_auto_carveout:
        n = compute_eval_carveout(
            len(training_ds), cfg.eval_carve_ratio, cfg.max_eval_seqs,
        )
        if n > 0:
            shuffled_indices = list(range(len(training_ds)))
            random.Random(cfg.seed).shuffle(shuffled_indices)
            eval_indices = shuffled_indices[:n]
            eval_index_set = set(eval_indices)
            eval_data = _render_eagerly(
                training_ds.with_indices(eval_indices), len(eval_indices),
            )
            training_ds = training_ds.with_indices(
                [idx for idx in range(len(training_ds)) if idx not in eval_index_set]
            )
            logger.info(
                "Auto carve-out: %d eval examples, %d training examples "
                "(seed=%d)",
                len(eval_data), len(training_ds), cfg.seed,
            )
            return training_ds, eval_data
        logger.warning("Dataset too small for auto carve-out, skipping eval")

    return training_ds, []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    render_samples_file: str = ""
    """Optional JSONL path for rendered token/mask diagnostic samples."""

    render_samples_limit: int | None = None
    """Global rendered datum sample cap. ``None`` means default; negative
    values mean full dump. Can be overridden by
    FIRETITAN_SFT_RENDER_SAMPLES_LIMIT."""

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str = ""
    tokenizer_model: str = ""  # HuggingFace model name for chat template, e.g. "Qwen/Qwen3-1.7B"
    tokenizer_revision: str = ""  # Optional HuggingFace revision for client-side tokenization
    renderer_name: str = ""
    train_on_what: str = "all_assistant_messages"

    learning_rate: float = 1e-4
    lr_scheduler: LRSchedulerSpec = field(default_factory=default_constant_schedule)
    """Per-step LR scheduler spec. Legacy ``warmup_steps`` is still accepted below."""

    epochs: int = 3
    batch_size: int = 32
    """Number of training samples per optimizer step. For managed (V2) jobs
    this is set from ``BaseTrainingConfig.batch_size_samples`` via the
    cookbook orchestrator."""
    max_seq_len: int | None = None
    max_examples: int | None = None
    lora_rank: int = 0
    output_model_id: str | None = None

    serverless: bool = False
    """When True, train against an already-provisioned shared/pooled serverless
    trainer reached via the tinker gateway intercept
    (``{FIREWORKS_BASE_URL}/training/v1/serverless``) instead of the SDK-managed
    dedicated-trainer path. Skips GPU provisioning, substitutes config values for
    the managed-metadata reads, and promotes the final adapter through the
    session-scoped ``PromoteTrainingSessionCheckpoint`` RPC. Set by the control
    plane (``CookbookTrainingConfig.serverless``) when a routable pooled trainer
    exists for the base model; requires ``lora_rank > 0`` and a concrete
    ``max_seq_len`` (there is no training shape to resolve it from)."""

    save_final_checkpoint: bool = True

    dcp_save_interval: int = 0  # save DCP checkpoint every N steps (0 = off)

    init_from_checkpoint: str | None = None
    """Load pretrained DCP weights on a fresh dataset. Supports cross-job
    format ``"job_id:checkpoint_name"``."""

    warm_start_from_adapter: str | None = None
    """GCS URI of an HF PEFT adapter directory. When set, initializes LoRA
    weights from the adapter at training start (weights-only, fresh optimizer).
    Mutually exclusive with ``init_from_checkpoint``. Requires ``lora_rank > 0``."""

    grad_clip_norm: float = 1.0
    """Max gradient norm for clipping. 0 = no clipping."""

    adam_beta2: float | None = None
    """Override Adam beta2 (default 0.999 via DEFAULT_ADAM). Lower values
    (e.g. 0.98) make the variance estimate converge faster — useful for
    short runs or recipes like slime's GLM5 SFT."""

    weight_decay: float | None = None
    """Override Adam weight decay (default 0.01 via DEFAULT_ADAM)."""

    warmup_steps: int = 0
    """Linear LR warmup from 0 → learning_rate over the first N optimizer
    steps. 0 disables warmup (lr is constant)."""

    seed: int = 0
    """Shuffle seed for the training dataset.

    Used both for deterministic eval auto-carveout membership and for
    per-epoch training shuffle as ``seed + epoch`` so fresh runs and
    resumes see the same raw-row order in epoch 0 before any skipped
    batches.
    """

    group_by_length: bool = False
    """Compose each batch from similarly-sized examples (bucket-then-
    shuffle) instead of a pure random shuffle. Cuts padding waste and,
    when context parallel is enabled, lets most batches run at a low CP
    degree -- only the long-sequence batches pay the high-CP cost. Uses
    raw JSONL byte length as a cheap token-count proxy, so it is most
    beneficial for text datasets with high length variance and should
    stay off for base64-image multimodal data where byte length is a
    poor proxy. Batch count and resume/cursor semantics are unchanged."""

    length_group_factor: int = 50
    """Mega-batch multiplier for ``group_by_length``: a fresh per-epoch
    permutation is cut into windows of ``batch_size * length_group_factor``
    that are sorted by length before chunking into batches. Larger ->
    tighter length homogeneity (less padding / lower CP) but weaker
    shuffling; smaller -> more randomness but looser grouping."""

    step_timeout: int = 0
    """Timeout in seconds for forward_backward / optim_step calls.
    0 = use DEFAULT_TIMEOUT_S from training.utils.client."""

    pipeline_depth: int = 1
    """Number of (forward_backward, optim_step) pairs in flight at once
    (inter-step pipelining). Intra-step fb/optim overlap is always on.
    Increase this number (i.e. set to 4) to improve throughput."""

    evaluation_dataset: str = ""
    """Path to an explicit eval dataset (JSONL).  When set, auto-carveout
    is skipped and this dataset is used for evaluation instead."""

    eval_auto_carveout: bool = False
    """When True and no eval_dataset is provided, automatically carve out
    the first N training examples as an eval set."""

    eval_carve_ratio: float = DEFAULT_EVAL_CARVE_RATIO
    """Max fraction of training data to use for eval carve-out."""

    max_eval_seqs: int = DEFAULT_MAX_EVAL_SEQS
    """Max number of eval sequences for carve-out."""

    render_workers: int | None = None
    """Number of worker processes for streaming dataset rendering. ``None``
    auto-selects ``min(os.cpu_count(), DEFAULT_RENDER_WORKERS)``. Set to 1
    to disable the spawn pool and render in-process (useful for unit tests
    that monkeypatch the renderer)."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="sft-tinker"))
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    """Optional orchestration outputs written during training.

    Paths can be set here or via environment variables:
      COOKBOOK_STATUS_FILE      -- training status + progress (JSON, overwritten each step)
      COOKBOOK_METADATA_FILE    -- tokens processed + accelerator-seconds (JSON)
      COOKBOOK_METRICS_FILE     -- per-step metrics (JSONL, appended each step)
      COOKBOOK_OUTPUT_MODEL_PATH -- final model info written on completion (JSON)

    All paths are optional; unset paths are silently skipped.
    See training/utils/runner.py for file format details.
    """


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def run_eval(
    eval_data: List[tinker.Datum],
    client: ReconnectableClient,
    batch_size: int,
    step: int,
    epoch: int,
) -> float | None:
    """Run evaluation without affecting model weights or optimizer state.

    Uses forward (no backward) so weights, gradient state, and Adam moments
    are untouched. Logprobs come back from the server; the training loss
    function is invoked client-side purely to compute metrics, and the
    returned loss tensor is discarded.
    """
    if not eval_data:
        return None

    logger.info("[Eval] Running evaluation (%d examples)...", len(eval_data))

    eval_loss_sum = 0.0
    eval_resp_tokens = 0

    train_loss_fn = make_batch_weighted_sft_loss_fn()

    def _eval_batch(b: List[tinker.Datum]) -> Dict[str, float]:
        fwd = client.forward(b, "cross_entropy")
        logprobs_list = [
            fwd.loss_fn_outputs[i]["logprobs"].to_torch() for i in range(len(b))
        ]
        _, metrics = train_loss_fn(b, logprobs_list)
        return metrics

    batch: List[tinker.Datum] = []
    for item in eval_data:
        batch.append(item)
        if len(batch) >= batch_size:
            m = _eval_batch(batch)
            eval_loss_sum += m.get("ce_loss_sum", 0.0)
            eval_resp_tokens += int(m.get("response_tokens", 0))
            batch = []

    if batch:
        m = _eval_batch(batch)
        eval_loss_sum += m.get("ce_loss_sum", 0.0)
        eval_resp_tokens += int(m.get("response_tokens", 0))

    if eval_resp_tokens == 0:
        logger.warning("[Eval] No valid eval tokens, skipping metrics")
        return None

    eval_loss = eval_loss_sum / eval_resp_tokens
    eval_ppl = torch.exp(torch.tensor(eval_loss)).item()

    logger.info(
        "[Eval] Epoch %d | Loss: %.4f | PPL: %.2f | Tokens: %d",
        epoch + 1, eval_loss, eval_ppl, eval_resp_tokens,
    )
    log_metrics_json(step, eval_loss=eval_loss, eval_ppl=eval_ppl)
    wandb_log(
        {"eval/loss": eval_loss, "eval/ppl": eval_ppl, "eval/tokens": eval_resp_tokens},
        step,
    )

    return eval_loss


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(
    config: Config,
):
    cfg = config
    runner = RunnerIO(cfg.runner)

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(cfg.base_model, cfg.dataset, output_model_id=cfg.output_model_id)
    validate_warm_start_config(
        warm_start_from_adapter=cfg.warm_start_from_adapter,
        init_from_checkpoint=cfg.init_from_checkpoint,
        lora_rank=cfg.lora_rank,
    )
    lr_scheduler = normalize_lr_scheduler_spec(
        cfg.lr_scheduler,
        legacy_warmup_steps=cfg.warmup_steps,
    )
    setup_wandb(
        cfg.wandb,
        {
            "lr": cfg.learning_rate,
            "lr_schedule": lr_scheduler.type,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
        },
    )

    if not cfg.tokenizer_model:
        raise ValueError(
            "Config.tokenizer_model is required for chat template formatting. "
            "Set it to the HuggingFace model name (e.g. 'Qwen/Qwen3-1.7B')."
        )

    # -- SDK-managed Tinker client -----------------------------------------

    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    runner.write_status(
        RunStatus.PENDING,
        message="attaching to serverless trainer pool" if cfg.serverless else "provisioning",
    )

    with runner, ExitStack() as stack:
        if cfg.serverless:
            # Attach to a shared, already-running pooled trainer through the
            # serverless surface instead of provisioning a dedicated trainer.
            service, client, ckpt, job_id, max_seq_len = setup_serverless_training(
                cfg, api_key=api_key, base_url=base_url, additional_headers=additional_headers, stack=stack,
            )
            stack.callback(service.close)
            # Per-token billed by the pool trainer; there is no dedicated
            # accelerator to report, and the control plane must skip its own
            # token-billing leg (mark_serverless) to avoid double-charging.
            runner.set_accelerator_info(None, None, profile=None)
            runner.mark_serverless()
        else:
            service = build_service_client(
                api_key=api_key,
                base_url=base_url,
                additional_headers=additional_headers,
                base_model=cfg.base_model,
                tokenizer_model=cfg.tokenizer_model,
                lora_rank=cfg.lora_rank,
                max_context_length=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                trainer=cfg.trainer,
            )
            stack.callback(service.close)
            training_client = service.create_training_client(cfg.base_model, lora_rank=cfg.lora_rank)
            runner.set_accelerator_info(
                service.accelerator_type,
                service.accelerator_count,
                profile=service.training_profile,
            )
            job_id = service.trainer_job_id
            max_seq_len = service.max_context_length
            client = ReconnectableClient.from_training_client(
                training_client,
                base_model=cfg.base_model,
                lora_rank=cfg.lora_rank,
                job_id=job_id,
                default_timeout=cfg.step_timeout or DEFAULT_TIMEOUT_S,
                service=service,
            )

            ckpt = TrainingCheckpoints(
                client,
                service,
                trainer_id=job_id,
                log_path=cfg.log_path,
                lora_rank=cfg.lora_rank,
            )

        # -- Prepare data ------------------------------------------------------
        # Render JSONL rows on the fly inside DataLoader workers so peak
        # RAM is O(num_workers * per_worker_render_footprint) instead of
        # O(dataset_size).
        num_workers = (
            cfg.render_workers
            if cfg.render_workers is not None
            else min(os.cpu_count() or 1, DEFAULT_RENDER_WORKERS)
        )
        base_init_args = (
            cfg.tokenizer_model,
            cfg.renderer_name,
            cfg.train_on_what,
            max_seq_len,
            cfg.tokenizer_revision,
        )
        _init_render_worker(*base_init_args)

        training_dataset, eval_data = _prepare_datasets(cfg)

        render_samples_local_dir = ""
        render_samples_limit = _resolve_render_samples_limit(cfg.render_samples_limit)
        if cfg.render_samples_file and render_samples_limit != 0:
            render_samples_local_dir = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="sft-render-samples-")
            )
            stack.callback(
                _finalize_render_samples,
                render_samples_local_dir,
                cfg.render_samples_file,
                render_samples_limit,
            )
            limit_label = "full dataset" if render_samples_limit is None else str(render_samples_limit)
            logger.info(
                "Capturing SFT render samples to %s (global limit=%s)",
                cfg.render_samples_file,
                limit_label,
            )
        elif cfg.render_samples_file:
            logger.info("SFT render samples disabled by limit=0 for %s", cfg.render_samples_file)

        init_args = (
            *base_init_args,
            render_samples_local_dir,
            render_samples_limit,
        )
        _configure_render_sample_state(
            render_samples_local_dir=render_samples_local_dir,
            render_samples_limit=render_samples_limit,
        )
        worker_init_fn = functools.partial(_init_render_worker, *init_args)

        training_count = len(training_dataset)
        effective_batch_size = max(1, min(cfg.batch_size, training_count))
        if effective_batch_size < cfg.batch_size:
            logger.warning(
                "Training examples (%d) < batch_size (%d); reducing effective "
                "batch_size to %d.",
                training_count, cfg.batch_size, effective_batch_size,
            )

        loader_generator = torch.Generator()
        loader = make_render_dataloader(
            training_dataset,
            batch_size=effective_batch_size,
            num_workers=num_workers,
            shuffle=True,
            generator=loader_generator,
            worker_init_fn=worker_init_fn,
            group_by_length=cfg.group_by_length,
            length_group_factor=cfg.length_group_factor,
            sizes=training_dataset.approx_row_sizes() if cfg.group_by_length else None,
        )
        # Pre-filter upper bound; filtered rows make actual batches
        # smaller but never larger.
        total_batches_per_epoch = (training_count + effective_batch_size - 1) // effective_batch_size
        logger.info(
            "Dataset: %d examples from %s (renderer=%s, train_on_what=%s,"
            " workers=%d, seed=%d) -> ~%d batches/epoch x %d epochs%s",
            training_count, cfg.dataset,
            resolve_renderer_name(cfg.tokenizer_model, cfg.renderer_name),
            cfg.train_on_what, num_workers, cfg.seed,
            total_batches_per_epoch, cfg.epochs,
            f" + {len(eval_data)} eval examples" if eval_data else "",
        )

        # -- Resume ---------------------------------------------------------------

        resume_info = ckpt.resume(
            init_from_checkpoint=cfg.init_from_checkpoint,
            warm_start_from_adapter=cfg.warm_start_from_adapter,
        )
        step = resume_info.step if resume_info else 0
        total_raw_rows = training_count * cfg.epochs
        cursor = RawRowCursor(max_rows=total_raw_rows)
        cursor.resume(resume_info.data_consumed if resume_info else None)
        wandb_log({"train/step": step}, step)

        adam_kwargs = dict(DEFAULT_ADAM)
        adam_kwargs["grad_clip_norm"] = cfg.grad_clip_norm
        if cfg.adam_beta2 is not None:
            adam_kwargs["beta2"] = cfg.adam_beta2
        if cfg.weight_decay is not None:
            adam_kwargs["weight_decay"] = cfg.weight_decay

        def _current_lr(optim_step_idx: int) -> float:
            return compute_lr(
                lr_scheduler,
                step=optim_step_idx,
                base_lr=cfg.learning_rate,
                total_steps=total_steps_estimate,
            )

        # -- Training loop (batch-indexed) -------------------------------------

        completed_epochs = cursor.value // training_count if training_count else 0
        rows_into_current_epoch = cursor.value % training_count if training_count else 0
        start_batch = rows_into_current_epoch // effective_batch_size
        remaining_raw_rows = max(0, total_raw_rows - cursor.value)
        total_steps_estimate = step + (
            (remaining_raw_rows + effective_batch_size - 1) // effective_batch_size
        )

        # Always-on intra-step async + optional inter-step pipelining
        in_flight: deque = deque()
        pipe_started = time.time()
        pipe_total_tokens = 0
        # Track previous fb/opt completion times in pipeline mode   .
        last_t_fb_done: float | None = None
        last_t_opt_done: float | None = None
        last_optim_time: float | None = None

        def _pipe_submit(batch: list[tinker.Datum], step: int) -> int:
            """Submit one fwd_bwd + optim_step pair (non-blocking). Returns updated step."""
            tokens = sum(
                len(c.tokens) for d in batch for c in d.model_input.chunks if hasattr(c, "tokens")
            )
            step += 1
            adam = tinker.AdamParams(learning_rate=_current_lr(step), **adam_kwargs)
            t_submit = time.time()
            in_flight.append((
                step, tokens, t_submit,
                client.submit_forward_backward(batch, loss_fn="cross_entropy"),
                client.submit_optim_step(adam),
            ))
            return step

        def _pipe_collect() -> None:
            """Pop the oldest pair, emit per-step metrics + checkpoint."""
            nonlocal pipe_total_tokens, last_t_fb_done, last_t_opt_done, last_optim_time
            s, tokens, t_submit, fb_fut, opt_fut = in_flight.popleft()
            result = fb_fut.result(timeout=cfg.step_timeout or DEFAULT_TIMEOUT_S)
            t_fb_done = time.time()
            loss_sum = result.metrics.get("loss:sum", 0.0)
            response_tokens = result.metrics.get("response_tokens")
            optim_result = opt_fut.result(timeout=cfg.step_timeout or DEFAULT_TIMEOUT_S)
            t_opt_done = time.time()

            pipe_total_tokens += tokens
            tps = pipe_total_tokens / max(1e-9, time.time() - pipe_started)

            if cfg.dcp_save_interval > 0 and s % cfg.dcp_save_interval == 0:
                with timer("dcp_save"):
                    logger.info("Saving DCP checkpoint at step %d", s)
                    ckpt.save(f"step-{s}", resumable=True, promotable=False, data_consumed=cursor.value)

            # Metrics logging
            step_metrics: Dict[str, Any] = flush_timing()
            # Real per-step fb_compute and optim_compute (pipeline-agnostic).
            #   gap_fb  = t_fb_done_N - t_fb_done_(N-1)  = optim_(N-1) + fb_N
            #   gap_opt = t_opt_done_N - t_opt_done_(N-1) = fb_N + optim_N
            if last_t_fb_done is None or last_t_opt_done is None:
                fb_time = t_fb_done - t_submit
                opt_time = t_opt_done - t_fb_done
            else:
                gap_fb = t_fb_done - last_t_fb_done
                gap_opt = t_opt_done - last_t_opt_done
                fb_time = max(0.0, gap_fb - (last_optim_time or 0.0))
                opt_time = max(0.0, gap_opt - fb_time)
            last_t_fb_done = t_fb_done
            last_t_opt_done = t_opt_done
            last_optim_time = opt_time
            step_metrics["perf/fwd_bwd_time"] = fb_time
            step_metrics["perf/optim_step_time"] = opt_time
            if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
                for k, v in optim_result.metrics.items():
                    step_metrics[f"train/{k}"] = v

            step_metrics["train/total_tokens"] = tokens
            step_metrics["train/tokens_per_sec"] = tps

            if response_tokens and response_tokens > 0:
                avg_loss = loss_sum / response_tokens
                ppl = torch.exp(torch.tensor(avg_loss)).item()
                current_lr = _current_lr(s)
                logger.info(
                    "Step %d/%d | Loss: %.4f | PPL: %.2f | tok/s: %.0f | tokens: %d | depth: %d",
                    s, total_steps_estimate, avg_loss, ppl, tps, tokens, cfg.pipeline_depth,
                )
                log_metrics_json(s, ce_loss=avg_loss, ppl=ppl, tokens_per_sec=tps, lr=current_lr)
                step_metrics.update({
                    "train/step": s, "train/ce_loss": avg_loss,
                    "train/loss": avg_loss, "train/ppl": ppl,
                })
                step_metrics["train/lr"] = current_lr
                wandb_log(step_metrics, s)

            write_running_step(
                runner,
                step=s,
                total_steps=total_steps_estimate,
                metrics=step_metrics,
                tokens=tokens,
            )

        def _pipe_drain_safe() -> None:
            """Drain remaining in-flight ops for cleanup (e.g. in ``finally``)."""
            while in_flight:
                try:
                    _pipe_collect()
                except Exception as e:
                    logger.warning("pipeline drain: %s", e)

        start_running(runner, total_steps=total_steps_estimate)

        try:
            for epoch in range(completed_epochs, cfg.epochs):
                loader_generator.manual_seed(cfg.seed + epoch)
                batch_iter = iter(loader)
                epoch_start_batch = start_batch if epoch == completed_epochs else 0
                if epoch_start_batch > 0:
                    batch_iter = islice(batch_iter, epoch_start_batch, None)

                epoch_valid_examples = 0
                for raw_batch_idx, batch in enumerate(batch_iter, start=epoch_start_batch):
                    raw_batch_size = min(
                        effective_batch_size,
                        training_count - raw_batch_idx * effective_batch_size,
                    )
                    cursor.record(raw_batch_size)
                    batch = _flatten_rendered_batch(batch)
                    if not batch:
                        continue  # entire batch was filtered (None render); skip
                    epoch_valid_examples += len(batch)
                    step = _pipe_submit(batch, step)
                    if len(in_flight) >= cfg.pipeline_depth:  # collect once queue is full
                        _pipe_collect()

                # Drain in-flight ops so eval sees fully-applied gradients.
                while in_flight:
                    _pipe_collect()

                # Reset pipe timing baselines so the next epoch's first step
                # doesn't fold the eval / data-loader-setup gap into perf/fwd_bwd_time.
                last_t_fb_done = last_t_opt_done = last_optim_time = None

                if epoch == 0 and completed_epochs == 0 and start_batch == 0:
                    filtered_count = training_count - epoch_valid_examples
                    if filtered_count > 0:
                        logger.info(
                            "Seq-length / format filter: %d/%d raw rows filtered",
                            filtered_count,
                            training_count,
                        )
                    if epoch_valid_examples == 0:
                        raise RuntimeError("No valid training examples after tokenization")

                # Run eval after each epoch
                if eval_data:
                    try:
                        eval_loss = run_eval(
                            eval_data=eval_data,
                            client=client,
                            batch_size=cfg.batch_size,
                            step=step,
                            epoch=epoch,
                        )
                        if eval_loss is not None:
                            runner.append_metrics(
                                step,
                                {"eval/loss": eval_loss, "eval/ppl": torch.exp(torch.tensor(eval_loss)).item()},
                            )
                    except Exception as e:
                        logger.warning("Eval failed at epoch %d, continuing: %s", epoch + 1, e)
        finally:
            # Drain in-flight ops on exit (incl. SIGTERM) to avoid leaking partial batches.
            _pipe_drain_safe()

        # -- Final checkpoint --------------------------------------------------

        start_step = resume_info.step if resume_info else 0
        if cfg.save_final_checkpoint and step > start_step:
            logger.info("Saving final checkpoint (step %d)...", step)
            cp_name = f"step-{step}"
            ckpt.save(
                cp_name,
                resumable=True,
                promotable=True,
                data_consumed=cursor.value,
            )
            if getattr(cfg, "output_model_id", None):
                ckpt.promote_latest(cfg.output_model_id, cfg.base_model)
                runner.write_output_model(
                    model_id=cfg.output_model_id, checkpoint=cp_name, job_id=job_id,
                )

        write_completed(runner, step=step, total_steps=step)
        logger.info("Training complete: %d optimizer steps", step)
        wandb_finish()
        return {"steps": step, "job_id": job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        log_path="./sft_logs",
        dataset="kimi2_deid_sample_100_formatted.jsonl",
        tokenizer_model="Qwen/Qwen3-8B",
        max_seq_len=4096,
        max_examples=10,
        trainer=TrainerConfig(
            training_shape_id="your-training-shape",
        ),
    )
    main(cfg)
