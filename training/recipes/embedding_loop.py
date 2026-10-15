#!/usr/bin/env python3
"""Minimal embedding (retrieval) fine-tuning loop.

For each batch of ``B`` (query, positive) pairs:

1. tokenize the pairs into ``2B`` sequences (queries first, then documents),
2. the trainer returns one pooled embedding per sequence, the client computes
   bidirectional in-batch InfoNCE (``training.utils.embedding``) and sends the
   gradients back, then one optimizer step.

Each query's paired document is its only positive; the other ``B-1`` documents
in the batch are random negatives (and vice versa). Fork this file to change
the loss or the data format.

Dataset format (JSONL, one pair per line):
    {"query": "...", "positive": "..."}

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.embedding_loop
"""

from __future__ import annotations

import logging
import os
import random
import signal
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

from fireworks.training import sdk
from fireworks.training.sdk import types
from training.utils import (
    DEFAULT_ADAM,
    ReconnectableClient,
    TrainerConfig,
    WandBConfig,
    build_service_client,
    load_jsonl_dataset,
    load_tokenizer,
    log_metrics,
    log_metrics_json,
    read_api_extra_headers_env,
    setup_wandb,
    validate_config,
    wandb_finish,
)
from training.utils.checkpoints import TrainingCheckpoints
from training.utils.client import DEFAULT_TIMEOUT_S
from training.utils.embedding import info_nce_loss_fn
from training.utils.logging import log_run_resources
from training.utils.metrics import forward_backward_metrics, optimizer_metrics
from training.utils.termination import TerminatedBySignal
from training.utils.timer import elapsed_timer, flush_timing

logger = logging.getLogger(__name__)

# Qwen3-Embedding convention: queries get an instruction prefix, documents do
# not. Keeping train/eval tokenization identical is what makes the fine-tune
# transfer to retrieval eval.
_QWEN3_INSTRUCTION_TEMPLATE = "Instruct: {}\nQuery:"


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    base_model: str = "accounts/fireworks/models/qwen3-embedding-8b"
    dataset: str = ""
    tokenizer_model: str = ""  # HuggingFace model name, e.g. "Qwen/Qwen3-Embedding-8B"
    tokenizer_revision: str = ""

    query_instruction: str = "Given a web search query, retrieve relevant passages that answer the query"
    """Qwen3-Embedding query-side instruction. Documents get no prefix."""

    temperature: float = 0.02
    learning_rate: float = 1e-5
    grad_clip_norm: float = 0.0

    epochs: int = 1
    batch_size: int = 16
    """Number of (query, positive) pairs per step. Must be >= 2 so there is at
    least one in-batch negative; the trainer sees 2*batch_size sequences."""
    max_examples: int | None = None
    max_query_len: int = 64
    max_doc_len: int = 256
    lora_rank: int = 0
    lora_alpha: int | None = 32
    """LoRA alpha scaling factor. Ignored when ``lora_rank == 0``.

    Defaults to ``32`` to match Tinker and the Training API SDK client
    ``DEFAULT_LORA_ALPHA``. Override when you need a different scaling factor."""
    output_model_id: str | None = None
    save_final_checkpoint: bool = True
    seed: int = 0

    step_timeout: int = 0
    """Timeout in seconds for forward_backward / optim_step. 0 = default."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="embedding-tinker"))


# ---------------------------------------------------------------------------
# Data -- customize these for your dataset
# ---------------------------------------------------------------------------


def _load_pairs(dataset: str, max_examples: int | None) -> list[tuple[str, str]]:
    rows = load_jsonl_dataset(dataset, max_examples)
    pairs: list[tuple[str, str]] = []
    for row in rows:
        query = row.get("query")
        positive = row.get("positive")
        if query and positive:
            pairs.append((str(query), str(positive)))
    return pairs


def _build_batch_datums(
    pairs: list[tuple[str, str]],
    *,
    tokenizer,
    query_instruction: str,
    max_query_len: int,
    max_doc_len: int,
) -> list[types.Datum]:
    """Tokenize ``B`` (query, positive) pairs into ``2B`` Datums.

    Layout: ``[Q_0..Q_{B-1}, D_0..D_{B-1}]`` — queries first, then documents,
    so the loss closure can split embeddings as ``[:B]`` / ``[B:2B]``.
    """
    query_texts = [_QWEN3_INSTRUCTION_TEMPLATE.format(query_instruction) + q for q, _ in pairs]
    doc_texts = [d for _, d in pairs]

    # truncation vs. pooling="last": HF tokenizers reserve room for special
    # tokens and truncate content first, so the trailing special token that
    # pooling="last" reads is preserved even for over-length inputs. This
    # assumes the tokenizer appends a trailing special (Qwen3-Embedding adds
    # <|endoftext|>); if yours appends none, prefer mean pooling or set
    # tokenizer.truncation_side="left" so a real end token isn't dropped.
    query_ids = tokenizer(
        query_texts,
        add_special_tokens=True,
        truncation=True,
        max_length=max_query_len,
        padding=False,
    )["input_ids"]
    doc_ids = tokenizer(
        doc_texts,
        add_special_tokens=True,
        truncation=True,
        max_length=max_doc_len,
        padding=False,
    )["input_ids"]

    return [
        types.Datum(model_input=types.ModelInput.from_ints(list(ids)), loss_fn_inputs={})
        for ids in [*query_ids, *doc_ids]
    ]


def main(config: Config) -> dict[str, Any]:
    cfg = config
    validate_config(cfg.base_model, cfg.dataset, output_model_id=cfg.output_model_id)
    if cfg.batch_size < 2:
        raise ValueError(f"batch_size must be >= 2 for in-batch negatives (got {cfg.batch_size}).")
    if not cfg.tokenizer_model:
        raise ValueError(
            "Config.tokenizer_model is required for client-side tokenization. "
            "Set it to the HuggingFace model name (e.g. 'Qwen/Qwen3-Embedding-8B')."
        )

    def _signal_handler(signum, _):
        name = signal.Signals(signum).name
        raise TerminatedBySignal(name)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    setup_wandb(
        cfg.wandb,
        {
            "lr": cfg.learning_rate,
            "temperature": cfg.temperature,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
        },
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()
    with ExitStack() as stack:
        # Queries and docs are separate datums, so the longest single sequence
        # is the larger of the two caps -- not their sum.
        service = build_service_client(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
            base_model=cfg.base_model,
            tokenizer_model=cfg.tokenizer_model,
            max_lora_rank=cfg.lora_rank,
            max_context_length=max(cfg.max_query_len, cfg.max_doc_len),
            learning_rate=cfg.learning_rate,
            trainer=cfg.trainer,
            cleanup_trainer_on_close=True,
        )
        stack.callback(service.close)
        training_client = service.create_training_client(
            cfg.base_model,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
        )
        job_id = service.trainer_job_id
        policy = ReconnectableClient.from_training_client(
            training_client,
            base_model=cfg.base_model,
            lora_rank=cfg.lora_rank,
            job_id=job_id,
            default_timeout=cfg.step_timeout or DEFAULT_TIMEOUT_S,
            service=service,
        )
        checkpoint = TrainingCheckpoints(
            policy,
            service,
            trainer_id=job_id,
            log_path=cfg.log_path,
            lora_rank=cfg.lora_rank,
        )
        log_run_resources(service)

        tokenizer = load_tokenizer(cfg.tokenizer_model, cfg.tokenizer_revision)
        pairs = _load_pairs(cfg.dataset, cfg.max_examples)
        if len(pairs) < cfg.batch_size:
            raise RuntimeError(
                f"Dataset has {len(pairs)} (query, positive) pair(s) from {cfg.dataset}, "
                f"fewer than batch_size={cfg.batch_size}: no full batch can be formed and "
                f"nothing would be trained. Add more examples or lower batch_size."
            )
        # Every step needs exactly batch_size pairs (a short tail batch would
        # weaken the contrast), so each epoch drops its remainder.
        steps_per_epoch = len(pairs) // cfg.batch_size
        total_steps = steps_per_epoch * cfg.epochs
        logger.info(
            "Training on %d pairs from %s: %d steps of %d (dropping %d pair(s) per epoch)",
            len(pairs),
            cfg.dataset,
            total_steps,
            cfg.batch_size,
            len(pairs) % cfg.batch_size,
        )

        # -- Training loop (VISIBLE algorithm phases) -------------------------

        loss_fn = info_nce_loss_fn(cfg.batch_size, cfg.temperature)
        adam_params = sdk.AdamParams(
            learning_rate=cfg.learning_rate,
            **{**DEFAULT_ADAM, "grad_clip_norm": cfg.grad_clip_norm},
        )
        step = 0
        for epoch in range(cfg.epochs):
            epoch_pairs = list(pairs)
            random.Random(cfg.seed + epoch).shuffle(epoch_pairs)
            for start in range(0, steps_per_epoch * cfg.batch_size, cfg.batch_size):
                datums = _build_batch_datums(
                    epoch_pairs[start : start + cfg.batch_size],
                    tokenizer=tokenizer,
                    query_instruction=cfg.query_instruction,
                    max_query_len=cfg.max_query_len,
                    max_doc_len=cfg.max_doc_len,
                )

                # 1. Pooled embeddings out, client-side InfoNCE gradients back.
                #    For one server-side round trip, use
                #    policy.forward_backward_contrastive(datums, num_queries=B, temperature=T).
                with elapsed_timer("fwd_bwd"):
                    fwd_bwd_result = policy.forward_backward_custom(
                        datums,
                        loss_fn,
                        output="embedding",
                        pooling="last",
                    )
                # 2. One optimizer step.
                with elapsed_timer("optim_step"):
                    optim_result = policy.optim_step(adam_params)
                step += 1

                metrics = flush_timing()
                metrics.update(forward_backward_metrics([fwd_bwd_result]))
                metrics.update(optimizer_metrics(optim_result))
                metrics["train/step"] = step
                metrics["train/epoch"] = epoch + 1
                logger.info(
                    "Step %d/%d | epoch %d | loss=%.4f | recall@1=%.3f",
                    step,
                    total_steps,
                    epoch + 1,
                    metrics.get("train/loss", float("nan")),
                    metrics.get("train/in_batch_recall_at_1", float("nan")),
                )
                log_metrics_json(step, **dict(fwd_bwd_result.metrics))
                log_metrics(metrics, step=step)

        if cfg.save_final_checkpoint and step > 0:
            checkpoint.save(
                f"step-{step}",
                resumable=True,
                promotable=True,
                data_consumed=step * cfg.batch_size,
            )
            if cfg.output_model_id:
                checkpoint.promote_latest(cfg.output_model_id, cfg.base_model)

        logger.info("Training complete: %d optimizer steps", step)
        wandb_finish(metrics_file=os.environ.get("COOKBOOK_METRICS_FILE"))
        return {"steps": step, "job_id": job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(
        Config(
            log_path="./embedding_logs",
            dataset=os.environ.get("EMBEDDING_DATASET_PATH", "retrieval_pairs.jsonl"),
            tokenizer_model=os.environ.get("EMBEDDING_TOKENIZER", "Qwen/Qwen3-Embedding-8B"),
            trainer=TrainerConfig(training_shape_id=os.environ.get("EMBEDDING_TRAINING_SHAPE", "")),
        )
    )
