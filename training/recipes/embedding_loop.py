#!/usr/bin/env python3
"""Minimal embedding (retrieval) fine-tuning loop.

A readable, modifiable contrastive fine-tuning loop for embedding models,
built on the Fireworks Training SDK. For a batch of ``B`` (query, positive)
pairs it trains with bidirectional in-batch InfoNCE: each query's paired
document is the only positive, and the other ``B-1`` documents in the batch
act as random negatives (and vice versa).

The SDK exposes three interchangeable ways to drive the same loss; this recipe
demonstrates all of them via ``Config.output_mode``:

  - ``embedding`` (default): the trainer returns ``2B`` pooled vectors, the
    client computes bidirectional InfoNCE locally and ships gradients back.
    Per-datum and therefore safe to split across HTTP chunks.

  - ``cos_similarity_matrix``: the trainer returns rows of the in-batch
    ``[2B, 2B]`` cosine-similarity matrix and the client computes the same
    loss over the submatrix. Requires the whole batch to fit in one request
    (single-GPU / DP=1 trainer), so keep ``batch_size`` modest.

  - ``contrastive_loss``: the trainer runs pool + normalize + similarity +
    cross-entropy + backward server-side in a single round trip (fastest).

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
import time
from contextlib import ExitStack
from dataclasses import dataclass, field

import tinker
import torch
import torch.nn.functional as F
from tinker import types

from training.utils import (
    DEFAULT_ADAM,
    ReconnectableClient,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    TrainerConfig,
    WandBConfig,
    build_service_client,
    load_jsonl_dataset,
    load_tokenizer,
    log_metrics_json,
    read_api_extra_headers_env,
    setup_wandb,
    validate_config,
    wandb_finish,
    wandb_log,
)
from training.utils.checkpoints import TrainingCheckpoints
from training.utils.client import DEFAULT_TIMEOUT_S
from training.utils.runner_state import start_running, write_completed, write_running_step

logger = logging.getLogger(__name__)

# Qwen3-Embedding convention: queries get an instruction prefix, documents do
# not. Keeping train/eval tokenization identical is what makes the fine-tune
# transfer to retrieval eval.
_QWEN3_INSTRUCTION_TEMPLATE = "Instruct: {}\nQuery:"


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def bidirectional_info_nce_loss(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Bidirectional in-batch InfoNCE over ``B`` (query, positive) pairs.

    Row ``i`` of each tensor is paired (query ``i`` <-> positive ``i``). Every
    other document in the batch is an in-batch negative.
    """
    z_q = F.normalize(query_embeddings, p=2, dim=-1)
    z_d = F.normalize(doc_embeddings, p=2, dim=-1)
    sim = (z_q @ z_d.t()) / temperature  # [B, B]
    targets = torch.arange(sim.shape[0], device=sim.device)
    loss_q2d = F.cross_entropy(sim, targets)
    loss_d2q = F.cross_entropy(sim.t(), targets)
    loss = 0.5 * (loss_q2d + loss_d2q)
    with torch.no_grad():
        recall_at_1 = (sim.argmax(dim=1) == targets).float().mean().item()
        metrics = {
            "loss": float(loss.item()),
            "loss_q2d": float(loss_q2d.item()),
            "loss_d2q": float(loss_d2q.item()),
            "in_batch_recall_at_1": recall_at_1,
        }
    return loss, metrics


def _build_embedding_loss_fn(batch_queries: int, temperature: float):
    """``output="embedding"``: trainer returns ``2B`` pooled vectors."""

    def loss_fn(_data, embeddings, _B=batch_queries, _T=temperature):
        q = torch.stack(embeddings[:_B], dim=0).to(torch.float32)
        d = torch.stack(embeddings[_B:2 * _B], dim=0).to(torch.float32)
        return bidirectional_info_nce_loss(q, d, temperature=_T)

    return loss_fn


def _build_cosine_similarity_loss_fn(batch_queries: int, temperature: float):
    """``output="cos_similarity_matrix"``: trainer returns rows of ``[2B, 2B]``.

    The client slices the query-vs-doc submatrix and computes the same
    bidirectional InfoNCE the embedding path does.
    """

    def loss_fn(_data, embeddings, _B=batch_queries, _T=temperature):
        sim_full = torch.stack(embeddings, dim=0).to(torch.float32)  # [2B, 2B]
        sim_q2d = sim_full[:_B, _B:2 * _B] / _T  # queries vs paired docs
        sim_d2q = sim_full[_B:2 * _B, :_B] / _T  # paired docs vs queries
        targets = torch.arange(_B, device=sim_q2d.device)
        loss_q2d = F.cross_entropy(sim_q2d, targets)
        loss_d2q = F.cross_entropy(sim_d2q, targets)
        loss = 0.5 * (loss_q2d + loss_d2q)
        with torch.no_grad():
            recall_at_1 = (sim_q2d.argmax(dim=1) == targets).float().mean().item()
            metrics = {
                "loss": float(loss.item()),
                "loss_q2d": float(loss_q2d.item()),
                "loss_d2q": float(loss_d2q.item()),
                "in_batch_recall_at_1": recall_at_1,
            }
        return loss, metrics

    return loss_fn


def _format_metrics(metrics: dict[str, float]) -> str:
    # The server renames client-side ``loss`` to ``loss:sum`` when it flows back
    # through forward_backward (contrastive_loss mode); honor either name.
    loss_val = metrics.get("loss", metrics.get("loss:sum", float("nan")))
    parts = [f"loss={loss_val:.4f}"]
    if "loss_q2d" in metrics:
        parts.append(f"q2d={metrics['loss_q2d']:.4f}")
    if "loss_d2q" in metrics:
        parts.append(f"d2q={metrics['loss_d2q']:.4f}")
    if "in_batch_recall_at_1" in metrics:
        parts.append(f"recall@1={metrics['in_batch_recall_at_1']:.3f}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


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
        query_texts, add_special_tokens=True, truncation=True,
        max_length=max_query_len, padding=False,
    )["input_ids"]
    doc_ids = tokenizer(
        doc_texts, add_special_tokens=True, truncation=True,
        max_length=max_doc_len, padding=False,
    )["input_ids"]

    data: list[types.Datum] = []
    for ids in [*query_ids, *doc_ids]:
        data.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(list(ids)),
                loss_fn_inputs={},
            )
        )
    return data


def _load_pairs(dataset: str, max_examples: int | None) -> list[tuple[str, str]]:
    rows = load_jsonl_dataset(dataset, max_examples)
    pairs: list[tuple[str, str]] = []
    for row in rows:
        query = row.get("query")
        positive = row.get("positive")
        if query and positive:
            pairs.append((str(query), str(positive)))
    return pairs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    base_model: str = "accounts/fireworks/models/qwen3-embedding-8b"
    dataset: str = ""
    tokenizer_model: str = ""  # HuggingFace model name, e.g. "Qwen/Qwen3-Embedding-8B"
    tokenizer_revision: str = ""

    output_mode: str = "embedding"
    """One of "embedding", "cos_similarity_matrix", "contrastive_loss".
    "cos_similarity_matrix" requires the batch to fit in a single request
    (single-GPU / DP=1 trainer), so keep batch_size modest in that mode."""

    query_instruction: str = "Given a web search query, retrieve relevant passages that answer the query"
    """Qwen3-Embedding query-side instruction. Documents get no prefix."""

    temperature: float = 0.02
    learning_rate: float = 1e-5
    grad_clip_norm: float = 1.0

    epochs: int = 1
    batch_size: int = 16
    """Number of (query, positive) pairs per step. Must be >= 2 so there is at
    least one in-batch negative; the trainer sees 2*batch_size sequences."""
    max_examples: int | None = None
    max_query_len: int = 64
    max_doc_len: int = 256
    lora_rank: int = 0
    output_model_id: str | None = None
    save_final_checkpoint: bool = True
    seed: int = 0

    step_timeout: int = 0
    """Timeout in seconds for forward_backward / optim_step. 0 = default."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="embedding-tinker"))
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    """Optional orchestration outputs (status / metrics / output-model) written
    during training. See training/utils/runner.py for the file formats."""


_VALID_MODES = ("embedding", "cos_similarity_matrix", "contrastive_loss")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config: Config):
    cfg = config
    runner = RunnerIO(cfg.runner)

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(cfg.base_model, cfg.dataset, output_model_id=cfg.output_model_id)
    if cfg.output_mode not in _VALID_MODES:
        raise ValueError(
            f"Unsupported output_mode={cfg.output_mode!r}; expected one of {_VALID_MODES}."
        )
    if cfg.batch_size < 2:
        raise ValueError(
            f"batch_size must be >= 2 for in-batch negatives (got {cfg.batch_size})."
        )
    if not cfg.tokenizer_model:
        raise ValueError(
            "Config.tokenizer_model is required for client-side tokenization. "
            "Set it to the HuggingFace model name (e.g. 'Qwen/Qwen3-Embedding-8B')."
        )
    setup_wandb(
        cfg.wandb,
        {
            "output_mode": cfg.output_mode,
            "lr": cfg.learning_rate,
            "temperature": cfg.temperature,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
        },
    )

    # -- SDK-managed Tinker client -----------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    runner.write_status(RunStatus.PENDING, message="provisioning")

    # Provision the trainer with enough context for the longest tokenized
    # sequence. Queries and docs are separate datums, so the longest single
    # sequence is the larger of the two caps -- not their sum.
    max_context_length = max(cfg.max_query_len, cfg.max_doc_len)

    with runner, ExitStack() as stack:
        service = build_service_client(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
            base_model=cfg.base_model,
            tokenizer_model=cfg.tokenizer_model,
            lora_rank=cfg.lora_rank,
            max_context_length=max_context_length,
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

        # -- Data ----------------------------------------------------------

        tokenizer = load_tokenizer(cfg.tokenizer_model, cfg.tokenizer_revision)
        pairs = _load_pairs(cfg.dataset, cfg.max_examples)
        # In-batch InfoNCE needs every step to have exactly batch_size queries
        # (a partial/size-1 tail batch would weaken or break the contrast), so
        # the loop below drops the per-epoch remainder. Require at least one
        # full batch, and surface how many pairs are dropped so it isn't silent.
        if len(pairs) < cfg.batch_size:
            raise RuntimeError(
                f"Dataset has {len(pairs)} (query, positive) pair(s) from {cfg.dataset}, "
                f"fewer than batch_size={cfg.batch_size}: no full batch can be formed and "
                f"nothing would be trained. Add more examples or lower batch_size."
            )
        logger.info("Loaded %d (query, positive) pairs from %s", len(pairs), cfg.dataset)
        dropped_per_epoch = len(pairs) % cfg.batch_size
        if dropped_per_epoch:
            logger.info(
                "Dropping %d remainder pair(s) each epoch to keep full batch_size=%d "
                "batches (uniform in-batch negatives).",
                dropped_per_epoch, cfg.batch_size,
            )

        # -- Training loop -------------------------------------------------

        adam_kwargs = dict(DEFAULT_ADAM)
        adam_kwargs["grad_clip_norm"] = cfg.grad_clip_norm
        adam = tinker.AdamParams(learning_rate=cfg.learning_rate, **adam_kwargs)

        total_steps = (len(pairs) // cfg.batch_size) * cfg.epochs
        logger.info(
            "Training: mode=%s steps=%d batch_size=%d lr=%.2e temp=%.3f",
            cfg.output_mode, total_steps, cfg.batch_size, cfg.learning_rate, cfg.temperature,
        )
        start_running(runner, total_steps=total_steps)

        step = 0
        for epoch in range(cfg.epochs):
            epoch_pairs = list(pairs)
            random.Random(cfg.seed + epoch).shuffle(epoch_pairs)
            # Step only over full batches; the trailing remainder is dropped (see above).
            for start in range(0, len(epoch_pairs) - cfg.batch_size + 1, cfg.batch_size):
                batch = epoch_pairs[start:start + cfg.batch_size]
                step_t0 = time.monotonic()
                datums = _build_batch_datums(
                    batch,
                    tokenizer=tokenizer,
                    query_instruction=cfg.query_instruction,
                    max_query_len=cfg.max_query_len,
                    max_doc_len=cfg.max_doc_len,
                )

                if cfg.output_mode == "contrastive_loss":
                    result = client.forward_backward_contrastive(
                        datums, num_queries=cfg.batch_size, temperature=cfg.temperature,
                        pooling="last",
                    )
                elif cfg.output_mode == "cos_similarity_matrix":
                    loss_fn = _build_cosine_similarity_loss_fn(cfg.batch_size, cfg.temperature)
                    result = client.forward_backward_custom(
                        datums, loss_fn, output="cos_similarity_matrix", pooling="last",
                    )
                else:
                    loss_fn = _build_embedding_loss_fn(cfg.batch_size, cfg.temperature)
                    result = client.forward_backward_custom(
                        datums, loss_fn, output="embedding", pooling="last",
                    )
                client.optim_step(adam)
                step += 1

                metrics = dict(result.metrics)
                step_elapsed = time.monotonic() - step_t0
                logger.info(
                    "Step %d/%d | epoch %d | %s | %.1fs",
                    step, total_steps, epoch + 1, _format_metrics(metrics), step_elapsed,
                )
                log_metrics_json(step, **metrics)
                step_metrics = {f"train/{k}": v for k, v in metrics.items()}
                step_metrics["train/step"] = step
                step_metrics["train/epoch"] = epoch + 1
                wandb_log(step_metrics, step)
                write_running_step(
                    runner, step=step, total_steps=total_steps, metrics=step_metrics,
                )

        # -- Final checkpoint ----------------------------------------------

        if cfg.save_final_checkpoint and step > 0:
            logger.info("Saving final checkpoint (step %d)...", step)
            cp_name = f"step-{step}"
            ckpt.save(
                cp_name,
                resumable=True,
                promotable=True,
                data_consumed=step * cfg.batch_size,
            )
            if cfg.output_model_id:
                ckpt.promote_latest(cfg.output_model_id, cfg.base_model)
                runner.write_output_model(
                    model_id=cfg.output_model_id, checkpoint=cp_name, job_id=job_id,
                )

        write_completed(runner, step=step, total_steps=total_steps)
        logger.info("Training complete: %d optimizer steps", step)
        wandb_finish()
        return {"steps": step, "job_id": job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        log_path="./embedding_logs",
        dataset=os.environ.get("EMBEDDING_DATASET_PATH", "retrieval_pairs.jsonl"),
        tokenizer_model=os.environ.get("EMBEDDING_TOKENIZER", "Qwen/Qwen3-Embedding-8B"),
        trainer=TrainerConfig(training_shape_id=os.environ.get("EMBEDDING_TRAINING_SHAPE", "")),
    )
    main(cfg)
