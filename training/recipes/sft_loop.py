#!/usr/bin/env python3
"""Minimal SFT (Supervised Fine-Tuning) training loop.

A readable, modifiable fine-tuning loop using the Fireworks RLOR API.
Uses a single RLOR trainer job with cross-entropy loss on response tokens.

Dataset format (JSONL, OpenAI chat format):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    export FIREWORKS_API_KEY=...
    export FIREWORKS_ACCOUNT_ID=...
    python -m recipes.sft_loop
"""

from __future__ import annotations

import os
import signal
import logging
from typing import Any, Dict, List
from dataclasses import field, dataclass

import torch
import tinker

import json
import datasets as hf_datasets
import transformers
from dotenv import load_dotenv

from fireworks.training.sdk import TrainerJobManager
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from training.utils import (
    DEFAULT_ADAM,
    InfraConfig,
    ResourceCleanup,
    WandBConfig,
    ReconnectableClient,
    wandb_log,
    setup_wandb,
    wandb_finish,
    validate_config,
    log_metrics_json,
    create_trainer_job,
    make_batch_weighted_sft_loss_fn,
    build_renderer,
    parse_train_on_what,
    render_messages_to_datum,
    resolve_renderer_name,
)
from training.utils.checkpoint_utils import (
    resolve_resume,
    save_checkpoint,
    CheckpointKind,
)
from training.utils.timer import timer, flush_timing


load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str = ""
    tokenizer_model: str = ""  # HuggingFace model name for chat template, e.g. "Qwen/Qwen3-1.7B"
    renderer_name: str = ""
    train_on_what: str = "all_assistant_messages"

    learning_rate: float = 1e-4
    epochs: int = 3
    batch_size: int = 32
    grad_accum: int = 4
    max_seq_len: int | None = None
    max_examples: int | None = None
    lora_rank: int = 0
    output_model_id: str | None = None

    dcp_save_interval: int = 0  # save DCP checkpoint every N steps (0 = off)

    init_from_checkpoint: str | None = None
    """Load pretrained DCP weights on a fresh dataset. Supports cross-job
    format ``"job_id:checkpoint_name"``."""

    grad_accumulation_normalization: str | None = "num_loss_tokens"
    """Normalization mode for accumulated gradients at optim_step.
    ``"num_loss_tokens"``: per-token mean (verl ``token-mean``).
    ``"num_sequences"``: per-sequence mean (verl ``seq-mean-token-sum``
    if loss is raw sum, ``seq-mean-token-mean`` if loss is pre-normalized).
    ``"none"``: no normalization.
    ``None``: server default (currently per-token).
    Requires the loss function to return raw token sums (raw_sum=True),
    which is set automatically when this is not ``"none"``."""

    grad_clip_norm: float = 0.0
    """Max gradient norm for clipping. 0 = no clipping."""

    infra: InfraConfig = field(default_factory=InfraConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="sft-tinker"))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
):
    cfg = config

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(cfg.base_model, cfg.dataset)
    setup_wandb(
        cfg.wandb,
        {
            "lr": cfg.learning_rate,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "grad_accum": cfg.grad_accum,
        },
    )

    if not cfg.tokenizer_model:
        raise ValueError(
            "Config.tokenizer_model is required for chat template formatting. "
            "Set it to the HuggingFace model name (e.g. 'Qwen/Qwen3-1.7B')."
        )

    # -- Setup infrastructure ----------------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    account = os.environ.get("FIREWORKS_ACCOUNT_ID", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(api_key=api_key, account_id=account, base_url=base_url)

    profile = None
    if cfg.infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)

    if profile and cfg.max_seq_len is None:
        cfg.max_seq_len = profile.max_supported_context_length
        logger.info("max_seq_len from training shape: %d", cfg.max_seq_len)
    if cfg.max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )

    with ResourceCleanup(rlor_mgr) as cleanup:
        endpoint = create_trainer_job(
            rlor_mgr,
            base_model=cfg.base_model,
            infra=cfg.infra,
            profile=profile,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            display_name="sft-trainer",
        )
        job_id = endpoint.job_id
        cleanup.trainer(job_id)
        client = ReconnectableClient(rlor_mgr, job_id, cfg.base_model, cfg.lora_rank, fw_api_key=api_key)

        # -- Prepare data ------------------------------------------------------
        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer_model, trust_remote_code=True)
        renderer = build_renderer(tokenizer, cfg.tokenizer_model, cfg.renderer_name)
        train_on_what = parse_train_on_what(cfg.train_on_what)
        logger.info(
            "Using renderer=%s train_on_what=%s",
            resolve_renderer_name(cfg.tokenizer_model, cfg.renderer_name),
            train_on_what.value,
        )

        raw_data: List[Dict[str, Any]] = []
        with open(cfg.dataset) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_data.append(json.loads(line))
                if cfg.max_examples and len(raw_data) >= cfg.max_examples:
                    break
        logger.info("Loaded %d examples from %s", len(raw_data), cfg.dataset)

        max_seq_len = cfg.max_seq_len
        filtered_count = 0

        def _map_fn(row: dict) -> tinker.Datum | None:
            nonlocal filtered_count
            messages = row.get("messages", [])
            if not messages:
                filtered_count += 1
                return None
            rendered = render_messages_to_datum(
                messages, renderer=renderer, train_on_what=train_on_what,
            )
            if len(rendered.token_ids) > max_seq_len or len(rendered.token_ids) < 2:
                filtered_count += 1
                return None
            return rendered.datum

        training_data = [d for row in raw_data if (d := _map_fn(row)) is not None]
        if filtered_count > 0:
            logger.info(
                "Seq-length filter: %d/%d examples filtered (len > %d or len < 2)",
                filtered_count, len(raw_data), max_seq_len,
            )
        logger.info("Prepared %d training examples", len(training_data))
        if not training_data:
            raise RuntimeError("No valid training examples after tokenization")

        sft_dataset = SupervisedDatasetFromHFDataset(
            hf_datasets.Dataset.from_dict({"datum_idx": list(range(len(training_data)))}),
            batch_size=cfg.batch_size,
            map_fn=lambda row: training_data[row["datum_idx"]],
        )
        total_batches_per_epoch = len(sft_dataset)
        logger.info("Dataset: %d examples, %d batches/epoch, %d epochs",
                     len(training_data), total_batches_per_epoch, cfg.epochs)

        # -- Resume ---------------------------------------------------------------

        resume_info = resolve_resume(client, cfg.log_path, cfg.init_from_checkpoint)
        step = resume_info.step if resume_info else 0
        data_consumed = resume_info.data_consumed if resume_info else 0
        wandb_log({"train/step": step}, step)

        adam_kwargs = dict(DEFAULT_ADAM)
        if cfg.grad_clip_norm > 0:
            adam_kwargs["grad_clip_norm"] = cfg.grad_clip_norm
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **adam_kwargs)

        # -- Training loop (batch-indexed) -------------------------------------

        start_batch = data_consumed // cfg.batch_size
        total_steps_estimate = (total_batches_per_epoch * cfg.epochs) // cfg.grad_accum
        accum = 0
        agg_loss_sum = 0.0
        agg_resp_tokens = 0
        agg_sequences = 0

        def _flush_batch(batch_buf: list[tinker.Datum], step: int, accum: int) -> tuple[int, int]:
            nonlocal agg_loss_sum, agg_resp_tokens, agg_sequences

            # NOTE: raw_sum=True when server-side normalization is active to
            # avoid double-normalization (client divides by token count AND
            # server divides again). raw_sum=False only with "none".
            use_raw_sum = cfg.grad_accumulation_normalization != "none"
            loss_fn = make_batch_weighted_sft_loss_fn(raw_sum=use_raw_sum)
            with timer("fwd_bwd"):
                result = client.forward_backward_custom(batch_buf, loss_fn)

            fwd_metrics = result.metrics
            agg_loss_sum += fwd_metrics.get("ce_loss_sum", 0.0)
            agg_resp_tokens += fwd_metrics.get("response_tokens", 0)
            agg_sequences += fwd_metrics.get("batch_size", len(batch_buf))
            accum += 1

            if accum >= cfg.grad_accum:
                with timer("optim_step"):
                    optim_result = client.optim_step(
                        adam_params,
                        grad_accumulation_normalization=cfg.grad_accumulation_normalization,
                    )
                step += 1
                accum = 0

                if cfg.dcp_save_interval > 0 and step % cfg.dcp_save_interval == 0:
                    with timer("dcp_save"):
                        logger.info("Saving DCP checkpoint at step %d", step)
                        save_checkpoint(client, f"step-{step}", cfg.log_path, {
                            "step": step,
                            "data_consumed": data_consumed,
                            "source_job_id": job_id,
                        }, kind=CheckpointKind.STATE)

                step_metrics: Dict[str, Any] = flush_timing()

                if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
                    for k, v in optim_result.metrics.items():
                        step_metrics[f"train/{k}"] = v

                norm_mode = cfg.grad_accumulation_normalization
                if norm_mode == "num_loss_tokens":
                    expected_norm_factor = agg_resp_tokens
                elif norm_mode == "num_sequences":
                    expected_norm_factor = agg_sequences
                else:
                    expected_norm_factor = 0

                step_metrics["grad_acc/accumulated_sequences"] = agg_sequences
                step_metrics["grad_acc/accumulated_tokens"] = agg_resp_tokens
                step_metrics["grad_acc/loss_sum_raw"] = agg_loss_sum
                step_metrics["grad_acc/norm_mode"] = norm_mode or "none"
                step_metrics["grad_acc/expected_norm_factor"] = expected_norm_factor
                step_metrics["grad_acc/num_microbatches"] = cfg.grad_accum

                if agg_resp_tokens > 0:
                    avg_loss = agg_loss_sum / agg_resp_tokens
                    ppl = torch.exp(torch.tensor(avg_loss)).item()
                    logger.info(
                        "Step %d/%d | Loss: %.4f | PPL: %.2f | accum_tokens=%d accum_seqs=%d norm=%s",
                        step, total_steps_estimate, avg_loss, ppl,
                        agg_resp_tokens, agg_sequences, norm_mode or "none",
                    )
                    log_metrics_json(step, ce_loss=avg_loss, ppl=ppl)
                    step_metrics.update({
                        "train/step": step,
                        "train/ce_loss": avg_loss,
                        "train/ppl": ppl,
                    })
                    wandb_log(step_metrics, step)

                agg_loss_sum = 0.0
                agg_resp_tokens = 0
                agg_sequences = 0

            return step, accum

        for epoch in range(cfg.epochs):
            sft_dataset.set_epoch(epoch)
            epoch_start = start_batch if epoch == 0 else 0
            for i_batch in range(epoch_start, total_batches_per_epoch):
                batch = sft_dataset.get_batch(i_batch)
                data_consumed += len(batch)
                step, accum = _flush_batch(batch, step, accum)

        if accum > 0:
            client.optim_step(
                adam_params,
                grad_accumulation_normalization=cfg.grad_accumulation_normalization,
            )
            step += 1

        # -- Final checkpoint (skip if dcp_save_interval == -1) ----------------

        start_step = resume_info.step if resume_info else 0
        if cfg.dcp_save_interval != -1 and step > start_step:
            logger.info("Saving final checkpoint (step %d)...", step)
            cp_name = f"step-{step}"
            save_checkpoint(client, cp_name, cfg.log_path, {
                "step": step,
                "data_consumed": data_consumed,
                "source_job_id": job_id,
            }, kind=CheckpointKind.BOTH)
            
            if getattr(cfg, "output_model_id", None):
                from training.utils.checkpoint_utils import promote_checkpoint
                promote_checkpoint(
                    rlor_mgr,
                    job_id,
                    cp_name,
                    cfg.output_model_id,
                )

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
        infra=InfraConfig(
            training_shape_id="your-training-shape",
        ),
    )
    main(cfg)
