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
import transformers
from dotenv import load_dotenv

from fireworks.training.sdk import TrainerJobManager
from training.utils import (
    DEFAULT_ADAM,
    InfraConfig,
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
    load_dcp,
    save_loop_state,
    dataset_fingerprint,
    validate_dataset,
)
from training.utils.timer import timer, flush_timing


load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
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

    dcp_save_interval: int = 0  # save DCP checkpoint every N steps (0 = off)

    log_path: str = "./sft_logs"
    init_from_checkpoint: str | None = None
    """Load pretrained DCP weights on a fresh dataset. Supports cross-job
    format ``"job_id:checkpoint_name"``."""

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

    validate_config(cfg.base_model, cfg.dataset, infra=cfg.infra)
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
    client = ReconnectableClient(rlor_mgr, job_id, cfg.base_model, cfg.lora_rank, fw_api_key=api_key)

    # -- Prepare data ------------------------------------------------------
    try:
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

        training_data: List[tinker.Datum] = []
        filtered_count = 0
        for row in raw_data:
            messages = row.get("messages", [])
            if not messages:
                continue

            rendered = render_messages_to_datum(
                messages,
                renderer=renderer,
                train_on_what=train_on_what,
            )
            if len(rendered.token_ids) > cfg.max_seq_len or len(rendered.token_ids) < 2:
                filtered_count += 1
                continue

            training_data.append(rendered.datum)

        if filtered_count > 0:
            logger.info(
                "Seq-length filter: %d/%d examples filtered (len > %d or len < 2)",
                filtered_count,
                len(raw_data),
                cfg.max_seq_len,
            )
        logger.info("Prepared %d training examples", len(training_data))
        if not training_data:
            raise RuntimeError("No valid training examples after tokenization")

        # -- Resume ---------------------------------------------------------------

        state = resolve_resume(cfg.log_path, cfg.init_from_checkpoint)
        dcp_load_time = load_dcp(client, state)
        if dcp_load_time > 0:
            wandb_log({"perf/dcp_load_time": dcp_load_time}, state.step)

        fp = dataset_fingerprint(raw_data)
        validate_dataset(state.dataset_fingerprint, fp, state.data_consumed)

        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)

        # -- Training loop (batched) -------------------------------------------

        batch_size = cfg.batch_size
        all_examples = training_data * cfg.epochs
        examples_to_process = all_examples[state.data_consumed:]
        data_consumed = state.data_consumed

        step = state.step
        total_steps = len(all_examples) // (cfg.grad_accum * batch_size)
        accum = 0
        agg_loss_sum = 0.0
        agg_resp_tokens = 0

        def _flush_batch(batch_buf: list[tinker.Datum], step: int, accum: int) -> tuple[int, int]:
            """Send a batch through forward_backward_custom and return (step, accum)."""
            nonlocal agg_loss_sum, agg_resp_tokens

            loss_fn = make_batch_weighted_sft_loss_fn()
            with timer("fwd_bwd"):
                result = client.forward_backward_custom(batch_buf, loss_fn)

            fwd_metrics = result.metrics
            agg_loss_sum += fwd_metrics.get("ce_loss_sum", 0.0)
            agg_resp_tokens += fwd_metrics.get("response_tokens", 0)
            accum += 1

            if accum >= cfg.grad_accum:
                with timer("optim_step"):
                    optim_result = client.optim_step(adam_params)
                step += 1
                accum = 0

                if cfg.dcp_save_interval > 0 and step % cfg.dcp_save_interval == 0:
                    with timer("dcp_save"):
                        logger.info("Saving DCP checkpoint at step %d", step)
                        client.inner.save_state(f"step-{step}")
                    save_loop_state(cfg.log_path, {
                        "step": step,
                        "data_consumed": data_consumed,
                        "dcp_name": f"step-{step}",
                        "dataset_fingerprint": fp,
                        "training_shape_id": getattr(cfg.infra, "training_shape_id", None),
                        "source_job_id": job_id,
                    })

                step_metrics: Dict[str, Any] = flush_timing()

                if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
                    for k, v in optim_result.metrics.items():
                        step_metrics[f"train/{k}"] = v

                if agg_resp_tokens > 0:
                    avg_loss = agg_loss_sum / agg_resp_tokens
                    ppl = torch.exp(torch.tensor(avg_loss)).item()
                    logger.info(
                        "Step %d/%d | Loss: %.4f | PPL: %.2f",
                        step,
                        total_steps,
                        avg_loss,
                        ppl,
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

            return step, accum

        batch_buffer: list[tinker.Datum] = []
        for ex in examples_to_process:
            batch_buffer.append(ex)
            data_consumed += 1
            if len(batch_buffer) >= batch_size:
                step, accum = _flush_batch(batch_buffer, step, accum)
                batch_buffer = []

        if batch_buffer:
            step, accum = _flush_batch(batch_buffer, step, accum)

        if accum > 0:
            client.optim_step(adam_params)
            step += 1

        # -- Final checkpoint --------------------------------------------------

        if step > state.step:
            logger.info("Saving final DCP checkpoint (step %d)...", step)
            client.inner.save_state(f"step-{step}")
            save_loop_state(cfg.log_path, {
                "step": step,
                "data_consumed": data_consumed,
                "dcp_name": f"step-{step}",
                "dataset_fingerprint": fp,
                "training_shape_id": getattr(cfg.infra, "training_shape_id", None),
                "source_job_id": job_id,
            })

            logger.info("Saving final base checkpoint (step %d)...", step)
            result = client.inner.save_weights_for_sampler_ext(
                f"final-step-{step}", checkpoint_type="base"
            )
            logger.info("Final base checkpoint saved: %s", result.path)

        logger.info("Training complete: %d optimizer steps", step)
        return {"steps": step, "job_id": job_id}
    finally:
        wandb_finish()
        try:
            logger.info("Cleanup: deleting trainer job %s", job_id)
            rlor_mgr.delete(job_id)
        except Exception as e:
            logger.warning("Cleanup: failed to delete trainer job %s: %s", job_id, e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        dataset="kimi2_deid_sample_100_formatted.jsonl",
        tokenizer_model="Qwen/Qwen3-8B",
        max_seq_len=4096,
        max_examples=10,
        infra=InfraConfig(
            training_shape_id="your-training-shape",
        ),
    )
    main(cfg)
