# Branched from training/_vendor/tinker_cookbook_fw/distillation/sdft.py
# (Config and main()), which is adapted from thinking-machines-lab/tinker-cookbook
# (Apache-2.0, Copyright 2025 Thinking Machines Lab) via the
# fw-ai-external/tinker-cookbook fork at b1223e5 (tinker_cookbook/distillation/sdft.py).
# Added by Fireworks AI (not in the fork). Config and main() follow sdft.py and
# differ only where Fireworks serverless training requires it:
#   - Connect to the serverless pool instead of a dedicated trainer job, and
#     sample from save_weights_for_sampler snapshots instead of hot-loading a
#     deployment with WeightSyncer.
#   - Teacher: a never-trained LoRA model on its own serverless session (the
#     pool is LoRA-only, so there is no base-only model).
#   - Checkpoints: plain save_state recorded in checkpoints.jsonl, instead of
#     CheckpointManager's trainer-job checkpoint listing.
#   - Tokenizer and renderer from the cookbook (training.renderer), which cover
#     models upstream tinker-cookbook does not (e.g. GLM-5.3-Flash).
# All SDFT helpers are imported from sdft.py. See
# training/_vendor/tinker_cookbook_fw/README.md.
"""
Self-Distillation Fine-Tuning (SDFT) on Fireworks serverless training.

Same algorithm and loop as :mod:`sdft` (see its docstring), run on a shared,
already-running serverless trainer instead of a dedicated trainer job plus an
inference deployment:

- **Student**: a LoRA model on a serverless session. Each step saves its
  weights for sampling and rolls out on-policy from that snapshot.
- **Teacher**: the frozen base model. The serverless pool is LoRA-only, so the
  teacher is a LoRA model that is never trained (LoRA is zero-initialized, so it
  equals the base model). A service holds one training client per base model
  and rank, so the teacher gets its own serverless session.

Entry point: ``training/recipes/sdft_loop.py``.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import chz
import tinker
from fireworks.training.sdk import FiretitanServiceClient, FiretitanTrainingClient
from tinker.types import LossFnType
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import (
    SamplingClientEvaluator,
    SamplingClientEvaluatorBuilder,
)
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import (
    RLTestSetEvaluator,
    compute_trajectory_metrics,
)
from tinker_cookbook.rl.rollouts import do_group_rollout_and_filter_constant_reward
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.git_rev import recipe_user_metadata

from training._vendor.tinker_cookbook_fw import checkpoint_utils
from training._vendor.tinker_cookbook_fw.distillation.sdft import (
    DEFAULT_DEMO_TEMPLATE,
    SDFTBatchProvider,
    _SDFTEvalDatasetAdapter,
    _train_step_reverse_kl,
    build_reverse_kl_datums,
    build_sdft_teacher_prompt,
    build_topk_distillation_datums,
    compute_sdft_advantages,
)
from training._vendor.tinker_cookbook_fw.rl.train import train_step
from training.renderer import get_renderer
from training.utils.supervised import resolve_renderer_name
from training.utils.tokenizers import load_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.fireworks.ai"


@chz.chz
class Config:
    """Configuration for SDFT on Fireworks serverless training.

    Same as :class:`sdft.Config` minus the dedicated-trainer / deployment
    fields, plus ``lora_alpha``. ``model_name`` is the Hugging Face tokenizer
    id; ``fireworks_base_model`` is the model trained on the serverless pool.
    """

    # Model
    model_name: str
    recipe_name: str
    fireworks_base_model: str
    renderer_name: str | None = None
    lora_rank: int = 128
    lora_alpha: int = 32
    base_url: str | None = None  # None = FIREWORKS_BASE_URL or the public API

    # Training
    learning_rate: float = 2e-5
    max_tokens: int = 2048
    temperature: float = 1.0
    loss_fn: LossFnType = "cross_entropy"

    # SDFT-specific
    topk: int = 20
    reverse: bool = False
    demo_template: str = DEFAULT_DEMO_TEMPLATE
    system_prompt: str | None = None
    teacher_sync_every: int | None = None
    max_context_length: int = 32768

    # Evaluation
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    eval_every: int = 20
    save_every: int = 20

    # Standard infra
    num_substeps: int = 1
    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    wandb_project: str | None = None
    wandb_name: str | None = None
    load_checkpoint_path: str | None = None
    max_steps: int | None = None

    enable_trace: bool = False
    span_chart_every: int = 0


def _serverless_base_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/training/v1/serverless"):
        return root
    if root.endswith("/training/v1"):
        return f"{root}/serverless"
    return f"{root}/training/v1/serverless"


async def _get_sampling_client(
    service_client: FiretitanServiceClient,
    training_client: FiretitanTrainingClient,
    name: str,
    tokenizer: Any,
) -> tinker.SamplingClient:
    """Snapshot the current student weights and bind a sampler to them."""
    save_future = await training_client.save_weights_for_sampler_async(name)
    path = (await save_future.result_async()).path
    if not path:
        raise RuntimeError(f"save_weights_for_sampler({name!r}) returned no path")
    return await asyncio.to_thread(
        service_client.create_sampling_client, model_path=path, tokenizer=tokenizer
    )


async def _save_state(
    training_client: FiretitanTrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    store: Any,
) -> str:
    """``save_state`` and append a ``checkpoints.jsonl`` record for resume."""
    save_future = await training_client.save_state_async(f"{name}-state")
    state_path = (await save_future.result_async()).path
    record = checkpoint_utils.CheckpointRecord.from_dict(
        {"name": name, **loop_state, "state_path": state_path}
    )
    if store is not None:
        store.write_checkpoint(record.to_dict())
    else:
        with open(Path(log_path) / "checkpoints.jsonl", "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
    logger.info(f"Saved checkpoint {name}: {state_path}")
    return state_path


@trace.scope
async def main(
    cfg: Config,
    sdft_dataset: SDFTBatchProvider,
    test_dataset: SDFTBatchProvider | None = None,
) -> None:
    """Main training loop for SDFT on Fireworks serverless training.

    Same as :func:`sdft.main`, on a serverless session.

    Args:
        cfg: Training configuration. See :class:`Config`.
        sdft_dataset: Dataset providing (builders, questions, golden_answers)
            batches. Use :class:`~tinker_cookbook.recipes.sdft.datasets.SDFTDataset`
            built with the cookbook renderer.
        test_dataset: Optional test dataset for periodic evaluation.
    """
    if cfg.reverse and cfg.topk == 0:
        raise ValueError(
            "reverse=True requires topk>0: the analytical reverse KL runs over "
            "the teacher's top-K set, which doesn't exist when topk=0."
        )
    if cfg.topk < 0:
        raise ConfigurationError(f"topk must be non-negative, got {cfg.topk}")
    if cfg.max_context_length < 2:
        raise ConfigurationError(
            f"max_context_length must be at least 2, got {cfg.max_context_length}"
        )
    if cfg.lora_rank <= 0:
        raise ConfigurationError(
            f"lora_rank must be positive (the serverless pool is LoRA-only), got {cfg.lora_rank}"
        )
    if cfg.teacher_sync_every is not None:
        raise ConfigurationError(
            "teacher_sync_every is not supported by the Firetitan SDFT backend; "
            "use a static teacher"
        )
    fireworks_api_key = os.environ.get("FIREWORKS_API_KEY")
    if not fireworks_api_key:
        raise ConfigurationError("FIREWORKS_API_KEY must be set")
    serverless_url = _serverless_base_url(
        cfg.base_url or os.environ.get("FIREWORKS_BASE_URL") or DEFAULT_BASE_URL
    )

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    store = ml_logger.store
    if cfg.enable_trace:
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = str(Path(cfg.log_path) / "trace_events.jsonl")
        logger.info(f"Tracing enabled. Events saved to {trace_events_path}")
        trace.trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    # Resume handling
    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    start_batch = (resume_info.batch or 0) if resume_info else 0

    # Service and training client setup
    service_client = FiretitanServiceClient(
        api_key=fireworks_api_key,
        base_url=serverless_url,
        user_metadata=recipe_user_metadata(cfg.recipe_name),
    )
    teacher_service_client: FiretitanServiceClient | None = None
    try:
        user_metadata: dict[str, str] = {}
        if wandb_link := ml_logger.get_logger_url():
            user_metadata["wandb_link"] = wandb_link

        if resume_info:
            assert resume_info.state_path is not None
            training_client = (
                await service_client.create_training_client_from_state_with_optimizer_async(
                    resume_info.state_path
                )
            )
            logger.info(f"Resumed training from {resume_info.state_path}")
        elif cfg.load_checkpoint_path:
            training_client = await service_client.create_training_client_from_state_async(
                cfg.load_checkpoint_path
            )
            logger.info(f"Loaded weights from {cfg.load_checkpoint_path}")
        else:
            training_client = await service_client.create_lora_training_client_async(
                base_model=cfg.fireworks_base_model,
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
                user_metadata=user_metadata,
            )

        # Fireworks model IDs are not Hugging Face tokenizer IDs.
        tokenizer = load_tokenizer(cfg.model_name)
        renderer_name = resolve_renderer_name(cfg.model_name, cfg.renderer_name or "")
        renderer = get_renderer(renderer_name, tokenizer)

        num_batches = len(sdft_dataset)
        if cfg.max_steps is not None:
            num_batches = min(cfg.max_steps, num_batches)
        logger.info(f"Will train on {num_batches} batches")

        # Evaluators
        evaluators: list[SamplingClientEvaluator] = [e() for e in cfg.evaluator_builders]
        if test_dataset is not None:
            evaluators.append(
                RLTestSetEvaluator(
                    _SDFTEvalDatasetAdapter(test_dataset),
                    max_tokens=cfg.max_tokens,
                )
            )

        # Static teacher: a never-trained LoRA model (= the frozen base model)
        # on its own serverless session. See the module docstring.
        teacher_service_client = FiretitanServiceClient(
            api_key=fireworks_api_key, base_url=serverless_url
        )
        teacher_client = await teacher_service_client.create_lora_training_client_async(
            base_model=cfg.fireworks_base_model,
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
        )
        logger.info(f"Created static serverless teacher client for {cfg.fireworks_base_model}")

        sampling_client = await _get_sampling_client(
            service_client, training_client, f"sampler-{start_batch:06d}", tokenizer
        )

        log_path = Path(cfg.log_path)

        for i_batch in range(start_batch, num_batches):
            metrics: dict[str, Any] = {
                "progress/batch": i_batch,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
            }

            with trace.trace_iteration(step=i_batch) as window:
                # Evaluation
                if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
                    async with trace.scope_span("run_evals"):
                        for evaluator in evaluators:
                            eval_metrics = await evaluator(sampling_client)
                            metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

                # Get batch: builders + questions + golden answers
                builders_P, questions_P, golden_answers_P = sdft_dataset.get_batch(i_batch)

                # Rollout: student generates completions on-policy.
                async with trace.scope_span("sample"):
                    trajectory_groups_raw = await asyncio.gather(
                        *[
                            asyncio.create_task(
                                do_group_rollout_and_filter_constant_reward(
                                    sampling_client,
                                    builder,
                                    temperature=cfg.temperature,
                                    max_tokens=cfg.max_tokens,
                                    do_remove_constant_reward_groups=False,
                                ),
                                name=f"sample_task_{i}",
                            )
                            for i, builder in enumerate(builders_P)
                        ],
                    )
                successful_rollouts = [
                    (builder, question, golden_answer, trajectory_group)
                    for builder, question, golden_answer, trajectory_group in zip(
                        builders_P,
                        questions_P,
                        golden_answers_P,
                        trajectory_groups_raw,
                        strict=True,
                    )
                    if trajectory_group is not None
                ]
                if not successful_rollouts:
                    logger.warning("Skipping batch %d because every rollout failed", i_batch)
                    continue

                builders_P = [item[0] for item in successful_rollouts]
                questions_P = [item[1] for item in successful_rollouts]
                golden_answers_P = [item[2] for item in successful_rollouts]
                trajectory_groups_P: list[TrajectoryGroup] = [
                    item[3] for item in successful_rollouts
                ]

                # Compute trajectory metrics
                taglist_P = [b.logging_tags() for b in builders_P]
                metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

                # Assemble training data (advantages start as 0 since rewards are all 0)
                async with trace.scope_span("assemble_training_data"):
                    advantages_P = compute_advantages(trajectory_groups_P)
                    data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

                # Log one example
                if data_D:
                    logger.info(colorize_example(data_D[0], tokenizer, key="mask"))

                # Build teacher prompts (one per problem)
                teacher_prompts_P = [
                    build_sdft_teacher_prompt(
                        question=question,
                        golden_answer=golden_answer,
                        renderer=renderer,
                        system_prompt=cfg.system_prompt,
                        demo_template=cfg.demo_template,
                    )
                    for question, golden_answer in zip(questions_P, golden_answers_P)
                ]

                if cfg.topk > 0 and cfg.reverse:
                    # Analytical reverse KL over teacher top-K via custom loss
                    async with trace.scope_span("build_reverse_kl_datums"):
                        rev_datums, rev_metrics = await build_reverse_kl_datums(
                            data_D,
                            metadata_D,
                            teacher_client,
                            teacher_prompts_P,
                            topk=cfg.topk,
                            max_context_length=cfg.max_context_length,
                            vocab_size=len(tokenizer),
                        )
                    metrics.update(rev_metrics)

                    async with trace.scope_span("train"):
                        await _train_step_reverse_kl(
                            data_D=rev_datums,
                            training_client=training_client,
                            learning_rate=cfg.learning_rate,
                            num_substeps=cfg.num_substeps,
                            metrics=metrics,
                        )
                elif cfg.topk > 0:
                    # Top-K CE distillation (forward KL)
                    async with trace.scope_span("build_topk_distillation_datums"):
                        topk_datums, topk_metrics = await build_topk_distillation_datums(
                            data_D,
                            metadata_D,
                            teacher_client,
                            teacher_prompts_P,
                            topk=cfg.topk,
                            max_context_length=cfg.max_context_length,
                            vocab_size=len(tokenizer),
                        )
                    metrics.update(topk_metrics)

                    async with trace.scope_span("train"):
                        # Top-K CE only (no IS, no student renormalization)
                        await train_step(
                            data_D=topk_datums,
                            training_client=training_client,
                            learning_rate=cfg.learning_rate,
                            num_substeps=cfg.num_substeps,
                            loss_fn="cross_entropy",
                            metrics=metrics,
                        )
                else:
                    # DEPRECATED: single-sample importance-sampling approximation of
                    # reverse KL. Superseded by `reverse=True`.
                    async with trace.scope_span("compute_sdft_advantages"):
                        is_metrics = await compute_sdft_advantages(
                            data_D,
                            metadata_D,
                            teacher_client,
                            teacher_prompts_P,
                            max_context_length=cfg.max_context_length,
                        )
                    metrics.update(is_metrics)

                    async with trace.scope_span("train"):
                        await train_step(
                            data_D=data_D,
                            training_client=training_client,
                            learning_rate=cfg.learning_rate,
                            num_substeps=cfg.num_substeps,
                            loss_fn=cfg.loss_fn,
                            metrics=metrics,
                        )

                # Save a periodic checkpoint and refresh the sampling client
                done = i_batch + 1
                if cfg.save_every > 0 and done % cfg.save_every == 0 and done < num_batches:
                    async with trace.scope_span("save_checkpoint"):
                        await _save_state(
                            training_client,
                            f"{done:06d}",
                            cfg.log_path,
                            {"batch": done},
                            store,
                        )
                if done < num_batches:
                    async with trace.scope_span("refresh_sampling_client"):
                        sampling_client.close()
                        sampling_client = await _get_sampling_client(
                            service_client, training_client, f"sampler-{done:06d}", tokenizer
                        )

            # Log timing
            metrics.update(window.get_timing_metrics())
            window.save_timing(i_batch, store=store)
            if cfg.span_chart_every > 0 and i_batch % cfg.span_chart_every == 0:
                trace.save_gantt_chart_html(
                    window, i_batch, log_path / f"timing_gantt_{i_batch:06d}.html"
                )
            ml_logger.log_metrics(metrics, step=i_batch)

        sampling_client.close()

        # Final checkpoint
        if start_batch < num_batches:
            await _save_state(
                training_client,
                "final",
                cfg.log_path,
                {"batch": num_batches, "final": True},
                store,
            )

        logger.info("SDFT training completed successfully")
    finally:
        ml_logger.close()
        if teacher_service_client is not None:
            teacher_service_client.close()
        service_client.close()
