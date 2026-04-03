#!/usr/bin/env python3
"""GRPO training loop with concurrent rollout.

A readable, modifiable RL training loop using the Fireworks RLOR API.
Fork this script and customise the reward function, loss, or sampling
strategy to fit your task.

Each optimizer step samples ``prompt_groups_per_step`` prompts concurrently,
then runs a single training update + ``optim_step`` (1:1 ratio).

RL losses can execute in two places:
- Server-side builtin path: ``forward_backward(...)`` with a builtin kernel
  resolved by :func:`training.utils.rl.losses.resolve_builtin_loss`.
- Client-side custom path: ``forward_backward_custom(...)`` with a Python
  loss closure built by :func:`training.utils.rl.losses.build_loss_fn`.

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.rl_loop
"""

from __future__ import annotations

import os
import re
import json
import signal
import asyncio
import logging
from contextlib import ExitStack
from typing import List, Optional
from dataclasses import field, dataclass
from concurrent.futures import ThreadPoolExecutor

import tinker

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.client import GradAccNormalization
from training.utils import (
    DEFAULT_ADAM,
    ConcurrencyConfig,
    InfraConfig,
    ResourceCleanup,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    DeployConfig,
    WeightSyncConfig,
    ReconnectableClient,
    RLPromptDataset,
    wandb_log,
    setup_wandb,
    wandb_finish,
    validate_config,
    log_metrics_json,
    get_deployment_gpu_count,
    setup_deployment,
    compute_advantages,
    create_trainer_job,
    load_jsonl_dataset,
    prepare_sampling_messages,
    ShapeSelectionRequest,
    materialize_profile_infra,
    select_validated_launch_shapes,
)
from training.utils.checkpoint_utils import (
    resolve_resume,
    save_checkpoint,
    CheckpointKind,
)
from fireworks.training.sdk.deployment import DeploymentSampler

from fireworks.training.sdk.deployment import AdaptiveConcurrencyController, FixedConcurrencyController
from training.utils.rl import PromptGroup
from training.utils.rl.tis import TISConfig
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils.timer import timer, flush_timing
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.train import TrainStepFns, run_rl_loop
from training.utils.rl.losses import (
    build_builtin_loss_datums,
    build_loss_fn,
    combine_prompt_groups,
    resolve_builtin_loss,
)
from training.utils.rl.metrics import compute_step_metrics
from training.utils.rl.router_replay import build_r3_routing_matrices

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"

    learning_rate: float = 1e-5
    kl_beta: float = 0.001
    completions_per_prompt: int = 4
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    max_rows: int = 100
    max_seq_len: int | None = None
    """Max sequence length for sampling and training.  When using training
    shapes, this is auto-populated from the shape's
    ``max_supported_context_length``.  Must be set manually on the
    manual path (no training shape)."""
    lora_rank: int = 0

    prompt_groups_per_step: int = 1
    """Number of prompt groups per optimizer step.

    All groups are collected before a single ``forward_backward_custom`` +
    ``optim_step`` pair fires (1:1 ratio)."""

    router_replay: bool = False
    router_replay_completion_only: bool = True

    grad_accumulation_normalization: GradAccNormalization | str | None = GradAccNormalization.NUM_LOSS_TOKENS
    """Normalization mode for accumulated gradients at optim_step.
    Defaults to ``GradAccNormalization.NUM_LOSS_TOKENS`` (per-token mean)."""

    policy_loss: str = "grpo"
    """``"grpo"``, ``"importance_sampling"``, ``"dapo"``, ``"dro"``, ``"gspo"``, ``"reinforce"``, or ``"cispo"``.

    If an eligible builtin kernel exists for the selected loss, training uses
    the server-side ``forward_backward(...)`` path. Otherwise it falls back to
    the client-side ``forward_backward_custom(...)`` path.
    """

    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    eps_clip: float = 0.2
    """PPO clip epsilon for the off-policy ratio (GRPO only)."""
    eps_clip_high: float | None = None
    """Asymmetric upper clip bound (GRPO only)."""
    ratio_log_cap: float = 20.0
    """Log-ratio clamp for ``policy_loss="importance_sampling"``."""
    tis: TISConfig = field(default_factory=TISConfig)
    """TIS (Train-Inference IS) weight correction config."""

    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    """Concurrency control for inference sampling.  ``"fixed"`` (default)
    uses a static semaphore; ``"adaptive"`` adjusts the window based on
    server-side prefill queue latency.  Adaptive mode requires ``stream=True``."""

    trajectory_dir: str | None = None
    """Directory to save per-step trajectory JSONL files.  Each file contains
    prompts, completions, and rewards for every prompt group in that step."""

    policy_job_id: str | None = None
    """Pre-created RLOR policy trainer job ID (skip creation if set)."""

    policy_base_url: str | None = None
    """Base URL for the policy trainer (bypass direct route)."""

    reference_job_id: str | None = None
    """Pre-created RLOR reference trainer job ID (skip creation if set)."""

    reference_base_url: str | None = None
    """Base URL for the reference trainer (bypass direct route)."""

    init_from_checkpoint: str | None = None
    """Load pretrained DCP weights on a fresh dataset. Supports cross-job
    format ``"job_id:checkpoint_name"``."""

    output_model_id: str | None = None
    save_final_checkpoint: bool = True

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="grpo-tinker"))
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
# Reward function -- customise this for your task
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    digits = re.search(r"(-?\d+)", match.group(1))
    return digits.group(1) if digits else None


def reward_fn(completion: str, row: dict) -> float:
    """Return 1.0 if the model's numeric answer matches the ground truth."""
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


# ---------------------------------------------------------------------------
# Rollout filter -- customise this for your task
# ---------------------------------------------------------------------------


def should_accept(pg: PromptGroup) -> bool:
    """Reject groups where all rewards are identical (zero-variance).

    Passed to ``run_rl_loop`` as a pluggable filter.  Replace with your
    own logic (e.g. minimum reward threshold, response length filter).
    """
    return len(set(pg.rewards)) > 1


# ---------------------------------------------------------------------------
# Trajectory logging
# ---------------------------------------------------------------------------


def _dump_trajectory(trajectory_dir: str, step: int, prompt_groups: list[PromptGroup]) -> None:
    """Write per-step trajectory JSONL: one line per individual completion."""
    os.makedirs(trajectory_dir, exist_ok=True)
    path = os.path.join(trajectory_dir, f"step_{step:04d}.jsonl")
    n_records = 0
    with open(path, "w") as f:
        for pg_idx, pg in enumerate(prompt_groups):
            completions = pg.completions or []
            for comp_idx, comp_text in enumerate(completions):
                record = {
                    "step": step,
                    "prompt_group": pg_idx,
                    "completion_index": comp_idx,
                    "prompt": pg.prompt,
                    "completion": comp_text,
                    "reward": pg.rewards[comp_idx] if comp_idx < len(pg.rewards) else None,
                    "advantage": pg.advantages[comp_idx] if comp_idx < len(pg.advantages) else None,
                    "completion_len": pg.completion_lens[comp_idx] if comp_idx < len(pg.completion_lens) else None,
                    "truncated": pg.truncated[comp_idx] if comp_idx < len(pg.truncated) else None,
                    "ground_truth": pg.row_meta.get("ground_truth") if pg.row_meta else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_records += 1
    logger.info(
        "[step %d] Saved trajectory to %s (%d completions from %d groups)", step, path, n_records, len(prompt_groups)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
    deploy_mgr: DeploymentManager | None = None,
    cleanup_on_exit: bool = False,
):
    cfg = config
    runner = RunnerIO(cfg.runner)

    # Convert SIGTERM/SIGINT into exceptions so the finally block runs cleanup.
    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(
        cfg.base_model,
        cfg.dataset,
        cfg.weight_sync,
        cfg.deployment,
        output_model_id=cfg.output_model_id,
    )
    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step
    if not cfg.deployment.tokenizer_model:
        raise ValueError(
            "deployment.tokenizer_model is required for client-side tokenization. "
            "Set it to the HuggingFace model name (e.g. 'Qwen/Qwen3-1.7B')."
        )
    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": completions_per_prompt,
            "prompt_groups_per_step": prompt_groups_per_step,
            "kl_beta": cfg.kl_beta,
            "lr": cfg.learning_rate,
        },
    )

    # -- Setup infrastructure -----------------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)

    policy_selection = select_validated_launch_shapes(
        rlor_mgr,
        deploy_mgr=deploy_mgr,
        request=ShapeSelectionRequest(
            base_model=cfg.base_model,
            max_seq_len=cfg.max_seq_len,
            trainer_role="policy",
            needs_deployment=True,
            lora_rank=cfg.lora_rank,
            explicit_training_shape_id=cfg.infra.training_shape_id,
            explicit_deployment_shape=cfg.deployment.deployment_shape,
        ),
    )
    if policy_selection.training_shape_id:
        cfg.infra.training_shape_id = policy_selection.training_shape_id
    if policy_selection.deployment_shape:
        cfg.deployment.deployment_shape = policy_selection.deployment_shape
    if policy_selection.inferred_training_shape:
        logger.info(
            "Using validated policy training shape for %s: %s",
            cfg.base_model,
            policy_selection.training_shape_id,
        )
    if policy_selection.inferred_deployment_shape:
        logger.info(
            "Using validated deployment shape for %s: %s",
            cfg.base_model,
            policy_selection.deployment_shape,
        )

    # -- Resolve training shapes -----------------------------------------------

    profile = policy_selection.training_profile
    policy_infra = materialize_profile_infra(cfg.infra, profile) if profile else cfg.infra
    policy_profile = profile

    if profile and cfg.max_seq_len is None:
        cfg.max_seq_len = profile.max_supported_context_length
        logger.info("max_seq_len from training shape: %d", cfg.max_seq_len)
    if cfg.max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )

    reference_needed = cfg.kl_beta > 0 or cfg.infra.ref_training_shape_id is not None
    ref_profile = None
    reference_infra = cfg.infra
    reference_launch_profile = None
    if reference_needed:
        reference_selection = select_validated_launch_shapes(
            rlor_mgr,
            request=ShapeSelectionRequest(
                base_model=cfg.base_model,
                max_seq_len=cfg.max_seq_len,
                trainer_role="reference",
                needs_deployment=False,
                lora_rank=cfg.lora_rank,
                explicit_training_shape_id=cfg.infra.ref_training_shape_id,
            ),
        )
        if reference_selection.training_shape_id:
            cfg.infra.ref_training_shape_id = reference_selection.training_shape_id
        if reference_selection.inferred_training_shape:
            logger.info(
                "Using validated reference training shape for %s: %s",
                cfg.base_model,
                reference_selection.training_shape_id,
            )
        ref_profile = reference_selection.training_profile
        reference_infra = (
            materialize_profile_infra(cfg.infra, ref_profile) if ref_profile else cfg.infra
        )
        reference_launch_profile = ref_profile

    use_reference = ref_profile is not None
    if not use_reference:
        logger.info("No ref_training_shape_id set, skipping reference model")

    import time as _time

    runner.set_accelerator_info(
        policy_infra.accelerator_type,
        policy_infra.accelerator_count,
        profile=profile,
    )
    runner.write_status(RunStatus.RUNNING, message="provisioning")

    def _on_trainer_status(msg: str) -> None:
        runner.write_status(RunStatus.RUNNING, message=msg)

    _infra_start = _time.time()

    with runner, ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup, ExitStack() as stack:
        _on_trainer_status("provisioning deployment")
        dep_info = setup_deployment(deploy_mgr, cfg.deployment, cfg.base_model, policy_infra)
        if cleanup_on_exit:
            cleanup.deployment(cfg.deployment.deployment_id, action="scale_to_zero")

        logger.info(
            "Training: prompt_groups_per_step=%d | completions_per_prompt=%d",
            prompt_groups_per_step,
            completions_per_prompt,
        )

        if use_reference:
            _on_trainer_status("provisioning policy and reference trainers")
            with ThreadPoolExecutor(max_workers=2) as pool:
                pol_fut = pool.submit(
                    create_trainer_job,
                    rlor_mgr,
                    base_model=cfg.base_model,
                    infra=policy_infra,
                    profile=policy_profile,
                    lora_rank=cfg.lora_rank,
                    max_seq_len=cfg.max_seq_len,
                    learning_rate=cfg.learning_rate,
                    display_name="grpo-policy",
                    hot_load_deployment_id=cfg.deployment.deployment_id,
                    job_id=cfg.policy_job_id,
                    base_url_override=cfg.policy_base_url,
                    cleanup=cleanup if not cfg.policy_job_id else None,
                    on_status=_on_trainer_status,
                )
                ref_fut = pool.submit(
                    create_trainer_job,
                    rlor_mgr,
                    base_model=cfg.base_model,
                    infra=reference_infra,
                    profile=reference_launch_profile,
                    lora_rank=cfg.lora_rank,
                    max_seq_len=cfg.max_seq_len,
                    learning_rate=cfg.learning_rate,
                    display_name="grpo-reference",
                    forward_only=True,
                    job_id=cfg.reference_job_id,
                    base_url_override=cfg.reference_base_url,
                    cleanup=cleanup if not cfg.reference_job_id else None,
                    on_status=_on_trainer_status,
                )
                # Collect both results so that if both fail we report
                # both errors instead of swallowing the second one.
                errors: list[str] = []
                policy_ep = reference_ep = None
                try:
                    policy_ep = pol_fut.result()
                except Exception as e:
                    errors.append(f"Policy trainer: {e}")
                try:
                    reference_ep = ref_fut.result()
                except Exception as e:
                    errors.append(f"Reference trainer: {e}")
                if errors:
                    raise RuntimeError(
                        "Trainer creation failed:\n" + "\n".join(errors)
                    )
                policy_job_id = policy_ep.job_id
                reference_job_id = reference_ep.job_id
        else:
            policy_ep = create_trainer_job(
                rlor_mgr,
                base_model=cfg.base_model,
                infra=policy_infra,
                profile=policy_profile,
                lora_rank=cfg.lora_rank,
                max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name="grpo-policy",
                hot_load_deployment_id=cfg.deployment.deployment_id,
                job_id=cfg.policy_job_id,
                base_url_override=cfg.policy_base_url,
                cleanup=cleanup if not cfg.policy_job_id else None,
                on_status=_on_trainer_status,
            )
            policy_job_id = policy_ep.job_id
            reference_ep = None
            reference_job_id = None

        policy = ReconnectableClient(
            rlor_mgr,
            policy_ep.job_id,
            cfg.base_model,
            cfg.lora_rank,
            fw_api_key=api_key,
            endpoint=policy_ep if cfg.policy_base_url else None,
        )
        if hasattr(policy, "close"):
            stack.callback(policy.close)
        reference = (
            ReconnectableClient(
                rlor_mgr,
                reference_ep.job_id,
                cfg.base_model,
                cfg.lora_rank,
                fw_api_key=api_key,
                endpoint=reference_ep if cfg.reference_base_url else None,
            )
            if reference_ep
            else None
        )
        if reference is not None and hasattr(reference, "close"):
            stack.callback(reference.close)

        import transformers

        inference_model = dep_info.inference_model if dep_info else cfg.base_model
        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.deployment.tokenizer_model, trust_remote_code=True)

        # -- Concurrency controller ------------------------------------------------
        if cfg.concurrency.mode == "adaptive":
            gpu_count = get_deployment_gpu_count(deploy_mgr, cfg.deployment)
            _SLOTS_PER_GPU = 8  # Default concurrent requests per GPU.
            initial_window = cfg.concurrency.initial_window or (_SLOTS_PER_GPU * gpu_count)
            concurrency_controller = AdaptiveConcurrencyController(
                initial_window=initial_window,
                min_window=cfg.concurrency.min_window,
                max_window=cfg.concurrency.max_window,
                prefill_queue_target=cfg.concurrency.prefill_queue_target,
            )
            logger.info(
                "Using adaptive concurrency (initial=%d, range=%d-%d, target_pq=%.2fs)",
                initial_window,
                cfg.concurrency.min_window,
                cfg.concurrency.max_window,
                cfg.concurrency.prefill_queue_target,
            )
        elif cfg.concurrency.mode == "fixed":
            concurrency_controller = None
            logger.info("Using fixed concurrency: unlimited")
        elif cfg.concurrency.mode is None and cfg.concurrency.max_concurrency is not None:
            import warnings
            warnings.warn(
                "ConcurrencyConfig.max_concurrency is deprecated. "
                "Use mode='adaptive' (default) or mode='fixed' with "
                "FixedConcurrencyController instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            concurrency_controller = FixedConcurrencyController(cfg.concurrency.max_concurrency)
            logger.info("Using fixed concurrency (deprecated max_concurrency=%d)", cfg.concurrency.max_concurrency)
        else:
            raise ValueError(
                f"Unknown concurrency mode: {cfg.concurrency.mode!r}. Must be 'adaptive' or 'fixed'."
            )

        sampler = DeploymentSampler(
            inference_url=deploy_mgr.inference_url,
            model=inference_model,
            api_key=api_key,
            tokenizer=tokenizer,
            concurrency_controller=concurrency_controller,
        )
        weight_syncer = WeightSyncer(
            policy_client=policy.inner,
            deploy_mgr=deploy_mgr,
            deployment_id=cfg.deployment.deployment_id,
            base_model=cfg.base_model,
            hotload_timeout=cfg.weight_sync.weight_sync_timeout,
            first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
            lora_rank=cfg.lora_rank,
        )

        infra_boot_time = _time.time() - _infra_start
        boot_metrics: dict = {
            "train/step": 0,
            "infra/total_boot_time": infra_boot_time,
        }
        if deploy_mgr.boot_time_s is not None:
            boot_metrics["infra/deploy_boot_time"] = deploy_mgr.boot_time_s
        wandb_log(boot_metrics, step=0)

        # -- Resume ---------------------------------------------------------------

        resume_info = resolve_resume(policy, cfg.log_path, cfg.init_from_checkpoint)
        step_offset = resume_info.step if resume_info else 0
        wandb_log({"train/step": step_offset}, step_offset)

        if cfg.weight_sync.weight_sync_before_training and cfg.deployment.deployment_id:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            weight_syncer.save_and_hotload(name, checkpoint_type="base")

        # -- Prepare sampling and training --------------------------------------

        raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        all_rows = raw_dataset * cfg.epochs
        rl_dataset = RLPromptDataset(all_rows, prompts_per_step=prompt_groups_per_step)
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)
        # Client-side fallback: build the Python loss closure used by
        # forward_backward_custom(...) when no eligible builtin kernel exists.
        client_loss_builder = build_loss_fn(
            policy_loss=cfg.policy_loss,
            kl_beta=cfg.kl_beta,
            dapo_config=cfg.dapo,
            gspo_config=cfg.gspo,
            cispo_config=cfg.cispo,
            ratio_log_cap=cfg.ratio_log_cap,
            tis_config=cfg.tis,
            eps_clip=cfg.eps_clip,
            eps_clip_high=cfg.eps_clip_high,
        )

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=cfg.max_seq_len,
            http_timeout=cfg.deployment.sample_timeout,
        )
        if cfg.router_replay:
            sample_kwargs.update(include_routing_matrix=True, echo=True, logprobs=True)
        sample_kwargs["logprobs"] = True

        # -- Sample one prompt (VISIBLE -- customise this) ----------------------

        async def sample_one_prompt(row: dict) -> PromptGroup | None:
            """Sample completions for one prompt and return a PromptGroup."""
            messages = row.get("messages", [])
            input_messages = prepare_sampling_messages(messages)
            if not input_messages:
                return None

            try:
                sampled = await sampler.sample_with_tokens(
                    messages=input_messages,
                    n=completions_per_prompt,
                    **sample_kwargs,
                )
            except Exception as e:
                # TODO: HTTP 425 (deployment hot-loading after weight sync)
                # can cause transient failures here.  Currently the prompt is
                # silently dropped (counted as sample_fails).  Consider adding
                # a retry loop so no training data is lost during hotload.
                logger.warning("Sampling failed: %s", e)
                return None

            if not sampled or len(sampled) < completions_per_prompt:
                return None

            rewards = [reward_fn(s.text, row) for s in sampled]
            advantages = compute_advantages(rewards)

            prompt_len = sampled[0].prompt_len
            policy_data: List[tinker.Datum] = []
            reference_data: List[tinker.Datum] = []
            adv_filtered: List[float] = []
            inf_logprobs_aligned: List[List[float]] = []

            for idx, s in enumerate(sampled):
                tokens = s.full_tokens
                if len(tokens) < 2:
                    continue
                model_input_len = len(tokens) - 1

                rm = None
                if cfg.router_replay:
                    rm = build_r3_routing_matrices(
                        s.routing_matrices,
                        s.prompt_len,
                        model_input_len,
                        completion_only=cfg.router_replay_completion_only,
                    )

                policy_datum = tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(tokens[:-1], routing_matrices=rm),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData(data=tokens[1:], dtype="int64", shape=[model_input_len]),
                    },
                )
                policy_data.append(policy_datum)

                if use_reference:
                    reference_datum = tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData(
                                data=tokens[1:], dtype="int64", shape=[model_input_len]
                            ),
                        },
                    )
                    reference_data.append(reference_datum)

                adv_filtered.append(advantages[idx])

                if not s.inference_logprobs:
                    raise RuntimeError(
                        f"Inference logprobs required but sample {idx} has none. "
                        f"Ensure the deployment returns logprobs."
                    )
                response_start = max(0, prompt_len - 1)
                echoed = getattr(s, "logprobs_echoed", False)
                aligned = list(s.inference_logprobs) if echoed else [0.0] * response_start + list(s.inference_logprobs)
                inf_logprobs_aligned.append(aligned)

            if not policy_data:
                return None

            comp_lens = [len(s.full_tokens) - s.prompt_len for s in sampled]
            trunc = [s.finish_reason == "length" for s in sampled]

            return PromptGroup(
                data=policy_data,
                ref_data=reference_data,
                advantages=adv_filtered,
                ref_logprobs=None,
                prompt_len=prompt_len,
                rewards=rewards,
                inf_logprobs=inf_logprobs_aligned,
                completion_lens=comp_lens,
                truncated=trunc,
                prompt=input_messages if cfg.trajectory_dir else None,
                completions=[s.text for s in sampled] if cfg.trajectory_dir else None,
                row_meta={"ground_truth": row.get("ground_truth", "")} if cfg.trajectory_dir else None,
            )

        # -- Training callbacks ----------------------------------------------------

        def ref_forward(groups: list[PromptGroup]) -> None:
            """Compute reference logprobs for all prompt groups (one call)."""
            if not use_reference:
                return
            all_ref_data = [d for pg in groups for d in pg.ref_data]
            try:
                ref_fwd = reference.forward(all_ref_data, "cross_entropy")
            except Exception as e:
                raise RuntimeError(
                    f"Reference forward failed (batch of {len(all_ref_data)} datums): {e}\n"
                    "Possible causes: reference trainer crashed, NCCL timeout, or "
                    "request exceeded the default timeout. Check reference trainer "
                    "logs and consider increasing the client timeout."
                ) from e
            idx = 0
            for pg in groups:
                n = len(pg.ref_data)
                pg.ref_logprobs = [ref_fwd.loss_fn_outputs[idx + i]["logprobs"].data for i in range(n)]
                idx += n

        # Server-side fast path: resolve the builtin kernel/config used by
        # forward_backward(...). Returns None when this loss has no builtin
        # implementation, and raises when the current profile is ineligible.
        builtin_server_loss = resolve_builtin_loss(
            cfg.policy_loss,
            profile,
            dapo_config=cfg.dapo,
            gspo_config=cfg.gspo,
            cispo_config=cfg.cispo,
            ratio_log_cap=cfg.ratio_log_cap,
            eps_clip=cfg.eps_clip,
            eps_clip_high=cfg.eps_clip_high,
        )

        def fwd_bwd_one(prompt_groups: list[PromptGroup]):
            """One minibatch update using the builtin or client-side loss path."""
            if not prompt_groups:
                raise ValueError("fwd_bwd_one requires at least one prompt group")

            data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(prompt_groups)

            t0 = _time.time()
            prox_fwd = policy.forward(data, "cross_entropy")
            prox_lp = [prox_fwd.loss_fn_outputs[i]["logprobs"].data for i in range(len(data))]
            logger.info("policy_forward: done (%.1fs)", _time.time() - t0)

            t0 = _time.time()
            if builtin_server_loss is not None:
                # Server-side builtin path: pre-pack the rollout tensors into
                # datums the trainer kernel understands, then call
                # forward_backward(...).
                kernel_loss, kernel_config = builtin_server_loss
                rl_datums = build_builtin_loss_datums(
                    data,
                    adv,
                    prox_lp,
                    inf_lp,
                    prompt_lens,
                    cfg.tis,
                    policy_loss=cfg.policy_loss,
                )
                fwd_bwd_result = policy.forward_backward(
                    rl_datums,
                    kernel_loss,
                    loss_fn_config=kernel_config,
                )
            else:
                # Client-side custom path: execute the Python loss closure
                # returned by build_loss_fn(...) via forward_backward_custom(...).
                fwd_bwd_result = policy.forward_backward_custom(
                    data,
                    client_loss_builder(adv, ref_lp, prompt_lens, inf_lp, prox_lp),
                )
            logger.info("fwd_bwd: done (%.1fs)", _time.time() - t0)
            return fwd_bwd_result

        def train_step(
            step: int,
            prompt_groups: list[PromptGroup],
            loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            """ref_forward + fwd_bwd + optim_step + metrics (1:1)."""
            t0 = _time.time()
            ref_forward(prompt_groups)
            logger.info("[step %d] ref_forward: done (%.1fs)", step + 1, _time.time() - t0)

            t0 = _time.time()
            fwd_bwd_result = fwd_bwd_one(prompt_groups)
            logger.info("[step %d] fwd_bwd: done (%.1fs)", step + 1, _time.time() - t0)

            t0 = _time.time()
            optim_result = policy.optim_step(
                adam_params,
                grad_accumulation_normalization=cfg.grad_accumulation_normalization,
            )
            step += 1
            logger.info("[step %d] optim_step: done (%.1fs)", step, _time.time() - t0)

            if cfg.weight_sync.dcp_save_interval > 0 and step % cfg.weight_sync.dcp_save_interval == 0:
                logger.info("[step %d] dcp_save...", step)
                t0 = _time.time()
                logger.info("[step %d] dcp_save: done (%.1fs)", step, _time.time() - t0)
                _data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    step - step_offset
                ) * prompt_groups_per_step
                save_checkpoint(
                    policy,
                    f"step-{step}",
                    cfg.log_path,
                    {
                        "step": step,
                        "data_consumed": _data_consumed,
                        "source_job_id": policy_job_id,
                    },
                    kind=CheckpointKind.STATE,
                    base_model=cfg.base_model,
                    training_shape=cfg.infra.training_shape_id,
                )

            metrics = compute_step_metrics(
                prompt_groups=prompt_groups,
                fwd_bwd_results=[fwd_bwd_result],
                optim_result=optim_result,
                n_accum=1,
                timing_metrics=flush_timing(),
                loop_stats=loop_stats,
                completions_per_prompt=completions_per_prompt,
            )
            metrics["train/step"] = step

            step_tokens = sum(
                len(d.loss_fn_inputs["target_tokens"].data) for pg in prompt_groups for d in pg.data
            )
            avg_reward = metrics.get("rollout/reward", 0.0)
            avg_acc = metrics.get("rollout/accuracy", 0.0)
            avg_kl = metrics.get("train/mean_kl", 0.0)
            logger.info(
                "Step %d | Reward: %.3f | Acc: %.1f%% | KL: %.4f",
                step,
                avg_reward,
                avg_acc * 100,
                avg_kl,
            )
            log_metrics_json(step, reward=avg_reward, accuracy=avg_acc, kl=avg_kl)
            wandb_log(metrics, step)

            total_rl_steps = len(rl_dataset) - step_offset
            runner.append_metrics(step, metrics, tokens=step_tokens)
            runner.write_status(
                RunStatus.RUNNING, step=step, total_steps=total_rl_steps, message="training",
            )
            runner.write_metadata()

            if cfg.trajectory_dir:
                _dump_trajectory(cfg.trajectory_dir, step, prompt_groups)

            return step, metrics

        # -- Run ----------------------------------------------------------------

        def _weight_sync(step: int) -> None:
            logger.info("[step %d] weight_sync: saving + loading...", step)
            t0 = _time.time()
            with timer("weight_sync"):
                weight_syncer.save_and_hotload(f"step-{step}")
            logger.info("[step %d] weight_sync: done (%.1fs)", step, _time.time() - t0)

        def _loop_metrics_callback(loop_metrics: dict) -> None:
            """Called by run_rl_loop after each train step with loop-level metrics."""
            if concurrency_controller is not None:
                cc_summary = concurrency_controller.step_completed()
                for k, v in cc_summary.items():
                    loop_metrics[f"concurrency/{k}"] = v
            wandb_log(loop_metrics, step=loop_metrics.get("train/step", 0))

        train_fns = TrainStepFns(train_step=train_step)

        remaining_rows = []
        for i_step in range(step_offset, len(rl_dataset)):
            remaining_rows.extend(rl_dataset.get_batch(i_step))

        total_rl_steps = len(rl_dataset) - step_offset
        runner.start_training()
        runner.write_status(RunStatus.RUNNING, total_steps=total_rl_steps, message="training")

        global_step = asyncio.run(
            run_rl_loop(
                sample_fns=(sample_one_prompt(row) for row in remaining_rows),
                train_fns=train_fns,
                prompt_groups_per_step=prompt_groups_per_step,
                dynamic_filter_fn=should_accept,
                global_step=step_offset,
                metrics_callback=_loop_metrics_callback,
                weight_sync_fn=_weight_sync if cfg.weight_sync.weight_sync_interval > 0 else None,
                weight_sync_interval=cfg.weight_sync.weight_sync_interval,
            )
        )

        # -- Final checkpoint ----------------------------------------------------

        if cfg.save_final_checkpoint and global_step > step_offset:
            try:
                _data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    global_step - step_offset
                ) * prompt_groups_per_step
                cp_name = f"step-{global_step}"
                paths = save_checkpoint(
                    policy,
                    cp_name,
                    cfg.log_path,
                    {
                        "step": global_step,
                        "data_consumed": _data_consumed,
                        "source_job_id": policy_job_id,
                    },
                    kind=CheckpointKind.BOTH,
                    base_model=cfg.base_model,
                    training_shape=cfg.infra.training_shape_id,
                )

                if getattr(cfg, "output_model_id", None):
                    rlor_mgr.promote_checkpoint(
                        policy_job_id,
                        paths["sampler_path"],
                        cfg.output_model_id,
                    )
                    runner.write_output_model(
                        model_id=cfg.output_model_id, checkpoint=cp_name, job_id=policy_job_id,
                    )
            except Exception as e:
                logger.warning("Failed to save final checkpoint: %s", e)

            runner.write_status(
                RunStatus.COMPLETED, step=global_step, total_steps=total_rl_steps, message="done",
            )
            runner.write_metadata()
            logger.info("Training complete: %d steps", global_step)
            wandb_finish()
            return {
                "steps": global_step,
                "policy_job_id": policy_job_id,
                "reference_job_id": reference_job_id,
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        log_path="./rl_logs",
        base_model="accounts/fireworks/models/qwen3-8b",
        infra=InfraConfig(
            training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200",
        ),
        deployment=DeployConfig(
            tokenizer_model="Qwen/Qwen3-8B",
        ),
    )
    main(cfg)
