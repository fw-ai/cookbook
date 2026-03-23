#!/usr/bin/env python3
"""Async GRPO training loop with streaming rollout-training overlap.

An async-first RL training loop that streams prompt groups to the trainer
as they arrive from sampling, instead of collecting a full batch first.
The tinker server accumulates gradients across ``ref_fwd_bwd`` calls;
``finish_step`` fires ``optim_step`` + weight sync after all groups in a
step have been processed.

Key differences from ``rl_loop.py``:

- **Streaming pipeline**: ``stream_groups()`` yields groups one at a time;
  each is immediately sent to the trainer for ``ref_forward + fwd_bwd``.
  Sampling for the next group continues in the background.
- **HTTP concurrency gate**: ``sample_max_concurrency`` limits actual HTTP
  requests via a semaphore on ``_do_one_completion``, independent of the
  rollout scheduler's policy window.
- **Extracted infra**: ``_setup_infra()`` separates infrastructure setup
  from the training loop.

Acknowledgements:

  The ``AsyncRolloutScheduler`` and two-level capacity gating (staleness
  cap + concurrency cap) are inspired by AReaL's ``BatchTaskDispatcher``
  and ``StalenessManager``:
  https://github.com/inclusionAI/AReaL

Usage::

    from training.recipes.async_rl_loop import main, Config

TODO: Once stable, merge the infra extraction (_Infra, _setup_infra) and
      the streaming pipeline back into rl_loop.py and retire this module.
"""

from __future__ import annotations

import os
import re
import signal
import asyncio
import logging
import warnings
import time as _time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List

import tinker
import transformers

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.client import GradAccNormalization
from fireworks.training.sdk.deployment import DeploymentSampler
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils import (
    DEFAULT_ADAM,
    ResourceCleanup,
    ReconnectableClient,
    RLPromptDataset,
    compute_advantages,
    wandb_log,
    setup_wandb,
    wandb_finish,
    validate_config,
    log_metrics_json,
    setup_deployment,
    create_trainer_job,
    load_jsonl_dataset,
    prepare_sampling_messages,
)
from training.utils.checkpoint_utils import (
    resolve_resume,
    save_checkpoint,
    CheckpointKind,
)
from training.utils.config import DeployConfig, InfraConfig, WeightSyncConfig, RewardFn, WandBConfig
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.losses import PromptGroup, build_loss_fn, resolve_builtin_loss
from training.utils.rl.router_replay import build_r3_routing_matrices
from training.utils.rl.tis import TISConfig
from training.utils.rl.train import TrainContext, RolloutStats, ref_fwd_bwd, finish_step

logger = logging.getLogger(__name__)

warnings.warn(
    "\n"
    "╔══════════════════════════════════════════════════════════════╗\n"
    "║  async_rl_loop is EXPERIMENTAL — APIs may change or break  ║\n"
    "║  without notice. For production training, use rl_loop.py.  ║\n"
    "╚══════════════════════════════════════════════════════════════╝",
    stacklevel=2,
)

# ---------------------------------------------------------------------------
# Config — everything the user can configure
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Full configuration for async RL training.

    Customisation points visible in this file:
    - ``reward_fn`` — module-level reward function (override via assignment)
    - ``should_accept`` — module-level rollout filter (override via assignment)
    - ``policy_loss`` — select a registered loss algorithm
    """

    log_path: str
    """Directory for checkpoints and logs. Required."""

    # -- Model & data -------------------------------------------------------

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
    """Auto-populated from training shape's max_supported_context_length."""
    lora_rank: int = 0

    # -- Sampling & batching ------------------------------------------------

    prompt_groups_per_step: int = 1
    """Number of prompt groups per optimizer step."""

    # -- Async rollout ------------------------------------------------------

    valid_prompt_groups_per_step: int | None = None
    """Target accepted groups per step.  Defaults to prompt_groups_per_step."""

    max_head_offpolicy_versions: int = 2
    """Maximum staleness: how many versions ahead rollouts can be."""

    sample_max_concurrency: int | None = None
    """Max concurrent HTTP requests to the deployment (resource window).

    sample_with_tokens(n=K) fans out into K individual HTTP requests.
    This gates each request, not each prompt.  With completions_per_prompt=8
    and sample_max_concurrency=32, at most 4 prompts sample concurrently."""

    # -- Router replay (R3) -------------------------------------------------

    router_replay: bool = False
    router_replay_completion_only: bool = True

    # -- Training -----------------------------------------------------------

    grad_accumulation_normalization: GradAccNormalization | str | None = GradAccNormalization.NUM_LOSS_TOKENS

    # -- Loss ---------------------------------------------------------------

    policy_loss: str = "grpo"
    """grpo, importance_sampling, dapo, dro, gspo, reinforce, or cispo."""

    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    ratio_log_cap: float = 20.0
    tis: TISConfig = field(default_factory=TISConfig)

    # -- Trajectory / resume ------------------------------------------------

    trajectory_dir: str | None = None
    policy_job_id: str | None = None
    policy_base_url: str | None = None
    reference_job_id: str | None = None
    reference_base_url: str | None = None
    init_from_checkpoint: str | None = None
    output_model_id: str | None = None

    # -- Sub-configs (infra plumbing) ---------------------------------------

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="grpo-tinker"))


# ---------------------------------------------------------------------------
# Reward function -- customise this for your task
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> str | None:
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

    Passed to ``AsyncRolloutScheduler`` as a pluggable filter.  Replace with your
    own logic (e.g. minimum reward threshold, response length filter).
    """
    return len(set(pg.rewards)) > 1


# ---------------------------------------------------------------------------
# Datum construction (inlined from datum.py — async path only)
# TODO: If sync rl_loop.py also adopts build_prompt_group, extract to a
#       shared module.
# ---------------------------------------------------------------------------


def build_prompt_group(
    sampled: list,
    row: dict,
    *,
    reward_fn_: RewardFn,
    completions_per_prompt: int,
    use_reference: bool = False,
    router_replay: bool = False,
    router_replay_completion_only: bool = True,
    trajectory_dir: str | None = None,
    input_messages: list[dict] | None = None,
) -> PromptGroup | None:
    """Build a PromptGroup from sampled completions.

    Reward computation, advantage normalisation, datum construction, and
    logprob alignment.  No I/O.
    """
    if not sampled or len(sampled) < completions_per_prompt:
        return None

    rewards = [reward_fn_(s.text, row) for s in sampled]
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
        if router_replay:
            rm = build_r3_routing_matrices(
                s.routing_matrices,
                s.prompt_len,
                model_input_len,
                completion_only=router_replay_completion_only,
            )

        policy_datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:-1], routing_matrices=rm),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=tokens[1:], dtype="int64", shape=[model_input_len]
                ),
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
        aligned = (
            list(s.inference_logprobs)
            if echoed
            else [0.0] * response_start + list(s.inference_logprobs)
        )
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
        prompt=input_messages if trajectory_dir else None,
        completions=[s.text for s in sampled] if trajectory_dir else None,
        row_meta={"ground_truth": row.get("ground_truth", "")} if trajectory_dir else None,
    )


# ---------------------------------------------------------------------------
# Async rollout scheduler (AReaL-style capacity gating)
#
# Manages in-flight sampling tasks with two-level gating:
#   1. Staleness cap (policy window): how far ahead rollouts can be
#   2. Concurrency cap (resource window): max in-flight HTTP requests
#
# stream_groups() yields accepted groups one at a time for immediate
# training — the core of the streaming pipeline.
# ---------------------------------------------------------------------------


class AsyncRolloutScheduler:
    """Async rollout scheduler with AReaL-style capacity gating."""

    def __init__(
        self,
        step_target: int,
        max_head_offpolicy_versions: int,
        filter_fn: Callable | None = None,
        global_step: int = 0,
        total_accepted: int = 0,
        total_rejected: int = 0,
        rows_submitted: int = 0,
        max_concurrent: int | None = None,
    ):
        self._step_target = step_target
        self._max_offpolicy = max_head_offpolicy_versions
        self._filter_fn = filter_fn

        self._current_version = global_step
        self._total_accepted = total_accepted
        self._total_rejected = total_rejected
        self._rows_submitted = rows_submitted

        policy_window = (max_head_offpolicy_versions + 1) * step_target
        self._max_concurrent = min(max_concurrent, policy_window) if max_concurrent is not None else policy_window

        self._in_flight: set[asyncio.Task] = set()
        self._result_queue: asyncio.Queue[tuple[PromptGroup | None, int]] = asyncio.Queue()
        self._rows_exhausted = False

    @property
    def current_version(self) -> int:
        return self._current_version

    @property
    def data_exhausted(self) -> bool:
        return self._rows_exhausted and len(self._in_flight) == 0

    def _staleness_cap(self) -> int:
        budget = (
            (self._max_offpolicy + self._current_version + 1) * self._step_target
            - (self._total_accepted + len(self._in_flight))
        )
        return max(budget, 0)

    def _concurrency_cap(self) -> int:
        return max(self._max_concurrent - len(self._in_flight), 0)

    def _capacity(self) -> int:
        return min(self._staleness_cap(), self._concurrency_cap())

    def _submit_one(self, sample_fn_factory, row: dict) -> None:
        version = self._current_version
        coro = sample_fn_factory(row)

        async def _worker():
            try:
                result = await coro
            except Exception as exc:
                logger.warning("Rollout task failed (%s): %s", type(exc).__name__, exc or repr(exc))
                result = None
            self._result_queue.put_nowait((result, version))

        task = asyncio.create_task(_worker())
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)
        self._rows_submitted += 1

    async def stream_groups(self, sample_fn_factory, rows):
        """Yield accepted (PromptGroup, version) one at a time.

        The core of the streaming pipeline: each accepted group is yielded
        immediately so the caller can send it to the trainer for fwd_bwd
        while sampling continues in the background.
        """
        accepted = 0

        def _try_submit():
            if self._rows_exhausted:
                return
            cap = self._capacity()
            for _ in range(cap):
                try:
                    row = next(rows)
                except StopIteration:
                    self._rows_exhausted = True
                    break
                self._submit_one(sample_fn_factory, row)

        _try_submit()

        while accepted < self._step_target:
            if not self._in_flight and self._result_queue.empty():
                break

            try:
                item, version = await asyncio.wait_for(
                    self._result_queue.get(), timeout=0.1,
                )
            except asyncio.TimeoutError:
                _try_submit()
                continue

            if item is None:
                _try_submit()
                continue

            if self._filter_fn is not None and not self._filter_fn(item):
                self._total_rejected += 1
                _try_submit()
                continue

            self._total_accepted += 1
            accepted += 1
            yield item, version
            _try_submit()

    async def collect_batch(self, sample_fn_factory, rows):
        """Collect step_target accepted groups (blocking).

        Re-implements the batch collection with full stats tracking.
        """
        accepted = []
        stats = RolloutStats()
        t0 = _time.time()

        def _process_one(item, version):
            if item is None:
                stats.sample_fails += 1
                stats.total_sampled += 1
                return
            stats.total_sampled += 1
            stats.raw_rewards.extend(item.rewards)
            if self._filter_fn is not None and not self._filter_fn(item):
                stats.filter_drops += 1
                self._total_rejected += 1
                return
            accepted.append(item)
            self._total_accepted += 1
            stats.version_offsets.append(self._current_version - version)

        def _drain():
            while not self._result_queue.empty():
                pair = self._result_queue.get_nowait()
                _process_one(*pair)
                if len(accepted) >= self._step_target:
                    break

        _drain()

        while len(accepted) < self._step_target:
            if not self._rows_exhausted:
                cap = self._capacity()
                for _ in range(cap):
                    try:
                        row = next(rows)
                    except StopIteration:
                        self._rows_exhausted = True
                        break
                    self._submit_one(sample_fn_factory, row)

            if not self._in_flight and self._result_queue.empty():
                break

            try:
                item, version = await asyncio.wait_for(self._result_queue.get(), timeout=0.1)
                _process_one(item, version)
                if len(accepted) < self._step_target:
                    _drain()
            except asyncio.TimeoutError:
                _drain()

        stats.valid_groups = len(accepted)
        stats.wall_time = _time.time() - t0
        return accepted, stats

    def bump_version(self) -> None:
        """Increment version after optim_step. Opens capacity budget."""
        self._current_version += 1

    def get_state(self) -> dict:
        """Snapshot counters for checkpointing."""
        return {
            "rows_submitted": self._rows_submitted,
            "total_accepted": self._total_accepted,
            "total_rejected": self._total_rejected,
        }


# ---------------------------------------------------------------------------
# Infrastructure setup
# TODO: Extract this into a shared module (e.g. training.utils.infra_setup)
#       so both rl_loop.py and async_rl_loop.py can reuse it without
#       duplicating the deployment/trainer/client creation logic.
# ---------------------------------------------------------------------------


@dataclass
class _Infra:
    """Everything the training loop needs from infrastructure setup."""

    ctx: TrainContext
    sampler: Any  # DeploymentSampler
    use_reference: bool
    step_offset: int
    resume_info: Any
    remaining_rows: list
    resume_data_consumed: int
    reference_job_id: str | None


def _setup_infra(
    cfg: Config,
    rlor_mgr: TrainerJobManager,
    deploy_mgr: DeploymentManager,
    cleanup: ResourceCleanup,
    cleanup_on_exit: bool,
    api_key: str,
    base_url: str,
) -> _Infra:
    """Create deployment, trainers, clients, dataset, and build TrainContext.

    Called once at the start of ``main()``; the return value feeds the loop.

    TODO: Share this with rl_loop.py via a common helper module.
    """
    # -- Resolve training shapes ----------------------------------------

    profile = None
    if cfg.infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)
        dep_shape = getattr(profile, "deployment_shape", None) or getattr(
            profile, "deployment_shape_version", None
        )
        if dep_shape and not cfg.deployment.deployment_shape:
            cfg.deployment.deployment_shape = dep_shape

    if profile and cfg.max_seq_len is None:
        cfg.max_seq_len = profile.max_supported_context_length
        logger.info("max_seq_len from training shape: %d", cfg.max_seq_len)
    if cfg.max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )

    ref_profile = None
    if cfg.infra.ref_training_shape_id:
        ref_profile = rlor_mgr.resolve_training_profile(cfg.infra.ref_training_shape_id)

    use_reference = ref_profile is not None
    if not use_reference:
        logger.info("No ref_training_shape_id set, skipping reference model")

    if not cfg.deployment.deployment_shape:
        extra = cfg.deployment.deployment_extra_args or []
        if "--hot-load-async-transition" not in extra:
            cfg.deployment.deployment_extra_args = extra + [
                "--hot-load-async-transition"
            ]
            logger.info(
                "Auto-injected --hot-load-async-transition for async rollout"
            )
    else:
        logger.info(
            "Async rollout: deployment shape is set, ensure it includes "
            "--hot-load-async-transition in its extra_args"
        )

    # -- Deployment + trainers ------------------------------------------

    _infra_start = _time.time()

    dep_info = setup_deployment(deploy_mgr, cfg.deployment, cfg.base_model, cfg.infra)
    if cleanup_on_exit:
        cleanup.deployment(cfg.deployment.deployment_id, action="scale_to_zero")

    prompt_groups_per_step = cfg.prompt_groups_per_step
    completions_per_prompt = cfg.completions_per_prompt
    logger.info(
        "Training: prompt_groups_per_step=%d | completions_per_prompt=%d",
        prompt_groups_per_step,
        completions_per_prompt,
    )

    if use_reference:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pol_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.base_model,
                infra=cfg.infra,
                profile=profile,
                lora_rank=cfg.lora_rank,
                max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name="grpo-policy",
                hot_load_deployment_id=cfg.deployment.deployment_id,
                job_id=cfg.policy_job_id,
                base_url_override=cfg.policy_base_url,
            )
            ref_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.base_model,
                infra=cfg.infra,
                profile=ref_profile,
                lora_rank=cfg.lora_rank,
                max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name="grpo-reference",
                forward_only=True,
                job_id=cfg.reference_job_id,
                base_url_override=cfg.reference_base_url,
            )
            policy_ep = pol_fut.result()
            policy_job_id = policy_ep.job_id
            if not cfg.policy_job_id:
                cleanup.trainer(policy_job_id)
            reference_ep = ref_fut.result()
            reference_job_id = reference_ep.job_id
            if not cfg.reference_job_id:
                cleanup.trainer(reference_job_id)
    else:
        policy_ep = create_trainer_job(
            rlor_mgr,
            base_model=cfg.base_model,
            infra=cfg.infra,
            profile=profile,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            display_name="grpo-policy",
            hot_load_deployment_id=cfg.deployment.deployment_id,
            job_id=cfg.policy_job_id,
            base_url_override=cfg.policy_base_url,
        )
        policy_job_id = policy_ep.job_id
        if not cfg.policy_job_id:
            cleanup.trainer(policy_job_id)
        reference_ep = None
        reference_job_id = None

    # -- Clients, sampler, weight syncer --------------------------------

    policy = ReconnectableClient(
        rlor_mgr,
        policy_ep.job_id,
        cfg.base_model,
        cfg.lora_rank,
        fw_api_key=api_key,
        endpoint=policy_ep if cfg.policy_base_url else None,
    )
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

    inference_model = dep_info.inference_model if dep_info else cfg.base_model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.deployment.tokenizer_model, trust_remote_code=True
    )
    sampler = DeploymentSampler(
        inference_url=deploy_mgr.inference_url,
        model=inference_model,
        api_key=api_key,
        tokenizer=tokenizer,
        max_concurrency=cfg.sample_max_concurrency,
    )
    weight_syncer = WeightSyncer(
        policy_client=policy.inner,
        deploy_mgr=deploy_mgr,
        deployment_id=cfg.deployment.deployment_id,
        base_model=cfg.base_model,
        hotload_timeout=cfg.weight_sync.weight_sync_timeout,
        first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
        dcp_timeout=cfg.weight_sync.dcp_timeout,
    )

    infra_boot_time = _time.time() - _infra_start
    boot_metrics: dict = {
        "train/step": 0,
        "infra/total_boot_time": infra_boot_time,
    }
    if deploy_mgr.boot_time_s is not None:
        boot_metrics["infra/deploy_boot_time"] = deploy_mgr.boot_time_s
    wandb_log(boot_metrics, step=0)

    # -- Resume ---------------------------------------------------------

    resume_info = resolve_resume(policy, cfg.log_path, cfg.init_from_checkpoint)
    step_offset = resume_info.step if resume_info else 0
    wandb_log({"train/step": step_offset}, step_offset)

    if cfg.weight_sync.weight_sync_before_training and cfg.deployment.deployment_id:
        name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
        weight_syncer.save_and_hotload(name, checkpoint_type="base")

    # -- Dataset + loss -------------------------------------------------

    raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
    all_rows = raw_dataset * cfg.epochs
    rl_dataset = RLPromptDataset(all_rows, prompts_per_step=prompt_groups_per_step)
    adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)

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

    # -- TrainContext ----------------------------------------------------

    ctx = TrainContext(
        policy=policy,
        reference=reference,
        weight_syncer=weight_syncer,
        adam_params=adam_params,
        grad_accumulation_normalization=cfg.grad_accumulation_normalization,
        builtin_server_loss=builtin_server_loss,
        client_loss_builder=client_loss_builder,
        tis_config=cfg.tis,
        policy_loss=cfg.policy_loss,
        log_path=cfg.log_path,
        policy_job_id=policy_job_id,
        completions_per_prompt=completions_per_prompt,
        trajectory_dir=cfg.trajectory_dir,
        weight_sync_interval=cfg.weight_sync.weight_sync_interval,
        dcp_save_interval=cfg.weight_sync.dcp_save_interval,
        wandb_log=wandb_log,
        log_metrics_json=log_metrics_json,
    )

    # -- Remaining rows -------------------------------------------------

    remaining_rows = []
    for i_step in range(step_offset, len(rl_dataset)):
        remaining_rows.extend(rl_dataset.get_batch(i_step))

    resume_data_consumed = resume_info.data_consumed if resume_info else 0

    return _Infra(
        ctx=ctx,
        sampler=sampler,
        use_reference=use_reference,
        step_offset=step_offset,
        resume_info=resume_info,
        remaining_rows=remaining_rows,
        resume_data_consumed=resume_data_consumed,
        reference_job_id=reference_job_id,
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
    """Async GRPO training with streaming rollout-training overlap.

    Uses ``AsyncRolloutScheduler.stream_groups()`` to yield groups one at a
    time, sending each to the trainer immediately via ``ref_fwd_bwd()``.
    The server accumulates gradients; ``finish_step()`` fires ``optim_step``
    after all groups in the step.
    """
    cfg = config

    # -- Signal handling ------------------------------------------------

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # -- Validation -----------------------------------------------------

    validate_config(
        cfg.base_model,
        cfg.dataset,
        cfg.weight_sync,
        cfg.deployment,
        output_model_id=cfg.output_model_id,
    )
    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step

    step_target = (
        cfg.valid_prompt_groups_per_step
        if cfg.valid_prompt_groups_per_step is not None
        else prompt_groups_per_step
    )
    if step_target < 1:
        raise ValueError("valid_prompt_groups_per_step must be >= 1")
    if cfg.max_head_offpolicy_versions < 0:
        raise ValueError("max_head_offpolicy_versions must be >= 0")
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

    # -- Setup infrastructure -------------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)

    # Force 1:1 weight sync for streaming pipeline
    if cfg.weight_sync.weight_sync_interval != 1:
        logger.warning(
            "Async pipeline requires weight_sync_interval=1, overriding configured value %d",
            cfg.weight_sync.weight_sync_interval,
        )
        cfg.weight_sync.weight_sync_interval = 1

    with ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup:
        infra = _setup_infra(
            cfg, rlor_mgr, deploy_mgr, cleanup, cleanup_on_exit, api_key, base_url
        )
        ctx = infra.ctx
        ctx.weight_sync_interval = 1

        # -- Sampling closure -------------------------------------------

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=cfg.max_seq_len,
            http_timeout=cfg.deployment.sample_timeout,
        )
        if cfg.router_replay:
            sample_kwargs.update(
                include_routing_matrix=True, echo=True, logprobs=True
            )
        sample_kwargs["logprobs"] = True

        async def sample_one_prompt(row: dict):
            """Sample completions for one prompt and return a PromptGroup."""
            messages = row.get("messages", [])
            input_messages = prepare_sampling_messages(messages)
            if not input_messages:
                return None

            try:
                sampled = await infra.sampler.sample_with_tokens(
                    messages=input_messages,
                    n=completions_per_prompt,
                    **sample_kwargs,
                )
            except Exception as e:
                logger.warning("Sampling failed (%s): %s", type(e).__name__, e or repr(e))
                return None

            return build_prompt_group(
                sampled,
                row,
                reward_fn_=reward_fn,
                completions_per_prompt=completions_per_prompt,
                use_reference=infra.use_reference,
                router_replay=cfg.router_replay,
                router_replay_completion_only=cfg.router_replay_completion_only,
                trajectory_dir=cfg.trajectory_dir,
                input_messages=input_messages,
            )

        # ==============================================================
        # Streaming training loop
        # ==============================================================

        async_state = (
            (infra.resume_info.async_state if infra.resume_info else None) or {}
        )

        async def _async_loop() -> int:
            scheduler = AsyncRolloutScheduler(
                step_target=step_target,
                max_head_offpolicy_versions=cfg.max_head_offpolicy_versions,
                filter_fn=should_accept,
                global_step=infra.step_offset,
                total_accepted=async_state.get("total_accepted", 0),
                total_rejected=async_state.get("total_rejected", 0),
                rows_submitted=async_state.get("rows_submitted", 0),
                max_concurrent=cfg.sample_max_concurrency,
            )
            row_iter = iter(infra.remaining_rows)
            if async_state.get("rows_submitted", 0) > 0:
                skip = async_state["rows_submitted"]
                for _ in range(min(skip, len(infra.remaining_rows))):
                    next(row_iter, None)
                logger.info("Async resume: skipped %d rows", skip)

            step = infra.step_offset

            while not scheduler.data_exhausted:
                # -- Stream groups: train each immediately, server accumulates --
                groups: list = []
                fwd_bwd_results: list = []
                version_offsets: list[int] = []
                raw_rewards: list[float] = []
                t0 = _time.time()

                async for group, version in scheduler.stream_groups(
                    sample_one_prompt, row_iter
                ):
                    groups.append(group)
                    version_offsets.append(scheduler.current_version - version)
                    raw_rewards.extend(group.rewards)

                    # Train this group now — server accumulates gradients
                    result = ref_fwd_bwd(ctx, group)
                    fwd_bwd_results.append(result)
                    logger.info(
                        "[async step %d] trained group %d/%d",
                        step + 1, len(groups), step_target,
                    )

                if not groups:
                    break

                wall_time = _time.time() - t0
                logger.info(
                    "[async step %d] streamed %d groups (%.1fs)",
                    step + 1, len(groups), wall_time,
                )

                loop_stats = {
                    "valid_prompt_groups": len(groups),
                    "total_sampled": len(groups),
                    "filter_drops": 0,
                    "sample_fails": 0,
                    "sample_wait_time": wall_time,
                    "step_wall_time": wall_time,
                    "all_raw_rewards": raw_rewards,
                    "version_offsets": version_offsets,
                }

                step, metrics = finish_step(
                    ctx,
                    step,
                    groups,
                    fwd_bwd_results,
                    loop_stats,
                    save_checkpoint_fn=lambda name, extra: save_checkpoint(
                        ctx.policy,
                        name,
                        cfg.log_path,
                        {**extra, "async_state": scheduler.get_state()},
                        kind=CheckpointKind.STATE,
                    ),
                    step_target=step_target,
                    resume_data_consumed=infra.resume_data_consumed,
                    step_offset=infra.step_offset,
                )
                metrics["async/version"] = scheduler.current_version
                scheduler.bump_version()

            return step

        global_step = asyncio.run(_async_loop())

        # -- Final checkpoint -------------------------------------------

        if global_step > infra.step_offset:
            try:
                _data_consumed = infra.resume_data_consumed + (
                    global_step - infra.step_offset
                ) * prompt_groups_per_step
                cp_name = f"step-{global_step}"
                paths = save_checkpoint(
                    ctx.policy,
                    cp_name,
                    cfg.log_path,
                    {
                        "step": global_step,
                        "data_consumed": _data_consumed,
                        "source_job_id": ctx.policy_job_id,
                    },
                    kind=CheckpointKind.BOTH,
                )

                if getattr(cfg, "output_model_id", None):
                    rlor_mgr.promote_checkpoint(
                        ctx.policy_job_id,
                        paths["sampler_path"],
                        cfg.output_model_id,
                    )
            except Exception as e:
                logger.warning("Failed to save final checkpoint: %s", e)

            logger.info("Training complete: %d steps", global_step)
            wandb_finish()
            return {
                "steps": global_step,
                "policy_job_id": ctx.policy_job_id,
                "reference_job_id": infra.reference_job_id,
            }
