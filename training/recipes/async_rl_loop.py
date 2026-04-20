#!/usr/bin/env python3
"""Async GRPO with streaming rollout-training overlap — library-style recipe.

A self-contained training script. Direct calls only: this recipe owns
the scheduler (a small recipe-local class for staleness + concurrency
gating), the per-group ``ref_forward + policy_forward + forward_backward``
sequence, and the end-of-step ``optim_step + weight_sync + metrics``.
The cookbook contributes only pure functions and data shapes.

Compared to ``rl_loop.py``:

* Streams groups one at a time. Each accepted group is sent to the
  trainer immediately; the server accumulates gradients across calls.
* Two-level capacity gating: ``max_head_offpolicy_versions`` (staleness)
  + ``sample_max_concurrency`` (in-flight HTTP requests).
* Forces ``weight_sync_interval=1`` (the streaming pipeline requires
  hotload after every optimizer step).

Acknowledgements: scheduler design (staleness budget + concurrency cap)
is inspired by AReaL's ``BatchTaskDispatcher`` + ``StalenessManager``
(https://github.com/inclusionAI/AReaL).

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.async_rl_loop
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import time
import warnings
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, List, Optional

import tinker
import transformers

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.client import GradAccNormalization
from fireworks.training.sdk.deployment import DeploymentSampler
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils import (
    DEFAULT_ADAM,
    ConcurrencyConfig,
    DeployConfig,
    InfraConfig,
    ReconnectableClient,
    ResourceCleanup,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    WeightSyncConfig,
    compute_advantages,
    load_jsonl_dataset,
    log_metrics_json,
    prepare_sampling_messages,
    read_api_extra_headers_env,
    setup_or_reattach_deployment,
    setup_wandb,
    validate_config,
    wandb_finish,
    wandb_log,
)
from training.utils.checkpoint_utils import (
    CheckpointKind,
    resolve_resume,
    save_checkpoint,
)
from training.utils.client import DEFAULT_TIMEOUT_S
from training.utils.rl import (
    PromptGroup,
    TrainContext,
    align_inference_logprobs,
    finish_step,
    make_concurrency_controller,
    make_policy_datum,
    make_reference_datum,
    provision_trainer_pair,
    ref_fwd_bwd,
    resolve_policy_profile,
    resolve_reference_profile,
)
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.losses import build_loss_fn, resolve_builtin_loss
from training.utils.rl.router_replay import build_r3_routing_matrices
from training.utils.rl.tis import TISConfig

logger = logging.getLogger(__name__)


warnings.warn(
    "async_rl_loop is EXPERIMENTAL — APIs may change without notice. "
    "For production training, use rl_loop.py.",
    stacklevel=2,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required."""

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    rollout_base_model: str | None = None
    dataset: str = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"

    learning_rate: float = 1e-5
    kl_beta: float = 0.001
    completions_per_prompt: int = 4
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    max_rows: int = 100
    max_seq_len: int | None = None
    lora_rank: int = 0

    # -- Step sizing --------------------------------------------------------

    prompt_groups_per_step: int = 1
    """Default number of accepted groups per optimizer step."""

    valid_prompt_groups_per_step: int | None = None
    """Target accepted groups per step (defaults to ``prompt_groups_per_step``)."""

    # -- Async-specific knobs -----------------------------------------------

    max_head_offpolicy_versions: int = 2
    """Max staleness in optimizer steps an accepted rollout may lag the
    current weight version. 0 = strict on-policy."""

    sample_max_concurrency: int | None = None
    """Max in-flight HTTP requests to the deployment. ``None`` = capped
    by the staleness window only."""

    # -- Router replay (R3) -------------------------------------------------

    router_replay: bool = False
    router_replay_completion_only: bool = True

    # -- Training ----------------------------------------------------------

    grad_accumulation_normalization: GradAccNormalization | str | None = (
        GradAccNormalization.NUM_LOSS_TOKENS
    )

    # -- Loss --------------------------------------------------------------

    policy_loss: str = "grpo"
    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    ratio_log_cap: float = 20.0
    tis: TISConfig = field(default_factory=TISConfig)

    # -- Trajectory / resume / output --------------------------------------

    trajectory_dir: str | None = None
    policy_job_id: str | None = None
    policy_base_url: str | None = None
    reference_job_id: str | None = None
    reference_base_url: str | None = None
    init_from_checkpoint: str | None = None
    output_model_id: str | None = None
    save_final_checkpoint: bool = True
    step_timeout: int = 0

    # -- Sub-configs -------------------------------------------------------

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="grpo-tinker"))
    runner: RunnerConfig = field(default_factory=RunnerConfig)


# ---------------------------------------------------------------------------
# Reward + filter (customise per task)
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    digits = re.search(r"(-?\d+)", match.group(1))
    return digits.group(1) if digits else None


def reward_fn(completion: str, row: dict) -> float:
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


def should_accept(pg: PromptGroup) -> bool:
    return len(set(pg.rewards)) > 1


# ---------------------------------------------------------------------------
# Recipe-local scheduler — staleness + concurrency gating.
# Lives in the recipe (not the library) because the cookbook should not
# call user code; this class invokes ``sample_fn`` and ``filter_fn``.
# ---------------------------------------------------------------------------


SampleFn = Callable[[dict], Awaitable[PromptGroup | None]]
FilterFn = Callable[[PromptGroup], bool]


class _StreamingScheduler:
    """Two-level capacity gating: staleness + concurrency.

    Internal to this recipe. If you want streaming async with different
    behavior, fork this class and edit it — it is intentionally not
    exposed as cookbook API.
    """

    def __init__(
        self,
        *,
        step_target: int,
        max_head_offpolicy_versions: int,
        global_step: int = 0,
        total_accepted: int = 0,
        total_rejected: int = 0,
        rows_submitted: int = 0,
        max_concurrent: int | None = None,
    ):
        self._step_target = step_target
        self._max_offpolicy = max_head_offpolicy_versions
        self._current_version = global_step
        self._total_accepted = total_accepted
        self._total_rejected = total_rejected
        self._rows_submitted = rows_submitted
        policy_window = (max_head_offpolicy_versions + 1) * step_target
        self._max_concurrent = (
            min(max_concurrent, policy_window) if max_concurrent is not None
            else policy_window
        )
        self._in_flight: set[asyncio.Task] = set()
        self._results: asyncio.Queue[tuple[PromptGroup | None, int]] = asyncio.Queue()
        self._rows_exhausted = False

    @property
    def current_version(self) -> int:
        return self._current_version

    @property
    def data_exhausted(self) -> bool:
        return self._rows_exhausted and not self._in_flight

    def bump_version(self) -> None:
        self._current_version += 1

    def get_state(self) -> dict:
        return {
            "rows_submitted": self._rows_submitted,
            "total_accepted": self._total_accepted,
            "total_rejected": self._total_rejected,
        }

    def _capacity(self) -> int:
        staleness = (
            (self._max_offpolicy + self._current_version + 1) * self._step_target
            - (self._total_accepted + len(self._in_flight))
        )
        concurrency = self._max_concurrent - len(self._in_flight)
        return max(0, min(staleness, concurrency))

    def _submit(self, sample_fn: SampleFn, row: dict) -> None:
        version = self._current_version

        async def _worker() -> None:
            try:
                result = await sample_fn(row)
            except Exception as exc:
                logger.warning(
                    "Rollout failed (%s): %s", type(exc).__name__, exc or repr(exc),
                )
                result = None
            self._results.put_nowait((result, version))

        task = asyncio.create_task(_worker())
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)
        self._rows_submitted += 1

    def _refill(self, sample_fn: SampleFn, rows: Iterator[dict]) -> None:
        if self._rows_exhausted:
            return
        for _ in range(self._capacity()):
            try:
                row = next(rows)
            except StopIteration:
                self._rows_exhausted = True
                return
            self._submit(sample_fn, row)

    async def stream(
        self,
        sample_fn: SampleFn,
        filter_fn: FilterFn | None,
        rows: Iterator[dict],
    ) -> AsyncIterator[tuple[PromptGroup, int]]:
        """Yield ``(group, version)`` until ``step_target`` accepted or rows exhausted."""
        accepted = 0
        self._refill(sample_fn, rows)
        while accepted < self._step_target:
            if not self._in_flight and self._results.empty():
                return
            try:
                item, version = await asyncio.wait_for(
                    self._results.get(), timeout=0.1,
                )
            except asyncio.TimeoutError:
                self._refill(sample_fn, rows)
                continue
            if item is None:
                self._refill(sample_fn, rows)
                continue
            if filter_fn is not None and not filter_fn(item):
                self._total_rejected += 1
                self._refill(sample_fn, rows)
                continue
            self._total_accepted += 1
            accepted += 1
            yield item, version
            self._refill(sample_fn, rows)


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

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(
        cfg.base_model, cfg.dataset, cfg.weight_sync, cfg.deployment,
        output_model_id=cfg.output_model_id,
    )
    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step
    step_target = cfg.valid_prompt_groups_per_step or prompt_groups_per_step
    if step_target < 1:
        raise ValueError("valid_prompt_groups_per_step must be >= 1")
    if cfg.max_head_offpolicy_versions < 0:
        raise ValueError("max_head_offpolicy_versions must be >= 0")
    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required.")

    # Streaming pipeline requires hotload after every optimizer step.
    if cfg.weight_sync.weight_sync_interval != 1:
        logger.warning(
            "Async streaming pipeline forces weight_sync_interval=1 (was %d)",
            cfg.weight_sync.weight_sync_interval,
        )
        cfg.weight_sync.weight_sync_interval = 1

    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": completions_per_prompt,
            "step_target": step_target,
            "max_head_offpolicy_versions": cfg.max_head_offpolicy_versions,
            "kl_beta": cfg.kl_beta,
            "lr": cfg.learning_rate,
        },
    )

    # -- Setup infrastructure (direct one-shot builder calls) --------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(
            api_key=api_key, base_url=base_url, additional_headers=additional_headers,
        )
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(
            api_key=api_key, base_url=base_url, additional_headers=additional_headers,
        )

    cfg.infra.training_shape_id, policy_profile = resolve_policy_profile(
        rlor_mgr,
        shape_id=cfg.infra.training_shape_id,
        base_model=cfg.base_model,
        lora_rank=cfg.lora_rank,
        max_seq_len=cfg.max_seq_len,
    )
    if not cfg.deployment.deployment_shape and policy_profile.deployment_shape_version:
        cfg.deployment.deployment_shape = policy_profile.deployment_shape_version
    if cfg.max_seq_len is None:
        cfg.max_seq_len = policy_profile.max_supported_context_length
    if cfg.max_seq_len is None:
        raise ValueError("max_seq_len is required.")
    cfg.infra.ref_training_shape_id, ref_profile = resolve_reference_profile(
        rlor_mgr,
        shape_id=cfg.infra.ref_training_shape_id,
        base_model=cfg.base_model,
        lora_rank=cfg.lora_rank,
        max_seq_len=cfg.max_seq_len,
        kl_beta=cfg.kl_beta,
    )

    runner.set_accelerator_info(profile=policy_profile)
    runner.write_status(RunStatus.PENDING, message="provisioning")

    def _on_trainer_status(msg: str) -> None:
        runner.write_status(RunStatus.PENDING, message=msg)

    boot_start = time.time()

    with runner, ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup, ExitStack() as stack:
        _cleanup = cleanup if cleanup_on_exit else None

        policy_ep, reference_ep = provision_trainer_pair(
            rlor_mgr,
            base_model=cfg.base_model,
            infra_config=cfg.infra,
            policy_profile=policy_profile,
            ref_profile=ref_profile,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            policy_job_id=cfg.policy_job_id,
            reference_job_id=cfg.reference_job_id,
            policy_base_url=cfg.policy_base_url,
            reference_base_url=cfg.reference_base_url,
            cleanup=_cleanup,
            on_status=_on_trainer_status,
        )
        policy_job_id = policy_ep.job_id
        reference_job_id = reference_ep.job_id if reference_ep else policy_ep.job_id

        dep_info = setup_or_reattach_deployment(
            deploy_mgr, cfg.deployment, cfg.base_model, cfg.infra, policy_ep.job_name,
        )
        if cleanup_on_exit:
            cleanup.deployment(cfg.deployment.deployment_id, action="scale_to_zero")

        timeout = cfg.step_timeout or DEFAULT_TIMEOUT_S

        def _make_client(ep):
            c = ReconnectableClient(
                rlor_mgr, ep.job_id, cfg.base_model,
                lora_rank=cfg.lora_rank,
                fw_api_key=api_key,
                default_timeout=timeout,
                endpoint=ep if (cfg.policy_base_url or cfg.reference_base_url) else None,
            )
            if hasattr(c, "close"):
                stack.callback(c.close)
            return c

        policy = _make_client(policy_ep)
        if reference_ep is not None:
            reference = _make_client(reference_ep)
        elif cfg.lora_rank > 0 and cfg.kl_beta > 0:
            reference = policy.create_base_reference()
            stack.callback(reference.close)
        else:
            reference = None

        inference_model = dep_info.inference_model if dep_info else cfg.base_model
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.deployment.tokenizer_model, trust_remote_code=True,
        )

        concurrency_controller = make_concurrency_controller(
            cfg.concurrency, deploy_mgr, cfg.deployment,
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
            base_model=cfg.rollout_base_model or cfg.base_model,
            hotload_timeout=cfg.weight_sync.weight_sync_timeout,
            first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
            lora_rank=cfg.lora_rank,
        )

        boot_metrics: dict = {
            "train/step": 0,
            "infra/total_boot_time": time.time() - boot_start,
        }
        if deploy_mgr.boot_time_s is not None:
            boot_metrics["infra/deploy_boot_time"] = deploy_mgr.boot_time_s
        wandb_log(boot_metrics, step=0)

        # -- Resume --------------------------------------------------------

        resume_info = resolve_resume(policy, cfg.log_path, cfg.init_from_checkpoint)
        step_offset = resume_info.step if resume_info else 0
        wandb_log({"train/step": step_offset}, step_offset)

        if cfg.weight_sync.weight_sync_before_training and cfg.deployment.deployment_id:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            weight_syncer.save_and_hotload(name, checkpoint_type="base")

        # -- Loss + adam ---------------------------------------------------

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
            cfg.policy_loss, policy_profile,
            dapo_config=cfg.dapo, gspo_config=cfg.gspo, cispo_config=cfg.cispo,
            ratio_log_cap=cfg.ratio_log_cap,
            eps_clip=cfg.eps_clip, eps_clip_high=cfg.eps_clip_high,
        )

        # -- Sampling closure (VISIBLE — customise this) -------------------

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=cfg.max_seq_len,
            http_timeout=cfg.deployment.sample_timeout,
            logprobs=True,
        )
        if cfg.router_replay:
            sample_kwargs.update(include_routing_matrix=True, echo=True)

        async def sample_one_prompt(row: dict) -> PromptGroup | None:
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
                logger.warning("Sampling failed (%s): %s", type(e).__name__, e or repr(e))
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
                        s.routing_matrices, s.prompt_len, model_input_len,
                        completion_only=cfg.router_replay_completion_only,
                    )
                policy_data.append(make_policy_datum(tokens, routing_matrices=rm))
                if reference is not None:
                    reference_data.append(make_reference_datum(tokens))
                adv_filtered.append(advantages[idx])
                inf_logprobs_aligned.append(
                    align_inference_logprobs(
                        s.inference_logprobs,
                        prompt_len=prompt_len,
                        total_len=model_input_len,
                        echoed=getattr(s, "logprobs_echoed", False),
                    )
                )

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

        # -- TrainContext: bundle handed to ref_fwd_bwd / finish_step ------

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
            # Streaming pipeline: hotload after every optimizer step.
            weight_sync_interval=1,
            dcp_save_interval=cfg.weight_sync.dcp_save_interval,
            wandb_log=wandb_log,
            log_metrics_json=log_metrics_json,
        )

        # -- Outer streaming loop (VISIBLE — recipe owns this) -------------

        raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        all_rows = raw_dataset * cfg.epochs
        async_state = (resume_info.async_state if resume_info else None) or {}

        async def _run() -> int:
            scheduler = _StreamingScheduler(
                step_target=step_target,
                max_head_offpolicy_versions=cfg.max_head_offpolicy_versions,
                global_step=step_offset,
                total_accepted=async_state.get("total_accepted", 0),
                total_rejected=async_state.get("total_rejected", 0),
                rows_submitted=async_state.get("rows_submitted", 0),
                max_concurrent=cfg.sample_max_concurrency,
            )
            row_iter = iter(all_rows)
            for _ in range(min(async_state.get("rows_submitted", 0), len(all_rows))):
                next(row_iter, None)

            step = step_offset
            total_steps = max(0, len(all_rows) // step_target)
            runner.start_training()
            runner.write_status(
                RunStatus.RUNNING, total_steps=total_steps, message="training",
            )

            while not scheduler.data_exhausted:
                groups: list[PromptGroup] = []
                fwd_results: list = []
                version_offsets: list[int] = []
                raw_rewards: list[float] = []
                t0 = time.time()

                async for group, version in scheduler.stream(
                    sample_one_prompt, should_accept, row_iter,
                ):
                    groups.append(group)
                    version_offsets.append(scheduler.current_version - version)
                    raw_rewards.extend(group.rewards)
                    # Train this group now — server accumulates gradients.
                    fwd_results.append(
                        await asyncio.to_thread(ref_fwd_bwd, ctx, group),
                    )
                    logger.info(
                        "[async step %d] trained group %d/%d",
                        step + 1, len(groups), step_target,
                    )

                if not groups:
                    break

                wall_time = time.time() - t0
                stats = {
                    "valid_prompt_groups": len(groups),
                    "total_sampled": len(groups),
                    "filter_drops": 0,
                    "sample_fails": 0,
                    "sample_wait_time": wall_time,
                    "step_wall_time": wall_time,
                    "all_raw_rewards": raw_rewards,
                    "version_offsets": version_offsets,
                }

                # finish_step does optim_step + (per ctx) hotload + metrics +
                # (optional) DCP save. Recipe carries scheduler state into
                # the checkpoint extras.
                scheduler_state = scheduler.get_state()

                def _save(name: str, extra: dict) -> object:
                    return save_checkpoint(
                        policy, name, cfg.log_path,
                        {**extra, "async_state": scheduler_state},
                        kind=CheckpointKind.STATE,
                        base_model=cfg.base_model,
                        training_shape=cfg.infra.training_shape_id,
                    )

                step, metrics = await asyncio.to_thread(
                    finish_step, ctx, step, groups, fwd_results,
                    stats,
                    save_checkpoint_fn=_save,
                    step_target=step_target,
                    resume_data_consumed=resume_info.data_consumed if resume_info else 0,
                    step_offset=step_offset,
                )
                runner.append_metrics(step, metrics)
                runner.write_status(
                    RunStatus.RUNNING, step=step, message="training",
                )
                runner.write_metadata()

                scheduler.bump_version()

            return step

        global_step = asyncio.run(_run())

        # -- Final checkpoint ----------------------------------------------

        if cfg.save_final_checkpoint and global_step > step_offset:
            try:
                data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    global_step - step_offset
                ) * step_target
                cp_name = f"step-{global_step}"
                paths = save_checkpoint(
                    policy, cp_name, cfg.log_path,
                    {
                        "step": global_step,
                        "data_consumed": data_consumed,
                        "source_job_id": policy_job_id,
                    },
                    kind=CheckpointKind.BOTH,
                    base_model=cfg.base_model,
                    training_shape=cfg.infra.training_shape_id,
                )
                if cfg.output_model_id:
                    rlor_mgr.promote_checkpoint(
                        policy_job_id,
                        paths["sampler_path"],
                        cfg.output_model_id,
                        cfg.base_model,
                    )
                    runner.write_output_model(
                        model_id=cfg.output_model_id,
                        checkpoint=cp_name, job_id=policy_job_id,
                    )
            except Exception as e:
                logger.warning("Failed to save final checkpoint: %s", e)

        runner.write_status(
            RunStatus.COMPLETED, step=global_step, message="done",
        )
        runner.write_metadata()
        logger.info("Async training complete: %d steps", global_step)
        wandb_finish()
        return {
            "steps": global_step,
            "policy_job_id": policy_job_id,
            "reference_job_id": reference_job_id,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = Config(
        log_path="./async_rl_logs",
        infra=InfraConfig(
            training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200",
        ),
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
    )
    main(cfg)
