#!/usr/bin/env python3
"""IGPO training on multi-hop QA with interleaved IG scoring.

Each question is paired with a paragraph pool (e.g. HotpotQA).  The model
calls ``search(query)`` to retrieve paragraphs, then ``submit_answer(answer)``
to finish.  IG scoring uses the ground-truth answer as the ``answer_tokens``
and fires in parallel with generation via ``turn_callback``.

Usage:
    pip install --pre "fireworks-ai>=1.0.0a36" tinker-cookbook eval-protocol datasets
    python prepare_data.py                     # download HotpotQA → dataset.jsonl
    python train_multihop_qa_igpo.py --training-shape <shape_id> --output-model-id <id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, cast

import tinker

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from eval_protocol.models import EvaluationRow, InputMetadata, Message
from eval_protocol.pytest.types import RolloutProcessorConfig

from training.examples.frozen_lake.masking import (
    compute_model_output_spans,
    build_ui_token_mask,
)
from training.examples.multihop_qa.multihop_qa_rollout import (
    MultiHopQARolloutProcessor,
    MultiHopQARolloutService,
)

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.client import GradAccNormalization
from fireworks.training.sdk.weight_syncer import WeightSyncer

from training.utils import (
    DEFAULT_ADAM,
    InfraConfig,
    ResourceCleanup,
    WandBConfig,
    DeployConfig,
    WeightSyncConfig,
    ReconnectableClient,
    wandb_log,
    setup_wandb,
    wandb_finish,
    log_metrics_json,
    setup_deployment,
    create_trainer_job,
    validate_config,
    load_jsonl_dataset,
    build_datum_from_token_mask,
)
from training.utils.rl import PromptGroup
from training.utils.rl.rollout import make_remote_rollout_fn
from training.utils.rl.rollout import Rollout, rollout_to_prompt_group
from training.utils.rl.rollout import pack_payload_to_sample
from training.utils.rl.train import TrainStepFns, run_rl_loop
from training.utils.rl.losses import combine_prompt_groups
from training.utils.rl.tis import TISConfig
from training.utils.rl.metrics import compute_step_metrics
from training.utils.rl.igpo import (
    IGPOTurnScorer,
    compute_turn_advantages,
    expand_turn_advantages_from_spans,
    make_igpo_loss_fn,
)
from training.utils.timer import timer, flush_timing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question by searching for "
    "relevant information. You have access to two tools:\n"
    "- search(query): Search for information about a topic. Returns "
    "relevant paragraphs.\n"
    "- submit_answer(answer): Submit your final answer.\n\n"
    "Search as many times as needed, then submit your answer. "
    "Always respond with exactly one tool call and no additional text."
)


@dataclass
class MultiHopQAIGPOConfig:
    log_path: str = "./multihop_qa_igpo_logs"

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    tokenizer_model: str = "Qwen/Qwen3-8B"

    dataset_path: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "dataset.jsonl")
    )

    learning_rate: float = 1e-5
    kl_beta: float = 0.001
    completions_per_prompt: int = 4
    max_completion_tokens: int = 512
    temperature: float = 1.0
    epochs: int = 3
    max_rows: int = 200
    max_steps: int = 8
    lora_rank: int = 0
    max_seq_len: int | None = None

    prompt_groups_per_step: int = 4
    max_concurrent: int = 16

    # IGPO-specific — ig_weight enables IG scoring (0 = pure GRPO)
    gamma: float = 0.95
    ig_weight: float = 1.0
    scoring_workers: int = 8
    eps_clip: float = 0.2
    skip_ig_last_turn: bool = True

    # Search env
    search_top_k: int = 2

    training_shape: str = ""
    deployment_shape: str = ""
    accelerator_type: str = ""
    accelerator_count: int | None = None
    custom_image_tag: str = ""
    deployment_id: str | None = None
    region: str | None = None
    deployment_region: str | None = None
    deployment_replica_count: int | None = None

    wandb_entity: str = field(
        default_factory=lambda: os.environ.get("WANDB_ENTITY", "")
    )
    wandb_project: str = field(
        default_factory=lambda: os.environ.get("WANDB_PROJECT", "multihop-qa-igpo")
    )

    policy_job_id: str | None = None
    reference_job_id: str | None = None
    inference_base_url: str | None = None
    output_model_id: str | None = None


def parse_args() -> MultiHopQAIGPOConfig:
    parser = argparse.ArgumentParser(
        description="IGPO training on multi-hop QA with search tools"
    )
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3-8b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset-path",
                        default=os.path.join(os.path.dirname(__file__), "dataset.jsonl"))
    parser.add_argument("--training-shape",
                        default=os.environ.get("TRAINING_SHAPE", ""))
    parser.add_argument("--deployment-shape", default="")
    parser.add_argument("--accelerator-type", default="")
    parser.add_argument("--accelerator-count", type=int, default=None)
    parser.add_argument("--custom-image-tag", default="")
    parser.add_argument("--deployment-id", default=None)
    parser.add_argument("--region", default="US_VIRGINIA_1")
    parser.add_argument("--deployment-region", default=None)
    parser.add_argument("--deployment-replica-count", type=int, default=None)

    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--completions-per-prompt", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-beta", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--prompt-groups-per-step", type=int, default=4)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=0)

    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--ig-weight", type=float, default=1.0,
                        help="Enable IG intrinsic rewards (any non-zero value). "
                             "Set to 0 for pure GRPO baseline. IG and outcome "
                             "rewards are z-normalized separately per the paper.")
    parser.add_argument("--scoring-workers", type=int, default=8)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--skip-ig-last-turn", action="store_true", default=True)
    parser.add_argument("--no-skip-ig-last-turn", dest="skip_ig_last_turn",
                        action="store_false")
    parser.add_argument("--search-top-k", type=int, default=2)

    parser.add_argument("--wandb-entity",
                        default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-project",
                        default=os.environ.get("WANDB_PROJECT", "multihop-qa-igpo"))

    parser.add_argument("--policy-job-id", default=None)
    parser.add_argument("--reference-job-id", default=None)
    parser.add_argument("--inference-base-url", default=None)
    parser.add_argument("--output-model-id", type=str, required=True)

    cfg = cast(
        MultiHopQAIGPOConfig,
        parser.parse_args(namespace=MultiHopQAIGPOConfig()),
    )
    if cfg.ig_weight != 0.0 and cfg.ig_weight != 1.0:
        logger.info(
            "ig_weight=%.2f — note this is a flag (0=off, non-zero=on); "
            "IG rewards are z-normalized independently, not scaled by this value.",
            cfg.ig_weight,
        )
    return cfg


# ---------------------------------------------------------------------------
# EvaluationRow -> training data with per-turn IG advantages
# ---------------------------------------------------------------------------


def evaluation_row_to_igpo_training_data(
    row: EvaluationRow,
    turn_advantages: List[float],
) -> tuple[list[tinker.Datum], int, list[float], list[float], list[float]]:
    """Convert a completed rollout into training data with per-token IGPO advantages.

    Returns (datums, prompt_len, inf_logprobs, [episode_reward], per_token_advantages).
    """
    extra = row.execution_metadata.extra or {}
    token_turn_traces = extra.get("token_turn_traces") or []
    model_request_traces = extra.get("model_request_traces") or []
    step_rewards = extra.get("step_rewards") or []

    if not token_turn_traces:
        return [], 0, [], [], []

    last_trace = token_turn_traces[-1]
    last_prompt_ids = [int(x) for x in (last_trace.get("prompt_ids") or [])]
    last_completion_ids = [int(x) for x in (last_trace.get("completion_ids") or [])]
    full_tokens = last_prompt_ids + last_completion_ids

    if len(full_tokens) < 2:
        return [], 0, [], [], []

    first_prompt_len = len(
        [int(x) for x in (token_turn_traces[0].get("prompt_ids") or [])]
    )

    spans = compute_model_output_spans(token_turn_traces, model_request_traces)
    token_mask = build_ui_token_mask(spans, len(full_tokens))
    rendered = build_datum_from_token_mask(
        full_tokens, token_mask, include_loss_mask=True
    )
    datum = rendered.datum
    model_input_len = len(rendered.token_ids) - 1

    inf_logprobs = [0.0] * model_input_len
    for trace in token_turn_traces:
        turn_prompt_len = len(trace.get("prompt_ids") or [])
        turn_completion_logprobs = trace.get("completion_logprobs") or []
        start_pos = max(0, turn_prompt_len - 1)
        for i, lp in enumerate(turn_completion_logprobs):
            pos = start_pos + i
            if pos < model_input_len:
                inf_logprobs[pos] = float(lp)

    episode_reward = step_rewards[-1] if step_rewards else 0.0

    per_token_adv = expand_turn_advantages_from_spans(
        turn_advantages, spans, model_input_len
    )

    return [datum], first_prompt_len, [inf_logprobs], [episode_reward], per_token_adv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: MultiHopQAIGPOConfig | None = None) -> dict:
    if cfg is None:
        cfg = parse_args()

    logger.info(
        "Multi-hop QA IGPO training: %s (gamma=%.2f, ig_weight=%.2f)",
        cfg.base_model, cfg.gamma, cfg.ig_weight,
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step

    infra = InfraConfig(
        training_shape_id=cfg.training_shape or None,
        region=cfg.region,
        accelerator_type=cfg.accelerator_type or None,
        accelerator_count=cfg.accelerator_count,
        custom_image_tag=cfg.custom_image_tag or None,
    )
    deploy_cfg = DeployConfig(
        deployment_id=cfg.deployment_id,
        deployment_shape=cfg.deployment_shape or None,
        deployment_accelerator_type=cfg.accelerator_type or None,
        deployment_region=cfg.deployment_region,
        replica_count=cfg.deployment_replica_count,
        tokenizer_model=cfg.tokenizer_model,
        sample_timeout=1200,
    )
    weight_sync_cfg = WeightSyncConfig(
        weight_sync_interval=1,
        dcp_save_interval=0,  # skip DCP checkpoints; enable (e.g. 20) for production runs
        dcp_timeout=2700,
        first_checkpoint_type="base",
        weight_sync_before_training=bool(cfg.deployment_id),
        weight_sync_timeout=1800,
    )
    wandb_cfg = WandBConfig(
        entity=cfg.wandb_entity or None,
        project=cfg.wandb_project,
        run_name=cfg.deployment_id or f"multihop-qa-igpo-{int(time.time()) % 100000}",
    )

    validate_config(
        cfg.base_model,
        cfg.dataset_path,
        weight_sync_cfg,
        deploy_cfg,
        output_model_id=cfg.output_model_id,
    )

    # Load dataset
    dataset = load_jsonl_dataset(cfg.dataset_path, max_rows=cfg.max_rows)
    logger.info("Loaded %d rows from %s", len(dataset), cfg.dataset_path)

    setup_wandb(wandb_cfg, {
        "completions_per_prompt": completions_per_prompt,
        "prompt_groups_per_step": prompt_groups_per_step,
        "kl_beta": cfg.kl_beta,
        "lr": cfg.learning_rate,
        "gamma": cfg.gamma,
        "ig_weight": cfg.ig_weight,
        "skip_ig_last_turn": cfg.skip_ig_last_turn,
        "max_rows": cfg.max_rows,
    })

    rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    deploy_mgr = DeploymentManager(
        api_key=api_key,
        base_url=base_url,
        hotload_api_url=base_url,
        inference_url=cfg.inference_base_url or base_url,
    )

    profile = None
    if infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(infra.training_shape_id)
        dsv = profile.deployment_shape_version or ""
        if dsv and not deploy_cfg.deployment_shape:
            idx = dsv.find("/versions/")
            deploy_cfg.deployment_shape = dsv[:idx] if idx >= 0 else dsv

    if profile and cfg.max_seq_len is None:
        cfg.max_seq_len = profile.max_supported_context_length
    if cfg.max_seq_len is None:
        cfg.max_seq_len = 4096

    ref_profile = None
    if infra.ref_training_shape_id:
        ref_profile = rlor_mgr.resolve_training_profile(infra.ref_training_shape_id)
    use_reference = ref_profile is not None

    _infra_start = time.time()
    policy_job_id: str | None = None
    global_step = step_offset = 0
    reward_history: list[float] = []
    _shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal _shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.warning("Received %s — initiating graceful shutdown...", sig_name)
        _shutdown_requested = True
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    scoring_executor = ThreadPoolExecutor(max_workers=cfg.scoring_workers)

    # Tokenizer for answer tokens (loaded once, used per-prompt)
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.tokenizer_model, trust_remote_code=True,
    )

    with ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup:
        # Create trainer jobs first (trainer owns the hot-load bucket)
        def _make_job(label, precreated_id, job_profile=None, **extra_kw):
            if precreated_id:
                ep = create_trainer_job(
                    rlor_mgr, base_model=cfg.base_model, infra=infra,
                    job_id=precreated_id,
                )
                return ep, precreated_id, True
            ep = create_trainer_job(
                rlor_mgr,
                base_model=cfg.base_model,
                infra=infra,
                profile=job_profile,
                lora_rank=cfg.lora_rank,
                max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name=f"multihop-qa-igpo-{label}",
                **extra_kw,
            )
            return ep, ep.job_id, False

        with ThreadPoolExecutor(max_workers=2) as pool:
            pol_fut = pool.submit(
                _make_job, "policy", cfg.policy_job_id, job_profile=profile
            )
            ref_fut = (
                pool.submit(
                    _make_job, "reference", cfg.reference_job_id,
                    job_profile=ref_profile, forward_only=True,
                )
                if use_reference
                else None
            )
            policy_ep, policy_job_id, precreated_policy = pol_fut.result()
            if ref_fut:
                reference_ep, reference_job_id, _ = ref_fut.result()
            else:
                reference_ep, reference_job_id = None, None

            if not precreated_policy:
                cleanup.trainer(policy_job_id)
            if reference_job_id:
                cleanup.trainer(reference_job_id)

        # Create deployment referencing the trainer's hot-load bucket
        if cfg.policy_job_id and cfg.deployment_id:
            dep_info = None
        else:
            deploy_cfg.hot_load_trainer_job = policy_ep.job_name
            dep_info = setup_deployment(deploy_mgr, deploy_cfg, cfg.base_model, infra)
            if (
                not cfg.deployment_id
                and deploy_cfg.deployment_id
                and os.environ.get("KEEP_DEPLOYMENT", "0") != "1"
            ):
                cleanup.deployment(deploy_cfg.deployment_id)

        policy = ReconnectableClient(
            rlor_mgr, policy_ep.job_id, cfg.base_model, cfg.lora_rank,
            fw_api_key=api_key, endpoint=policy_ep,
        )
        reference = (
            ReconnectableClient(
                rlor_mgr, reference_ep.job_id, cfg.base_model, cfg.lora_rank,
                fw_api_key=api_key, endpoint=reference_ep,
            )
            if reference_ep
            else None
        )

        if dep_info:
            inference_model = dep_info.inference_model
        elif deploy_cfg.deployment_id:
            inference_model = (
                f"{cfg.base_model}#accounts/{deploy_mgr.account_id}"
                f"/deployments/{deploy_cfg.deployment_id}"
            )
        else:
            inference_model = cfg.base_model

        weight_syncer = WeightSyncer(
            policy_client=policy.inner,
            deploy_mgr=deploy_mgr,
            deployment_id=deploy_cfg.deployment_id,
            base_model=cfg.base_model,
            hotload_timeout=weight_sync_cfg.weight_sync_timeout,
            first_checkpoint_type=weight_sync_cfg.first_checkpoint_type,
        )

        wandb_log(
            {"train/step": 0, "infra/total_boot_time": time.time() - _infra_start},
            step=0,
        )

        from training.utils.checkpoints import TrainingCheckpoints
        ckpt = TrainingCheckpoints(
            policy,
            rlor_mgr,
            trainer_id=policy_job_id,
            log_path=cfg.log_path,
            lora_rank=cfg.lora_rank,
        )
        resume_info = ckpt.resume()
        step_offset = resume_info.step if resume_info else 0
        if weight_sync_cfg.weight_sync_before_training and deploy_cfg.deployment_id:
            weight_syncer._deployment_checked = True
            name = (
                f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            )
            for _hotload_attempt in range(3):
                try:
                    weight_syncer.save_and_hotload(name, checkpoint_type="base")
                    ckpt.invalidate_promotable_snapshot_cache()
                    break
                except RuntimeError as e:
                    if _hotload_attempt < 2:
                        logger.warning(
                            "Hotload attempt %d failed (%s), checking status...",
                            _hotload_attempt + 1, e,
                        )
                        import time as _time
                        _time.sleep(15)
                        status = deploy_mgr.hotload_check_status(
                            deploy_cfg.deployment_id, cfg.base_model,
                        )
                        replicas = status.get("replicas", [])
                        if replicas and replicas[0].get("readiness") and replicas[0].get("current_snapshot_identity"):
                            logger.info(
                                "Hotload recovered: identity=%s",
                                replicas[0]["current_snapshot_identity"],
                            )
                            break
                        logger.info("Retrying hotload...")
                    else:
                        raise

        # Readiness check
        inference_url = deploy_mgr.inference_url
        import httpx
        _inference_prefix = "/v1" if cfg.inference_base_url else "/inference/v1"
        _readiness_url = (
            inference_url.rstrip("/") + _inference_prefix + "/completions"
        )
        for _ready_attempt in range(600):
            try:
                _resp = httpx.post(
                    _readiness_url,
                    json={
                        "model": inference_model,
                        "prompt": "test",
                        "max_tokens": 1,
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=15,
                )
                if _resp.status_code == 200:
                    logger.info("Deployment is ready for inference")
                    break
                time.sleep(5)
            except Exception:
                time.sleep(5)

        # Rollout processor
        rollout_base_url = inference_url.rstrip("/") + (
            "" if cfg.inference_base_url else "/inference"
        )
        rollout_processor = MultiHopQARolloutProcessor(
            model_id=inference_model,
            tokenizer_name_or_path=cfg.tokenizer_model,
            api_key=api_key,
            base_url=rollout_base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_completion_tokens,
            system_prompt=SYSTEM_PROMPT,
            logprobs=True,
            enable_thinking=False,
            search_top_k=cfg.search_top_k,
        )
        rollout_config = RolloutProcessorConfig(
            completion_params={"model": inference_model},
            mcp_config_path="",
            steps=cfg.max_steps,
            semaphore=asyncio.Semaphore(cfg.max_concurrent),
        )

        adam_params = tinker.AdamParams(
            learning_rate=cfg.learning_rate, **DEFAULT_ADAM
        )

        trajectory_path = (
            f"/tmp/multihop_qa_igpo_trajectories_{int(time.time())}.jsonl"
        )
        trajectory_log = open(trajectory_path, "a")
        logger.info("Logging trajectories to %s", trajectory_path)

        try:
            # -- Sample one prompt with interleaved IG scoring -----------------

            # Cookbook-native rollout-fn wiring: ``MultiHopQARolloutService``
            # implements the cookbook ``RolloutService`` contract.  We drive
            # it through ``make_remote_rollout_fn`` (now extras-aware) so the
            # IGPO step-reward / row-id metadata each payload carries
            # survives into ``rollout.row_meta["payload_extras"]``.

            class _RolloutCtx:
                def __init__(self, version_cell: list[int]) -> None:
                    self.completions_per_prompt = completions_per_prompt
                    self.sample_kwargs: Dict[str, Any] = {}
                    self.tokenizer_id = cfg.tokenizer_model
                    self._version_cell = version_cell

                def current_version(self) -> int:
                    return self._version_cell[0]

            _version_cell = [step_offset]
            _rollout_ctx = _RolloutCtx(_version_cell)

            def _assistant_spans(loss_mask: List[int]) -> List[tuple[int, int]]:
                """Return (start, end) indices of each assistant span in
                ``loss_mask`` (assistant tokens masked 1, gap tokens 0).

                One span per assistant turn; empty list when no assistant
                tokens are present."""
                spans: List[tuple[int, int]] = []
                start: int | None = None
                for i, m in enumerate(loss_mask):
                    if m == 1 and start is None:
                        start = i
                    elif m == 0 and start is not None:
                        spans.append((start, i))
                        start = None
                if start is not None:
                    spans.append((start, len(loss_mask)))
                return spans

            async def sample_one_prompt(
                row_data: Dict[str, Any],
            ) -> PromptGroup | None:
                ground_truth = str(row_data.get("ground_truth", ""))
                answer_tokens = tokenizer.encode(
                    ground_truth, add_special_tokens=False
                )
                if not answer_tokens:
                    logger.warning("Empty answer tokens for GT: %r", ground_truth)
                    return None

                messages_raw = row_data.get("messages") or []
                question = ""
                for m in messages_raw:
                    if m.get("role") == "user":
                        question = m.get("content", "")
                        break

                # IGPO turn scorer — restored online IG behavior.  The
                # service is constructed per-call with the scorer's
                # ``on_turn_complete`` callback so the processor's
                # in-flight ``turn_callback`` hook drives the scorer
                # during generation.
                scorer = IGPOTurnScorer(
                    answer_tokens=answer_tokens,
                    executor=scoring_executor,
                    ig_weight=cfg.ig_weight,
                    skip_ig_last_turn=cfg.skip_ig_last_turn,
                    inference_url=rollout_base_url,
                    model_id=inference_model,
                    api_key=api_key,
                    tokenizer=tokenizer,
                )

                service_row = {
                    "context": row_data.get("context") or {},
                    "ground_truth": ground_truth,
                    "question": question,
                    "messages": list(messages_raw),
                }
                # Pre-register turn_futs for the row_ids the service will
                # emit (matches legacy entrypoint shape).
                pre_row_ids = MultiHopQARolloutService.prepare_row_ids(
                    n=completions_per_prompt, row=service_row,
                )
                for rid in pre_row_ids:
                    scorer._turn_futs[rid] = []

                rollout_service = MultiHopQARolloutService(
                    processor=rollout_processor,
                    rollout_config=rollout_config,
                    tokenizer_id=cfg.tokenizer_model,
                    turn_callback=scorer.on_turn_complete,
                )
                _multihop_rollout_fn = make_remote_rollout_fn(rollout_service)

                rollout = await _multihop_rollout_fn(service_row, _rollout_ctx)
                if rollout is None or len(rollout.samples) < 2:
                    return None

                # ``payload_extras`` carries per-payload step_rewards +
                # row_id (from the service).  Match payloads to scorer
                # state via the row_id so the IG bookkeeping uses the
                # actual scorer record, not a stub.
                payload_extras_list: List[Dict[str, Any]] = list(
                    (rollout.row_meta or {}).get("payload_extras") or []
                )

                # ``on_rollout_start`` runs after rollouts complete because
                # the prompt_tokens it baselines on come from the first
                # turn's prompt_ids in the assembled trajectory (mirrors
                # legacy timing).
                for sample, extras in zip(rollout.samples, payload_extras_list):
                    rid = extras.get("row_id")
                    if rid is None:
                        continue
                    spans = _assistant_spans(sample.loss_mask)
                    if not spans:
                        continue
                    # Initial prompt span = tokens before the first
                    # assistant span (the user/system seed prefix).
                    first_assistant_start = spans[0][0]
                    prompt_tokens = list(sample.tokens[:first_assistant_start])
                    if prompt_tokens:
                        scorer.on_rollout_start(rid, prompt_tokens)

                all_ig_rewards: List[List[float]] = []
                all_outcome_rewards: List[List[float]] = []
                for sample, extras in zip(rollout.samples, payload_extras_list):
                    step_rewards = list(extras.get("step_rewards") or [])
                    rid = extras.get("row_id")
                    if rid is not None and rid in scorer._baselines:
                        ig_r, outcome_r = await asyncio.to_thread(
                            scorer.collect_rewards, rid, step_rewards
                        )
                    else:
                        # Fallback (no callback fired or scorer baseline
                        # missing — same shape as legacy fallback branch).
                        ig_r = [0.0] * len(step_rewards)
                        outcome_r = list(step_rewards)
                    all_ig_rewards.append(ig_r)
                    all_outcome_rewards.append(outcome_r)

                if trajectory_log:
                    for sample, extras in zip(rollout.samples, payload_extras_list):
                        sr = list(extras.get("step_rewards") or [])
                        entry = {
                            "question": question[:100],
                            "ground_truth": ground_truth,
                            "step_rewards": sr,
                            "final_reward": sr[-1] if sr else 0.0,
                            "num_turns": len(sr),
                        }
                        trajectory_log.write(json.dumps(entry) + "\n")
                    trajectory_log.flush()

                turn_adv = compute_turn_advantages(
                    ig_rewards=all_ig_rewards,
                    outcome_rewards=all_outcome_rewards,
                    gamma=cfg.gamma,
                )

                # Build per-sample per_token_advantages in target-token
                # coordinates (length ``len(tokens) - 1``, advantage
                # written at indices ``[s-1, e-1)`` for each
                # full-token-coord assistant span ``[s, e)``).  This
                # matches the coordinate system ``make_igpo_loss_fn``
                # consumes (``response_start = prompt_len - 1``).
                all_per_token_adv: List[List[float]] = []
                for sample, row_turn_adv in zip(rollout.samples, turn_adv):
                    spans = _assistant_spans(sample.loss_mask)
                    n_targets = max(0, len(sample.tokens) - 1)
                    pta = [0.0] * n_targets
                    for span_idx, (s, e) in enumerate(spans):
                        if span_idx >= len(row_turn_adv):
                            break
                        adv = float(row_turn_adv[span_idx])
                        # Full-token assistant span [s, e) maps to target
                        # indices [s-1, e-1).  Spans always start at s>=1
                        # (the prompt is the first user/system span), so
                        # s-1 >= 0 in practice; clamp defensively.
                        t_start = max(0, s - 1)
                        t_end = max(0, e - 1)
                        for k in range(t_start, t_end):
                            pta[k] = adv
                    all_per_token_adv.append(pta)

                # Stash IGPO row_meta on the rollout (rollout_to_prompt_group
                # forwards row_meta to the resulting PromptGroup unchanged).
                rich_row_meta = dict(rollout.row_meta or {})
                rich_row_meta.update({
                    "ground_truth": ground_truth,
                    "question": question,
                    "per_token_advantages": all_per_token_adv,
                    "turn_rewards": all_outcome_rewards,
                    "ig_rewards": all_ig_rewards,
                    "outcome_rewards": all_outcome_rewards,
                })
                rollout.row_meta = rich_row_meta

                pg = rollout_to_prompt_group(rollout, with_reference=use_reference)
                if pg is None:
                    return None
                return pg

            # -- Training callbacks --------------------------------------------

            def ref_forward_batch(groups: list[PromptGroup]) -> None:
                if not use_reference or reference is None:
                    return
                all_ref_data = [d for pg in groups for d in pg.ref_data]
                if not all_ref_data:
                    return
                ref_fwd = reference.forward(all_ref_data, "cross_entropy")
                idx = 0
                for pg in groups:
                    n = len(pg.ref_data)
                    pg.ref_logprobs = [
                        ref_fwd.loss_fn_outputs[idx + i]["logprobs"].data
                        for i in range(n)
                    ]
                    idx += n

            def fwd_bwd_one(sub: list[PromptGroup]):
                data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(
                    sub
                )

                all_pta: List[List[float]] = []
                for pg in sub:
                    meta = pg.row_meta or {}
                    pta = meta.get("per_token_advantages")
                    if pta:
                        all_pta.extend(pta)
                    else:
                        for i in range(len(pg.data)):
                            n_lp = len(
                                pg.data[i].loss_fn_inputs["target_tokens"].data
                            )
                            all_pta.append([pg.advantages[i]] * n_lp)

                loss_fn = make_igpo_loss_fn(
                    per_token_advantages=all_pta,
                    ref_logprobs=ref_lp,
                    prompt_lens=prompt_lens,
                    inf_logprobs=inf_lp,
                    prox_logprobs=None,
                    kl_beta=cfg.kl_beta,
                    eps_clip=cfg.eps_clip,
                )
                return policy.forward_backward_custom(data, loss_fn)

            def train_step(
                step: int,
                prompt_groups: list[PromptGroup],
                loop_stats: dict | None = None,
            ) -> tuple[int, dict]:
                t0 = time.time()
                ref_forward_batch(prompt_groups)
                logger.info(
                    "[step %d] ref_forward: done (%.1fs)",
                    step + 1, time.time() - t0,
                )

                t0 = time.time()
                fwd_bwd_result = fwd_bwd_one(prompt_groups)
                logger.info(
                    "[step %d] fwd_bwd: done (%.1fs)",
                    step + 1, time.time() - t0,
                )

                t0 = time.time()
                optim_result = policy.optim_step(
                    adam_params,
                    grad_accumulation_normalization=GradAccNormalization.NUM_LOSS_TOKENS,
                )
                step += 1
                logger.info(
                    "[step %d] optim_step: done (%.1fs)",
                    step, time.time() - t0,
                )

                if (
                    weight_sync_cfg.weight_sync_interval > 0
                    and step % weight_sync_cfg.weight_sync_interval == 0
                ):
                    with timer("weight_sync"):
                        weight_syncer.save_and_hotload(f"step-{step}")
                    ckpt.invalidate_promotable_snapshot_cache()

                if (
                    weight_sync_cfg.dcp_save_interval > 0
                    and step % weight_sync_cfg.dcp_save_interval == 0
                ):
                    with timer("dcp_save"):
                        _data_consumed = (
                            (resume_info.data_consumed if resume_info else 0)
                            + (step - step_offset) * prompt_groups_per_step
                        )
                        ckpt.save(
                            f"step-{step}",
                            resumable=True,
                            promotable=False,
                            data_consumed=_data_consumed,
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

                all_turn_rewards = [
                    r
                    for pg in prompt_groups
                    for r in (pg.row_meta or {}).get("turn_rewards", [])
                ]
                if all_turn_rewards:
                    flat_ig = [r for rlist in all_turn_rewards for r in rlist]
                    metrics["igpo/mean_turn_reward"] = (
                        sum(flat_ig) / len(flat_ig) if flat_ig else 0.0
                    )
                    metrics["igpo/avg_turns"] = sum(
                        len(r) for r in all_turn_rewards
                    ) / len(all_turn_rewards)

                avg_reward = metrics.get("rollout/reward", 0.0)
                avg_kl = metrics.get("train/mean_kl", 0.0)
                logger.info(
                    "Step %d | Reward: %.3f | KL: %.4f | Turns: %.1f",
                    step,
                    avg_reward,
                    avg_kl,
                    metrics.get("igpo/avg_turns", 0.0),
                )
                reward_history.append(avg_reward)
                log_metrics_json(step, reward=avg_reward, kl=avg_kl)
                _wandb_step[0] = max(_wandb_step[0] + 1, step)
                wandb_log(metrics, _wandb_step[0])
                return step, metrics

            train_fns = TrainStepFns(train_step=train_step)

            def dynamic_filter(pg: PromptGroup) -> bool:
                return len(set(pg.rewards)) > 1

            all_prompts = dataset * cfg.epochs
            logger.info(
                "Training: %d rows x %d epochs = %d prompt groups, "
                "%d completions/prompt, %d groups/step",
                len(dataset),
                cfg.epochs,
                len(all_prompts),
                completions_per_prompt,
                prompt_groups_per_step,
            )

            _wandb_step = [step_offset]

            def _filtered_step_callback(loop_metrics: dict) -> None:
                _wandb_step[0] += 1
                wandb_log(loop_metrics, step=_wandb_step[0])

            global_step = asyncio.run(
                run_rl_loop(
                    sample_fns=(
                        sample_one_prompt(row) for row in all_prompts
                    ),
                    train_fns=train_fns,
                    prompt_groups_per_step=prompt_groups_per_step,
                    dynamic_filter_fn=dynamic_filter,
                    global_step=step_offset,
                    metrics_callback=_filtered_step_callback,
                )
            )

            if global_step > step_offset:
                try:
                    cp_name = f"step-{global_step}"
                    _data_consumed = (
                        resume_info.data_consumed if resume_info else 0
                    ) + (global_step - step_offset) * prompt_groups_per_step
                    ckpt.save(
                        cp_name,
                        resumable=True,
                        promotable=True,
                        data_consumed=_data_consumed,
                    )
                    if getattr(cfg, "output_model_id", None):
                        ckpt.promote_latest(cfg.output_model_id, cfg.base_model)
                except Exception as e:
                    logger.warning("Failed to save final checkpoint: %s", e)
                logger.info("Training complete: %d steps", global_step)

        finally:
            if trajectory_log and not trajectory_log.closed:
                trajectory_log.close()
            wandb_finish()
            scoring_executor.shutdown(wait=False)

    return {"steps": global_step - step_offset, "rewards": reward_history}


if __name__ == "__main__":
    main()
