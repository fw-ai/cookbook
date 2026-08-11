#!/usr/bin/env python3
"""Serverless multimodal RL on VisualToolBench.

This is the multimodal counterpart to ``examples/serverless_rl/countdown_rl.py``:
it runs GRPO against Fireworks **serverless training** -- you connect to a
shared, already-running pooled trainer through the gateway and get back a
Tinker-compatible training client. There is **no trainer job to provision and
no inference deployment to stand up**; the same service hands you both a
training client and, per step, a sampling client bound to that step's LoRA
snapshot. That is the serverless contract documented in ``countdown_rl.py`` and
is what this file preserves -- it does *not* route through the managed
``async_rl_loop`` (which provisions a dedicated trainer + a persistent
deployment sampler with hot-load, and needs a validated training shape).

The loop is the standard GRPO shape, but multimodal:

    service = FiretitanServiceClient(base_url=".../training/v1/serverless")
    training_client = service.create_lora_training_client(base_model, rank)
    for step in range(steps):
        snapshot = training_client.save_weights_for_sampler(name).result().path
        sampler  = service.create_sampling_client(model_path=snapshot, tokenizer=...)
        # multi-turn tool rollout (crop/zoom/rotate/adjust) + rubric LLM judge,
        # one RolloutRun per prompt; group-relative advantages -> policy datums.
        training_client.forward_backward_custom(datums, loss_fn=grpo_closure)
        training_client.optim_step(adam)

Loss path: **client-side GRPO** (``forward_backward_custom`` +
``make_grpo_loss_fn``), not the built-in ``importance_sampling``. Multimodal
inputs, targets, weights, behavior logprobs, raw inference logprobs, and Router
Replay data all use the same image-expanded coordinate system.

Provide aligned training JSONL in the schema documented by the
``visual_toolbench`` example, then:

    export FIREWORKS_API_KEY=fw_...
    python -m training.examples.serverless_rl.visual_toolbench_rl \
        --dataset /absolute/path/to/aligned-vtb.jsonl
    # or: python examples/serverless_rl/visual_toolbench_rl.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tinker
from fireworks.training.sdk import FiretitanServiceClient
from tinker_cookbook.tokenizer_utils import get_tokenizer

try:  # Load env vars from a local .env if present.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from training.examples.rl.visual_toolbench.rollout import (
    DEFAULT_SYSTEM_PROMPT,
    make_rollout_fn,
)
from training.examples.rl.visual_toolbench.reward import (
    DEFAULT_CRITICAL_REWARD_WEIGHT,
    DEFAULT_JUDGE_MAX_CONCURRENCY,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TIMEOUT_S,
)
from training.recipes.async_rl_loop import RolloutSetup
from training.utils import GradAccNormalization
from training.utils.rl.grpo import make_grpo_loss_fn, validate_grpo_config
from training.utils.rl.metrics import add_optimizer_metrics
from training.utils.rl.rollout import Rollout, rollout_to_prompt_group
from training.utils.rl.tis import TISConfig
from training.utils.service import resolve_router_replay_enabled


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "rl" / "visual_toolbench"
DEFAULT_DATASET = EXAMPLE_DIR / "train.jsonl"
DEFAULT_BASE_MODEL = "accounts/fireworks/models/qwen3p6-27b"
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_RENDERER_NAME = "qwen3_6_disable_thinking_interleaved"
FIREWORKS_API_BASE_URL = "https://api.fireworks.ai"
FIREWORKS_SERVERLESS_BASE_URL = f"{FIREWORKS_API_BASE_URL}/training/v1/serverless"
_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")


def _mean_loss(fb_output: Any) -> float | None:
    metrics = getattr(fb_output, "metrics", None) or {}
    loss_sum = metrics.get("sampler_loss:sum", metrics.get("loss:sum"))
    tokens = (
        metrics.get("response_tokens")
        or metrics.get("num_loss_tokens")
        or metrics.get("active_tokens")
        or 1.0
    )
    return float(loss_sum) / max(float(tokens), 1.0) if loss_sum is not None else None


_TRAIN_DIAGNOSTIC_KEYS = (
    "ppo_clip_frac",
    "ppo_ratio_mean",
    "ppo_kl",
    "ref_kl",
    "tis/weight_mean",
    "tis/clip_frac",
    "active_tokens",
    "total_resp_tokens",
    "mask_ratio",
    "mean_adv_loss",
    "mean_kl_penalty",
    "mean_loss",
    "policy_gradient/coefficient_variance_proxy",
    "policy_gradient/estimator_variance_proxy",
    "policy_gradient/estimator_std_error_proxy",
    "policy_gradient/sample_count",
    "raw_inference_logprob_coverage",
    "inference_diff",
    "inference_k1",
    "inference_kld",
)


def _extract_train_diagnostics(fb_output: Any) -> dict[str, float]:
    """Extract stable scalar diagnostics returned by the custom loss."""
    raw = getattr(fb_output, "metrics", None) or {}
    diagnostics: dict[str, float] = {}
    for key in _TRAIN_DIAGNOSTIC_KEYS:
        value = raw.get(key, raw.get(f"{key}:last"))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            diagnostics[f"train/{key}"] = float(value)
    return diagnostics


def _validate_checkpoint_prefix(name: str, field_name: str) -> None:
    # save_weights_for_sampler appends a short unique suffix; step checkpoints
    # also add "-0000". Keep enough room below the DNS-label limit of 63.
    if len(name) > 49 or _DNS_LABEL_RE.fullmatch(name) is None:
        raise ValueError(
            f"{field_name} must be a lowercase DNS-label prefix of at most 49 "
            f"characters, got {name!r}"
        )


@dataclass
class Config:
    """Configuration for an isolated pooled TrainingSession."""

    # --- What to train ------------------------------------------------------
    base_model: str = DEFAULT_BASE_MODEL
    # HuggingFace tokenizer (or local tokenizer dir) matching ``base_model``.
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
    # Qwen defaults to the renderer used by the validated run. Override this
    # together with base_model/tokenizer_model when using another model.
    renderer_name: str = DEFAULT_RENDERER_NAME
    dataset: str = str(DEFAULT_DATASET)
    eval_dataset: str = ""
    lora_rank: int = 64
    # Serverless has no training shape from which to infer this bound.
    max_seq_len: int = 131072

    # --- RL loop shape ------------------------------------------------------
    steps: int = 15
    prompt_groups_per_step: int = 8
    # Completions per prompt (the GRPO group the advantage is computed over).
    group_size: int = 8
    # Number of individual trajectory calls in flight. Keep this conservative
    # for a shared serverless pool; it need not equal the full step size.
    rollout_concurrency: int = 8
    max_completion_tokens: int = 32768
    temperature: float = 1.0
    learning_rate: float = 3e-5
    adam_beta2: float = 0.95
    adam_eps: float = 1e-12
    adam_weight_decay: float = 0.0
    # Drop prompt groups whose samples all share one reward (zero GRPO
    # advantage) -- standard GRPO filtering on a hard benchmark.
    filter_constant_reward: bool = True
    filter_truncated_rollouts: bool = True
    # Shuffle the dataset each epoch so consecutive steps do not preserve the
    # source-file order.
    shuffle: bool = True
    seed: int = 0
    epochs: int = 1
    eval_interval: int = 0
    eval_upfront: bool = False
    eval_at_end: bool = False
    eval_group_size: int = 1
    eval_temperature: float = 1.0
    # Eval sampling can differ from rollout sampling. Training rollouts always
    # use top_p=1/top_k=0 so sampler and trainer score the same distribution.
    eval_top_p: float = 0.95
    eval_top_k: int = 20
    # Keep evaluation independent from the longer stochastic training budget.
    # Set this to None in Config to inherit max_completion_tokens.
    eval_max_completion_tokens: int | None = 26666
    require_complete_eval: bool = False

    # --- Rollout (per-turn tool loop + rubric judge) ------------------------
    max_turns: int = 6
    max_workspace_images: int = 6
    max_prompt_tokens: int = 57344
    tool_image_dim: int = 1024
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    judge_max_concurrency: int = DEFAULT_JUDGE_MAX_CONCURRENCY
    judge_timeout_s: float = DEFAULT_JUDGE_TIMEOUT_S
    critical_reward_weight: float = DEFAULT_CRITICAL_REWARD_WEIGHT

    # --- GRPO loss ----------------------------------------------------------
    kl_beta: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    grad_accumulation_normalization: GradAccNormalization = (
        GradAccNormalization.NUM_LOSS_TOKENS
    )

    # --- R3 / Router Replay -------------------------------------------------
    # Dense Qwen does not need expert-route replay. Enable this when switching
    # the example to a supported MoE policy.
    router_replay: bool = False
    router_replay_completion_only: bool = False

    # --- Authentication -----------------------------------------------------
    api_key: str = field(
        default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", "")
    )

    # --- Bookkeeping --------------------------------------------------------
    checkpoint_name: str = "vtb-serverless"
    final_checkpoint_name: str = "vtb-serverless-final"
    # Persist model + optimizer state after every N successful optimizer
    # updates. Sampler snapshots above are weights-only and cannot resume Adam.
    dcp_save_interval: int = 2
    sampling_timeout_s: float = 900.0
    run_dir: str = ""
    # Requires matplotlib; set False (or don't install it) to skip the plot.
    plot_reward_curve: bool = True
    # Optional W&B logging (set entity/project or WANDB_* env).
    wandb_entity: str = field(
        default_factory=lambda: os.environ.get("WANDB_ENTITY", "")
    )
    wandb_project: str = field(
        default_factory=lambda: os.environ.get("WANDB_PROJECT", "serverless-rl-vtb")
    )
    wandb_run_name: str = ""
    require_tool_aligned_data: bool = True


def _load_rows(
    path: Path,
    *,
    require_tool_aligned: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                alignment = row.get("tool_alignment")
                if require_tool_aligned and (
                    not isinstance(alignment, dict)
                    or alignment.get("eligible") is not True
                ):
                    raise ValueError(
                        f"{path}:{line_number} row {row.get('id', '<unknown>')} "
                        "is not four-image-tool aligned; set "
                        "tool_alignment.eligible=true only after verifying "
                        "the row uses supported image tools"
                    )
                rows.append(row)
    return rows


def _iter_epochs(rows: list[dict[str, Any]], epochs: int, shuffle: bool, seed: int):
    """Yield row lists per epoch, optionally shuffled with a per-epoch seed."""
    import random

    for epoch in range(max(1, epochs)):
        order = list(range(len(rows)))
        if shuffle:
            random.Random(seed + epoch).shuffle(order)
        yield [rows[i] for i in order]


def _iter_training_batches(
    rows: list[dict[str, Any]],
    *,
    epochs: int,
    shuffle: bool,
    seed: int,
    batch_size: int,
):
    """Batch complete epochs without dropping rows at epoch boundaries."""
    pending: list[dict[str, Any]] = []
    for epoch_rows in _iter_epochs(rows, epochs, shuffle, seed):
        pending.extend(epoch_rows)
        while len(pending) >= batch_size:
            yield pending[:batch_size]
            del pending[:batch_size]
    if pending:
        yield pending


def _inference_kld(
    policy_logprobs: list[float],
    raw_inference_logprobs: list[float],
    *,
    response_start: int,
) -> list[float]:
    """Trainer-policy vs raw sampler KL: ``exp(d) - d - 1``.

    Both inputs live in the SAME shifted whole-sequence index space, so both
    must be sliced at ``response_start`` (this mirrors the canonical slicing in
    ``training/utils/rl/observability.py``: ``raw_inf_lp[response_start:...]``).

    This intentionally uses ``PromptGroup.raw_inf_logprobs``. Sampling
    logprobs are the behavior-policy stream used by PPO/TIS and can include
    sampling transformations; substituting them into this diagnostic measures
    a different quantity.
    """
    out: list[float] = []
    resp_policy = policy_logprobs[response_start:]
    resp_inference = raw_inference_logprobs[response_start:]
    for pi_lp, inf_lp in zip(resp_policy, resp_inference):
        d = float(pi_lp) - float(inf_lp)
        out.append(math.exp(d) - d - 1.0)
    return out


def _extract_output_logprobs(fwd_output: Any) -> list[list[float]]:
    rows: list[list[float]] = []
    for out in getattr(fwd_output, "loss_fn_outputs", None) or []:
        lp = (
            out.get("logprobs")
            if isinstance(out, dict)
            else getattr(out, "logprobs", None)
        )
        data = lp.get("data") if isinstance(lp, dict) else getattr(lp, "data", None)
        rows.append([float(x) for x in (data or [])])
    return rows


def _validate_eval_completeness(
    *,
    label: str,
    required: bool,
    returned_episodes: int,
    expected_episodes: int,
    truncated_samples: int,
    prompt_budget_exhaustions: int,
) -> None:
    if not required or (
        returned_episodes == expected_episodes
        and truncated_samples == 0
        and prompt_budget_exhaustions == 0
    ):
        return
    raise RuntimeError(
        f"eval {label} was incomplete: returned {returned_episodes}/"
        f"{expected_episodes} episodes, length-truncated sampling calls="
        f"{truncated_samples}, prompt-budget exhaustions="
        f"{prompt_budget_exhaustions}"
    )


def _maybe_init_wandb(cfg: Config, run_dir: Path) -> Any:
    entity = (cfg.wandb_entity or "").strip()
    project = (cfg.wandb_project or "").strip() or "serverless-rl-vlm"
    if not entity or not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
    except Exception:
        print("wandb not installed; skipping W&B logging", flush=True)
        return None
    run = wandb.init(
        entity=entity,
        project=project,
        name=cfg.wandb_run_name or f"vtb-serverless-{run_dir.name}",
        config={
            "base_model": cfg.base_model,
            "tokenizer_model": cfg.tokenizer_model,
            "dataset": cfg.dataset,
            "eval_dataset": cfg.eval_dataset or None,
            "lora_rank": cfg.lora_rank,
            "steps": cfg.steps,
            "epochs": cfg.epochs,
            "group_size": cfg.group_size,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "eval_interval": cfg.eval_interval,
            "eval_upfront": cfg.eval_upfront,
            "eval_at_end": cfg.eval_at_end,
            "eval_group_size": cfg.eval_group_size,
            "eval_temperature": cfg.eval_temperature,
            "eval_top_p": cfg.eval_top_p,
            "eval_top_k": cfg.eval_top_k,
            "eval_max_completion_tokens": (
                cfg.eval_max_completion_tokens or cfg.max_completion_tokens
            ),
            "require_complete_eval": cfg.require_complete_eval,
            "dcp_save_interval": cfg.dcp_save_interval,
            "max_turns": cfg.max_turns,
            "max_completion_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "top_p": 1.0,
            "top_k": 0,
            "learning_rate": cfg.learning_rate,
            "adam_beta2": cfg.adam_beta2,
            "adam_eps": cfg.adam_eps,
            "adam_weight_decay": cfg.adam_weight_decay,
            "judge_model": cfg.judge_model,
            "judge_max_tokens": cfg.judge_max_tokens,
            "judge_max_concurrency": cfg.judge_max_concurrency,
            "judge_timeout_s": cfg.judge_timeout_s,
            "critical_reward_weight": cfg.critical_reward_weight,
            "router_replay": cfg.router_replay,
            "router_replay_completion_only": cfg.router_replay_completion_only,
            "grad_accumulation_normalization": (
                cfg.grad_accumulation_normalization.value
            ),
            "loss": "client_grpo",
        },
    )
    wandb.define_metric("rollout/*", step_metric="train/step")
    wandb.define_metric("eval/*", step_metric="train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("eval/return_ratio", step_metric="train/step", summary="min")
    wandb.define_metric("eval/truncated_ratio", step_metric="train/step", summary="max")
    wandb.define_metric("kld/*", step_metric="train/step")
    print(f"W&B: {run.url}", flush=True)
    return run


def _log_wandb_step(step: int, rec: dict[str, Any]) -> None:
    try:
        import wandb

        if wandb.run is None:
            return
        payload: dict[str, float] = {"train/step": int(step)}
        for key, value in rec.items():
            if (
                key != "train/step"
                and key.startswith(("rollout/", "train/", "kld/"))
                and isinstance(value, (int, float))
            ):
                payload[key] = float(value)
        kld = rec.get("train/inference_kld")
        if isinstance(kld, (int, float)):
            payload["kld/inference_step_mean"] = float(kld)
            payload["train/inference_kld"] = float(kld)
        if isinstance(rec.get("kld/max"), (int, float)):
            payload["kld/max"] = float(rec["kld/max"])
        wandb.log(payload, step=int(step))
    except Exception as exc:  # observability only
        print(f"wandb log skipped: {exc}", flush=True)


def _log_wandb_eval(completed_steps: int, rec: dict[str, Any]) -> None:
    try:
        import wandb

        if wandb.run is None:
            return
        payload: dict[str, float] = {"train/step": int(completed_steps)}
        for key, value in rec.items():
            if key.startswith("eval/") and isinstance(value, (int, float)):
                payload[key] = float(value)
        wandb.log(payload, step=int(completed_steps))
    except Exception as exc:  # observability only
        print(f"wandb eval log skipped: {exc}", flush=True)


class ServerlessVisualToolbenchRL:
    """One serverless multimodal RL run over VisualToolBench."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        if cfg.lora_rank <= 0:
            raise ValueError("serverless training requires lora_rank > 0")
        if cfg.max_seq_len <= 0:
            raise ValueError("serverless training requires max_seq_len > 0")
        if cfg.steps <= 0 or cfg.prompt_groups_per_step <= 0 or cfg.group_size < 2:
            raise ValueError(
                "steps and prompt_groups_per_step must be positive and "
                "group_size must be at least 2"
            )
        if cfg.rollout_concurrency < cfg.group_size:
            raise ValueError("rollout_concurrency must be at least group_size")
        if cfg.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        if (
            cfg.eval_max_completion_tokens is not None
            and cfg.eval_max_completion_tokens <= 0
        ):
            raise ValueError("eval_max_completion_tokens must be positive")
        if cfg.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        largest_completion_budget = max(
            cfg.max_completion_tokens,
            cfg.eval_max_completion_tokens or cfg.max_completion_tokens,
        )
        if cfg.max_prompt_tokens + largest_completion_budget > cfg.max_seq_len:
            raise ValueError(
                "max_prompt_tokens plus the largest completion budget must fit "
                f"max_seq_len ({cfg.max_prompt_tokens} + "
                f"{largest_completion_budget} > {cfg.max_seq_len})"
            )
        if not 0.0 < cfg.adam_beta2 < 1.0:
            raise ValueError("adam_beta2 must be between 0 and 1")
        if cfg.adam_eps <= 0.0:
            raise ValueError("adam_eps must be positive")
        if cfg.adam_weight_decay < 0.0:
            raise ValueError("adam_weight_decay must be non-negative")
        if cfg.eval_temperature < 0:
            raise ValueError("eval_temperature must be non-negative")
        if not 0.0 < cfg.eval_top_p <= 1.0:
            raise ValueError("eval_top_p must be in (0, 1]")
        if cfg.eval_top_k < 0:
            raise ValueError("eval_top_k must be non-negative")
        if cfg.eval_interval < 0:
            raise ValueError("eval_interval must be non-negative")
        if cfg.eval_group_size <= 0:
            raise ValueError("eval_group_size must be positive")
        if cfg.dcp_save_interval < 0:
            raise ValueError("dcp_save_interval must be non-negative")
        eval_requested = cfg.eval_upfront or cfg.eval_at_end or cfg.eval_interval > 0
        if eval_requested and not cfg.eval_dataset:
            raise ValueError("--eval-dataset is required when evaluation is enabled")
        _validate_checkpoint_prefix(cfg.checkpoint_name, "checkpoint_name")
        _validate_checkpoint_prefix(cfg.final_checkpoint_name, "final_checkpoint_name")
        validate_grpo_config(
            kl_beta=cfg.kl_beta,
            eps_clip=cfg.eps_clip,
            eps_clip_high=cfg.eps_clip_high,
            reference_training_shape_id=None,
            reference_job_id=None,
            anchor_logp="old_policy",
        )
        self.rows = _load_rows(
            Path(cfg.dataset),
            require_tool_aligned=cfg.require_tool_aligned_data,
        )
        if not self.rows:
            raise SystemExit(f"dataset is empty: {cfg.dataset}")
        self.eval_rows = (
            _load_rows(
                Path(cfg.eval_dataset),
                require_tool_aligned=cfg.require_tool_aligned_data,
            )
            if cfg.eval_dataset
            else []
        )
        overlap = {str(row.get("id", "")) for row in self.rows} & {
            str(row.get("id", "")) for row in self.eval_rows
        }
        if overlap:
            first = sorted(overlap)[0]
            raise ValueError(
                f"train/eval datasets overlap on {len(overlap)} row ids; "
                f"first overlap: {first}"
            )
        if eval_requested and not self.eval_rows:
            raise ValueError(f"eval dataset is empty: {cfg.eval_dataset}")
        available_rows = len(self.rows) * max(1, cfg.epochs)
        available_steps = math.ceil(available_rows / cfg.prompt_groups_per_step)
        if cfg.steps > available_steps:
            raise ValueError(
                f"{cfg.steps} steps were requested but {available_rows} "
                f"row-epochs produce only {available_steps} batches at "
                f"prompt_groups_per_step={cfg.prompt_groups_per_step}"
            )
        if not cfg.tokenizer_model:
            raise ValueError("--tokenizer-model is required")

        self.tokenizer = get_tokenizer(cfg.tokenizer_model)
        self.run_dir = (
            Path(cfg.run_dir).resolve()
            if cfg.run_dir
            else Path("/tmp") / f"serverless_vtb_{int(time.time())}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.dcp_metrics_path = self.run_dir / "dcp_metrics.jsonl"
        self.completions_dir = self.run_dir / "completions"
        self.completions_dir.mkdir(parents=True, exist_ok=True)
        self.eval_metrics_path = self.run_dir / "eval_metrics.jsonl"
        self.eval_completions_dir = self.run_dir / "eval_completions"
        if self.eval_rows:
            self.eval_completions_dir.mkdir(parents=True, exist_ok=True)
        self._wandb = _maybe_init_wandb(cfg, self.run_dir)

        # The one connection that gives us BOTH training and (per-step)
        # sampling clients. No trainer job, no deployment -- pooled serverless.
        self.service = FiretitanServiceClient(
            api_key=cfg.api_key,
            base_url=FIREWORKS_SERVERLESS_BASE_URL,
        )
        self.training_client = self.service.create_lora_training_client(
            base_model=cfg.base_model,
            rank=cfg.lora_rank,
        )
        self.router_replay_enabled = resolve_router_replay_enabled(
            requested=cfg.router_replay,
            api_key=cfg.api_key,
            base_url=FIREWORKS_API_BASE_URL,
            additional_headers=None,
            base_model=cfg.base_model,
        )

        session = getattr(self.service, "training_session_name", None) or getattr(
            self.service, "training_session_id", None
        )
        run_id = getattr(self.training_client, "run_id", None)
        self.session = str(session or "")
        self.run_id = str(run_id or "")
        manifest = {
            "session": self.session,
            "run_id": self.run_id,
            "base_model": cfg.base_model,
            "tokenizer_model": cfg.tokenizer_model,
            "renderer": cfg.renderer_name,
            "dataset": str(Path(cfg.dataset).resolve()),
            "dataset_rows": len(self.rows),
            "eval_dataset": (
                str(Path(cfg.eval_dataset).resolve()) if cfg.eval_dataset else None
            ),
            "eval_dataset_rows": len(self.eval_rows),
            "eval_interval": cfg.eval_interval,
            "eval_upfront": cfg.eval_upfront,
            "eval_at_end": cfg.eval_at_end,
            "eval_group_size": cfg.eval_group_size,
            "eval_temperature": cfg.eval_temperature,
            "eval_top_p": cfg.eval_top_p,
            "eval_top_k": cfg.eval_top_k,
            "eval_max_completion_tokens": (
                cfg.eval_max_completion_tokens or cfg.max_completion_tokens
            ),
            "require_complete_eval": cfg.require_complete_eval,
            "epochs": cfg.epochs,
            "steps": cfg.steps,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "group_size": cfg.group_size,
            "rollout_concurrency": cfg.rollout_concurrency,
            "max_completion_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "top_p": 1.0,
            "top_k": 0,
            "max_seq_len": cfg.max_seq_len,
            "lora_rank": cfg.lora_rank,
            "learning_rate": cfg.learning_rate,
            "adam_beta2": cfg.adam_beta2,
            "adam_eps": cfg.adam_eps,
            "adam_weight_decay": cfg.adam_weight_decay,
            "grad_accumulation_normalization": (
                cfg.grad_accumulation_normalization.value
            ),
            "router_replay": self.router_replay_enabled,
            "router_replay_completion_only": cfg.router_replay_completion_only,
            "judge_model": cfg.judge_model,
            "judge_max_tokens": cfg.judge_max_tokens,
            "judge_max_concurrency": cfg.judge_max_concurrency,
            "judge_timeout_s": cfg.judge_timeout_s,
            "critical_reward_weight": cfg.critical_reward_weight,
            "dcp_save_interval": cfg.dcp_save_interval,
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"connected serverless session={session} run={run_id}\n"
            f"base_model={cfg.base_model} tokenizer={cfg.tokenizer_model} "
            f"renderer={cfg.renderer_name}\n"
            f"steps={cfg.steps} prompt_groups_per_step={cfg.prompt_groups_per_step} "
            f"group_size={cfg.group_size} max_turns={cfg.max_turns} "
            f"epochs={cfg.epochs} "
            f"lora_rank={cfg.lora_rank} max_seq_len={cfg.max_seq_len} "
            f"lr={cfg.learning_rate} max_completion_tokens="
            f"{cfg.max_completion_tokens} temperature={cfg.temperature} "
            "top_p=1.0 top_k=0\n"
            f"eval_rows={len(self.eval_rows)} eval_interval={cfg.eval_interval} "
            f"eval_upfront={cfg.eval_upfront} eval_at_end={cfg.eval_at_end} "
            f"eval_group_size={cfg.eval_group_size} "
            f"eval_temperature={cfg.eval_temperature} "
            f"eval_top_p={cfg.eval_top_p} eval_top_k={cfg.eval_top_k} "
            "eval_max_completion_tokens="
            f"{cfg.eval_max_completion_tokens or cfg.max_completion_tokens} "
            f"require_complete_eval={cfg.require_complete_eval}\n"
            f"adam_beta2={cfg.adam_beta2} adam_eps={cfg.adam_eps} "
            f"adam_weight_decay={cfg.adam_weight_decay}\n"
            f"dcp_save_interval={cfg.dcp_save_interval}\n"
            f"router_replay={self.router_replay_enabled} "
            f"completion_only={cfg.router_replay_completion_only} "
            f"grad_normalization={cfg.grad_accumulation_normalization.value}\n"
            "training_shape_id=None region=None (pooled session; no provisioning)\n"
            f"run_dir={self.run_dir}",
            flush=True,
        )

    def _rollout_setup(
        self,
        snapshot: str,
        *,
        sampler: Any,
        event_sink,
        completions_per_prompt: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_completion_tokens: int | None = None,
        include_router_replay: bool = True,
    ) -> RolloutSetup:
        """Per-step rollout deps: sampler model = the just-saved snapshot."""
        cfg = self.cfg
        completion_count = completions_per_prompt or cfg.group_size
        sample_kwargs: dict[str, Any] = {
            "max_tokens": (
                cfg.max_completion_tokens
                if max_completion_tokens is None
                else max_completion_tokens
            ),
            "temperature": cfg.temperature if temperature is None else temperature,
            "top_p": 1.0 if top_p is None else top_p,
            "top_k": 0 if top_k is None else top_k,
            "http_timeout": cfg.sampling_timeout_s,
        }
        if self.router_replay_enabled and include_router_replay:
            sample_kwargs.update(
                include_routing_matrix=True,
                echo=not cfg.router_replay_completion_only,
            )
        return RolloutSetup(
            tokenizer=self.tokenizer,
            tokenizer_id=cfg.tokenizer_model,
            sample_kwargs=sample_kwargs,
            inference_base_url=FIREWORKS_SERVERLESS_BASE_URL,
            api_key=cfg.api_key,
            # The injected sampler is already session-bound to ``snapshot``.
            # Keep the public checkpoint path as the rollout's model identity
            # instead of reaching into a private SDK name-resolution helper.
            model=snapshot,
            completions_per_prompt=completion_count,
            sampler=sampler,
            extras={
                "max_turns": cfg.max_turns,
                "max_workspace_images": cfg.max_workspace_images,
                "max_prompt_tokens": cfg.max_prompt_tokens,
                "tool_image_dim": cfg.tool_image_dim,
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "judge_model": cfg.judge_model,
                "judge_max_tokens": cfg.judge_max_tokens,
                "judge_max_concurrency": cfg.judge_max_concurrency,
                "judge_timeout_s": cfg.judge_timeout_s,
                "critical_reward_weight": cfg.critical_reward_weight,
                "renderer_name": cfg.renderer_name,
                "filter_truncated_rollouts": cfg.filter_truncated_rollouts,
                "event_sink": event_sink,
            },
        )

    async def _collect_runs(
        self,
        setup: RolloutSetup,
        batch_rows: list[dict[str, Any]],
        *,
        completions_per_prompt: int | None = None,
    ) -> list[list[Any]]:
        """Roll out completions per prompt, with individual calls concurrency-capped."""
        # Build the rollout fn (and its DeploymentSampler + RubricJudge) inside
        # the running event loop so async clients bind to this loop.
        rollout_fn = make_rollout_fn(setup)
        sem = asyncio.Semaphore(max(1, self.cfg.rollout_concurrency))
        completion_count = completions_per_prompt or self.cfg.group_size

        async def one(row: dict[str, Any]) -> Any:
            async with sem:
                return await rollout_fn(dict(row))

        flat = [one(row) for row in batch_rows for _ in range(completion_count)]
        try:
            done = await asyncio.gather(*flat)
            grouped: list[list[Any]] = []
            idx = 0
            for _ in batch_rows:
                grouped.append(
                    [r for r in done[idx : idx + completion_count] if r is not None]
                )
                idx += completion_count
            return grouped
        finally:
            close = getattr(rollout_fn, "close", None)
            if callable(close):
                result = close()
                if asyncio.iscoroutine(result):
                    await result

    def _evaluate(self, *, completed_steps: int, label: str) -> dict[str, Any]:
        """Evaluate the current policy snapshot without any trainer update."""
        t0 = time.time()
        cfg = self.cfg
        suffix = "b0000" if label == "baseline" else f"e{completed_steps:04d}"
        save_name = f"{cfg.checkpoint_name}-{suffix}"
        snapshot = (
            self.training_client.save_weights_for_sampler(save_name).result().path
        )
        if not snapshot:
            raise RuntimeError(
                f"save_weights_for_sampler({save_name!r}) returned no path"
            )

        sampling_client = self.service.create_sampling_client(
            model_path=snapshot, tokenizer=self.tokenizer
        )
        rollout_events: list[dict[str, Any]] = []
        setup = self._rollout_setup(
            snapshot,
            sampler=sampling_client.deployment_sampler,
            event_sink=rollout_events.append,
            completions_per_prompt=cfg.eval_group_size,
            temperature=cfg.eval_temperature,
            top_p=cfg.eval_top_p,
            top_k=cfg.eval_top_k,
            max_completion_tokens=(
                cfg.eval_max_completion_tokens or cfg.max_completion_tokens
            ),
            include_router_replay=False,
        )
        try:
            grouped_runs = asyncio.run(
                self._collect_runs(
                    setup,
                    self.eval_rows,
                    completions_per_prompt=cfg.eval_group_size,
                )
            )
        finally:
            try:
                sampling_client.close()
            except Exception:
                pass

        rewards: list[float] = []
        official_scores: list[float] = []
        critical_fractions: list[float] = []
        judge_passes: list[bool] = []
        tool_call_counts: list[int] = []
        episode_records: list[dict[str, Any]] = []
        category_scores: dict[str, list[float]] = {}
        focus_scores: dict[str, list[float]] = {}

        for row, runs in zip(self.eval_rows, grouped_runs):
            for run in runs:
                meta = run.metadata or {}
                reward = float(run.segments[0].reward)
                official_score = float(meta.get("mean_official_score", 0.0))
                critical_fraction = float(meta.get("mean_critical_fraction", 0.0))
                judge_passed = bool(meta.get("judge_passed"))
                num_tool_calls = int(meta.get("num_tool_calls", 0))
                rewards.append(reward)
                official_scores.append(official_score)
                critical_fractions.append(critical_fraction)
                judge_passes.append(judge_passed)
                tool_call_counts.append(num_tool_calls)
                category = str(row.get("category", "") or "missing")
                eval_focus = str(row.get("eval_focus", "") or "missing")
                category_scores.setdefault(category, []).append(official_score)
                focus_scores.setdefault(eval_focus, []).append(official_score)
                episode_records.append(
                    {
                        "label": label,
                        "completed_steps": completed_steps,
                        "row_id": row.get("id", ""),
                        "category": category,
                        "eval_focus": eval_focus,
                        "prompt": str(row.get("prompt", ""))[:200],
                        "reward": reward,
                        "official_score": official_score,
                        "critical_fraction": critical_fraction,
                        "judge_passed": judge_passed,
                        "num_turns": int(meta.get("num_turns", 0)),
                        "num_tool_calls": num_tool_calls,
                        "final_answer": (run.segments[-1].text if run.segments else "")[
                            :1000
                        ],
                        "snapshot": snapshot,
                    }
                )

        sample_events = [e for e in rollout_events if e.get("event") == "sample"]
        completion_lens = [
            int(e["completion_tokens"])
            for e in sample_events
            if isinstance(e.get("completion_tokens"), int)
        ]
        truncated_samples = sum(
            str(e.get("finish_reason", "")).lower() == "length" for e in sample_events
        )
        prompt_budget_exhaustions = sum(
            e.get("event") == "prompt_budget_exhausted" for e in rollout_events
        )
        judge_events = [e for e in rollout_events if e.get("event") == "judge"]
        judge_latencies = [
            float(e["latency_s"])
            for e in judge_events
            if isinstance(e.get("latency_s"), (int, float))
        ]
        expected_episodes = len(self.eval_rows) * cfg.eval_group_size
        category_expected = Counter(
            str(row.get("category", "") or "missing") for row in self.eval_rows
        )
        focus_expected = Counter(
            str(row.get("eval_focus", "") or "missing") for row in self.eval_rows
        )

        def mean(values):
            return statistics.fmean(values) if values else 0.0

        def fixed_mean(values, expected):
            return sum(values) / expected if expected else 0.0

        returned_reward = mean(rewards)
        returned_official_score = mean(official_scores)

        rec: dict[str, Any] = {
            "label": label,
            "completed_steps": completed_steps,
            "train/step": completed_steps,
            "snapshot": snapshot,
            # Missing or truncated episodes count as zero so every checkpoint
            # uses the same denominator. Keep returned-only means separately
            # for diagnosing sampling failures without biasing the main curve.
            "eval/reward": fixed_mean(rewards, expected_episodes),
            "eval/reward_returned": returned_reward,
            "eval/official_score": fixed_mean(official_scores, expected_episodes),
            "eval/official_score_returned": returned_official_score,
            "eval/critical_fraction": fixed_mean(critical_fractions, expected_episodes),
            "eval/judge_pass": fixed_mean(judge_passes, expected_episodes),
            "eval/mean_tool_calls": mean(tool_call_counts),
            "eval/rows": len(self.eval_rows),
            "eval/returned_episodes": len(rewards),
            "eval/return_ratio": (
                len(rewards) / expected_episodes if expected_episodes else 0.0
            ),
            "eval/completion_len/mean": mean(completion_lens),
            "eval/truncated_ratio": (
                truncated_samples / len(sample_events) if sample_events else 0.0
            ),
            "eval/prompt_budget_exhaustions": prompt_budget_exhaustions,
            "eval/judge_latency_s": mean(judge_latencies),
            "eval/judge_failures": sum(e.get("success") is False for e in judge_events),
            "perf/eval_wall_time": time.time() - t0,
        }
        for category, expected_rows in category_expected.items():
            rec[f"eval/category/{category}/official_score"] = fixed_mean(
                category_scores.get(category, []), expected_rows * cfg.eval_group_size
            )
        for eval_focus, expected_rows in focus_expected.items():
            rec[f"eval/focus/{eval_focus}/official_score"] = fixed_mean(
                focus_scores.get(eval_focus, []), expected_rows * cfg.eval_group_size
            )

        with self.eval_metrics_path.open("a") as handle:
            handle.write(json.dumps(rec) + "\n")
        completions_path = self.eval_completions_dir / f"{label}.jsonl"
        with completions_path.open("w") as handle:
            for episode in episode_records:
                handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
        print(
            f"eval {label} after_steps={completed_steps} "
            f"reward={rec['eval/reward']:.3f} "
            f"reward_returned={returned_reward:.3f} "
            f"official={rec['eval/official_score']:.3f} "
            f"critical={rec['eval/critical_fraction']:.3f} "
            f"judge_pass={rec['eval/judge_pass']:.3f} "
            f"episodes={len(rewards)}/{expected_episodes} "
            f"completion_len={rec['eval/completion_len/mean']:.1f} "
            f"truncated={rec['eval/truncated_ratio']:.3f} "
            f"prompt_budget_exhaustions={prompt_budget_exhaustions} "
            f"elapsed={rec['perf/eval_wall_time']:.1f}s",
            flush=True,
        )
        _log_wandb_eval(completed_steps, rec)
        _validate_eval_completeness(
            label=label,
            required=cfg.require_complete_eval,
            returned_episodes=len(rewards),
            expected_episodes=expected_episodes,
            truncated_samples=truncated_samples,
            prompt_budget_exhaustions=prompt_budget_exhaustions,
        )
        return rec

    def _save_dcp(
        self,
        *,
        completed_steps: int,
        trained_steps: int,
        final: bool = False,
    ) -> str:
        """Persist resumable model + optimizer state after an optimizer update."""
        started = time.time()
        name = f"{self.cfg.checkpoint_name}-d{trained_steps:04d}s{completed_steps:04d}"
        saved = self.training_client.save_state(name).result()
        path = str(getattr(saved, "path", "") or "")
        if not path:
            raise RuntimeError(f"save_state({name!r}) returned no path")

        rec = {
            "completed_steps": completed_steps,
            "trained_steps": trained_steps,
            "name": name,
            "path": path,
            "final": final,
            "perf/dcp_wall_time": time.time() - started,
        }
        with self.dcp_metrics_path.open("a") as handle:
            handle.write(json.dumps(rec) + "\n")
        print(
            f"DCP saved after_steps={completed_steps} "
            f"trained_steps={trained_steps} path={path} "
            f"elapsed={rec['perf/dcp_wall_time']:.1f}s",
            flush=True,
        )
        return path

    def _step(self, step: int, batch_rows: list[dict[str, Any]]) -> dict[str, Any]:
        t0 = time.time()
        cfg = self.cfg

        # 1. Save the current LoRA weights and create a sampling client for
        #    that snapshot.
        save_name = f"{cfg.checkpoint_name}-{step:04d}"
        snapshot = (
            self.training_client.save_weights_for_sampler(save_name).result().path
        )
        if not snapshot:
            raise RuntimeError(
                f"save_weights_for_sampler({save_name!r}) returned no path"
            )
        sampling_client = self.service.create_sampling_client(
            model_path=snapshot, tokenizer=self.tokenizer
        )
        rollout_events: list[dict[str, Any]] = []
        setup = self._rollout_setup(
            snapshot,
            sampler=sampling_client.deployment_sampler,
            event_sink=rollout_events.append,
        )
        try:
            grouped_runs = asyncio.run(self._collect_runs(setup, batch_rows))
        finally:
            try:
                sampling_client.close()
            except Exception:
                pass

        # 2. Build GRPO prompt groups; drop constant-reward groups (zero
        #    advantage, no learning signal).
        all_data: list[Any] = []
        all_advantages: list[float] = []
        # Sampling logprobs drive PPO/TIS. Raw inference logprobs are a
        # separate stream used only for trainer-vs-inference KLD.
        all_inf_logprobs: list[list[float]] = []
        all_raw_inf_logprobs: list[list[float]] = []
        all_prompt_lens: list[int] = []
        raw_rewards: list[float] = []
        filtered_rewards: list[float] = []
        official_scores: list[float] = []
        critical_fractions: list[float] = []
        judge_passes: list[bool] = []
        tool_call_counts: list[int] = []
        completion_lens: list[int] = []
        episode_records: list[dict[str, Any]] = []
        degenerate = 0
        returned_episodes = 0

        for row, runs in zip(batch_rows, grouped_runs):
            returned_episodes += len(runs)
            if not runs:
                degenerate += 1
                continue
            rewards = [float(r.segments[0].reward) for r in runs]
            raw_rewards.extend(rewards)
            for r in runs:
                meta = r.metadata or {}
                official_scores.append(float(meta.get("mean_official_score", 0.0)))
                critical_fractions.append(
                    float(meta.get("mean_critical_fraction", 0.0))
                )
                judge_passes.append(bool(meta.get("judge_passed")))
                tool_call_counts.append(int(meta.get("num_tool_calls", 0)))
                episode_records.append(
                    {
                        "step": step,
                        "row_id": row.get("id", ""),
                        "prompt": str(row.get("prompt", ""))[:200],
                        "reward": float(r.segments[0].reward),
                        "official_score": float(meta.get("mean_official_score", 0.0)),
                        "critical_fraction": float(
                            meta.get("mean_critical_fraction", 0.0)
                        ),
                        "judge_passed": bool(meta.get("judge_passed")),
                        "num_turns": int(meta.get("num_turns", 0)),
                        "num_tool_calls": int(meta.get("num_tool_calls", 0)),
                        "final_answer": (r.segments[-1].text if r.segments else "")[
                            :1000
                        ],
                        "snapshot": snapshot,
                    }
                )
            if cfg.filter_constant_reward and len(set(rewards)) <= 1:
                degenerate += 1
                continue
            pg = rollout_to_prompt_group(
                Rollout(runs=runs, row_meta={"id": row.get("id", "")}),
                router_replay_completion_only=(
                    self.router_replay_enabled and cfg.router_replay_completion_only
                ),
            )
            if pg is None:
                degenerate += 1
                continue
            all_data.extend(pg.data)
            all_advantages.extend(pg.advantages)
            all_inf_logprobs.extend(pg.inf_logprobs)
            all_raw_inf_logprobs.extend(pg.raw_inf_logprobs)
            prompt_lens = pg.prompt_lens or [pg.prompt_len] * len(pg.data)
            all_prompt_lens.extend(prompt_lens)
            filtered_rewards.extend(pg.rewards)
            completion_lens.extend(pg.completion_lens)

        # 3. One client-GRPO update + optimizer step.
        trained = False
        loss = None
        kld_values: list[float] = []
        train_diagnostics: dict[str, float] = {}
        if all_data:
            # Cross-entropy forward yields the pre-update policy logprobs used
            # by the client-side GRPO closure.
            old_fwd = self.training_client.forward(all_data, "cross_entropy")
            if hasattr(old_fwd, "result"):
                old_fwd = old_fwd.result()
            old_policy_lp = _extract_output_logprobs(old_fwd)
            loss_fn = make_grpo_loss_fn(
                advantages=all_advantages,
                ref_logprobs=old_policy_lp,  # kl_beta=0 disables the ref term
                prompt_len=all_prompt_lens,
                inf_logprobs=all_inf_logprobs,
                old_policy_logprobs=old_policy_lp,
                kl_beta=cfg.kl_beta,
                eps_clip=cfg.eps_clip,
                eps_clip_high=cfg.eps_clip_high,
                tis_config=TISConfig(),
                raw_inf_logprobs=all_raw_inf_logprobs,
            )
            fb = self.training_client.forward_backward_custom(all_data, loss_fn=loss_fn)
            if hasattr(fb, "result"):
                fb = fb.result()
            trained = True
            loss = _mean_loss(fb)
            train_diagnostics.update(_extract_train_diagnostics(fb))
            # Compare the trainer's pre-update logits with the sampler's raw
            # logits. The behavior stream above remains reserved for PPO/TIS.
            for policy_lps, raw_inference_lps, plen in zip(
                old_policy_lp, all_raw_inf_logprobs, all_prompt_lens
            ):
                kld_values.extend(
                    _inference_kld(
                        policy_lps,
                        raw_inference_lps,
                        response_start=max(0, plen - 1),
                    )
                )
            adam = tinker.AdamParams(
                learning_rate=cfg.learning_rate,
                beta1=0.9,
                beta2=cfg.adam_beta2,
                eps=cfg.adam_eps,
                weight_decay=cfg.adam_weight_decay,
            )
            optim_result = self.training_client.optim_step(
                adam,
                grad_accumulation_normalization=(cfg.grad_accumulation_normalization),
            ).result()
            add_optimizer_metrics(train_diagnostics, optim_result)

        raw_reward = sum(raw_rewards) / len(raw_rewards) if raw_rewards else 0.0
        filtered_reward = (
            sum(filtered_rewards) / len(filtered_rewards) if filtered_rewards else 0.0
        )
        official_score = (
            sum(official_scores) / len(official_scores) if official_scores else 0.0
        )
        critical_fraction = (
            sum(critical_fractions) / len(critical_fractions)
            if critical_fractions
            else 0.0
        )
        judge_pass = sum(judge_passes) / len(judge_passes) if judge_passes else 0.0
        mean_tool_calls = (
            sum(tool_call_counts) / len(tool_call_counts) if tool_call_counts else 0.0
        )
        finite_kld = [v for v in kld_values if math.isfinite(v)]
        kld = sum(finite_kld) / len(finite_kld) if finite_kld else None
        sample_events = [e for e in rollout_events if e.get("event") == "sample"]
        sampled_completion_lens = [
            int(e["completion_tokens"])
            for e in sample_events
            if isinstance(e.get("completion_tokens"), int)
        ]
        truncated_samples = sum(
            str(e.get("finish_reason", "")).lower() == "length" for e in sample_events
        )
        prompt_budget_exhaustions = sum(
            e.get("event") == "prompt_budget_exhausted" for e in rollout_events
        )
        judge_events = [e for e in rollout_events if e.get("event") == "judge"]
        judge_latencies = [
            float(e["latency_s"])
            for e in judge_events
            if isinstance(e.get("latency_s"), (int, float))
        ]
        expected_episodes = len(batch_rows) * cfg.group_size
        rec = {
            "step": step,
            "train/step": step,
            "snapshot": snapshot,
            "rollout/raw_reward": raw_reward,
            "rollout/filtered_reward": filtered_reward,
            "rollout/official_score": official_score,
            "rollout/critical_fraction": critical_fraction,
            "rollout/judge_pass": judge_pass,
            "rollout/mean_tool_calls": mean_tool_calls,
            "rollout/raw_samples": len(raw_rewards),
            "rollout/returned_episodes": returned_episodes,
            "rollout/return_ratio": (
                returned_episodes / expected_episodes if expected_episodes else 0.0
            ),
            "rollout/completion_len/mean": (
                statistics.fmean(sampled_completion_lens)
                if sampled_completion_lens
                else 0.0
            ),
            "rollout/truncated_ratio": (
                truncated_samples / len(sample_events) if sample_events else 0.0
            ),
            "rollout/prompt_budget_exhaustions": prompt_budget_exhaustions,
            "rollout/judge_latency_s": (
                statistics.fmean(judge_latencies) if judge_latencies else 0.0
            ),
            "rollout/judge_failures": sum(
                e.get("success") is False for e in judge_events
            ),
            "rollout/valid_prompt_groups": len(batch_rows) - degenerate,
            "rollout/filter_reject_ratio": degenerate / len(batch_rows)
            if batch_rows
            else 0.0,
            "train/loss": loss,
            "train/trained": trained,
            "train/num_loss_tokens": sum(completion_lens),
            "train/inference_kld": kld,
            "kld/token_count": len(finite_kld),
            "kld/max": max(finite_kld) if finite_kld else None,
            "perf/step_wall_time": time.time() - t0,
        }
        rec.update(train_diagnostics)
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        comp_path = self.completions_dir / f"step_{step:04d}.jsonl"
        with comp_path.open("w") as f:
            for c in episode_records:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(
            f"step {step:02d} reward={raw_reward:.3f} "
            f"official={official_score:.3f} critical={critical_fraction:.3f} "
            f"judge_pass={judge_pass:.3f} "
            f"tool_calls={mean_tool_calls:.2f} "
            f"valid_groups={len(batch_rows) - degenerate}/{len(batch_rows)} "
            f"episodes={len(raw_rewards)}/{len(batch_rows) * cfg.group_size} "
            f"completion_len={rec['rollout/completion_len/mean']:.1f} "
            f"loss_tokens={rec['train/num_loss_tokens']} "
            f"kld={('n/a' if kld is None else f'{kld:.6f}')} "
            f"kld_tokens={len(finite_kld)} "
            f"trained={trained} elapsed={rec['perf/step_wall_time']:.1f}s",
            flush=True,
        )
        _log_wandb_step(step, rec)
        return rec

    def run(self) -> list[dict[str, Any]]:
        cfg = self.cfg
        records: list[dict[str, Any]] = []
        last_eval_step: int | None = None
        trained_steps = 0
        last_dcp_trained_steps = 0
        if cfg.eval_upfront:
            self._evaluate(completed_steps=0, label="baseline")
            last_eval_step = 0

        batches = _iter_training_batches(
            self.rows,
            epochs=cfg.epochs,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
            batch_size=cfg.prompt_groups_per_step,
        )
        for step, batch in enumerate(batches):
            if step >= cfg.steps:
                break
            rec = self._step(step, batch)
            records.append(rec)
            completed_steps = step + 1
            if rec["train/trained"]:
                trained_steps += 1
                if (
                    cfg.dcp_save_interval > 0
                    and trained_steps % cfg.dcp_save_interval == 0
                ):
                    self._save_dcp(
                        completed_steps=completed_steps,
                        trained_steps=trained_steps,
                    )
                    last_dcp_trained_steps = trained_steps
            if cfg.eval_interval and completed_steps % cfg.eval_interval == 0:
                self._evaluate(
                    completed_steps=completed_steps,
                    label=f"step_{completed_steps:04d}",
                )
                last_eval_step = completed_steps

        # Do not leave the final successful update unprotected when the run
        # length is not an exact multiple of the periodic cadence.
        if (
            cfg.dcp_save_interval > 0
            and trained_steps > 0
            and trained_steps != last_dcp_trained_steps
        ):
            self._save_dcp(
                completed_steps=len(records),
                trained_steps=trained_steps,
                final=True,
            )

        final = self.training_client.save_weights_for_sampler(
            cfg.final_checkpoint_name
        ).result()
        print(f"final sampler checkpoint: {getattr(final, 'path', None)}", flush=True)

        if len(records) != cfg.steps:
            raise RuntimeError(
                f"requested {cfg.steps} steps but dataset produced {len(records)}"
            )
        if cfg.eval_at_end and last_eval_step != len(records):
            self._evaluate(completed_steps=len(records), label="final")
        if records:
            rewards = [r["rollout/raw_reward"] for r in records]
            trained_steps = sum(bool(r["train/trained"]) for r in records)
            klds = [
                float(r["train/inference_kld"])
                for r in records
                if isinstance(r.get("train/inference_kld"), (int, float))
            ]
            kld_summary = (
                f"kld_mean={statistics.fmean(klds):.6f} kld_max={max(klds):.6f}"
                if klds
                else "kld=n/a"
            )
            print(
                f"\nreward: {rewards[0]:.3f} -> {rewards[-1]:.3f} (peak {max(rewards):.3f}) "
                f"over {len(records)} steps; trained_steps={trained_steps}; "
                f"{kld_summary}",
                flush=True,
            )
        if cfg.plot_reward_curve:
            self._plot(records)
        print(f"metrics: {self.metrics_path}", flush=True)
        return records

    def _plot(self, records: list[dict[str, Any]]) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping reward curve", flush=True)
            return
        steps = [r["step"] for r in records]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            steps,
            [r["rollout/raw_reward"] for r in records],
            marker="o",
            label="raw_reward",
        )
        ax.plot(
            steps,
            [r["rollout/judge_pass"] for r in records],
            marker="s",
            linestyle="--",
            label="judge_pass",
        )
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("score")
        ax.set_ylim(bottom=0.0)
        ax.set_title(
            f"Serverless VisualToolBench RL ({self.cfg.base_model}, client-GRPO)"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        plot_path = self.run_dir / "reward_curve.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"reward curve: {plot_path}", flush=True)
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({"reward_curve": wandb.Image(str(plot_path))})
        except Exception:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisualToolBench GRPO on a pooled serverless trainer"
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument(
        "--eval-dataset",
        default="",
        help="Held-out aligned JSONL evaluated without backward/optimizer steps.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--renderer-name", default=DEFAULT_RENDERER_NAME)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--prompt-groups-per-step", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rollout-concurrency", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-seq-len", type=int, default=131072)
    parser.add_argument("--max-prompt-tokens", type=int, default=57344)
    parser.add_argument("--max-workspace-images", type=int, default=6)
    parser.add_argument("--tool-image-dim", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-12)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help="Evaluate after every N completed optimizer steps (0 disables).",
    )
    parser.add_argument(
        "--eval-upfront",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the initial policy before the first optimizer step.",
    )
    parser.add_argument(
        "--eval-at-end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate final weights when the last step is not an eval boundary.",
    )
    parser.add_argument(
        "--eval-group-size",
        type=int,
        default=1,
        help="Independent completions per held-out eval row.",
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for held-out evaluation.",
    )
    parser.add_argument(
        "--eval-top-p",
        type=float,
        default=0.95,
        help="Nucleus-sampling probability for held-out evaluation.",
    )
    parser.add_argument(
        "--eval-top-k",
        type=int,
        default=20,
        help="Top-k sampling cutoff for held-out evaluation (0 disables).",
    )
    parser.add_argument(
        "--eval-max-completion-tokens",
        type=int,
        default=26666,
        help=(
            "Per-assistant-call generation cap for evaluation. Config callers "
            "may set None to inherit --max-completion-tokens."
        ),
    )
    parser.add_argument(
        "--require-complete-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail an eval boundary if any expected episode is missing or any "
            "sampling call reaches a length/context limit."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--filter-constant-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--filter-truncated-rollouts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--judge-max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS
    )
    parser.add_argument(
        "--judge-max-concurrency",
        type=int,
        default=DEFAULT_JUDGE_MAX_CONCURRENCY,
    )
    parser.add_argument(
        "--judge-timeout-s",
        type=float,
        default=DEFAULT_JUDGE_TIMEOUT_S,
        help="Per-request judge timeout; does not constrain reasoning or output length.",
    )
    parser.add_argument(
        "--critical-reward-weight",
        type=float,
        default=DEFAULT_CRITICAL_REWARD_WEIGHT,
    )
    parser.add_argument("--sampling-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--router-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--router-replay-completion-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--grad-accumulation-normalization",
        choices=[member.value for member in GradAccNormalization],
        default=GradAccNormalization.NUM_LOSS_TOKENS.value,
    )
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--checkpoint-name", default="vtb-serverless")
    parser.add_argument("--final-checkpoint-name", default="vtb-serverless-final")
    parser.add_argument(
        "--dcp-save-interval",
        type=int,
        default=2,
        help=(
            "Save resumable model+optimizer state every N successful optimizer "
            "updates (0 disables periodic and final DCP saves)."
        ),
    )
    parser.add_argument(
        "--plot-reward-curve",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-tool-aligned-data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "serverless-rl-vlm"),
    )
    parser.add_argument("--wandb-run-name", default="")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        dataset=args.dataset,
        eval_dataset=args.eval_dataset,
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        renderer_name=args.renderer_name,
        steps=args.steps,
        prompt_groups_per_step=args.prompt_groups_per_step,
        group_size=args.group_size,
        rollout_concurrency=args.rollout_concurrency,
        max_completion_tokens=args.max_completion_tokens,
        max_turns=args.max_turns,
        max_seq_len=args.max_seq_len,
        max_prompt_tokens=args.max_prompt_tokens,
        max_workspace_images=args.max_workspace_images,
        tool_image_dim=args.tool_image_dim,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        adam_weight_decay=args.adam_weight_decay,
        temperature=args.temperature,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        eval_upfront=args.eval_upfront,
        eval_at_end=args.eval_at_end,
        eval_group_size=args.eval_group_size,
        eval_temperature=args.eval_temperature,
        eval_top_p=args.eval_top_p,
        eval_top_k=args.eval_top_k,
        eval_max_completion_tokens=args.eval_max_completion_tokens,
        require_complete_eval=args.require_complete_eval,
        seed=args.seed,
        shuffle=args.shuffle,
        filter_constant_reward=args.filter_constant_reward,
        filter_truncated_rollouts=args.filter_truncated_rollouts,
        judge_model=args.judge_model,
        judge_max_tokens=args.judge_max_tokens,
        judge_max_concurrency=args.judge_max_concurrency,
        judge_timeout_s=args.judge_timeout_s,
        critical_reward_weight=args.critical_reward_weight,
        sampling_timeout_s=args.sampling_timeout_s,
        router_replay=args.router_replay,
        router_replay_completion_only=args.router_replay_completion_only,
        grad_accumulation_normalization=GradAccNormalization(
            args.grad_accumulation_normalization
        ),
        run_dir=args.run_dir,
        checkpoint_name=args.checkpoint_name,
        final_checkpoint_name=args.final_checkpoint_name,
        dcp_save_interval=args.dcp_save_interval,
        plot_reward_curve=args.plot_reward_curve,
        require_tool_aligned_data=args.require_tool_aligned_data,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


def main(cfg: Config | None = None) -> None:
    cfg = cfg or config_from_args(parse_args())
    if not cfg.api_key:
        raise SystemExit(
            "FIREWORKS_API_KEY is required (export it or set Config.api_key)"
        )
    if not Path(cfg.dataset).exists():
        raise SystemExit(f"dataset not found at {cfg.dataset}")
    if cfg.eval_dataset and not Path(cfg.eval_dataset).exists():
        raise SystemExit(f"eval dataset not found at {cfg.eval_dataset}")
    ServerlessVisualToolbenchRL(cfg).run()


if __name__ == "__main__":
    main()
