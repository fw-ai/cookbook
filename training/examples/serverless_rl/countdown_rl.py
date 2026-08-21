#!/usr/bin/env python3
"""Serverless RL on the Countdown task -- a self-contained, Tinker-style demo.

One file, one loop: GRPO-style reinforcement learning against Fireworks
**serverless training**. There is no trainer job to provision and no inference
deployment to stand up -- a single service object hands you both a training
client and a sampling client:

    service = FiretitanServiceClient(base_url=".../training/v1/serverless")
    training_client = service.create_lora_training_client(base_model, rank)
    for step in range(steps):
        snapshot = training_client.save_weights_for_sampler(name).result().path
        sampler = service.create_sampling_client(model_path=snapshot, tokenizer=...)
        # sample a group of completions per prompt -> score -> group-relative
        # advantages -> importance-sampling training datums
        training_client.forward_backward(datums, "importance_sampling").result()
        training_client.optim_step(adam).result()

Model-agnostic: pass any ``--base-model`` / ``--tokenizer-model`` /
``--renderer-name`` triple. For Kimi K3 use the ready-made launcher
``runs/run_countdown_k3_serverless.sh`` (it also sets HF_TRUST_REMOTE_CODE=1,
which the Kimi tokenizer needs).

Dataset: the default is ``data/countdown_3to4_train.jsonl``, materialized once
from HuggingFace (Jiayi-Pan/Countdown-Tasks-3to4, the TinyZero countdown tasks)
with:

    python -m training.examples.serverless_rl.countdown_rl --prepare-dataset

Rows are ``{"messages": [...], "ground_truth": {"numbers": [...], "target": N}}``.

Each optimizer step: save LoRA weights for the sampler, roll out
``group_size`` completions for each of ``prompt_groups_per_step`` prompts,
score with ``countdown_rewards.composite_reward``, standardize rewards within
each prompt group (GRPO, dropping zero-variance groups), then one
``forward_backward(..., "importance_sampling")`` + ``optim_step``.

Metrics stream to ``metrics.jsonl`` (and to W&B with ``--wandb-entity``);
a ``reward_curve.png`` is plotted at the end. Everything lands in ``--run-dir``.

Usage:
    export FIREWORKS_API_KEY=fw_...
    python -m training.examples.serverless_rl.countdown_rl \
        --base-model accounts/fireworks/models/kimi-k3 \
        --tokenizer-model moonshotai/Kimi-K3 --renderer-name kimi_k3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tinker
from fireworks.training.sdk import (
    FireworksClient,
    FiretitanSamplingParams,
    FiretitanServiceClient,
    validate_output_model_id,
)
from tinker_cookbook.renderers import get_renderer, get_text_content

# Registers the cookbook-local renderers ("kimi_k3", "kimi_k3_disable_thinking", ...).
import training.renderer  # noqa: F401

try:  # Load FIREWORKS_API_KEY / FIREWORKS_BASE_URL from a local .env if present.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from training.examples.serverless_rl.countdown_rewards import composite_reward
from training.utils import resolve_router_replay_enabled
from training.utils.rl.router_replay import (
    build_r3_routing_matrices,
    validate_r3_routing_matrices,
)
from training.utils.supervised import resolve_renderer_name
from training.utils.tokenizers import load_tokenizer

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "data" / "countdown_3to4_train.jsonl"

HF_DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"

# Session checkpoint ids embed the run id and an 8-character suffix against a
# 63-character resource-id limit. Keep the logical sampler name short enough
# to remain promotable after those server-owned components are added.
MAX_PROMOTABLE_CHECKPOINT_NAME_LEN = 17

SYSTEM_PROMPT = (
    "You are a math puzzle solver. Given a target number and a set of available "
    "numbers, find an arithmetic expression using each number exactly once with "
    "operations +, -, *, / to reach the target.\n\n"
    "Show your reasoning inside <think>...</think> tags, then put your final "
    "equation inside <answer>...</answer> tags.\n\n"
    "Example:\n"
    "Target: 24, Numbers: [1, 2, 3, 4]\n"
    "<think>I need to reach 24. Let me try 1 * 2 * 3 * 4 = 24.</think>\n"
    "<answer>1 * 2 * 3 * 4</answer>"
)

USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use +, -, *, / and each number must be used exactly once."
)


def prepare_dataset(output: Path, num_rows: int, seed: int) -> None:
    """Download the TinyZero Countdown dataset and write it as training JSONL."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, split="train")
    if num_rows and num_rows < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for rec in ds:
            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(numbers=list(rec["nums"]), target=rec["target"])},
                ],
                "ground_truth": json.dumps({"numbers": list(rec["nums"]), "target": int(rec["target"])}),
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(ds)} rows -> {output}")


@dataclass
class Config:
    """Everything you might want to tune. All fields map 1:1 onto CLI flags."""

    # --- Model --------------------------------------------------------------
    # Any Fireworks base model available on the serverless pool, with the
    # matching HuggingFace tokenizer and chat renderer.
    base_model: str = "accounts/fireworks/models/kimi-k3"
    tokenizer_model: str = "moonshotai/Kimi-K3"
    # Leave "" to auto-resolve the renderer from the tokenizer (handles Kimi K3,
    # DeepSeek V4, and other models tinker_cookbook's table does not know).
    renderer_name: str = ""
    lora_rank: int = 32
    lora_alpha: int = 64
    # Serverless has no training shape from which to infer this bound.
    max_seq_len: int = 32768

    # --- Data ---------------------------------------------------------------
    dataset: str = str(DEFAULT_DATASET)
    # Cycle through the dataset in a shuffled order (seeded, so reproducible).
    shuffle: bool = True
    seed: int = 0

    # --- RL loop shape ------------------------------------------------------
    steps: int = 20
    # Prompts per optimizer step (each prompt becomes one GRPO group).
    prompt_groups_per_step: int = 16
    # Completions per prompt (the group the advantage is computed over).
    group_size: int = 8
    # How many prompts' sample() calls are in flight at once.
    prompt_concurrency: int = 8
    max_sample_tokens: int = 4096
    temperature: float = 1.0
    learning_rate: float = 1e-4
    # Request inference-time expert routes for MoE models and replay them in
    # the trainer. Dense models are detected through model metadata and skip
    # this automatically. The Tinker-shaped sampler returns completion routes,
    # so this example deliberately uses completion-only Router Replay.
    router_replay: bool = True

    # --- W&B ----------------------------------------------------------------
    # Set ``wandb_entity`` to stream metrics to W&B (requires WANDB_API_KEY).
    wandb_entity: str = ""
    wandb_project: str = "serverless-rl-countdown"
    wandb_run_name: str = ""

    # --- Connection / bookkeeping -------------------------------------------
    # Defaults to the public Fireworks API; the "/training/v1/serverless"
    # suffix is added automatically.
    api_base_url: str = field(default_factory=lambda: os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"))
    api_key: str = field(default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", ""))
    # Sampler snapshots used for rollouts. These are promotable, but cannot
    # restore optimizer state.
    checkpoint_name: str = "cd-sample"
    final_checkpoint_name: str = "countdown-final"
    # Full trainer-state checkpoints used for resume. A positive interval also
    # guarantees one final DCP save after the last optimizer step.
    dcp_save_interval: int = 0
    training_checkpoint_name: str = "cd-state"
    resume_from: str = ""
    # Empty keeps the historical behavior (save, but do not promote). Set an id
    # to promote the final sampler checkpoint to a Fireworks LoRA model.
    output_model_id: str = ""
    dcp_timeout_s: float = 900.0
    sampling_timeout_s: float = 1800.0
    run_dir: str = ""
    plot_reward_curve: bool = True


def _validate_config(cfg: Config) -> None:
    if cfg.max_seq_len <= 0:
        raise ValueError("serverless training requires max_seq_len > 0")
    if cfg.lora_rank <= 0:
        raise ValueError("serverless training requires lora_rank > 0")
    if cfg.steps <= 0:
        raise ValueError("steps must be > 0")
    if cfg.dcp_save_interval < 0:
        raise ValueError("dcp_save_interval must be >= 0 (0 disables resumable checkpoints)")
    for name in (cfg.checkpoint_name, cfg.final_checkpoint_name, cfg.training_checkpoint_name):
        if not name:
            raise ValueError("checkpoint names must not be empty")
    if cfg.output_model_id:
        if len(cfg.final_checkpoint_name) > MAX_PROMOTABLE_CHECKPOINT_NAME_LEN:
            raise ValueError(
                f"final checkpoint name {cfg.final_checkpoint_name!r} is too long to promote; "
                f"use at most {MAX_PROMOTABLE_CHECKPOINT_NAME_LEN} characters"
            )
        errors = validate_output_model_id(cfg.output_model_id)
        if errors:
            raise ValueError("\n".join(errors))
    if cfg.resume_from:
        _validate_resume_reference(cfg.resume_from, cfg.training_checkpoint_name)


def _validate_length(what: str, length: int, max_seq_len: int) -> None:
    if length > max_seq_len:
        raise ValueError(f"{what} length {length} exceeds max_seq_len {max_seq_len}")


def _serverless_base_url(base_url: str) -> str:
    """Serverless training + sampling both hang off ``/training/v1/serverless``."""
    root = base_url.rstrip("/")
    if root.endswith("/training/v1/serverless"):
        return root
    if root.endswith("/training/v1"):
        return f"{root}/serverless"
    return f"{root}/training/v1/serverless"


def _control_plane_base_url(base_url: str) -> str:
    """Checkpoint list/promote use the regular API, not the serverless URL."""
    root = base_url.rstrip("/")
    for suffix in ("/training/v1/serverless", "/training/v1"):
        if root.endswith(suffix):
            return root[: -len(suffix)]
    return root


def _validate_resume_reference(reference: str, training_checkpoint_name: str = "cd-state") -> None:
    """Require this recipe's numbered DCP checkpoint, never a sampler save."""
    tail = reference.rstrip("/").split("/")[-1]
    prefix, separator, suffix = tail.rpartition("-")
    if separator != "-" or prefix != training_checkpoint_name or not suffix.isdigit():
        raise ValueError(
            "--resume-from must name a numbered training checkpoint with prefix "
            f"{training_checkpoint_name!r}, such as {training_checkpoint_name + '-0002'!r} "
            f"(got {reference!r}); sampler checkpoints are not resumable"
        )


def _step_from_checkpoint_name(reference: str) -> int:
    """Recover the number of completed optimizer steps from a DCP reference."""
    tail = reference.rstrip("/").split("/")[-1]
    suffix = tail.rsplit("-", 1)[-1] if "-" in tail else ""
    return int(suffix) if suffix.isdigit() else 0


def _account_from_session(session_name: str | None) -> str | None:
    """Parse ``accounts/{account}/trainingSessions/{id}``."""
    parts = (session_name or "").split("/")
    if len(parts) >= 2 and parts[0] == "accounts" and parts[1]:
        return parts[1]
    return None


def _checkpoint_label(checkpoint: dict[str, Any]) -> str:
    return str(checkpoint.get("name", "")).rstrip("/").split("/")[-1]


def _find_promotable(
    checkpoints: list[dict[str, Any]],
    final_name: str,
    run_id: str | None,
) -> dict[str, Any] | None:
    """Find this run's final sampler row using server-owned id prefixes."""
    prefixes = [final_name]
    if run_id:
        prefixes.append(f"{run_id}-{final_name}")

    def matches(checkpoint: dict[str, Any]) -> bool:
        label = _checkpoint_label(checkpoint)
        return any(label == prefix or label.startswith(prefix + "-") for prefix in prefixes)

    return next(
        (row for row in checkpoints if matches(row) and bool(row.get("promotable"))),
        None,
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _group_relative_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Standardize rewards within a group (GRPO): ``(r - mean) / std``.

    A near-zero std (a group where every sample scored the same) is floored to
    1.0 so we do not divide by ~0; otherwise the true std is used.
    """
    if len(rewards) <= 1:
        return [0.0 for _ in rewards]
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
    std = math.sqrt(variance)
    if std < 1e-6:
        std = 1.0
    return [(r - mean) / (std + eps) for r in rewards]


def _mean_loss(fb_output: Any) -> float | None:
    """Mean NLL from a forward_backward result: loss:sum / response_tokens."""
    metrics = getattr(fb_output, "metrics", None) or {}
    loss_sum = metrics.get("loss:sum")
    tokens = metrics.get("response_tokens") or metrics.get("num_loss_tokens") or 1.0
    return float(loss_sum) / max(float(tokens), 1.0) if loss_sum is not None else None


def _mean_policy_sample_logprob_gap(datums: list[Any], fb_output: Any) -> float | None:
    """Mean per-token k1 estimator between the pre-update policy and the sampler.

    The forward pass recomputes the current policy's logprobs on the sampled
    tokens (``loss_fn_outputs[*]["logprobs"]``), and the sampler's behavior
    logprobs live on each datum (``logprobs``). The mean gap over the response
    region is E[log pi_policy - log pi_sample], the k1 estimator matching
    ``training/utils/rl/observability.py``. Returns None when the server does
    not surface per-datum logprobs. Logging-only; no reference forward pass.

    The response region is where ``target_tokens != 0`` (prompt positions are
    left-padded with 0; real token ids are >= 1). This is robust to response
    tokens whose GRPO advantage happens to be exactly 0.
    """
    outputs = getattr(fb_output, "loss_fn_outputs", None)
    if not outputs:
        return None
    total_gap = 0.0
    total_tokens = 0
    for datum, out in zip(datums, outputs, strict=False):
        raw_logprobs = (
            out.get("logprobs") if isinstance(out, dict) else getattr(out, "logprobs", None)
        )
        policy_logprobs = _as_float_list(raw_logprobs)
        if policy_logprobs is None:
            continue
        sample_logprobs = _as_float_list(datum.loss_fn_inputs.get("logprobs"))
        target_tokens = _as_float_list(datum.loss_fn_inputs.get("target_tokens"))
        n = min(len(policy_logprobs), len(sample_logprobs), len(target_tokens))
        for i in range(n):
            if target_tokens[i] != 0:  # response region
                total_gap += policy_logprobs[i] - sample_logprobs[i]
                total_tokens += 1
    return total_gap / total_tokens if total_tokens else None


def _as_float_list(value: Any) -> list[float] | None:
    """Coerce a loss_fn_inputs value (plain list, or Tinker TensorData with
    ``.data`` / ``.to_torch()``) to a flat ``list[float]``. None if unavailable."""
    if value is None:
        return None
    if hasattr(value, "to_torch"):
        return [float(x) for x in value.to_torch().flatten().tolist()]
    data = getattr(value, "data", None)
    if data is not None:
        return [float(x) for x in data]
    return [float(x) for x in value]


def _maybe_init_wandb(
    cfg: Config,
    run_dir: Path,
    renderer_name: str,
    *,
    base_model: str,
    start_step: int,
) -> Any:
    """Initialize W&B when ``--wandb-entity`` is set; returns the run or None.

    ``renderer_name`` is the resolved renderer actually in use (``cfg.renderer_name``
    may be empty when auto-resolved), so the run config is labeled correctly.
    """
    entity = (cfg.wandb_entity or "").strip()
    if not entity:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set; skipping W&B logging", flush=True)
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping W&B logging", flush=True)
        return None
    run = wandb.init(
        entity=entity,
        project=(cfg.wandb_project or "").strip() or "serverless-rl-countdown",
        name=cfg.wandb_run_name or f"countdown-{run_dir.name}",
        config={
            "base_model": base_model,
            "tokenizer_model": cfg.tokenizer_model,
            "renderer_name": renderer_name,
            "dataset": cfg.dataset,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "max_seq_len": cfg.max_seq_len,
            "steps": cfg.steps,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "group_size": cfg.group_size,
            "max_sample_tokens": cfg.max_sample_tokens,
            "temperature": cfg.temperature,
            "learning_rate": cfg.learning_rate,
            "router_replay": cfg.router_replay,
            "shuffle": cfg.shuffle,
            "seed": cfg.seed,
            "resume_from": cfg.resume_from or None,
            "start_step": start_step,
            "loss": "importance_sampling",
        },
    )
    wandb.define_metric("rollout/*", step_metric="train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("kld/*", step_metric="train/step")
    wandb.define_metric("perf/*", step_metric="train/step")
    wandb.define_metric("rollout/raw_reward", step_metric="train/step", summary="max")
    print(f"W&B: {run.url}", flush=True)
    return run


def _log_wandb_step(step: int, rec: dict[str, Any]) -> None:
    try:
        import wandb

        if wandb.run is None:
            return
        payload = {"train/step": step}
        payload.update(
            (k, v)
            for k, v in rec.items()
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))
        )
        wandb.log(payload, step=int(step))
    except Exception as exc:
        print(f"wandb log skipped: {exc}", flush=True)


class ServerlessCountdownRL:
    """One serverless RL run over the Countdown dataset."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        _validate_config(cfg)
        self.rows = _load_rows(Path(cfg.dataset))
        if not self.rows:
            raise SystemExit(f"dataset is empty: {cfg.dataset} (run with --prepare-dataset first)")

        # Cookbook loaders cover models the default Tinker tokenizer/renderer
        # tables do not, including DeepSeek V4 and Kimi K3.
        self.tokenizer = load_tokenizer(cfg.tokenizer_model)
        self.renderer_name = resolve_renderer_name(cfg.tokenizer_model, cfg.renderer_name)
        self.renderer = get_renderer(self.renderer_name, self.tokenizer)

        # The one connection that gives us BOTH training and sampling clients.
        # No trainer job, no deployment -- just a pooled serverless session.
        self.service = FiretitanServiceClient(
            api_key=cfg.api_key,
            base_url=_serverless_base_url(cfg.api_base_url),
        )
        self.start_step = 0
        self.base_model = cfg.base_model
        if cfg.resume_from:
            print(f"resuming from checkpoint: {cfg.resume_from}", flush=True)
            started = time.time()
            self.training_client = self.service.create_training_client_from_state_with_optimizer(
                cfg.resume_from
            )
            self.start_step = _step_from_checkpoint_name(cfg.resume_from)
            weights_info = self.service.create_rest_client().get_weights_info_by_tinker_path(
                cfg.resume_from
            ).result()
            self.base_model = weights_info.base_model
            print(
                f"resumed in {time.time() - started:.1f}s; continuing at step "
                f"{self.start_step} on base model {self.base_model}",
                flush=True,
            )
        else:
            self.training_client = self.service.create_lora_training_client(
                base_model=cfg.base_model,
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
            )

        # Resolve against the checkpoint's actual base model on resume rather
        # than trusting a potentially stale --base-model CLI value.
        self.router_replay_enabled = resolve_router_replay_enabled(
            requested=cfg.router_replay,
            api_key=cfg.api_key,
            base_url=_control_plane_base_url(cfg.api_base_url),
            additional_headers=None,
            base_model=self.base_model,
        )
        if cfg.router_replay and not self.router_replay_enabled:
            print(f"Router Replay skipped for dense model {self.base_model}", flush=True)

        self.session_name = getattr(self.service, "training_session_name", None)
        self.session_id = getattr(self.service, "training_session_id", None)
        self.run_id = getattr(self.training_client, "run_id", None)
        if not self.session_id:
            raise RuntimeError(
                "serverless service did not expose a training_session_id; "
                "cannot record or promote checkpoints for this run"
            )

        self.run_dir = (
            Path(cfg.run_dir).resolve()
            if cfg.run_dir
            else Path("/tmp") / f"serverless_countdown_{int(time.time())}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

        self.order = list(range(len(self.rows)))
        if cfg.shuffle:
            random.Random(cfg.seed).shuffle(self.order)
        # Resume the deterministic dataset order at the first row not consumed
        # by the checkpoint's completed optimizer-step count.
        self.row_cursor = self.start_step * cfg.prompt_groups_per_step
        self.saved_training_checkpoints: list[str] = []

        self._wandb = _maybe_init_wandb(
            cfg,
            self.run_dir,
            self.renderer_name,
            base_model=self.base_model,
            start_step=self.start_step,
        )

        print(
            f"connected serverless session={self.session_name or self.session_id} run={self.run_id}\n"
            f"base_model={self.base_model} tokenizer={cfg.tokenizer_model} renderer={self.renderer_name}\n"
            f"dataset={cfg.dataset} rows={len(self.rows)}\n"
            f"steps={cfg.steps} prompt_groups_per_step={cfg.prompt_groups_per_step} "
            f"group_size={cfg.group_size} lora_rank={cfg.lora_rank} lora_alpha={cfg.lora_alpha} "
            f"max_seq_len={cfg.max_seq_len} lr={cfg.learning_rate} "
            f"router_replay={self.router_replay_enabled}\n"
            f"run_dir={self.run_dir}",
            flush=True,
        )

    def _save_dcp(self, completed_step: int) -> str:
        """Save adapter weights plus optimizer state for exact resume."""
        name = f"{self.cfg.training_checkpoint_name}-{completed_step:04d}"
        started = time.time()
        self.training_client.save_state(name).result(timeout=self.cfg.dcp_timeout_s)
        self.saved_training_checkpoints.append(name)
        print(f"saved training checkpoint {name!r} ({time.time() - started:.1f}s)", flush=True)
        return name

    def _resume_reference(self, checkpoint_name: str) -> str:
        if not self.run_id:
            raise RuntimeError(
                "serverless training client did not expose a run_id; "
                "cannot build a portable resume checkpoint reference"
            )
        account = _account_from_session(self.session_name)
        if account is None:
            cp_client = FireworksClient(
                api_key=self.cfg.api_key,
                base_url=_control_plane_base_url(self.cfg.api_base_url),
            )
            try:
                account = cp_client.account_id
            finally:
                cp_client.close()
        return f"{account}/{self.run_id}/{checkpoint_name}"

    def _promote(self) -> str:
        """Promote this run's final sampler checkpoint through the CP API."""
        cfg = self.cfg
        cp_client = FireworksClient(
            api_key=cfg.api_key,
            base_url=_control_plane_base_url(cfg.api_base_url),
        )
        try:
            session_name = self.session_name or (
                f"accounts/{cp_client.account_id}/trainingSessions/{self.session_id}"
            )
            checkpoints = cp_client.list_training_session_checkpoints(session_name)
            target = _find_promotable(checkpoints, cfg.final_checkpoint_name, self.run_id)
            if target is None:
                labels = [
                    f"{_checkpoint_label(row)} promotable={row.get('promotable')}"
                    for row in checkpoints
                ]
                raise RuntimeError(
                    f"final sampler checkpoint {cfg.final_checkpoint_name!r} is not listed as "
                    f"promotable (saw {len(checkpoints)} rows: {labels})"
                )
            model = cp_client.promote_session_checkpoint(
                name=target["name"],
                output_model_id=cfg.output_model_id,
                base_model=self.base_model,
            )
            model_name = model.get("name") if isinstance(model, dict) else str(model)
            print(f"promoted model: {model_name}", flush=True)
            return model_name
        finally:
            cp_client.close()

    def _next_batch(self) -> list[dict[str, Any]]:
        """Take the next ``prompt_groups_per_step`` rows, wrapping around."""
        n = len(self.order)
        idx = [self.order[(self.row_cursor + i) % n] for i in range(self.cfg.prompt_groups_per_step)]
        self.row_cursor += self.cfg.prompt_groups_per_step
        return [self.rows[i] for i in idx]

    def _step(self, step: int) -> dict[str, Any]:
        t0 = time.time()
        cfg = self.cfg

        # 1. Save the current LoRA weights so the sampler can serve them, then
        #    open a sampling client bound to that exact snapshot.
        save_name = f"{cfg.checkpoint_name}-{step:04d}"
        snapshot = self.training_client.save_weights_for_sampler(save_name).result().path
        if not snapshot:
            raise RuntimeError(f"save_weights_for_sampler({save_name!r}) returned no path")

        batch = self._next_batch()
        prompts = [self.renderer.build_generation_prompt(row["messages"]) for row in batch]
        for prompt in prompts:
            _validate_length("prompt + max_sample_tokens", prompt.length + cfg.max_sample_tokens, cfg.max_seq_len)

        # 2. Roll out `group_size` completions per prompt, a few prompts in
        #    flight at a time.
        sampler = self.service.create_sampling_client(model_path=snapshot, tokenizer=self.tokenizer)
        try:
            params = FiretitanSamplingParams(
                max_tokens=cfg.max_sample_tokens,
                temperature=cfg.temperature,
                stop=self.renderer.get_stop_sequences(),
                include_routing_matrix=self.router_replay_enabled,
            )
            results: list[Any] = []
            chunk = max(1, cfg.prompt_concurrency)
            for start in range(0, len(prompts), chunk):
                futures = [
                    sampler.sample(prompt=p, num_samples=cfg.group_size, sampling_params=params)
                    for p in prompts[start : start + chunk]
                ]
                results.extend(f.result(timeout=cfg.sampling_timeout_s) for f in futures)
        finally:
            sampler.close()

        # 3. Score each completion and keep only groups with reward spread (a
        #    group where every sample scores the same yields zero advantage and
        #    no learning signal, so we drop it -- standard GRPO filtering).
        datums: list[Any] = []
        raw_rewards: list[float] = []
        filtered_rewards: list[float] = []
        response_token_count = 0

        for result, prompt, row in zip(results, prompts, batch):
            tokens_g: list[list[int]] = []
            logprobs_g: list[list[float]] = []
            routing_matrices_g: list[list[str] | None] = []
            rewards_g: list[float] = []
            for seq in getattr(result, "sequences", []) or []:
                tokens = list(getattr(seq, "tokens", []) or [])
                logprobs = getattr(seq, "logprobs", None)
                # The importance-sampling loss needs one logprob per sampled token.
                if not tokens or logprobs is None or len(logprobs) != len(tokens):
                    continue
                content = get_text_content(self.renderer.parse_response(tokens)[0])
                reward = float(composite_reward(content, row["ground_truth"]))
                tokens_g.append(tokens)
                logprobs_g.append([float(x) for x in logprobs])
                routing_matrices = getattr(seq, "routing_matrices", None)
                routing_matrices_g.append(
                    list(routing_matrices) if routing_matrices is not None else None
                )
                rewards_g.append(reward)
                raw_rewards.append(reward)
                response_token_count += len(tokens)

            if len(set(rewards_g)) <= 1:
                continue
            filtered_rewards.extend(rewards_g)

            advantages = _group_relative_advantages(rewards_g)
            response_start = prompt.length - 1
            for tokens, logprobs, routing_matrices, advantage in zip(
                tokens_g,
                logprobs_g,
                routing_matrices_g,
                advantages,
                strict=True,
            ):
                # Shifted next-token layout: the model sees prompt + all but the
                # last sampled token; targets/logprobs/advantages are aligned to
                # the response region and left-padded over the prompt.
                model_input = prompt.append(tinker.EncodedTextChunk(tokens=tokens[:-1]))
                _validate_length("training datum", model_input.length, cfg.max_seq_len)
                if self.router_replay_enabled:
                    validate_r3_routing_matrices(
                        routing_matrices,
                        prompt_len=prompt.length,
                        model_input_len=model_input.length,
                    )
                    aligned_routes = build_r3_routing_matrices(
                        routing_matrices,
                        prompt_len=prompt.length,
                        model_input_len=model_input.length,
                        completion_only=True,
                    )
                    model_input = model_input.model_copy(
                        update={"routing_matrices": aligned_routes}
                    )
                response_len = model_input.length - response_start
                datums.append(
                    tinker.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": [0] * response_start + tokens,
                            "logprobs": [0.0] * response_start + logprobs,
                            "advantages": [0.0] * response_start + [advantage] * response_len,
                        },
                    )
                )

        # 4. One importance-sampling update. The forward_backward response also
        #    carries the pre-update policy's logprobs on the sampled tokens;
        #    the mean (policy - sample) logprob gap is the per-token k1
        #    estimator, logged as kld/mean_k1. This is logging-only -- it
        #    does not change the loss and needs no reference forward pass.
        loss = None
        kld_k1 = None
        if datums:
            fb = self.training_client.forward_backward(datums, "importance_sampling").result()
            loss = _mean_loss(fb)
            kld_k1 = _mean_policy_sample_logprob_gap(datums, fb)
            adam = tinker.AdamParams(learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0)
            self.training_client.optim_step(adam).result()

        raw_reward = sum(raw_rewards) / len(raw_rewards) if raw_rewards else 0.0
        filtered_reward = sum(filtered_rewards) / len(filtered_rewards) if filtered_rewards else 0.0
        filter_ratio = 1.0 - len(filtered_rewards) / len(raw_rewards) if raw_rewards else 0.0
        rec = {
            "step": step,
            "snapshot": snapshot,
            "rollout/raw_reward": raw_reward,
            "rollout/filtered_reward": filtered_reward,
            "rollout/raw_samples": len(raw_rewards),
            "rollout/filtered_samples": len(filtered_rewards),
            "rollout/filter_ratio": filter_ratio,
            "rollout/mean_response_tokens": response_token_count / len(raw_rewards) if raw_rewards else 0.0,
            "train/loss": loss,
            "train/trained": bool(datums),
            "train/router_replay": self.router_replay_enabled,
            "kld/mean_k1": kld_k1,
            "perf/step_wall_time": time.time() - t0,
        }
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        _log_wandb_step(step, rec)
        print(
            f"step {step:02d} reward={raw_reward:.3f}/{filtered_reward:.3f} "
            f"samples={len(raw_rewards)}/{len(filtered_rewards)} "
            f"filter={filter_ratio:.1%} "
            f"loss={'n/a' if loss is None else f'{loss:.4f}'} "
            f"kld={'n/a' if kld_k1 is None else f'{kld_k1:+.4f}'} "
            f"elapsed={rec['perf/step_wall_time']:.1f}s",
            flush=True,
        )
        return rec

    def run(self) -> list[dict[str, Any]]:
        try:
            cfg = self.cfg
            records: list[dict[str, Any]] = []
            for step in range(self.start_step, self.start_step + cfg.steps):
                records.append(self._step(step))
                completed_step = step + 1
                if cfg.dcp_save_interval and completed_step % cfg.dcp_save_interval == 0:
                    self._save_dcp(completed_step)

            resume_ref = ""
            if cfg.dcp_save_interval:
                final_step = self.start_step + cfg.steps
                final_name = f"{cfg.training_checkpoint_name}-{final_step:04d}"
                if final_name not in self.saved_training_checkpoints:
                    final_name = self._save_dcp(final_step)
                resume_ref = self._resume_reference(final_name)
                print(f"resume this run with:\n  --resume-from {resume_ref}", flush=True)
                (self.run_dir / "resume_from.txt").write_text(f"{resume_ref}\n")

            final = self.training_client.save_weights_for_sampler(cfg.final_checkpoint_name).result()
            final_path = getattr(final, "path", None)
            print(f"final sampler checkpoint: {final_path}", flush=True)
            (self.run_dir / "final_checkpoint.txt").write_text(f"{final_path}\n")

            promoted_model = self._promote() if cfg.output_model_id else ""
            if promoted_model:
                (self.run_dir / "promoted_model.txt").write_text(f"{promoted_model}\n")

            lifecycle = {
                "session": self.session_name or self.session_id,
                "run_id": self.run_id,
                "base_model": self.base_model,
                "resume_from": resume_ref or None,
                "final_sampler_checkpoint": final_path,
                "promoted_model": promoted_model or None,
            }
            (self.run_dir / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2) + "\n")

            if records:
                rewards = [r["rollout/raw_reward"] for r in records]
                print(
                    f"\nreward: {rewards[0]:.3f} -> {rewards[-1]:.3f} (peak {max(rewards):.3f}) "
                    f"over {len(records)} steps",
                    flush=True,
                )
            if self.cfg.plot_reward_curve:
                self._plot(records)
            print(f"metrics: {self.metrics_path}", flush=True)
            return records
        finally:
            if self._wandb is not None:
                try:
                    import wandb

                    wandb.finish()
                except Exception:
                    pass
            self.service.close()

    def _plot(self, records: list[dict[str, Any]]) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping reward curve", flush=True)
            return
        steps = [r["step"] for r in records]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, [r["rollout/raw_reward"] for r in records], marker="o", label="raw_reward")
        ax.plot(steps, [r["rollout/filtered_reward"] for r in records], marker="s", linestyle="--", label="filtered_reward")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("score")
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"Serverless Countdown RL ({self.base_model}, importance_sampling)")
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
        description="Serverless RL on the Countdown task (GRPO, importance sampling)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default=Config.base_model)
    parser.add_argument("--tokenizer-model", default=Config.tokenizer_model)
    parser.add_argument("--renderer-name", default=Config.renderer_name)
    parser.add_argument("--lora-rank", type=int, default=Config.lora_rank)
    parser.add_argument("--lora-alpha", type=int, default=Config.lora_alpha)
    parser.add_argument("--max-seq-len", type=int, default=Config.max_seq_len)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--prompt-groups-per-step", type=int, default=Config.prompt_groups_per_step)
    parser.add_argument("--group-size", type=int, default=Config.group_size)
    parser.add_argument("--prompt-concurrency", type=int, default=Config.prompt_concurrency)
    parser.add_argument("--max-sample-tokens", type=int, default=Config.max_sample_tokens)
    parser.add_argument("--temperature", type=float, default=Config.temperature)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument(
        "--router-replay",
        action=argparse.BooleanOptionalAction,
        default=Config.router_replay,
        help="Replay completion-token expert routes for MoE train/inference alignment.",
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-project", default=Config.wandb_project)
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--checkpoint-name", default=Config.checkpoint_name)
    parser.add_argument("--final-checkpoint-name", default=Config.final_checkpoint_name)
    parser.add_argument(
        "--dcp-save-interval",
        type=int,
        default=Config.dcp_save_interval,
        help="Save resumable adapter+optimizer state every N completed steps (0 disables).",
    )
    parser.add_argument("--training-checkpoint-name", default=Config.training_checkpoint_name)
    parser.add_argument(
        "--resume-from",
        default=Config.resume_from,
        help="Old-run training checkpoint: '<account>/<run-id>/<checkpoint>'.",
    )
    parser.add_argument(
        "--output-model-id",
        default=Config.output_model_id,
        help="Promote the final sampler checkpoint to this model id; empty disables promotion.",
    )
    parser.add_argument("--dcp-timeout-s", type=float, default=Config.dcp_timeout_s)
    parser.add_argument("--sampling-timeout-s", type=float, default=Config.sampling_timeout_s)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--plot-reward-curve", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prepare-dataset",
        action="store_true",
        help=f"Download {HF_DATASET_ID} to --dataset and exit.",
    )
    parser.add_argument("--prepare-dataset-rows", type=int, default=20000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.prepare_dataset:
        prepare_dataset(Path(args.dataset), args.prepare_dataset_rows, args.seed)
        return
    cfg = Config(**{k: v for k, v in vars(args).items() if k in Config.__dataclass_fields__})
    if not cfg.api_key:
        raise SystemExit("FIREWORKS_API_KEY is required (export it or put it in training/.env)")
    ServerlessCountdownRL(cfg).run()


if __name__ == "__main__":
    main()
