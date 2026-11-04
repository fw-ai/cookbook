"""Anti-slop RL: GRPO on tweet briefs against a gated AI-text-detector reward.

Serverless, one file, forked from the Short Thinker loop (itself a fork of
``examples/archive/countdown/countdown_rl.py``). The reward is the same one that
ranked the DPO pack: a sampled tweet is ELIGIBLE if it meets the exact format
spec and the judge rates it faithful, coherent and on-format (>= 0.8 each);
an eligible tweet earns the detector's human probability, an ineligible one
earns 0. Tropes are measured every step and never rewarded, so the run itself
answers whether climbing the detector changes them.

Rollouts prefill ``</think>`` so the policy writes directly, matching how the
DPO pairs were rendered. ``--init-from-state`` starts from a DPO checkpoint's
weights and optimizer (HumanSound's DPO-then-GRPO recipe); otherwise the LoRA
starts from the base model.

Usage:
    export FIREWORKS_API_KEY=... GPT_ZERO_API_KEY=...
    python antislop_rl.py --prepare-dataset
    python antislop_rl.py --steps 20 [--init-from-state example/run-.../dpo-0027]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fireworks.training import sdk
from fireworks.training.sdk import (
    FireworksClient,
    FiretitanSamplingParams,
    FiretitanServiceClient,
    validate_output_model_id,
)
from fireworks.training.sdk.routing import copy_routing, routing_model_input_kwargs

# Registers the cookbook-local renderers ("kimi_k3", "kimi_k3_disable_thinking", ...).
import training.renderer  # noqa: F401
from training.renderer import get_renderer, get_text_content

try:  # Load FIREWORKS_API_KEY / FIREWORKS_BASE_URL from a local .env if present.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import sys as _sys
from concurrent.futures import ThreadPoolExecutor

import httpx

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from antislop_tropes import trope_report  # noqa: E402
from build_antislop_pack import (  # noqa: E402
    ELIGIBILITY_MIN,
    SYSTEM_PROMPT as BRIEF_SYSTEM_PROMPT,
    brief_prompt,
    gptzero_human,
    llm_judge,
    meets_format,
    read_jsonl,
)
from training.utils import phase_span, resolve_router_replay_enabled
from training.utils.rl.router_replay import (
    build_r3_routing_matrices,
    validate_r3_routing_matrices,
)
from training.utils.supervised import resolve_renderer_name
from training.utils.tokenizers import load_tokenizer

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "data" / "antislop_rl_train.jsonl"
BRIEFS_PATH = HERE / "pack" / "antislop_briefs.jsonl"

HF_DATASET_ID = "pack/antislop_briefs.jsonl (local)"
RL_PLATFORMS = ("tweet",)   # LinkedIn drafts score ~0 on the detector; nothing to climb

# Session checkpoint ids embed the run id and an 8-character suffix against a
# 63-character resource-id limit. Keep the logical sampler name short enough
# to remain promotable after those server-owned components are added.
MAX_PROMOTABLE_CHECKPOINT_NAME_LEN = 17

SYSTEM_PROMPT = BRIEF_SYSTEM_PROMPT
USER_TEMPLATE = "{question}"


def prepare_dataset(output: Path, num_rows: int, seed: int) -> None:
    """Training rows from the frozen briefs: train split, tweets only."""
    briefs = [b for b in read_jsonl(BRIEFS_PATH) if b["split"] == "train" and b["platform"] in RL_PLATFORMS and b["article"]]
    if num_rows and num_rows < len(briefs):
        briefs = briefs[:num_rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for b in briefs:
            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": brief_prompt(b["platform"], b["topic"], b["article"])},
                ],
                "ground_truth": json.dumps({"brief_id": b["id"], "platform": b["platform"], "topic": b["topic"], "article": b["article"]}),
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(briefs)} rows -> {output}")


@dataclass
class Config:
    """Everything you might want to tune. All fields map 1:1 onto CLI flags."""

    # --- Model --------------------------------------------------------------
    # Any Fireworks base model available on the serverless pool, with the
    # matching HuggingFace tokenizer and chat renderer.
    base_model: str = "accounts/fireworks/models/glm-5p3-flash"
    tokenizer_model: str = "zai-org/GLM-5.3-Flash"
    # Leave "" to auto-resolve the renderer from the tokenizer (handles Kimi K3,
    # DeepSeek V4, and other models tinker_cookbook's table does not know).
    renderer_name: str = ""
    # Rank 64 / alpha 128: what the GLM 5.3 Flash serverless pool and its
    # multi-LoRA serving host are built around.
    lora_rank: int = 64
    lora_alpha: int = 128
    # Serverless has no training shape from which to infer this bound.
    max_seq_len: int = 32768

    # --- Data ---------------------------------------------------------------
    dataset: str = str(DEFAULT_DATASET)
    # Cycle through the dataset in a shuffled order (seeded, so reproducible).
    shuffle: bool = True
    seed: int = 0
    # Deterministically remove this many prompts from the training pool and
    # reuse that exact held-out slice for every evaluation.
    eval_prompt_groups: int = 16
    eval_seed: int = 1

    # --- RL loop shape ------------------------------------------------------
    steps: int = 20
    # Prompts per optimizer step (each prompt becomes one GRPO group).
    prompt_groups_per_step: int = 16
    # Completions per prompt (the group the advantage is computed over).
    group_size: int = 8
    # How many prompts' sample() calls are in flight at once.
    prompt_concurrency: int = 8
    # Tweets: rollouts prefill </think>, so this is the visible text budget.
    max_sample_tokens: int = 256
    temperature: float = 1.0
    learning_rate: float = 2e-5
    # --- Anti-slop reward ---
    # eligible (format spec + judge >= ELIGIBILITY_MIN) -> detector P(human); else 0.
    score_concurrency: int = 8
    # Start from a DPO checkpoint's weights and optimizer, step counter at 0.
    init_from_state: str = ""
    # Request inference-time expert routes for MoE models and replay them in
    # the trainer. Dense models are detected through model metadata and skip
    # this automatically. The Tinker-shaped sampler returns completion routes,
    # so this example deliberately uses completion-only Router Replay.
    router_replay: bool = True

    # --- Fixed held-out evaluation -----------------------------------------
    # Evaluation samples the same carved-out prompts without backward or
    # optimizer calls. An interval of 0 disables periodic evaluation.
    eval_interval: int = 5
    eval_group_size: int = 8
    eval_at_start: bool = True
    eval_at_end: bool = True

    # --- W&B ----------------------------------------------------------------
    # Set ``wandb_entity`` to stream metrics to W&B (requires WANDB_API_KEY).
    wandb_entity: str = ""
    wandb_project: str = "serverless-rl-antislop"
    wandb_run_name: str = ""

    # --- Connection / bookkeeping -------------------------------------------
    # Defaults to the public Fireworks API; the "/training/v1/serverless"
    # suffix is added automatically.
    api_base_url: str = field(default_factory=lambda: os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"))
    api_key: str = field(default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", ""))
    # Sampler snapshots used for rollouts. These are promotable, but cannot
    # restore optimizer state.
    checkpoint_name: str = "as-sample"
    final_checkpoint_name: str = "as-final"
    # Full trainer-state checkpoints used for resume. A positive interval also
    # guarantees one final DCP save after the last optimizer step.
    dcp_save_interval: int = 0
    training_checkpoint_name: str = "as-state"
    # Diversity term: a sample whose first three words match other samples in its group loses
    # diversity_penalty * (share of the other group members sharing that opener). 0 disables.
    diversity_penalty: float = 0.0
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
    if cfg.prompt_groups_per_step <= 0:
        raise ValueError("prompt_groups_per_step must be > 0")
    if cfg.group_size <= 0:
        raise ValueError("group_size must be > 0")
    if cfg.prompt_concurrency <= 0:
        raise ValueError("prompt_concurrency must be > 0")
    if cfg.max_sample_tokens <= 0:
        raise ValueError("max_sample_tokens must be > 0")
    if cfg.eval_prompt_groups < 0:
        raise ValueError("eval_prompt_groups must be >= 0")
    if cfg.eval_interval < 0:
        raise ValueError("eval_interval must be >= 0")
    if cfg.eval_group_size <= 0:
        raise ValueError("eval_group_size must be > 0")
    if (cfg.eval_at_start or cfg.eval_at_end or cfg.eval_interval) and not cfg.eval_prompt_groups:
        raise ValueError("eval_prompt_groups must be > 0 when evaluation is enabled")
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


def _validate_resume_reference(reference: str, training_checkpoint_name: str = "as-state") -> None:
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


def _carve_out_eval_rows(
    rows: list[dict[str, Any]],
    eval_prompt_groups: int,
    eval_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    """Return disjoint training rows and a deterministic held-out eval slice."""
    if eval_prompt_groups == 0:
        return list(rows), [], []
    if eval_prompt_groups >= len(rows):
        raise ValueError(
            "eval_prompt_groups must leave at least one training row "
            f"(got {eval_prompt_groups} eval rows from {len(rows)} total)"
        )
    order = list(range(len(rows)))
    random.Random(eval_seed).shuffle(order)
    eval_indices = order[:eval_prompt_groups]
    eval_index_set = set(eval_indices)
    train_rows = [row for index, row in enumerate(rows) if index not in eval_index_set]
    eval_rows = [rows[index] for index in eval_indices]
    return train_rows, eval_rows, eval_indices


def _finish_reason(seq: Any) -> str:
    value = getattr(seq, "finish_reason", "")
    return str(getattr(value, "value", value)).lower()


def _is_truncated(seq: Any, max_tokens: int | None = None) -> bool:
    reason = _finish_reason(seq)
    if reason == "length" or "max_token" in reason:
        return True
    # Serverless sampler sequences do not always expose finish_reason. Reaching
    # the requested token cap is therefore the authoritative fallback.
    tokens = getattr(seq, "tokens", None) or []
    return max_tokens is not None and len(tokens) >= max_tokens


def _opener(text: str, n: int = 3) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", "", (text or "").lower()).split()[:n])


class ScorerFailure(RuntimeError):
    """A judge or detector call failed for a completion; its group carries no signal."""


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


def _mean_policy_sample_k3(datums: list[Any], fb_output: Any) -> float | None:
    """Mean per-token k3 estimator between the policy and sampler.

    For ``log_ratio = log(pi_policy) - log(pi_sample)``, Schulman's k3
    estimator is ``exp(log_ratio) - 1 - log_ratio``. It is non-negative and
    is computed over the same response tokens as ``kld/mean_k1``.
    """
    outputs = getattr(fb_output, "loss_fn_outputs", None)
    if not outputs:
        return None
    total_k3 = 0.0
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
                log_ratio = policy_logprobs[i] - sample_logprobs[i]
                total_k3 += math.expm1(log_ratio) - log_ratio
                total_tokens += 1
    return total_k3 / total_tokens if total_tokens else None


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
        project=(cfg.wandb_project or "").strip() or "serverless-rl-antislop",
        name=cfg.wandb_run_name or f"antislop-{run_dir.name}",
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
            "eval_prompt_groups": cfg.eval_prompt_groups,
            "eval_seed": cfg.eval_seed,
            "eval_interval": cfg.eval_interval,
            "eval_group_size": cfg.eval_group_size,
            "eval_at_start": cfg.eval_at_start,
            "eval_at_end": cfg.eval_at_end,
            "resume_from": cfg.resume_from or None,
            "start_step": start_step,
            "loss": "importance_sampling",
        },
    )
    wandb.define_metric("rollout/*", step_metric="train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("kld/*", step_metric="train/step")
    wandb.define_metric("kld/mean_k1", step_metric="train/step", summary="last")
    wandb.define_metric("kld/mean_k3", step_metric="train/step", summary="last")
    wandb.define_metric("eval/*", step_metric="train/step")
    wandb.define_metric("perf/*", step_metric="train/step")
    wandb.define_metric("rollout/raw_reward", step_metric="train/step", summary="max")
    wandb.define_metric("eval/raw_reward", step_metric="train/step", summary="max")
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
        # Let W&B advance its own monotonically increasing internal step. The
        # semantic training step remains ``train/step`` and is configured as
        # the chart axis above. Passing ``step=`` makes W&B default to
        # ``commit=False``; with multi-minute rollouts that can hide the most
        # recent KLD row for an entire step and collide with fixed eval rows.
        wandb.log(payload)
    except Exception as exc:
        print(f"wandb log skipped: {exc}", flush=True)


def _log_wandb_eval(completed_steps: int, rec: dict[str, Any]) -> None:
    try:
        import wandb

        if wandb.run is None:
            return
        payload = {"train/step": completed_steps}
        payload.update(
            (k, v)
            for k, v in rec.items()
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))
        )
        wandb.log(payload)
    except Exception as exc:
        print(f"wandb eval log skipped: {exc}", flush=True)


class ServerlessAntislopRL:
    """One serverless RL run over the Anti-slop briefs."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        _validate_config(cfg)
        all_rows = _load_rows(Path(cfg.dataset))
        if not all_rows:
            raise SystemExit(f"dataset is empty: {cfg.dataset} (run with --prepare-dataset first)")
        self.rows, self.eval_rows, self.eval_row_indices = _carve_out_eval_rows(
            all_rows,
            cfg.eval_prompt_groups,
            cfg.eval_seed,
        )

        # Cookbook loaders cover models the default Tinker tokenizer/renderer
        # tables do not, including DeepSeek V4 and Kimi K3.
        self.tokenizer = load_tokenizer(cfg.tokenizer_model)
        self.renderer_name = resolve_renderer_name(cfg.tokenizer_model, cfg.renderer_name)
        self.renderer = get_renderer(self.renderer_name, self.tokenizer)
        self.gptzero_key = os.environ.get("GPT_ZERO_API_KEY")
        if not self.gptzero_key:
            raise SystemExit("GPT_ZERO_API_KEY is required: the reward is the detector")
        self._score_cache: dict[tuple[str, str], tuple[float, bool, float]] = {}

        # The one connection that gives us BOTH training and sampling clients.
        # No trainer job, no deployment -- just a pooled serverless session.
        self.service = FiretitanServiceClient(
            api_key=cfg.api_key,
            base_url=_serverless_base_url(cfg.api_base_url),
        )
        self.start_step = 0
        self.base_model = cfg.base_model
        if cfg.resume_from or cfg.init_from_state:
            ref = cfg.resume_from or cfg.init_from_state
            print(f"{'resuming from' if cfg.resume_from else 'initializing from'} checkpoint: {ref}", flush=True)
            started = time.time()
            self.training_client = self.service.create_training_client_from_state_with_optimizer(ref)
            self.start_step = _step_from_checkpoint_name(cfg.resume_from) if cfg.resume_from else 0
            weights_info = self.service.create_rest_client().get_weights_info_by_tinker_path(ref).result()
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
            training_client=self.training_client,
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
            else Path("/tmp") / f"serverless_antislop_{int(time.time())}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.eval_metrics_path = self.run_dir / "eval_metrics.jsonl"
        self.eval_completions_dir = self.run_dir / "eval_completions"
        if self.eval_rows:
            self.eval_completions_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "dataset_split.json").write_text(
            json.dumps(
                {
                    "dataset": str(Path(cfg.dataset).resolve()),
                    "total_rows": len(all_rows),
                    "training_rows": len(self.rows),
                    "eval_rows": len(self.eval_rows),
                    "eval_seed": cfg.eval_seed,
                    "eval_row_indices": self.eval_row_indices,
                },
                indent=2,
            )
            + "\n"
        )

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
            f"dataset={cfg.dataset} train_rows={len(self.rows)} "
            f"eval_rows={len(self.eval_rows)} eval_interval={cfg.eval_interval} "
            f"eval_group_size={cfg.eval_group_size}\n"
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
        with phase_span(
            "checkpoint_save",
            category="checkpoint",
            attributes={"completed_step": completed_step, "resumable": True},
        ):
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

    def _evaluate(self, completed_steps: int) -> dict[str, Any]:
        """Sample the fixed held-out carve-out without any optimizer mutation."""
        with phase_span(
            "evaluation",
            category="evaluation",
            attributes={
                "completed_steps": completed_steps,
                "prompt_groups": len(self.eval_rows),
                "group_size": self.cfg.eval_group_size,
            },
        ) as span:
            result = self._evaluate_impl(completed_steps)
            if span is not None:
                span.set_attribute("samples", result["eval/samples"])
            return result

    def _evaluate_impl(self, completed_steps: int) -> dict[str, Any]:
        cfg = self.cfg
        started = time.time()
        save_name = f"as-eval-{completed_steps:04d}"
        snapshot = self.training_client.save_weights_for_sampler(save_name).result().path
        if not snapshot:
            raise RuntimeError(f"save_weights_for_sampler({save_name!r}) returned no path")

        prompts = [self.renderer.build_generation_prompt(row["messages"], prefill="</think>") for row in self.eval_rows]
        for prompt in prompts:
            _validate_length(
                "eval prompt + max_sample_tokens",
                prompt.length + cfg.max_sample_tokens,
                cfg.max_seq_len,
            )

        sampler = self.service.create_sampling_client(model_path=snapshot, tokenizer=self.tokenizer)
        try:
            params = FiretitanSamplingParams(
                max_tokens=cfg.max_sample_tokens,
                temperature=cfg.temperature,
                stop=self.renderer.get_stop_sequences(),
            )
            results: list[Any] = []
            chunk = max(1, cfg.prompt_concurrency)
            for start in range(0, len(prompts), chunk):
                futures = [
                    sampler.sample(
                        prompt=prompt,
                        num_samples=cfg.eval_group_size,
                        sampling_params=params,
                    )
                    for prompt in prompts[start : start + chunk]
                ]
                results.extend(future.result(timeout=cfg.sampling_timeout_s) for future in futures)
        finally:
            sampler.close()

        self._prescore(results, self.eval_rows)
        rewards: list[float] = []
        response_lengths: list[int] = []
        correct_count = 0
        think_lengths: list[float] = []
        truncated_samples = 0
        scorer_failed_samples = 0
        completions: list[dict[str, Any]] = []
        for dataset_row_index, row, result in zip(
            self.eval_row_indices,
            self.eval_rows,
            results,
            strict=True,
        ):
            ground_truth = json.loads(row["ground_truth"])
            for sample_index, seq in enumerate(getattr(result, "sequences", []) or []):
                tokens = list(getattr(seq, "tokens", []) or [])
                content = get_text_content(self.renderer.parse_response(tokens)[0])
                try:
                    reward, correct, think_tokens, _total = self._score(tokens, row)
                except ScorerFailure:
                    scorer_failed_samples += 1   # a missing score is not a zero; leave the sample out
                    continue
                truncated = _is_truncated(seq, cfg.max_sample_tokens)
                rewards.append(reward)
                correct_count += int(correct)
                think_lengths.append(think_tokens)
                response_lengths.append(len(tokens))
                truncated_samples += int(truncated)
                completions.append(
                    {
                        "completed_steps": completed_steps,
                        "snapshot": snapshot,
                        "dataset_row_index": dataset_row_index,
                        "brief_id": ground_truth["brief_id"],
                        "sample_index": sample_index,
                        "reward": reward,
                        "eligible": correct,
                        "tropes_per_100w": think_tokens,
                        "response_tokens": len(tokens),
                        "finish_reason": _finish_reason(seq),
                        "response": content,
                    }
                )

        expected_samples = len(self.eval_rows) * cfg.eval_group_size
        returned_samples = len(rewards)
        rec = {
            "completed_steps": completed_steps,
            "snapshot": snapshot,
            "eval/raw_reward": sum(rewards) / returned_samples if returned_samples else 0.0,
            "eval/eligible_rate": correct_count / returned_samples if returned_samples else 0.0,
            "eval/scorer_failed_samples": scorer_failed_samples,
            "eval/tropes_per_100w": sum(think_lengths) / returned_samples if returned_samples else 0.0,
            "eval/detector_mean_eligible": (sum(r for r in rewards if r > 0) / correct_count) if correct_count else 0.0,
            "eval/prompt_groups": len(self.eval_rows),
            "eval/samples": returned_samples,
            "eval/expected_samples": expected_samples,
            "eval/return_ratio": returned_samples / expected_samples if expected_samples else 0.0,
            "eval/mean_response_tokens": (
                sum(response_lengths) / returned_samples if returned_samples else 0.0
            ),
            "eval/max_response_tokens": max(response_lengths, default=0),
            "eval/truncated_samples": truncated_samples,
            "eval/truncation_ratio": (
                truncated_samples / returned_samples if returned_samples else 0.0
            ),
            "perf/eval_wall_time": time.time() - started,
        }
        with self.eval_metrics_path.open("a") as handle:
            handle.write(json.dumps(rec) + "\n")
        completions_path = self.eval_completions_dir / f"step-{completed_steps:04d}.jsonl"
        with completions_path.open("w") as handle:
            for completion in completions:
                handle.write(json.dumps(completion) + "\n")
        _log_wandb_eval(completed_steps, rec)
        print(
            f"eval after_steps={completed_steps:02d} "
            f"reward={rec['eval/raw_reward']:.3f} "
            f"eligible={rec['eval/eligible_rate']:.1%} "
            f"detector={rec['eval/detector_mean_eligible']:.3f} tropes={rec['eval/tropes_per_100w']:.2f} "
            f"samples={returned_samples}/{expected_samples} "
            f"truncated={truncated_samples} "
            f"max_tokens={rec['eval/max_response_tokens']} "
            f"elapsed={rec['perf/eval_wall_time']:.1f}s",
            flush=True,
        )
        return rec

    def _score_one(self, content: str, gt: dict[str, Any]) -> tuple[float, bool, float] | None:
        """(reward, eligible, tropes_per_100w) for one visible tweet; network calls inside.

        Returns None when a scorer call (judge or detector) failed. A failure
        is not a verdict: scoring it as 0 would hand a good tweet a negative
        advantage inside its GRPO group and train against API noise, so the
        caller drops the whole group and never caches the failure.
        """
        tropes = trope_report(content)["per_100_words"]
        if not content.strip() or not meets_format(content, gt["platform"]):
            return 0.0, False, tropes
        with httpx.Client() as client:
            jud = llm_judge(client, content, gt["platform"], gt["topic"], gt["article"])
            if jud is None:
                return None
            if any(jud[k] < ELIGIBILITY_MIN for k in ("faithful", "coherent", "on_format", "specific")):
                return 0.0, False, tropes
            hit = gptzero_human(client, content, self.gptzero_key)
        if hit is None:
            return None
        return float(hit[0]), True, tropes

    def _prescore(self, results: list[Any], rows: list[dict[str, Any]]) -> None:
        """Score every sampled completion concurrently, filling the cache keyed by (brief, text)."""
        items = []
        pending = set()
        for row, result in zip(rows, results):
            gt = json.loads(row["ground_truth"])
            for seq in getattr(result, "sequences", []) or []:
                tokens = list(getattr(seq, "tokens", []) or [])
                content = get_text_content(self.renderer.parse_response(tokens)[0])
                key = (gt["brief_id"], content)
                if key not in self._score_cache and key not in pending:
                    pending.add(key)
                    items.append((key, content, gt))
        with ThreadPoolExecutor(self.cfg.score_concurrency) as ex:
            for key, score in zip([k for k, _, _ in items], ex.map(lambda it: self._score_one(it[1], it[2]), items)):
                if score is not None:   # failures are retried on the next lookup, never cached
                    self._score_cache[key] = score

    def _score(self, tokens: list[int], row: dict[str, Any]) -> tuple[float, bool, float, int]:
        """(reward, eligible, tropes_per_100w, total_tokens) from the cache filled by _prescore."""
        gt = json.loads(row["ground_truth"])
        content = get_text_content(self.renderer.parse_response(tokens)[0])
        key = (gt["brief_id"], content)
        if key not in self._score_cache:
            score = self._score_one(content, gt)
            if score is None:
                raise ScorerFailure(key[0])
            self._score_cache[key] = score
        reward, eligible, tropes = self._score_cache[key]
        return reward, eligible, tropes, len(tokens)

    def _step(self, step: int) -> dict[str, Any]:
        with phase_span(
            "training_step",
            category="train",
            attributes={
                "step": step,
                "prompt_groups": self.cfg.prompt_groups_per_step,
                "group_size": self.cfg.group_size,
            },
        ) as span:
            result = self._step_impl(step)
            if span is not None:
                span.set_attribute("trained", result["train/trained"])
                span.set_attribute("rollout_samples", result["rollout/raw_samples"])
            return result

    def _step_impl(self, step: int) -> dict[str, Any]:
        t0 = time.time()
        cfg = self.cfg

        # 1. Save the current LoRA weights so the sampler can serve them, then
        #    open a sampling client bound to that exact snapshot.
        save_name = f"{cfg.checkpoint_name}-{step:04d}"
        with phase_span(
            "sampler_weight_snapshot",
            category="weight_sync",
            attributes={"step": step},
        ):
            snapshot = self.training_client.save_weights_for_sampler(save_name).result().path
        if not snapshot:
            raise RuntimeError(f"save_weights_for_sampler({save_name!r}) returned no path")

        batch = self._next_batch()
        prompts = [self.renderer.build_generation_prompt(row["messages"], prefill="</think>") for row in batch]
        for prompt in prompts:
            _validate_length("prompt + max_sample_tokens", prompt.length + cfg.max_sample_tokens, cfg.max_seq_len)

        # 2. Roll out `group_size` completions per prompt, a few prompts in
        #    flight at a time.
        with phase_span(
            "rollout_batch",
            category="rollout",
            attributes={
                "step": step,
                "prompt_groups": len(prompts),
                "group_size": cfg.group_size,
            },
        ):
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
        self._prescore(results, batch)
        datums: list[Any] = []
        raw_rewards: list[float] = []
        filtered_rewards: list[float] = []
        correct_count = 0
        think_token_count = 0.0
        response_token_count = 0
        max_response_tokens = 0
        truncated_samples = 0
        opener_distinct_total = 0
        opener_samples_total = 0

        scorer_failed_groups = 0
        for result, prompt, row in zip(results, prompts, batch):
            tokens_g: list[list[int]] = []
            logprobs_g: list[list[float]] = []
            routing_matrices_g: list[list[str] | None] = []
            rewards_g: list[float] = []
            try:
                scored = []
                for seq in getattr(result, "sequences", []) or []:
                    tokens = list(getattr(seq, "tokens", []) or [])
                    logprobs = getattr(seq, "logprobs", None)
                    # The importance-sampling loss needs one logprob per sampled token.
                    if not tokens or logprobs is None or len(logprobs) != len(tokens):
                        continue
                    scored.append((seq, tokens, logprobs, self._score(tokens, row)))
            except ScorerFailure as e:
                scorer_failed_groups += 1
                print(f"scorer failure on brief {e}; dropping its group this step", flush=True)
                continue
            for seq, tokens, logprobs, (reward, correct, think_tokens, _total) in scored:
                correct_count += int(correct)
                think_token_count += think_tokens
                tokens_g.append(tokens)
                logprobs_g.append([float(x) for x in logprobs])
                routing_matrices = getattr(seq, "routing_matrices", None)
                routing_matrices_g.append(
                    copy_routing(routing_matrices)
                )
                rewards_g.append(reward)
                raw_rewards.append(reward)
                response_token_count += len(tokens)
                max_response_tokens = max(max_response_tokens, len(tokens))
                truncated_samples += int(_is_truncated(seq, cfg.max_sample_tokens))

            if cfg.diversity_penalty > 0 and len(tokens_g) > 1:
                openers = [_opener(get_text_content(self.renderer.parse_response(t)[0])) for t in tokens_g]
                for i, op in enumerate(openers):
                    share = sum(1 for j, o in enumerate(openers) if j != i and o == op) / (len(openers) - 1)
                    rewards_g[i] = rewards_g[i] * (1.0 - cfg.diversity_penalty * share)
                opener_distinct_total += len(set(openers))
                opener_samples_total += len(openers)

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
                model_input = prompt.append(sdk.EncodedTextChunk(tokens=tokens[:-1]))
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
                        update=routing_model_input_kwargs(aligned_routes)
                    )
                response_len = model_input.length - response_start
                datums.append(
                    sdk.Datum(
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
        #    the mean (policy - sample) logprob gap and its non-negative k3
        #    transform are logged as kld/mean_k1 and kld/mean_k3. These are
        #    logging-only and need no reference forward pass.
        loss = None
        kld_k1 = None
        kld_k3 = None
        if datums:
            with phase_span(
                "forward_backward",
                category="train",
                attributes={"step": step, "datums": len(datums)},
            ):
                fb = self.training_client.forward_backward(datums, "importance_sampling").result()
            loss = _mean_loss(fb)
            kld_k1 = _mean_policy_sample_logprob_gap(datums, fb)
            kld_k3 = _mean_policy_sample_k3(datums, fb)
            adam = sdk.AdamParams(learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0)
            with phase_span(
                "optimizer_step",
                category="train",
                attributes={"step": step},
            ):
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
            "rollout/eligible_rate": correct_count / len(raw_rewards) if raw_rewards else 0.0,
            "rollout/scorer_failed_groups": scorer_failed_groups,
            "rollout/distinct_opener_ratio": (opener_distinct_total / opener_samples_total) if opener_samples_total else None,
            "rollout/tropes_per_100w": think_token_count / len(raw_rewards) if raw_rewards else 0.0,
            "rollout/mean_response_tokens": response_token_count / len(raw_rewards) if raw_rewards else 0.0,
            "rollout/max_response_tokens": max_response_tokens,
            "rollout/truncated_samples": truncated_samples,
            "rollout/truncation_ratio": (
                truncated_samples / len(raw_rewards) if raw_rewards else 0.0
            ),
            "train/loss": loss,
            "train/trained": bool(datums),
            "train/router_replay": self.router_replay_enabled,
            "train/inference_k1": kld_k1,
            "train/inference_k3": kld_k3,
            "kld/mean_k1": kld_k1,
            "kld/mean_k3": kld_k3,
            "perf/step_wall_time": time.time() - t0,
        }
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        _log_wandb_step(step, rec)
        print(
            f"step {step:02d} reward={raw_reward:.3f}/{filtered_reward:.3f} "
            f"eligible={rec['rollout/eligible_rate']:.1%} tropes={rec['rollout/tropes_per_100w']:.2f} "
            f"samples={len(raw_rewards)}/{len(filtered_rewards)} "
            f"filter={filter_ratio:.1%} "
            f"truncated={truncated_samples} "
            f"loss={'n/a' if loss is None else f'{loss:.4f}'} "
            f"kld_k1={'n/a' if kld_k1 is None else f'{kld_k1:+.4f}'} "
            f"kld_k3={'n/a' if kld_k3 is None else f'{kld_k3:.4f}'} "
            f"elapsed={rec['perf/step_wall_time']:.1f}s",
            flush=True,
        )
        return rec

    def run(self) -> list[dict[str, Any]]:
        try:
            cfg = self.cfg
            records: list[dict[str, Any]] = []
            last_eval_step: int | None = None
            if cfg.eval_at_start:
                self._evaluate(self.start_step)
                last_eval_step = self.start_step
            for step in range(self.start_step, self.start_step + cfg.steps):
                records.append(self._step(step))
                completed_step = step + 1
                if cfg.dcp_save_interval and completed_step % cfg.dcp_save_interval == 0:
                    self._save_dcp(completed_step)
                if cfg.eval_interval and completed_step % cfg.eval_interval == 0:
                    self._evaluate(completed_step)
                    last_eval_step = completed_step

            final_step = self.start_step + cfg.steps
            if cfg.eval_at_end and last_eval_step != final_step:
                self._evaluate(final_step)
                last_eval_step = final_step

            resume_ref = ""
            if cfg.dcp_save_interval:
                final_name = f"{cfg.training_checkpoint_name}-{final_step:04d}"
                if final_name not in self.saved_training_checkpoints:
                    final_name = self._save_dcp(final_step)
                resume_ref = self._resume_reference(final_name)
                print(f"resume this run with:\n  --resume-from {resume_ref}", flush=True)
                (self.run_dir / "resume_from.txt").write_text(f"{resume_ref}\n")

            with phase_span(
                "final_sampler_checkpoint",
                category="checkpoint",
                attributes={"completed_steps": final_step},
            ):
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
        ax.set_title(f"Anti-slop serverless RL ({self.base_model}, importance_sampling)")
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
        description="Anti-slop RL: serverless GRPO on tweet briefs against a gated detector reward",
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
    parser.add_argument("--eval-prompt-groups", type=int, default=Config.eval_prompt_groups)
    parser.add_argument("--eval-seed", type=int, default=Config.eval_seed)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--prompt-groups-per-step", type=int, default=Config.prompt_groups_per_step)
    parser.add_argument("--group-size", type=int, default=Config.group_size)
    parser.add_argument("--prompt-concurrency", type=int, default=Config.prompt_concurrency)
    parser.add_argument("--max-sample-tokens", type=int, default=Config.max_sample_tokens)
    parser.add_argument("--temperature", type=float, default=Config.temperature)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--score-concurrency", type=int, default=Config.score_concurrency)
    parser.add_argument("--init-from-state", default=Config.init_from_state,
                        help="Start from a DPO training checkpoint's weights + optimizer, e.g. '<account>/<run-id>/dpo-0027'.")
    parser.add_argument("--eval-interval", type=int, default=Config.eval_interval)
    parser.add_argument("--eval-group-size", type=int, default=Config.eval_group_size)
    parser.add_argument(
        "--eval-at-start",
        action=argparse.BooleanOptionalAction,
        default=Config.eval_at_start,
    )
    parser.add_argument(
        "--eval-at-end",
        action=argparse.BooleanOptionalAction,
        default=Config.eval_at_end,
    )
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
    parser.add_argument("--diversity-penalty", type=float, default=Config.diversity_penalty)
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
    parser.add_argument("--prepare-dataset-rows", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.prepare_dataset:
        prepare_dataset(Path(args.dataset), args.prepare_dataset_rows, args.seed)
        return
    cfg = Config(**{k: v for k, v in vars(args).items() if k in Config.__dataclass_fields__})
    if not cfg.api_key:
        raise SystemExit("FIREWORKS_API_KEY is required (export it or put it in training/.env)")
    ServerlessAntislopRL(cfg).run()


if __name__ == "__main__":
    main()
