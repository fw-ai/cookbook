#!/usr/bin/env python3
"""Serverless RL on the Countdown task -- a self-contained, Tinker-style loop.

This example shows how to run reinforcement learning against Fireworks
**serverless training**: you connect to a shared, already-running pooled trainer
through the gateway and get back a Tinker-compatible training client. There is
**no trainer job to provision and no inference deployment to stand up** -- the
same service hands you both a training client and a sampling client. That is the
whole pitch of the serverless path, and it is what makes this read like a native
Tinker RL example.

The loop is the standard GRPO/importance-sampling shape:

    service = FiretitanServiceClient(base_url=".../training/v1/serverless")
    training_client = service.create_lora_training_client(base_model, rank)
    for step in range(steps):
        snapshot = training_client.save_weights_for_sampler(name).result().path
        sampler  = service.create_sampling_client(model_path=snapshot, tokenizer=...)
        # sample a group of completions per prompt, score them, and turn
        # group-relative advantages into importance-sampling training datums
        training_client.forward_backward(datums, "importance_sampling").result()
        training_client.optim_step(adam).result()

Each step saves the current LoRA weights for the sampler, rolls out a batch of
Countdown prompts through that snapshot, scores completions with
``countdown_rewards.composite_reward``, computes group-relative advantages, and
takes one optimizer step. Reward should climb as the policy learns to emit valid
Countdown equations.

Distilled from the internal serverless Countdown e2e journey
(``serverless_countdown_rl_journey_v2.py``); the e2e-only stage assertions,
checkpoint list/promote gates, and multi-model isolation checks are dropped so
the training loop itself stays front and center.

Usage:
    export FIREWORKS_API_KEY=fw_...
    python -m examples.serverless_rl.countdown_rl
    # or run the file directly from the training/ directory:
    #   python examples/serverless_rl/countdown_rl.py
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tinker
from fireworks.training.sdk import FiretitanServiceClient
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

try:  # Load FIREWORKS_API_KEY / FIREWORKS_BASE_URL from a local .env if present.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from examples.serverless_rl.countdown_rewards import composite_reward

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "data" / "countdown_train.jsonl"


@dataclass
class Config:
    """Everything you might want to tune. Edit the ``__main__`` block below."""

    # --- What to train ------------------------------------------------------
    base_model: str = "accounts/fireworks/models/qwen3p5-27b"
    # HuggingFace tokenizer matching ``base_model`` -- used to render prompts and
    # decode sampled tokens client-side.
    tokenizer_model: str = "Qwen/Qwen3.5-27B"
    # Leave "" to auto-pick the recommended chat renderer for the tokenizer.
    renderer_name: str = ""
    dataset: str = str(DEFAULT_DATASET)
    lora_rank: int = 8

    # --- RL loop shape ------------------------------------------------------
    steps: int = 10
    # Prompts sampled per optimizer step (each prompt becomes one GRPO group).
    prompt_groups_per_step: int = 8
    # Completions sampled per prompt (the group the advantage is computed over).
    group_size: int = 8
    # How many prompts' sample() calls are in flight at once.
    prompt_concurrency: int = 4
    max_sample_tokens: int = 1024
    temperature: float = 1.0
    learning_rate: float = 2.5e-5

    # --- Connection ---------------------------------------------------------
    # Prod gateway by default; override with FIREWORKS_BASE_URL (e.g. a dev
    # gateway). The "/training/v1/serverless" suffix is added automatically.
    api_base_url: str = field(default_factory=lambda: os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"))
    api_key: str = field(default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", ""))

    # --- Bookkeeping --------------------------------------------------------
    checkpoint_name: str = "countdown"
    final_checkpoint_name: str = "countdown-final"
    sampling_timeout_s: float = 600.0
    run_dir: str = ""
    # Requires matplotlib; set False (or don't install it) to skip the plot.
    plot_reward_curve: bool = True


def _serverless_base_url(base_url: str) -> str:
    """Serverless training + sampling both hang off ``/training/v1/serverless``."""
    root = base_url.rstrip("/")
    if root.endswith("/training/v1/serverless"):
        return root
    if root.endswith("/training/v1"):
        return f"{root}/serverless"
    return f"{root}/training/v1/serverless"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _group_relative_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Standardize rewards within a group (GRPO): ``(r - mean) / std``."""
    if len(rewards) <= 1:
        return [0.0 for _ in rewards]
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
    std = math.sqrt(variance)
    if std < eps:
        std = 1.0
    return [(r - mean) / (std + eps) for r in rewards]


def _mean_loss(fb_output: Any) -> float | None:
    """Mean NLL from a forward_backward result: loss:sum / response_tokens."""
    metrics = getattr(fb_output, "metrics", None) or {}
    loss_sum = metrics.get("loss:sum")
    tokens = metrics.get("response_tokens") or metrics.get("num_loss_tokens") or 1.0
    if loss_sum is not None:
        return float(loss_sum) / max(float(tokens), 1.0)
    return None


class ServerlessCountdownRL:
    """One serverless RL run over the Countdown dataset."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rows = _load_rows(Path(cfg.dataset))
        if not self.rows:
            raise SystemExit(f"dataset is empty: {cfg.dataset}")

        self.tokenizer = get_tokenizer(cfg.tokenizer_model)
        renderer_name = cfg.renderer_name or get_recommended_renderer_name(cfg.tokenizer_model)
        self.renderer = get_renderer(renderer_name, self.tokenizer)

        # The one connection that gives us BOTH training and sampling clients.
        # No trainer job, no deployment -- just a pooled serverless session.
        self.service = FiretitanServiceClient(
            api_key=cfg.api_key,
            base_url=_serverless_base_url(cfg.api_base_url),
        )
        self.training_client = self.service.create_lora_training_client(
            base_model=cfg.base_model,
            rank=cfg.lora_rank,
        )

        self.run_dir = (
            Path(cfg.run_dir).resolve()
            if cfg.run_dir
            else Path("/tmp") / f"serverless_countdown_{int(time.time())}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.row_cursor = 0

        session = getattr(self.service, "training_session_name", None) or getattr(
            self.service, "training_session_id", None
        )
        run_id = getattr(self.training_client, "run_id", None)
        print(
            f"connected serverless session={session} run={run_id}\n"
            f"base_model={cfg.base_model} tokenizer={cfg.tokenizer_model} renderer={renderer_name}\n"
            f"steps={cfg.steps} prompt_groups_per_step={cfg.prompt_groups_per_step} "
            f"group_size={cfg.group_size} lora_rank={cfg.lora_rank} lr={cfg.learning_rate}\n"
            f"run_dir={self.run_dir}",
            flush=True,
        )

    def _next_batch(self) -> list[dict[str, Any]]:
        """Take the next ``prompt_groups_per_step`` rows, wrapping around so the
        bundled 32-row sample supports a multi-step run. Point ``cfg.dataset`` at
        a larger file for real training."""
        batch = [self.rows[(self.row_cursor + i) % len(self.rows)] for i in range(self.cfg.prompt_groups_per_step)]
        self.row_cursor += self.cfg.prompt_groups_per_step
        return batch

    def _sample_group(self, sampler: Any, prompt: Any) -> Any:
        params = tinker.SamplingParams(
            max_tokens=self.cfg.max_sample_tokens,
            temperature=self.cfg.temperature,
            stop=self.renderer.get_stop_sequences(),
        )
        return sampler.sample(prompt=prompt, num_samples=self.cfg.group_size, sampling_params=params)

    def _step(self, step: int) -> dict[str, Any]:
        t0 = time.time()
        cfg = self.cfg

        # 1. Save the current LoRA weights so the sampler can serve them, then
        #    open a sampling client bound to that exact snapshot. On the
        #    serverless path the rollout host is the same session -- no
        #    deployment to create or hot-load.
        save_name = f"{cfg.checkpoint_name}-{step:04d}"
        snapshot = self.training_client.save_weights_for_sampler(save_name).result().path
        if not snapshot:
            raise RuntimeError(f"save_weights_for_sampler({save_name!r}) returned no path")

        batch = self._next_batch()
        prompts = [self.renderer.build_generation_prompt(row["messages"]) for row in batch]

        # 2. Roll out `group_size` completions per prompt, a few prompts in
        #    flight at a time.
        sampler = self.service.create_sampling_client(model_path=snapshot, tokenizer=self.tokenizer)
        try:
            results: list[Any] = []
            chunk = max(1, cfg.prompt_concurrency)
            for start in range(0, len(prompts), chunk):
                futures = [self._sample_group(sampler, p) for p in prompts[start : start + chunk]]
                for fut in futures:
                    results.append(fut.result(timeout=cfg.sampling_timeout_s) if hasattr(fut, "result") else fut)
        finally:
            sampler.close()

        # 3. Score each completion and keep only groups with reward spread (a
        #    group where every sample scores the same yields a zero advantage
        #    and no learning signal, so we drop it -- standard GRPO filtering).
        datums: list[Any] = []
        raw_rewards: list[float] = []
        raw_correct = 0
        kept_groups = 0

        for result, prompt, row in zip(results, prompts, batch):
            tokens_g: list[list[int]] = []
            logprobs_g: list[list[float]] = []
            rewards_g: list[float] = []
            for seq in getattr(result, "sequences", []) or []:
                tokens = list(getattr(seq, "tokens", []) or [])
                logprobs = getattr(seq, "logprobs", None)
                # Serverless sampling returns generated-token logprobs; the
                # importance-sampling loss needs one logprob per sampled token.
                if not tokens or logprobs is None or len(logprobs) != len(tokens):
                    continue
                content = get_text_content(self.renderer.parse_response(tokens)[0])
                reward = float(composite_reward(content, row["ground_truth"]))
                tokens_g.append(tokens)
                logprobs_g.append([float(x) for x in logprobs])
                rewards_g.append(reward)
                raw_rewards.append(reward)
                raw_correct += 1 if reward >= 0.9 else 0

            if not rewards_g or len(set(rewards_g)) <= 1:
                continue
            kept_groups += 1

            advantages = _group_relative_advantages(rewards_g)
            response_start = prompt.length - 1
            for tokens, logprobs, advantage in zip(tokens_g, logprobs_g, advantages):
                # Shifted next-token layout: the model sees prompt + all but the
                # last sampled token; targets/logprobs/advantages are aligned to
                # the response region and left-padded over the prompt.
                model_input = prompt.append(tinker.EncodedTextChunk(tokens=tokens[:-1]))
                datums.append(
                    tinker.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": [0] * response_start + tokens,
                            "logprobs": [0.0] * response_start + logprobs,
                            "advantages": [0.0] * response_start + [advantage] * (model_input.length - response_start),
                        },
                    )
                )

        # 4. One importance-sampling update + optimizer step.
        loss = None
        if datums:
            fb = self.training_client.forward_backward(datums, "importance_sampling").result()
            loss = _mean_loss(fb)
            adam = tinker.AdamParams(learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)
            self.training_client.optim_step(adam).result()

        raw_reward = sum(raw_rewards) / len(raw_rewards) if raw_rewards else 0.0
        raw_accuracy = raw_correct / len(raw_rewards) if raw_rewards else 0.0
        rec = {
            "step": step,
            "snapshot": snapshot,
            "rollout/raw_reward": raw_reward,
            "rollout/raw_accuracy": raw_accuracy,
            "rollout/samples": len(raw_rewards),
            "rollout/kept_groups": kept_groups,
            "train/loss": loss,
            "train/trained": bool(datums),
            "perf/step_wall_time": time.time() - t0,
        }
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"step {step:02d} raw_reward={raw_reward:.3f} raw_acc={raw_accuracy:.3f} "
            f"kept_groups={kept_groups}/{len(batch)} samples={len(raw_rewards)} "
            f"loss={'n/a' if loss is None else f'{loss:.4f}'} "
            f"elapsed={rec['perf/step_wall_time']:.1f}s",
            flush=True,
        )
        return rec

    def run(self) -> list[dict[str, Any]]:
        records = [self._step(step) for step in range(self.cfg.steps)]

        final = self.training_client.save_weights_for_sampler(self.cfg.final_checkpoint_name).result()
        print(f"final sampler checkpoint: {getattr(final, 'path', None)}", flush=True)

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
        ax.plot(steps, [r["rollout/raw_accuracy"] for r in records], marker="s", linestyle="--", label="raw_accuracy")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("score")
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"Serverless Countdown RL ({self.cfg.base_model}, importance_sampling)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        plot_path = self.run_dir / "reward_curve.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"reward curve: {plot_path}", flush=True)


def main(cfg: Config) -> None:
    if not cfg.api_key:
        raise SystemExit("FIREWORKS_API_KEY is required (export it or set Config.api_key)")
    ServerlessCountdownRL(cfg).run()


if __name__ == "__main__":
    # Fork this: point `dataset` at your own JSONL, swap `base_model` /
    # `tokenizer_model`, and tune the RL loop shape.
    main(Config())
