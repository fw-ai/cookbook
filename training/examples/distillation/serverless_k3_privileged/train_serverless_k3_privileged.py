#!/usr/bin/env python3
"""Privileged self-distillation (SDFT-style reverse KL) on SERVERLESS Kimi K3.

This is the serverless counterpart to ``gsm8k_privileged`` (which targets the
dedicated/managed path: a training shape + an auto-deployed teacher inference
deployment). Here there is **no trainer job and no deployment** -- one pooled
serverless session hands us both a LoRA training client and a Tinker-shaped
sampling client, exactly like ``examples/serverless_rl/countdown_rl.py``.

The problem this solves (privileged information ``I``): a model such as Kimi K3
only exhibits a desired behavior when steered by a strong instruction ``I`` that
will NOT be present at deployment. SFT on the ``I``-conditioned trajectories
(with ``I`` stripped) shifts the data distribution and risks catastrophic
forgetting. On-policy reverse-KL self-distillation avoids that: the student
samples its own rollouts WITHOUT ``I``, and is pulled toward the same weights
conditioned WITH ``I`` via a per-token reverse-KL signal.

    teacher = base K3 weights, prompt = x + I     (privileged context)
    student = current LoRA weights, prompt = x    (deployable context)
    per-token advantage = teacher_logprob - sampling_logprob   (sampled reverse KL)
    loss = server-side builtin ``importance_sampling``

Why the teacher is the FROZEN step-0 snapshot (not a base-model sampler):
the serverless gateway parser
(``api_gateway/pkg/middlewares/serverless_rl_rollout.go``) still requires a
``/checkpoints/<id>`` segment in the sampling model name, so a checkpoint-less
base-model sampler is rejected on serverless (the fix is PR #38886, unmerged).
The step-0 LoRA snapshot is numerically the base weights (LoRA init is zero), is
a real checkpoint, and stays fixed for the whole run -- so it is a valid frozen
self-teacher that passes the parser today. We serve it by keeping its sampler
client open (each ``create_sampling_client`` hot-loads one snapshot onto the
session host; the teacher sampler holds the step-0 snapshot while the student
sampler is re-created each step on the freshly saved snapshot).

The reverse-KL datum math is reused from the dedicated recipe
(``training.utils.distillation.build_opd_server_datums``); only the serving
layer differs (serverless session instead of dedicated deployments).

Usage:
    export FIREWORKS_API_KEY=fw_...           # account-scoped key that can read kimi-k3
    export FIREWORKS_BASE_URL=https://dev.api.fireworks.ai   # dev gateway (optional)
    python -m training.examples.distillation.serverless_k3_privileged.train_serverless_k3_privileged
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tinker
from fireworks.training.sdk import FiretitanServiceClient

# Registering the "kimi_k3" renderer is a side effect of this import.
import training.renderer.kimi_k3  # noqa: F401
from training.utils.distillation import build_opd_server_datums
from training.utils.tokenizers import load_tokenizer
from tinker_cookbook.renderers import get_renderer

try:  # Load FIREWORKS_API_KEY / FIREWORKS_BASE_URL from a local .env if present.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "data" / "gsm8k_privileged.jsonl"

# The privileged instruction I. Present ONLY in the teacher prompt; it steers K3
# toward short, verifiable, final-answer-terminated reasoning. The student never
# sees it -- the goal is to internalize the behavior into the LoRA weights.
PRIVILEGED_INSTRUCTION = (
    "You are a meticulous mathematician. Reason step by step, keep the reasoning "
    "concise, and verify each arithmetic step before moving on. End your response "
    "with exactly one line of the form 'Final: <answer>'."
)


@dataclass
class Config:
    """Everything you might want to tune. Override via CLI flags."""

    # --- What to distill ----------------------------------------------------
    base_model: str = "accounts/fireworks/models/kimi-k3"
    # HuggingFace tokenizer matching ``base_model`` (renders prompts client-side).
    tokenizer_model: str = "moonshotai/Kimi-K3"
    renderer_name: str = "kimi_k3"
    dataset: str = str(DEFAULT_DATASET)
    privileged_instruction: str = PRIVILEGED_INSTRUCTION
    lora_rank: int = 8
    # Serverless has no training shape from which to infer this bound.
    max_seq_len: int = 8192

    # --- Distillation loop shape --------------------------------------------
    steps: int = 4
    prompt_groups_per_step: int = 4
    completions_per_prompt: int = 4
    # How many prompts' sample() calls are in flight at once (the rollout loop
    # chunks by this, mirroring serverless_rl/countdown_rl.py).
    prompt_concurrency: int = 4
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    learning_rate: float = 2e-5
    # Scales the per-token (teacher - sampling) advantage.
    loss_scale: float = 1.0

    # --- Connection ----------------------------------------------------------
    # Prod gateway by default; override with FIREWORKS_BASE_URL for a dev
    # gateway. The "/training/v1/serverless" suffix is added automatically.
    api_base_url: str = field(
        default_factory=lambda: os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    )
    api_key: str = field(default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", ""))

    # --- Bookkeeping ---------------------------------------------------------
    teacher_snapshot_name: str = "sdft-teacher-step0"
    student_snapshot_name: str = "sdft-student"
    # Saved once after the final optim_step so the fully trained weights are
    # hot-loadable (mirrors serverless_rl/countdown_rl.py's final checkpoint).
    final_snapshot_name: str = "sdft-student-final"
    sampling_timeout_s: float = 900.0
    run_dir: str = ""


def parse_args() -> Config:
    defaults = Config()
    p = argparse.ArgumentParser(description="Serverless privileged self-distillation on Kimi K3")
    p.add_argument("--base-model")
    p.add_argument("--tokenizer-model")
    p.add_argument("--renderer-name")
    p.add_argument("--dataset")
    p.add_argument("--privileged-instruction")
    p.add_argument("--lora-rank", type=int)
    p.add_argument("--max-seq-len", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--prompt-groups-per-step", type=int)
    p.add_argument("--completions-per-prompt", type=int)
    p.add_argument("--prompt-concurrency", type=int)
    p.add_argument("--max-completion-tokens", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--loss-scale", type=float)
    p.add_argument("--final-snapshot-name")
    p.add_argument("--run-dir")
    return p.parse_args(namespace=defaults)


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


def _student_messages(question: str) -> list[dict[str, str]]:
    # The deployable prompt: NO privileged instruction.
    return [{"role": "user", "content": question}]


def _teacher_messages(question: str, instruction: str) -> list[dict[str, str]]:
    # The privileged prompt: the SAME question, plus instruction I in the system turn.
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]


def _mean_loss(fb_output: Any) -> float | None:
    metrics = getattr(fb_output, "metrics", None) or {}
    loss_sum = metrics.get("loss:sum")
    tokens = metrics.get("response_tokens") or metrics.get("num_loss_tokens") or 1.0
    if loss_sum is not None:
        return float(loss_sum) / max(float(tokens), 1.0)
    return None


class ServerlessK3PrivilegedDistillation:
    """One serverless privileged self-distillation run over the dataset."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        if cfg.max_seq_len <= 0:
            raise ValueError("serverless training requires max_seq_len > 0")
        if cfg.lora_rank <= 0:
            raise ValueError("serverless training requires lora_rank > 0")
        self.rows = _load_rows(Path(cfg.dataset))
        if not self.rows:
            raise SystemExit(f"dataset is empty: {cfg.dataset}")

        # Kimi K3's tokenizer ships custom code, so it needs trust_remote_code;
        # load_tokenizer enables that by default.
        self.tokenizer = load_tokenizer(cfg.tokenizer_model)
        self.renderer = get_renderer(cfg.renderer_name, self.tokenizer)

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
            else Path("/tmp") / f"serverless_k3_distill_{int(time.time())}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.checkpoints_path = self.run_dir / "checkpoints.json"
        # Runtime record of every sampler checkpoint path saved this run, so a
        # later session can hot-load any of them (e.g. to compare an early vs a
        # late snapshot) without re-deriving the names. Not config; bookkeeping.
        self._checkpoints: list[dict[str, Any]] = []
        self.row_cursor = 0

        # Save the step-0 snapshot once and pin it as the frozen teacher. LoRA
        # init is zero, so this snapshot serves the base weights; it is a real
        # checkpoint, so its sampler passes the serverless gateway parser.
        self.teacher_snapshot = (
            self.training_client.save_weights_for_sampler(cfg.teacher_snapshot_name).result().path
        )
        if not self.teacher_snapshot:
            raise RuntimeError("save_weights_for_sampler(teacher) returned no path")
        self._checkpoints.append({"role": "teacher", "name": cfg.teacher_snapshot_name, "path": self.teacher_snapshot})
        self.teacher_sampler = self.service.create_sampling_client(
            model_path=self.teacher_snapshot, tokenizer=self.tokenizer
        )

        # Pre-render the prompts. Teacher prompt carries I; student prompt does not.
        self._student_prompts: list[Any] = []
        self._teacher_prompt_tokens: list[list[int]] = []
        for row in self.rows:
            sp = self.renderer.build_generation_prompt(_student_messages(row["question"]))
            tp = self.renderer.build_generation_prompt(
                _teacher_messages(row["question"], cfg.privileged_instruction)
            )
            self._student_prompts.append(sp)
            self._teacher_prompt_tokens.append(list(tp.to_ints()))

        session = getattr(self.service, "training_session_name", None) or getattr(
            self.service, "training_session_id", None
        )
        print(
            f"connected serverless session={session}\n"
            f"base_model={cfg.base_model} tokenizer={cfg.tokenizer_model} renderer={cfg.renderer_name}\n"
            f"teacher_snapshot={self.teacher_snapshot}\n"
            f"rows={len(self.rows)} steps={cfg.steps} groups/step={cfg.prompt_groups_per_step} "
            f"completions/prompt={cfg.completions_per_prompt} lora_rank={cfg.lora_rank} "
            f"max_seq_len={cfg.max_seq_len} lr={cfg.learning_rate} loss_scale={cfg.loss_scale}\n"
            f"run_dir={self.run_dir}",
            flush=True,
        )

    def _next_batch_indices(self) -> list[int]:
        idxs = [
            (self.row_cursor + i) % len(self.rows)
            for i in range(self.cfg.prompt_groups_per_step)
        ]
        self.row_cursor += self.cfg.prompt_groups_per_step
        return idxs

    def _teacher_logprobs(self, teacher_prompt_tokens: list[int], completion_tokens: list[int]) -> list[float | None]:
        """Echo-score the student's completion under the privileged teacher prompt.

        The sequence is teacher_prompt + completion, so the first token is BOS and
        has no logprob; compute_logprobs returns one entry per sequence token with
        a leading None. We slice off the teacher-prompt region and return one
        logprob per completion token.
        """
        full = list(teacher_prompt_tokens) + list(completion_tokens)
        scored = self.teacher_sampler.compute_logprobs(
            tinker.ModelInput.from_ints(full)
        ).result(timeout=self.cfg.sampling_timeout_s)
        return scored[len(teacher_prompt_tokens):]

    def _step(self, step: int) -> dict[str, Any]:
        t0 = time.time()
        cfg = self.cfg

        # 1. Save the current student weights and serve them on the session host.
        save_name = f"{cfg.student_snapshot_name}-{step:04d}"
        student_snapshot = self.training_client.save_weights_for_sampler(save_name).result().path
        if not student_snapshot:
            raise RuntimeError(f"save_weights_for_sampler({save_name!r}) returned no path")
        self._checkpoints.append(
            {"role": "student", "step": step, "name": save_name, "path": student_snapshot}
        )

        batch_idx = self._next_batch_indices()

        # 2. Student rollouts: sample WITHOUT the privileged instruction, capturing
        #    the behavior-policy (sampling) logprobs the IS loss needs.
        student_sampler = self.service.create_sampling_client(
            model_path=student_snapshot, tokenizer=self.tokenizer
        )
        params = tinker.SamplingParams(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            stop=self.renderer.get_stop_sequences(),
        )
        try:
            sample_results: list[Any] = []
            chunk = max(1, cfg.prompt_concurrency)
            for start in range(0, len(batch_idx), chunk):
                futures = [
                    student_sampler.sample(
                        prompt=self._student_prompts[i],
                        num_samples=cfg.completions_per_prompt,
                        sampling_params=params,
                    )
                    for i in batch_idx[start : start + chunk]
                ]
                for fut in futures:
                    sample_results.append(
                        fut.result(timeout=cfg.sampling_timeout_s) if hasattr(fut, "result") else fut
                    )
        finally:
            student_sampler.close()

        # 3. Build OPD prompt groups: for each completion, align tokens + sampling
        #    logprobs, and score the same completion under the privileged teacher.
        prompt_groups: list[dict[str, Any]] = []
        n_scored = 0
        for i, result in zip(batch_idx, sample_results):
            teacher_prompt_tokens = self._teacher_prompt_tokens[i]
            data: list[tinker.Datum] = []
            teacher_lps: list[list[float]] = []
            sampling_lps: list[list[float]] = []
            student_prompt = self._student_prompts[i]
            response_start = student_prompt.length - 1

            for seq in getattr(result, "sequences", []) or []:
                completion_tokens = list(getattr(seq, "tokens", []) or [])
                seq_logprobs = getattr(seq, "logprobs", None)
                if (
                    not completion_tokens
                    or seq_logprobs is None
                    or len(seq_logprobs) != len(completion_tokens)
                ):
                    continue
                teacher_lp = self._teacher_logprobs(teacher_prompt_tokens, completion_tokens)
                if teacher_lp is None or len(teacher_lp) != len(completion_tokens):
                    continue

                model_input = student_prompt.append(
                    tinker.EncodedTextChunk(tokens=completion_tokens[:-1])
                )
                target_len = model_input.length
                if target_len > cfg.max_seq_len:
                    continue
                datum = tinker.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": [0] * response_start + completion_tokens,
                    },
                )
                data.append(datum)
                teacher_lps.append([0.0] * response_start + [float(v) for v in teacher_lp])
                sampling_lps.append([0.0] * response_start + [float(v) for v in seq_logprobs])
                n_scored += 1

            if data:
                prompt_groups.append(
                    {
                        "data": data,
                        "teacher_logprobs": teacher_lps,
                        "sampling_logprobs": sampling_lps,
                        "prompt_len": student_prompt.length,
                    }
                )

        # 4. Turn groups into server datums (advantage = teacher - sampling on the
        #    response region) and take one importance-sampling step.
        datums: list[tinker.Datum] = []
        input_metrics: dict[str, float] = {}
        if prompt_groups:
            flat_data: list[tinker.Datum] = []
            flat_teacher: list[list[float]] = []
            flat_sampling: list[list[float]] = []
            flat_prompt_lens: list[int] = []
            for g in prompt_groups:
                flat_data.extend(g["data"])
                flat_teacher.extend(g["teacher_logprobs"])
                flat_sampling.extend(g["sampling_logprobs"])
                flat_prompt_lens.extend([g["prompt_len"]] * len(g["data"]))
            datums, input_metrics = build_opd_server_datums(
                flat_data,
                flat_teacher,
                flat_sampling,
                flat_prompt_lens,
                loss_scale=cfg.loss_scale,
            )

        loss = None
        if datums:
            fb = self.training_client.forward_backward(datums, "importance_sampling").result()
            loss = _mean_loss(fb)
            adam = tinker.AdamParams(
                learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0
            )
            self.training_client.optim_step(adam).result()

        rec = {
            "step": step,
            "student_snapshot": student_snapshot,
            "rollout/prompts": len(batch_idx),
            "rollout/scored_completions": n_scored,
            "train/datums": len(datums),
            "train/loss": loss,
            "train/trained": bool(datums),
            **{f"opd/{k}": v for k, v in input_metrics.items()},
            "time/step_s": round(time.time() - t0, 2),
        }
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        return rec

    def _write_checkpoints(self) -> None:
        """Persist every sampler checkpoint path saved so far to checkpoints.json.

        Called from the ``finally`` in :meth:`run` so already-saved snapshot
        paths survive even if a step raises mid-run. A fresh session can
        hot-load any of them later (serverless sessions expire ~30s after the
        last request, but the underlying GCS checkpoints persist). See README
        "Reusing checkpoints across sessions".
        """
        session = getattr(self.service, "training_session_name", None) or getattr(
            self.service, "training_session_id", None
        )
        with self.checkpoints_path.open("w") as f:
            json.dump(
                {"session": session, "base_model": self.cfg.base_model, "checkpoints": self._checkpoints},
                f,
                indent=2,
            )

    def run(self) -> None:
        try:
            for step in range(self.cfg.steps):
                self._step(step)
            # Save the fully trained weights once after the last optim_step so
            # the final LoRA snapshot is hot-loadable, not discarded. Mirrors
            # serverless_rl/countdown_rl.py's final checkpoint.
            final_snapshot = (
                self.training_client.save_weights_for_sampler(self.cfg.final_snapshot_name).result().path
            )
            if final_snapshot:
                self._checkpoints.append(
                    {"role": "student", "step": "final", "name": self.cfg.final_snapshot_name, "path": final_snapshot}
                )
                print(f"final_snapshot={final_snapshot}", flush=True)
        finally:
            self.teacher_sampler.close()
            self._write_checkpoints()
        print(f"done. metrics at {self.metrics_path}", flush=True)
        print(f"checkpoints at {self.checkpoints_path}", flush=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = parse_args()
    if not cfg.api_key:
        raise SystemExit("FIREWORKS_API_KEY is required (export it or pass it in Config).")
    ServerlessK3PrivilegedDistillation(cfg).run()


if __name__ == "__main__":
    main()
