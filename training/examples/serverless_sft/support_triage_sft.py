#!/usr/bin/env python3
"""Serverless SFT with DCP checkpoints, resume, and promotion -- a self-contained,
Tinker-style loop.

This example shows supervised fine-tuning against Fireworks **serverless
training**: you connect to a shared, already-running pooled trainer through the
gateway and get back a Tinker-compatible training client. There is **no trainer
job to provision and no inference deployment to stand up**.

Where the serverless RL example (``examples/serverless_rl/countdown_rl.py``)
focuses on the rollout/score/advantage loop, this one focuses on the parts of a
real SFT job that outlive a single process:

1. **DCP checkpoints** (``save_state``) -- the full trainer state (weights *and*
   optimizer), written on an interval so a crashed or preempted run does not
   start over.
2. **Resume** (``--resume-from``) -- a *fresh process* reattaches to a saved
   checkpoint via ``create_training_client_from_state_with_optimizer`` and keeps
   training, with the dataset cursor restored so it does not re-see rows.
3. **Promote** -- the final checkpoint is saved for the sampler, listed on the
   owning TrainingSession, and promoted into a first-class Fireworks model you
   can serve.

The loop itself is the standard supervised shape::

    service = FiretitanServiceClient(base_url=".../training/v1/serverless")
    training_client = service.create_lora_training_client(base_model, rank)
    for step in range(steps):
        # prompt tokens weighted 0, response tokens weighted 1
        training_client.forward_backward(datums, "cross_entropy").result()
        training_client.optim_step(adam).result()
        if step % dcp_save_interval == 0:
            training_client.save_state(f"ckpt-{step:04d}").result()

The bundled dataset is a small support-ticket triage task: given a customer
message, emit a strict JSON object with ``category`` / ``severity`` /
``needs_human``. Loss should fall steadily as the adapter learns the format and
the label mapping.

Usage::

    export FIREWORKS_API_KEY=fw_...
    # fresh run (trains, checkpoints, promotes)
    python -m examples.serverless_sft.support_triage_sft

    # resume a previous run's checkpoint in a brand-new process
    python -m examples.serverless_sft.support_triage_sft \\
        --resume-from <account>/<run-id>/<checkpoint-name>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tinker
from fireworks.training.sdk import FireworksClient, FiretitanServiceClient

try:  # Load FIREWORKS_API_KEY / FIREWORKS_BASE_URL from a local .env if present.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from training.renderer.tokenizer import get_tokenizer
from training.utils.supervised import (
    build_renderer,
    render_messages_to_datum,
    resolve_renderer_name,
)

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "data" / "support_triage_train.jsonl"

# The trainer composes a promotable checkpoint id as "{run_id}-{name}-{suffix}",
# where run_id is "run-" + 32 hex and suffix is 8 hex. That is 46 + len(name)
# characters against a 63-character limit, so checkpoint names must stay short.
MAX_CHECKPOINT_NAME_LEN = 17


@dataclass
class Config:
    """Everything you might want to tune. Edit the ``__main__`` block below."""

    # --- What to train ------------------------------------------------------
    base_model: str = "accounts/fireworks/models/kimi-k3"
    # HuggingFace tokenizer matching ``base_model`` -- used to render prompts
    # and tokenize targets client-side.
    tokenizer_model: str = "moonshotai/Kimi-K3"
    # Leave "" to auto-pick the recommended chat renderer for the tokenizer.
    renderer_name: str = ""
    dataset: str = str(DEFAULT_DATASET)
    lora_rank: int = 8
    # Serverless has no training shape from which to infer this bound.
    max_seq_len: int = 32768

    # --- SFT loop shape -----------------------------------------------------
    steps: int = 12
    batch_size: int = 8
    learning_rate: float = 1e-4

    # --- Checkpointing ------------------------------------------------------
    # Save a DCP checkpoint (weights + optimizer) every N optimizer steps.
    # Set to 0 to disable periodic checkpoints.
    dcp_save_interval: int = 4
    # Base name for periodic DCP checkpoints; the step index is appended.
    checkpoint_name: str = "triage"
    # Sampler checkpoint saved at the end of the run, and the one promoted.
    final_checkpoint_name: str = "triage-final"
    # Resume a previous run: pass the DCP path printed by an earlier run
    # (``<account>/<run-id>/<checkpoint>``). Empty means start from the base model.
    resume_from: str = ""

    # --- Promotion ----------------------------------------------------------
    # Promote the final checkpoint into a servable Fireworks model. The run id
    # is appended so repeated runs do not collide.
    promote: bool = True
    output_model_id: str = "serverless-sft-support-triage"

    # --- Connection ---------------------------------------------------------
    # Prod gateway by default; override with FIREWORKS_BASE_URL (e.g. a dev
    # gateway). The "/training/v1/serverless" suffix is added automatically.
    api_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai"
        )
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", "")
    )

    # --- Bookkeeping --------------------------------------------------------
    dcp_timeout_s: float = 900.0
    run_dir: str = ""
    # Requires matplotlib; set False (or don't install it) to skip the plot.
    plot_loss_curve: bool = True


def _validate_config(cfg: Config) -> None:
    if cfg.lora_rank <= 0:
        raise ValueError(
            "serverless training requires lora_rank > 0 (the pool is LoRA-only)"
        )
    if cfg.max_seq_len <= 0:
        raise ValueError("serverless training requires max_seq_len > 0")
    if cfg.steps <= 0:
        raise ValueError("steps must be > 0")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if cfg.dcp_save_interval < 0:
        raise ValueError(
            "dcp_save_interval must be >= 0 (0 disables periodic checkpoints)"
        )
    for name in (cfg.checkpoint_name, cfg.final_checkpoint_name):
        if not name:
            raise ValueError("checkpoint names must not be empty")
        if len(name) > MAX_CHECKPOINT_NAME_LEN:
            raise ValueError(
                f"checkpoint name {name!r} is {len(name)} characters; the promotable "
                f"checkpoint id must fit in 63 characters, so names are capped at "
                f"{MAX_CHECKPOINT_NAME_LEN}"
            )


def _validate_datum_length(datum_length: int, max_seq_len: int) -> None:
    if datum_length > max_seq_len:
        raise ValueError(
            f"training datum length {datum_length} exceeds max_seq_len {max_seq_len}"
        )


def _serverless_base_url(base_url: str) -> str:
    """Serverless training + sampling both hang off ``/training/v1/serverless``."""
    root = base_url.rstrip("/")
    if root.endswith("/training/v1/serverless"):
        return root
    if root.endswith("/training/v1"):
        return f"{root}/serverless"
    return f"{root}/training/v1/serverless"


def _control_plane_base_url(base_url: str) -> str:
    """Checkpoint list/promote use the regular gateway, not the serverless surface."""
    root = base_url.rstrip("/")
    for suffix in ("/training/v1/serverless", "/training/v1"):
        if root.endswith(suffix):
            return root[: -len(suffix)]
    return root


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean_loss(fb_output: Any) -> float | None:
    """Mean NLL from a forward_backward result: loss:sum / response_tokens."""
    metrics = getattr(fb_output, "metrics", None) or {}
    loss_sum = metrics.get("loss:sum")
    tokens = metrics.get("response_tokens") or metrics.get("num_loss_tokens") or 1.0
    if loss_sum is not None:
        return float(loss_sum) / max(float(tokens), 1.0)
    return None


def _fmt(value: float | None, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(value, spec)


class ServerlessTriageSFT:
    """One serverless SFT run over the support-triage dataset."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        _validate_config(cfg)
        self.rows = _load_rows(Path(cfg.dataset))
        if not self.rows:
            raise SystemExit(f"dataset is empty: {cfg.dataset}")

        renderer_name = cfg.renderer_name or resolve_renderer_name(cfg.tokenizer_model)
        self.tokenizer = get_tokenizer(cfg.tokenizer_model)
        self.renderer = build_renderer(
            self.tokenizer, cfg.tokenizer_model, renderer_name
        )

        # The one connection that gives us both the training client and, on
        # resume, a client rebuilt from a checkpoint. No trainer job, no
        # deployment -- just a pooled serverless session.
        self.service = FiretitanServiceClient(
            api_key=cfg.api_key,
            base_url=_serverless_base_url(cfg.api_base_url),
        )

        # Resume vs fresh start. Resuming derives base_model / lora_rank /
        # train_* from the checkpoint itself, so a resumed run cannot silently
        # disagree with the run that wrote it.
        self.start_step = 0
        self.base_model = cfg.base_model
        if cfg.resume_from:
            print(f"resuming from DCP checkpoint: {cfg.resume_from}", flush=True)
            t0 = time.time()
            self.training_client = (
                self.service.create_training_client_from_state_with_optimizer(
                    cfg.resume_from
                )
            )
            self.start_step = _step_from_checkpoint_name(cfg.resume_from)
            # The checkpoint is the source of truth for the base model --
            # promote and the log line must use it, not the CLI default.
            weights_info = (
                self.service.create_rest_client()
                .get_weights_info_by_tinker_path(cfg.resume_from)
                .result()
            )
            self.base_model = weights_info.base_model
            print(
                f"resumed in {time.time() - t0:.1f}s; continuing at step {self.start_step} "
                f"on base model {self.base_model}",
                flush=True,
            )
        else:
            self.training_client = self.service.create_lora_training_client(
                base_model=cfg.base_model,
                rank=cfg.lora_rank,
            )

        self.run_dir = (
            Path(cfg.run_dir).resolve()
            if cfg.run_dir
            else Path("/tmp") / f"serverless_triage_sft_{int(time.time())}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        # Restore the dataset cursor so a resumed run does not re-see the rows
        # the earlier process already trained on.
        self.row_cursor = self.start_step * cfg.batch_size
        self.saved_checkpoints: list[str] = []

        self.session_name = getattr(self.service, "training_session_name", None)
        self.session_id = getattr(self.service, "training_session_id", None)
        self.run_id = getattr(self.training_client, "run_id", None)
        if not self.session_id:
            raise RuntimeError(
                "serverless service did not expose a training_session_id; "
                "cannot list or promote checkpoints for this run"
            )
        print(
            f"connected serverless session={self.session_name or self.session_id} run={self.run_id}\n"
            f"base_model={self.base_model} tokenizer={cfg.tokenizer_model} renderer={renderer_name}\n"
            f"steps={cfg.steps} batch_size={cfg.batch_size} lora_rank={cfg.lora_rank} "
            f"lr={cfg.learning_rate} max_seq_len={cfg.max_seq_len} "
            f"dcp_save_interval={cfg.dcp_save_interval}\n"
            f"rows={len(self.rows)} run_dir={self.run_dir}",
            flush=True,
        )

    # -- data ----------------------------------------------------------------

    def _next_batch(self) -> list[dict[str, Any]]:
        """Take the next ``batch_size`` rows, wrapping around so the bundled
        sample supports a multi-step run. Point ``cfg.dataset`` at a larger file
        for real training."""
        batch = [
            self.rows[(self.row_cursor + i) % len(self.rows)]
            for i in range(self.cfg.batch_size)
        ]
        self.row_cursor += self.cfg.batch_size
        return batch

    def _build_datum(self, messages: list[dict[str, Any]]) -> Any:
        """Build one cross-entropy SFT datum from a chat row.

        ``render_messages_to_datum`` renders the whole conversation with the
        model's chat template and weights only the assistant turn, so the loss
        is computed on the response and not on the prompt. It is the same
        supervised seam the ``sft_loop`` recipe uses, which keeps multi-turn
        rows and chat-template details correct rather than hand-slicing tokens.
        """
        if messages[-1].get("role") != "assistant":
            raise ValueError("each dataset row must end with an assistant message")
        rendered = render_messages_to_datum(
            messages,
            renderer=self.renderer,
            max_seq_len=self.cfg.max_seq_len,
        )
        _validate_datum_length(len(rendered.token_ids), self.cfg.max_seq_len)
        if not any(w > 0 for w in rendered.token_weights):
            raise ValueError(
                "rendered row has no trainable tokens; check the assistant turn "
                "and the renderer's train_on_what handling"
            )
        return rendered.datum

    # -- checkpoints ---------------------------------------------------------

    def _save_dcp(self, step: int) -> str:
        """Save a DCP checkpoint: full trainer state (weights + optimizer).

        This is the checkpoint you resume from. It is distinct from
        ``save_weights_for_sampler``, which writes a *sampler* snapshot that can
        be served and promoted but cannot be passed to ``load_state``.
        """
        name = f"{self.cfg.checkpoint_name}-{step:04d}"
        t0 = time.time()
        self.training_client.save_state(name).result(timeout=self.cfg.dcp_timeout_s)
        self.saved_checkpoints.append(name)
        print(f"  saved DCP checkpoint {name!r} ({time.time() - t0:.1f}s)", flush=True)
        return name

    # -- loop ----------------------------------------------------------------

    def _step(self, step: int) -> dict[str, Any]:
        t0 = time.time()
        cfg = self.cfg

        batch = self._next_batch()
        datums = [self._build_datum(row["messages"]) for row in batch]

        # One supervised update: response-only cross-entropy, then Adam.
        fb = self.training_client.forward_backward(datums, "cross_entropy").result()
        loss = _mean_loss(fb)
        adam = tinker.AdamParams(
            learning_rate=cfg.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.0,
        )
        self.training_client.optim_step(adam).result()

        rec = {
            "step": step,
            "train/loss": loss,
            "train/ppl": math.exp(loss)
            if loss is not None and math.isfinite(loss)
            else None,
            "train/examples": len(datums),
            "perf/step_wall_time": time.time() - t0,
        }
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"step {step:02d} loss={_fmt(loss)} ppl={_fmt(rec['train/ppl'], '.2f')} "
            f"examples={len(datums)} elapsed={rec['perf/step_wall_time']:.1f}s",
            flush=True,
        )
        return rec

    def run(self) -> list[dict[str, Any]]:
        cfg = self.cfg
        records: list[dict[str, Any]] = []

        # Checkpoints are saved at the START of a step, before its
        # forward_backward, so the pending gradient buffer is clean (the
        # previous optim_step already applied and cleared it). The saved state
        # is therefore exactly the weights this step is about to measure.
        for step in range(self.start_step, self.start_step + cfg.steps):
            if (
                cfg.dcp_save_interval
                and step > self.start_step
                and step % cfg.dcp_save_interval == 0
            ):
                self._save_dcp(step)
            records.append(self._step(step))

        # Final DCP checkpoint: this is what a follow-up process passes to
        # --resume-from to continue training.
        final_step = self.start_step + cfg.steps
        final_dcp = self._save_dcp(final_step)
        resume_ref = self._resume_reference(final_dcp)
        print(f"\nresume this run with:\n  --resume-from {resume_ref}", flush=True)

        # Final sampler checkpoint: servable, listable, and promotable.
        final = self.training_client.save_weights_for_sampler(
            cfg.final_checkpoint_name
        ).result()
        snapshot = getattr(final, "path", None)
        if not snapshot:
            raise RuntimeError(
                f"save_weights_for_sampler({cfg.final_checkpoint_name!r}) returned no path"
            )
        print(f"final sampler checkpoint: {snapshot}", flush=True)

        if cfg.promote:
            self._promote()

        losses = [r["train/loss"] for r in records if r["train/loss"] is not None]
        if losses:
            print(
                f"\nloss: {losses[0]:.4f} -> {losses[-1]:.4f} (best {min(losses):.4f}) "
                f"over {len(records)} steps",
                flush=True,
            )
        if cfg.plot_loss_curve:
            self._plot(records)
        print(f"metrics: {self.metrics_path}", flush=True)
        return records

    def _resume_reference(self, checkpoint_name: str) -> str:
        """The cross-run reference a *new* process passes to ``--resume-from``.

        Serverless checkpoints are namespaced under the saving run, so a bare
        name only resolves inside the same session. The portable form is
        ``<account>/<run-id>/<checkpoint>``.
        """
        account = _account_from_session(self.session_name)
        if account and self.run_id:
            return f"{account}/{self.run_id}/{checkpoint_name}"
        return checkpoint_name

    # -- promote -------------------------------------------------------------

    def _promote(self) -> str | None:
        """List the session's checkpoints and promote the final one to a model.

        Promotion goes through the session-scoped control-plane endpoints
        (``accounts/{a}/trainingSessions/{s}/checkpoints/{c}:promote``), not the
        serverless training surface, so it uses a separate REST client.
        """
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
            match = _find_promotable(
                checkpoints, cfg.final_checkpoint_name, self.run_id
            )
            if match is None:
                labels = [
                    f"{_checkpoint_label(c)} promotable={c.get('promotable')}"
                    for c in checkpoints
                ]
                print(
                    f"promote skipped: {cfg.final_checkpoint_name!r} is not listed as "
                    f"promotable yet (saw {len(checkpoints)}: {labels})",
                    flush=True,
                )
                return None

            # Keep the output model id unique per run so re-runs do not collide
            # with an already-promoted model. Model ids are <=63 chars,
            # lowercase alphanumeric plus hyphens.
            output_model_id = _unique_model_id(cfg.output_model_id, self.run_id)
            model = cp_client.promote_session_checkpoint(
                name=match["name"],
                output_model_id=output_model_id,
                base_model=self.base_model,
            )
            model_name = model.get("name") if isinstance(model, dict) else str(model)
            print(f"promoted model: {model_name}", flush=True)
            return model_name
        finally:
            cp_client.close()

    # -- plotting ------------------------------------------------------------

    def _plot(self, records: list[dict[str, Any]]) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping loss curve", flush=True)
            return
        points = [
            (r["step"], r["train/loss"]) for r in records if r["train/loss"] is not None
        ]
        if not points:
            return
        steps, losses = zip(*points)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, losses, marker="o", label="train/loss")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("cross-entropy loss")
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"Serverless SFT ({self.base_model}, cross_entropy)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        plot_path = self.run_dir / "loss_curve.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"loss curve: {plot_path}", flush=True)


# -- helpers -------------------------------------------------------------------


def _account_from_session(session_name: str | None) -> str | None:
    """``accounts/{account}/trainingSessions/{id}`` -> ``{account}``."""
    parts = (session_name or "").split("/")
    if len(parts) >= 2 and parts[0] == "accounts" and parts[1]:
        return parts[1]
    return None


def _step_from_checkpoint_name(reference: str) -> int:
    """Recover the optimizer step encoded in a ``<name>-<step>`` checkpoint.

    Returns 0 when the reference carries no step suffix, which restarts the
    step counter (and dataset cursor) rather than guessing.
    """
    tail = reference.rstrip("/").split("/")[-1]
    suffix = tail.rsplit("-", 1)[-1] if "-" in tail else ""
    return int(suffix) if suffix.isdigit() else 0


def _checkpoint_label(checkpoint: dict[str, Any]) -> str:
    return str(checkpoint.get("name", "")).rstrip("/").split("/")[-1]


def _find_promotable(
    checkpoints: list[dict[str, Any]],
    final_name: str,
    run_id: str | None,
) -> dict[str, Any] | None:
    """The listed, promotable checkpoint matching ``final_name``.

    The trainer may return the bare name or a ``{run_id}-{name}-{suffix}`` form,
    so match on both shapes.
    """
    prefixes = [final_name]
    if run_id:
        prefixes.append(f"{run_id}-{final_name}")

    def matches(checkpoint: dict[str, Any]) -> bool:
        label = _checkpoint_label(checkpoint)
        return any(label == p or label.startswith(p + "-") for p in prefixes)

    return next(
        (c for c in checkpoints if matches(c) and bool(c.get("promotable"))), None
    )


def _unique_model_id(base_id: str, run_id: str | None) -> str:
    suffix = str(run_id or "").replace("run-", "")[:8]
    if not suffix:
        return base_id[:63]
    return f"{base_id[: 63 - len(suffix) - 1]}-{suffix}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-model", default=Config.base_model)
    p.add_argument("--tokenizer-model", default=Config.tokenizer_model)
    p.add_argument("--renderer-name", default=Config.renderer_name)
    p.add_argument("--dataset", default=Config.dataset)
    p.add_argument("--lora-rank", type=int, default=Config.lora_rank)
    p.add_argument("--max-seq-len", type=int, default=Config.max_seq_len)
    p.add_argument("--steps", type=int, default=Config.steps)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    p.add_argument(
        "--dcp-save-interval",
        type=int,
        default=Config.dcp_save_interval,
        help="Save a DCP checkpoint every N steps (0 disables periodic saves).",
    )
    p.add_argument(
        "--resume-from",
        default=Config.resume_from,
        help="DCP reference from an earlier run: '<account>/<run-id>/<checkpoint>'.",
    )
    p.add_argument("--output-model-id", default=Config.output_model_id)
    p.add_argument(
        "--no-promote", action="store_false", dest="promote", default=Config.promote
    )
    p.add_argument("--run-dir", default=Config.run_dir)
    p.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot_loss_curve",
        default=Config.plot_loss_curve,
    )
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        renderer_name=args.renderer_name,
        dataset=args.dataset,
        lora_rank=args.lora_rank,
        max_seq_len=args.max_seq_len,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dcp_save_interval=args.dcp_save_interval,
        resume_from=args.resume_from,
        promote=args.promote,
        output_model_id=args.output_model_id,
        run_dir=args.run_dir,
        plot_loss_curve=args.plot_loss_curve,
    )


def main(cfg: Config) -> list[dict[str, Any]]:
    if not cfg.api_key:
        raise SystemExit(
            "FIREWORKS_API_KEY is required (export it or set Config.api_key)"
        )
    return ServerlessTriageSFT(cfg).run()


if __name__ == "__main__":
    # Fork this: point `dataset` at your own JSONL, swap `base_model` /
    # `tokenizer_model`, and tune the loop shape.
    main(config_from_args(parse_args()))
