"""Terminal status line for recipes that report progress events.

A recipe that accepts ``progress_fn`` calls ``progress_fn(event, **fields)`` at
phase boundaries (see
:func:`training.recipes.experiment.async_rl_loop_serverless.main`).
:class:`TerminalProgress` turns those events into a spinner with one status
line, plus a permanent line for each milestone::

    with TerminalProgress(accuracy_metric="eval/accuracy") as progress:
        result = async_rl_loop_serverless.main(cfg, ..., progress_fn=progress)
    progress.print_summary(result)

Without a terminal (CI, output redirected to a file) it prints one plain line
per phase change instead of animating.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from rich.console import Console
from rich.table import Column
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

__all__ = ["ProgressFn", "TerminalProgress", "format_duration"]


class ProgressFn(Protocol):
    def __call__(self, event: str, /, **fields: Any) -> None: ...


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


# Without a terminal these events print no status line: counters would print one
# line per rollout, and milestones already print their own line.
_QUIET_EVENTS = frozenset(
    {
        "rollout_finished",
        "groups_ready",
        "evaluation_rollout_finished",
        "session_started",
        "step_finished",
        "evaluation_finished",
    }
)
_SAMPLING = "sampling"
_TRAINING = "training"
_WAITING = "waiting for evaluation"
_PUBLISHING = "publishing new weights"
# Phases shown after a "Step k/N:" prefix.
_STEP_PHASES = frozenset({_TRAINING, _WAITING, _PUBLISHING})


class TerminalProgress:
    """Spinner and status line for an RL run; also a valid ``progress_fn``.

    ``evaluation_samples`` is the number of rollouts in one evaluation, shown
    as ``24/64``. ``accuracy_metric`` names the evaluation metric printed as a
    percentage after each evaluation and in the final summary.
    """

    def __init__(
        self,
        *,
        evaluation_samples: int | None = None,
        accuracy_metric: str | None = None,
        console: Console | None = None,
    ) -> None:
        self.evaluation_samples = evaluation_samples
        self.accuracy_metric = accuracy_metric
        self.console = console or Console(stderr=True)
        self.session: str | None = None
        self.run: str | None = None
        self.evaluations: dict[int, Mapping[str, Any]] = {}
        self._animate = self.console.is_terminal and not self.console.is_dumb_terminal
        self._lock = threading.RLock()
        self._progress: Progress | None = None
        self._task: TaskID | None = None
        self._started_at = time.monotonic()
        self._phase = ""
        self._step = 0
        self._total_steps = 0
        self._step_started_at = self._started_at
        self._groups_ready = 0
        self._groups_target = 0
        self._rollouts = 0
        self._evaluation_step: int | None = None
        self._evaluation_started_at = self._started_at
        self._evaluation_rollouts = 0
        self._last_printed = ""

    def __enter__(self) -> TerminalProgress:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._progress is not None:
                self._progress.stop()
                self._progress = None

    def __call__(self, event: str, /, **fields: Any) -> None:
        with self._lock:
            handler = getattr(self, f"_on_{event}", None)
            if handler is None:
                return
            handler(**fields)
            self._refresh(significant=event not in _QUIET_EVENTS)

    def summary(self, result: Mapping[str, Any]) -> str:
        """One line: elapsed time, steps, and first → last evaluation accuracy."""
        line = f"✓ Done in {format_duration(time.monotonic() - self._started_at)}"
        if "steps" in result:
            line += f": {result['steps']} training steps"
        accuracies = [
            metrics[self.accuracy_metric]
            for _step, metrics in sorted(self.evaluations.items())
            if self.accuracy_metric in metrics and not metrics.get("eval/failed")
        ]
        if len(accuracies) > 1:
            line += f", accuracy {accuracies[0]:.1%} → {accuracies[-1]:.1%}"
        elif accuracies:
            line += f", accuracy {accuracies[0]:.1%}"
        return line

    def print_summary(self, result: Mapping[str, Any], *, run_dir: str | Path | None = None) -> None:
        self.close()
        self._print(self.summary(result))
        session = result.get("training_session") or self.session
        if session:
            self._print(f"  Training session: {session}")
        if run_dir is not None:
            self._print(f"  Metrics and checkpoint paths: {run_dir}")

    # Event handlers. Field names match the recipe's progress_fn calls.

    def _on_session_starting(self, *, base_model: str, **_: Any) -> None:
        self._phase = f"Starting a serverless training session for {base_model.rsplit('/', 1)[-1]}"

    def _on_session_started(self, *, session: str | None, run: str | None, **_: Any) -> None:
        self.session, self.run = session, run
        if session:
            self._print(f"Training session: {session}" + (f" (run {run})" if run else ""))
        self._phase = "Publishing the starting weights"

    def _on_training_started(self, *, step: int, total_steps: int, **_: Any) -> None:
        # Rollouts for the first step start before its batch is handed over.
        self._step, self._total_steps = step, total_steps
        self._step_started_at = time.monotonic()
        self._phase = _SAMPLING

    def _on_step_started(self, *, step: int, total_steps: int, target_groups: int, **_: Any) -> None:
        self._step, self._total_steps = step, total_steps
        self._groups_ready, self._groups_target = 0, target_groups
        self._phase = _SAMPLING

    def _on_rollout_finished(self, **_: Any) -> None:
        self._rollouts += 1

    def _on_groups_ready(self, *, count: int, **_: Any) -> None:
        self._groups_ready = count

    def _on_step_training(self, **_: Any) -> None:
        self._phase = _TRAINING

    def _on_step_waiting_for_evaluation(self, **_: Any) -> None:
        # The recipe joins an evaluation that may already be done.
        if self._evaluation_step is not None:
            self._phase = _WAITING

    def _on_sampler_publishing(self, **_: Any) -> None:
        self._phase = _PUBLISHING

    def _on_step_finished(self, *, step: int, total_steps: int, **_: Any) -> None:
        elapsed = format_duration(time.monotonic() - self._step_started_at)
        self._print(f"✓ Step {step}/{total_steps} done in {elapsed}")
        self._step_started_at = time.monotonic()
        self._groups_ready = self._groups_target = self._rollouts = 0
        if step < total_steps:
            self._step = step + 1
            self._phase = _SAMPLING
        else:
            self._phase = ""

    def _on_checkpoint_saving(self, *, step: int, **_: Any) -> None:
        self._phase = f"Saving checkpoint (step {step})"

    def _on_evaluation_started(self, *, step: int, **_: Any) -> None:
        self._evaluation_step = step
        self._evaluation_started_at = time.monotonic()
        self._evaluation_rollouts = 0

    def _on_evaluation_rollout_finished(self, **_: Any) -> None:
        self._evaluation_rollouts += 1

    def _on_evaluation_finished(self, *, step: int, metrics: Mapping[str, Any], **_: Any) -> None:
        self.evaluations[step] = dict(metrics)
        label = _model_label(step)
        if metrics.get("eval/failed"):
            self._print(f"✗ Evaluation of {label} failed; training continues")
        elif self.accuracy_metric in metrics:
            self._print(f"✓ {label[0].upper()}{label[1:]} accuracy: {metrics[self.accuracy_metric]:.1%}")
        else:
            elapsed = format_duration(time.monotonic() - self._evaluation_started_at)
            self._print(f"✓ Evaluated {label} in {elapsed}")
        self._evaluation_step = None

    # Rendering.

    def _status(self) -> str:
        parts = []
        step = f"Step {self._step}/{self._total_steps}"
        if self._phase == _SAMPLING:
            counts = f"{self._rollouts} done"
            if self._groups_target:
                counts += f", {self._groups_ready}/{self._groups_target} groups"
            parts.append(f"{step}: generating rollouts ({counts})")
        elif self._phase in _STEP_PHASES:
            parts.append(f"{step}: {self._phase}")
        elif self._phase:
            parts.append(self._phase)
        if self._evaluation_step is not None:
            done = str(self._evaluation_rollouts)
            if self.evaluation_samples:
                done += f"/{self.evaluation_samples}"
            parts.append(f"Evaluating {_model_label(self._evaluation_step)} ({done})")
        return " · ".join(parts) or "Finishing up"

    def _refresh(self, *, significant: bool) -> None:
        status = self._status()
        if not self._animate:
            if significant and status != self._last_printed:
                self._last_printed = status
                elapsed = format_duration(time.monotonic() - self._started_at)
                self._print(f"[{elapsed}] {status}")
            return
        if self._progress is None:
            # Started on the first event so tokenizer download bars print first.
            # Only the text column shrinks, so the spinner and clock stay visible.
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn(
                    "{task.description}",
                    markup=False,
                    table_column=Column(ratio=1, no_wrap=True, overflow="ellipsis"),
                ),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
                expand=True,
            )
            self._progress.start()
            self._task = self._progress.add_task(status, total=None)
        else:
            self._progress.update(self._task, description=status)

    def _print(self, line: str) -> None:
        self.console.print(line, markup=False, highlight=False, soft_wrap=True)


def _model_label(step: int) -> str:
    return "base model" if step == 0 else f"model after step {step}"
