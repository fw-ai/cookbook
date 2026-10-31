from __future__ import annotations

import io

from rich.console import Console
from training.utils.progress import TerminalProgress, format_duration


def _run_events(progress: TerminalProgress) -> None:
    progress("session_starting", base_model="accounts/fireworks/models/muse-glimmer")
    progress("session_started", session="accounts/a/trainingSessions/ts-123", run="run-1")
    progress("evaluation_started", step=0)
    progress("step_started", step=1, total_steps=2, target_groups=4)
    for _ in range(3):
        progress("rollout_finished")
        progress("evaluation_rollout_finished", step=0)
    progress("groups_ready", step=1, count=4)
    progress("step_training", step=1)
    progress("step_waiting_for_evaluation", step=1)
    progress("evaluation_finished", step=0, metrics={"eval/accuracy": 0.297, "eval/failed": 0.0})
    progress("sampler_publishing", step=1)
    progress("step_finished", step=1, total_steps=2)
    progress("evaluation_started", step=1)
    progress("evaluation_finished", step=1, metrics={"eval/failed": 1.0})
    progress("checkpoint_saving", step=2)
    progress("evaluation_started", step=2)
    progress("evaluation_finished", step=2, metrics={"eval/accuracy": 0.359, "eval/failed": 0.0})
    progress("some_future_event", anything=1)


def test_without_a_terminal_prints_one_line_per_phase_and_no_counters() -> None:
    out = io.StringIO()
    with TerminalProgress(
        evaluation_samples=64,
        accuracy_metric="eval/accuracy",
        console=Console(file=out, force_terminal=False, width=40),
    ) as progress:
        _run_events(progress)
    progress.print_summary({"steps": 2}, run_dir="./run")

    # Counters (rollouts, groups, eval samples) never print a line of their own.
    assert out.getvalue().splitlines() == [
        "[0s] Starting a serverless training session for muse-glimmer",
        "Training session: accounts/a/trainingSessions/ts-123 (run run-1)",
        "[0s] Publishing the starting weights · Evaluating base model (0/64)",
        "[0s] Step 1/2: generating rollouts (0 done, 0/4 groups) · Evaluating base model (0/64)",
        "[0s] Step 1/2: training · Evaluating base model (3/64)",
        "[0s] Step 1/2: waiting for evaluation · Evaluating base model (3/64)",
        "✓ Base model accuracy: 29.7%",
        "[0s] Step 1/2: publishing new weights",
        "✓ Step 1/2 done in 0s",
        "[0s] Step 2/2: generating rollouts (0 done) · Evaluating model after step 1 (0/64)",
        "✗ Evaluation of model after step 1 failed; training continues",
        "[0s] Saving checkpoint (step 2)",
        "[0s] Saving checkpoint (step 2) · Evaluating model after step 2 (0/64)",
        "✓ Model after step 2 accuracy: 35.9%",
        "✓ Done in 0s: 2 training steps, accuracy 29.7% → 35.9%",
        "  Training session: accounts/a/trainingSessions/ts-123",
        "  Metrics and checkpoint paths: ./run",
    ]


def test_terminal_animates_then_keeps_only_milestones() -> None:
    out = io.StringIO()
    progress = TerminalProgress(
        accuracy_metric="eval/accuracy",
        console=Console(file=out, force_terminal=True, width=120),
    )
    with progress:
        _run_events(progress)
        assert progress._progress is not None
    assert progress._progress is None
    text = out.getvalue()
    assert "Training session: accounts/a/trainingSessions/ts-123" in text
    assert "Model after step 2 accuracy: 35.9%" in text
    assert "accuracy 29.7% → 35.9%" in progress.summary({"steps": 2})


def test_summary_without_evaluations() -> None:
    progress = TerminalProgress(console=Console(file=io.StringIO(), force_terminal=False))
    assert progress.summary({"steps": 3}) == "✓ Done in 0s: 3 training steps"


def test_format_duration() -> None:
    assert format_duration(42.9) == "42s"
    assert format_duration(451) == "7m 31s"
    assert format_duration(3 * 3600 + 5 * 60) == "3h 05m"
