"""Map Harbor trials to the async RL loop's normal ``RolloutRun`` contract."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from training.examples.rl.harbor.trial import (
    DEFAULT_OPENCODE_VERSION,
    HarborTrialOutcome,
    load_harbor_trial_config,
    run_harbor_trial,
    task_config_from_row,
    task_name_from_row,
)
from training.examples.rl.harbor.openai_policy import (
    OpenCodePolicyServer,
    OpenCodePolicySession,
)
from training.examples.rl.vanilla_sampler import build_deployment_sampler
from training.utils.rl.async_rl.errors import RecoverableRolloutError
from training.utils.rl.rollout import (
    RolloutRun,
    RolloutSample,
    analyze_token_turn_traces,
)
from training.utils.rl.agent.sampling import token_segment_to_sample

if TYPE_CHECKING:
    from training.recipes.async_rl_loop import RolloutFn, RolloutSetup

logger = logging.getLogger(__name__)
DEFAULT_MAX_CONCURRENT_TRIALS = 24


def _artifact_stem(run_id: str, retry_index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip("-.")
    return f"{stem or 'harbor-opencode'}-attempt-{retry_index}"


def _sample_shape(sample: RolloutSample) -> dict[str, int | None]:
    """Return content-free alignment evidence for one trainer segment."""
    return {
        "tokens": len(sample.tokens),
        "loss_mask": len(sample.loss_mask),
        "logprobs": len(sample.logprobs),
        "raw_logprobs": (
            len(sample.raw_logprobs) if sample.raw_logprobs is not None else None
        ),
        "routing_matrices": (
            len(sample.routing_matrices)
            if sample.routing_matrices is not None
            else None
        ),
        "trainable_tokens": sum(sample.loss_mask),
    }


def _split_trace_segments(
    traces: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split diagnostic traces at the same exact-token ancestry boundary."""
    segments: list[list[dict[str, Any]]] = []
    accumulated: list[int] = []
    for trace in traces:
        prompt_ids = [int(token) for token in trace.get("prompt_ids") or []]
        starts_segment = (
            not segments
            or trace.get("turn_kind") == "wipe"
            or prompt_ids[: len(accumulated)] != accumulated
        )
        if starts_segment:
            segments.append([])
        segments[-1].append(trace)
        accumulated = [
            *prompt_ids,
            *[int(token) for token in trace.get("completion_ids") or []],
        ]
    return segments


class _HarborRolloutRunner:
    """Own the example-specific Harbor configuration and local policy server."""

    def __init__(self, setup: RolloutSetup) -> None:
        if setup.sample_kwargs.get("echo"):
            raise ValueError(
                "Fireworks Harbor supports completion-only Router Replay; "
                "set router_replay_completion_only=True"
            )

        self._setup = setup
        self._sampler = build_deployment_sampler(setup)
        self._rollout_retries = int(setup.extras.get("rollout_retries", 3))
        if self._rollout_retries < 0:
            raise ValueError("rollout_extras['rollout_retries'] must be >= 0")
        max_concurrent_trials = int(
            setup.extras.get(
                "max_concurrent_trials",
                DEFAULT_MAX_CONCURRENT_TRIALS,
            )
        )
        if max_concurrent_trials < 1:
            raise ValueError("rollout_extras['max_concurrent_trials'] must be >= 1")
        self._trial_semaphore = asyncio.Semaphore(max_concurrent_trials)
        terminal_failure_reward = setup.extras.get("terminal_failure_reward")
        self._terminal_failure_reward = (
            None if terminal_failure_reward is None else float(terminal_failure_reward)
        )
        if self._terminal_failure_reward is not None and not math.isfinite(
            self._terminal_failure_reward
        ):
            raise ValueError("rollout_extras['terminal_failure_reward'] must be finite")
        self._trial_config = load_harbor_trial_config(
            setup.extras.get("harbor_trial_config")
        )
        trials_dir = setup.extras.get("harbor_trials_dir")
        self._trials_dir = Path(trials_dir) if trials_dir else None
        self._max_seq_len = int(setup.sample_kwargs.get("max_seq_len") or 0)
        self._max_sample_tokens = int(setup.sample_kwargs.get("max_tokens") or 2048)
        self._context_limit = self._max_seq_len
        if self._max_sample_tokens >= self._context_limit:
            raise ValueError("OpenCode context leaves no room for one completion")
        self._opencode_version = str(
            setup.extras.get("opencode_version", DEFAULT_OPENCODE_VERSION)
        )
        self._task_selector = setup.extras.get("task_selector")
        self._renderer_name = str(
            setup.extras.get("renderer_name", "qwen3_5_interleaved")
        )
        self._openai_server: OpenCodePolicyServer | None = None
        self._server_lock = asyncio.Lock()

    async def __call__(
        self,
        sample_prompt: dict[str, Any],
        *,
        cursor_index: int = 0,
        sample_index: int = 0,
        epoch: int = 0,
        evaluation: bool = False,
        **_: Any,
    ) -> RolloutRun | None:
        use_selector = self._task_selector is not None and not evaluation
        if use_selector:
            sample_prompt = await self._task_selector.row_for_group(cursor_index)
        task_name = task_name_from_row(sample_prompt)
        task_config = task_config_from_row(sample_prompt)

        run_id = f"harbor-opencode:{task_name}:{epoch}:{cursor_index}:{sample_index}"
        result = await self._run_opencode(
            task_config=task_config,
            task_name=task_name,
            run_id=run_id,
        )
        if use_selector:
            await self._task_selector.record(
                cursor_index,
                sample_index,
                result.segments[0].reward if result is not None else None,
            )
        return result

    async def _run_opencode(
        self,
        *,
        task_config: Any,
        task_name: str,
        run_id: str,
    ) -> RolloutRun | None:
        server = await self._get_openai_server()
        for attempt in range(self._rollout_retries + 1):
            policy_key = server.register_session(run_id)
            outcome: HarborTrialOutcome | None = None
            session: OpenCodePolicySession | None = None
            session_pop_attempted = False
            try:
                outcome = await self._run_trial(
                    task_config=task_config,
                    policy_key=policy_key,
                    run_id=(f"{run_id}:retry-{attempt}" if attempt else run_id),
                    policy_port=server.port,
                    context_limit=self._context_limit,
                    output_limit=self._max_sample_tokens,
                    opencode_version=self._opencode_version,
                )
                session_pop_attempted = True
                session = await self._take_policy_session(server, policy_key)
                if session is None:
                    raise RecoverableRolloutError(
                        f"OpenCode policy session {policy_key!r} was unavailable "
                        "after the Harbor trial"
                    )
                return await self._opencode_result(
                    session=session,
                    outcome=outcome,
                    run_id=run_id,
                    retry_index=attempt,
                )
            except RecoverableRolloutError as exc:
                if session is None and not session_pop_attempted:
                    session_pop_attempted = True
                    session = await self._take_policy_session(server, policy_key)
                should_retry = attempt < self._rollout_retries
                await self._write_failed_attempt(
                    session=session,
                    task_name=task_name,
                    retry_index=attempt,
                    status="retry" if should_retry else "failed",
                    outcome=outcome,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if should_retry:
                    await self._wait_to_retry(
                        task_name=task_name,
                        attempt=attempt,
                        error=exc,
                    )
                    continue
                logger.warning(
                    "Harbor/OpenCode discarded %s after %d attempts: %s",
                    task_name,
                    self._rollout_retries + 1,
                    exc,
                )
                return None
            except asyncio.CancelledError:
                server.discard_session(policy_key)
                raise
            except Exception as exc:  # noqa: BLE001 - rollout boundary preserves artifact
                if session is None and not session_pop_attempted:
                    session_pop_attempted = True
                    session = await self._take_policy_session(server, policy_key)
                await self._write_failed_attempt(
                    session=session,
                    task_name=task_name,
                    retry_index=attempt,
                    status="failed",
                    outcome=outcome,
                    error=f"{type(exc).__name__}: {exc}",
                )
                logger.warning(
                    "Harbor/OpenCode discarded non-recoverable trajectory %s: %s: %s",
                    task_name,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                return None

    @staticmethod
    async def _take_policy_session(
        server: OpenCodePolicyServer,
        policy_key: str,
    ) -> OpenCodePolicySession | None:
        """Pop a policy session once without masking the trajectory failure."""
        try:
            return await server.pop_session(policy_key)
        except KeyError:
            logger.warning("OpenCode policy session %s was already retired", policy_key)
            return None

    async def _write_failed_attempt(
        self,
        *,
        session: OpenCodePolicySession | None,
        task_name: str,
        retry_index: int,
        status: str,
        outcome: HarborTrialOutcome | None,
        error: str,
    ) -> None:
        """Best-effort artifact write; diagnostics never change disposition."""
        if session is None:
            return
        try:
            await asyncio.to_thread(
                self._write_trajectory_artifact,
                session=session,
                task_name=task_name,
                retry_index=retry_index,
                status=status,
                outcome=outcome,
                error=error,
            )
        except Exception as artifact_error:  # noqa: BLE001 - diagnostics only
            logger.warning(
                "Could not write failed Harbor trajectory artifact for %s: %s",
                task_name,
                artifact_error,
            )

    async def _get_openai_server(self) -> OpenCodePolicyServer:
        if self._openai_server is not None and self._openai_server.port > 0:
            return self._openai_server

        async with self._server_lock:
            if self._openai_server is None:
                self._openai_server = OpenCodePolicyServer(
                    sampler=self._sampler,
                    tokenizer=self._setup.tokenizer,
                    sample_kwargs=dict(self._setup.sample_kwargs),
                    renderer_name=self._renderer_name,
                    max_seq_len=self._max_seq_len,
                    max_sample_tokens=self._max_sample_tokens,
                    capture_request_traces=self._trials_dir is not None,
                )
            if self._openai_server.port < 1:
                await self._openai_server.start()
        return self._openai_server

    async def _opencode_result(
        self,
        *,
        session: OpenCodePolicySession,
        outcome: HarborTrialOutcome,
        run_id: str,
        retry_index: int,
    ) -> RolloutRun:
        if session.trace_integrity_error is not None:
            raise RecoverableRolloutError(
                f"Harbor/OpenCode trace integrity failed for {outcome.task_name}: "
                f"{session.trace_integrity_error}"
            )
        if session.sampling_failures:
            raise RecoverableRolloutError(
                f"Harbor/OpenCode sampling failed for {outcome.task_name}: "
                f"{session.last_error or 'unknown sampling error'}"
            )

        token_segments = session.drain()
        if not token_segments:
            raise RecoverableRolloutError(
                f"Harbor/OpenCode trajectory for {outcome.task_name} "
                "produced no trainable segment"
            )
        logger.info(
            "[harbor-opencode] run=%s drained turns=%d segments=%d "
            "history_wipes=%d auxiliary=%d",
            run_id,
            len(session.match_events),
            len(token_segments),
            session.history_wipes,
            session.auxiliary_turns,
        )
        samples = [
            token_segment_to_sample(segment, reward=outcome.reward)
            for segment in token_segments
        ]
        segment_shapes = [_sample_shape(sample) for sample in samples]
        segment_diagnostics = [
            {
                "append_token_base_mismatches": int(
                    segment.metadata.get("append_token_base_mismatches") or 0
                ),
                "append_token_suffix_mismatches": int(
                    segment.metadata.get("append_token_suffix_mismatches") or 0
                ),
            }
            for segment in token_segments
        ]
        append_token_mismatches = sum(
            int(segment.metadata.get("append_token_mismatches") or 0)
            for segment in token_segments
        )
        if session.history_wipes or append_token_mismatches:
            logger.info(
                "[harbor-opencode-history-shapes] run=%s segments=%s diagnostics=%s",
                run_id,
                segment_shapes,
                segment_diagnostics,
            )
        artifact = await asyncio.to_thread(
            self._write_trajectory_artifact,
            session=session,
            task_name=outcome.task_name,
            retry_index=retry_index,
            status="completed",
            outcome=outcome,
            segment_shapes=segment_shapes,
            segment_diagnostics=segment_diagnostics,
        )
        metadata = {
            "task_name": outcome.task_name,
            "trial_name": outcome.trial_name,
            "harbor_rewards": outcome.rewards,
            "harbor_exception_type": outcome.exception_type,
            "history_wipes": session.history_wipes,
            "append_token_mismatches": append_token_mismatches,
        }
        if artifact is not None:
            metadata["trajectory_artifact"] = artifact
        return RolloutRun(
            segments=samples,
            run_id=run_id,
            metadata=metadata,
        )

    def _write_trajectory_artifact(
        self,
        *,
        session: OpenCodePolicySession,
        task_name: str,
        retry_index: int,
        status: str,
        outcome: HarborTrialOutcome | None = None,
        error: str | None = None,
        segment_shapes: list[dict[str, int | None]] | None = None,
        segment_diagnostics: list[dict[str, int]] | None = None,
    ) -> str | None:
        if self._trials_dir is None:
            return None
        trainable_traces = [
            trace for trace in session.request_traces if trace.get("trainable")
        ]
        trace_segments = _split_trace_segments(trainable_traces)
        analyses = [
            analyze_token_turn_traces(
                traces,
                source="harbor_opencode",
                metadata={
                    "run_id": session.run_id,
                    "task_name": task_name,
                    "segment_index": index,
                },
            )
            for index, traces in enumerate(trace_segments)
        ]
        analysis_summary = {
            "segment_count": len(analyses),
            "turn_count": sum(len(analysis.turns) for analysis in analyses),
            "token_count": sum(len(analysis.tokens) for analysis in analyses),
            "generated_token_count": sum(
                analysis.summary()["generated_token_count"] for analysis in analyses
            ),
            "issue_count": sum(len(analysis.issues) for analysis in analyses),
        }
        analysis_issues = [
            {
                "segment_index": segment_index,
                "code": issue.code,
                "severity": issue.severity,
                "message": issue.message,
                "turn_idx": issue.turn_idx,
                "token_idx": issue.token_idx,
            }
            for segment_index, analysis in enumerate(analyses)
            for issue in analysis.issues
        ]
        shapes = list(segment_shapes or [])
        aligned = all(
            shape["tokens"] == shape["loss_mask"] == shape["logprobs"]
            and shape["raw_logprobs"] in {None, shape["tokens"]}
            # Router replay is expressed in trainer-input coordinates. The
            # model input is ``tokens[:-1]``, so one route per input position
            # is the aligned representation.
            and shape["routing_matrices"] in {None, max(0, shape["tokens"] - 1)}
            for shape in shapes
        )
        document = {
            "schema_version": 2,
            "run_id": session.run_id,
            "task_name": task_name,
            "retry_index": retry_index,
            "status": status,
            "error": error,
            "outcome": (
                {
                    "reward": outcome.reward,
                    "rewards": outcome.rewards,
                    "exception_type": outcome.exception_type,
                    "exception_message": outcome.exception_message,
                    "environment_type": outcome.environment_type,
                }
                if outcome is not None
                else None
            ),
            "session": {
                "history_wipes": session.history_wipes,
                "auxiliary_turns": session.auxiliary_turns,
                "sampling_failures": session.sampling_failures,
                "trace_integrity_failures": session.trace_integrity_failures,
                "trace_integrity_error": session.trace_integrity_error,
                "context_overflows": session.context_overflows,
                "toolless_tool_turns": session.toolless_tool_turns,
                "last_error": session.last_error,
                "match_events": session.match_events,
            },
            "segment_shapes": shapes,
            "segment_diagnostics": list(segment_diagnostics or []),
            "invariants": {
                "trace_integrity_ok": session.trace_integrity_error is None,
                "segment_arrays_aligned": aligned,
                "trajectory_issue_count": len(analysis_issues),
            },
            "analysis": {
                "summary": analysis_summary,
                "issues": analysis_issues,
            },
            "request_traces": session.request_traces,
        }
        artifact_dir = self._trials_dir / "_fireworks_trajectories"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = (
            artifact_dir / f"{_artifact_stem(session.run_id, retry_index)}.json.gz"
        )
        with gzip.open(artifact_path, "wt", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, separators=(",", ":"))
        return str(artifact_path.resolve())

    async def _run_trial(
        self,
        *,
        task_config: Any,
        policy_key: str,
        run_id: str,
        **agent_kwargs: Any,
    ) -> HarborTrialOutcome:
        async with self._trial_semaphore:
            return await run_harbor_trial(
                task_config=task_config,
                policy_key=policy_key,
                run_id=run_id,
                trial_config=self._trial_config,
                trials_dir=self._trials_dir,
                terminal_failure_reward=self._terminal_failure_reward,
                **agent_kwargs,
            )

    async def _wait_to_retry(
        self,
        *,
        task_name: str,
        attempt: int,
        error: RecoverableRolloutError,
    ) -> None:
        delay = 15 * (attempt + 1)
        logger.warning(
            "Harbor/OpenCode rollout attempt failed for %s "
            "(attempt %d/%d); retrying in %ds: %s",
            task_name,
            attempt + 1,
            self._rollout_retries + 1,
            delay,
            error,
        )
        await asyncio.sleep(delay)


def make_rollout_fn(setup: RolloutSetup) -> RolloutFn:
    """Create one logical Harbor rollout with bounded fresh-trial retries."""
    return _HarborRolloutRunner(setup)


__all__ = ["make_rollout_fn"]
