"""Black-box coding-agent rollout function for ``async_rl_loop``.

``make_rollout_fn(setup)`` starts the Anthropic shim once (background thread),
then returns ``rollout_fn(sample_prompt) -> RolloutRun | None``.  Each
call is one agent run = one GRPO group member:

    open shim session -> boot local SWE-Gym Docker runtime
    -> run claude-code (dials the shim) -> capture git diff
    -> close work runtime -> grade in a FRESH SWE-Gym runtime
    -> drain the session into >=1 segments -> one RolloutSample per segment.

A run is usually one ``final`` segment, but compaction or a sub-agent
excursion splits it into multiple disjoint segments (see ``shim.py``).  All
segments of a run share the run reward and are returned in one ``RolloutRun``,
so the loop treats them as a SINGLE group member and broadcasts the run's GRPO
advantage across them.

A wall-clock guard wraps the whole thing; on timeout/error we return ``None``
(the loop counts it as one dropped run in the row's group).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import secrets
from typing import TYPE_CHECKING, Any

from training.examples.rl.coding_agent import sandbox as sbx
from training.examples.rl.coding_agent import shim
from training.examples.rl.coding_agent.swegym_data import (
    prompt_text,
    registry_image_for_instance_id,
)
from training.examples.rl.coding_agent.aiohttp_threaded import run_app_in_thread
from training.examples.rl.coding_agent.trajectory import TokenSegment
from training.examples.rl.vanilla_sampler import build_deployment_sampler
from training.utils.rl.rollout import RolloutRun, RolloutSample
from training.utils.supervised import build_renderer

# Register cookbook-local renderers (qwen3 split, glm5, ...) for parse_response.
import training.renderer  # noqa: F401

if TYPE_CHECKING:
    from training.recipes.async_rl_loop import RolloutFn, RolloutSetup

logger = logging.getLogger(__name__)

SWE_TIME_BUDGET_SEC = int(os.environ.get("SWE_TIME_BUDGET_SEC", "1800"))
SWE_EVAL_TIMEOUT_SEC = int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "600"))
SWE_GENERATE_GUARD_SEC = int(os.environ.get("SWE_GENERATE_GUARD_SEC", "0") or 0) or (
    SWE_TIME_BUDGET_SEC + SWE_EVAL_TIMEOUT_SEC + 180
)

# Host the Docker runtime dials back to reach the shim.
HEAD_HOST = os.environ.get("AGENT_HEAD_HOST", "127.0.0.1")
SHIM_BIND_HOST = os.environ.get("SHIM_BIND_HOST", "0.0.0.0")
SHIM_PORT = int(os.environ.get("SHIM_PORT", "0"))

# When set, each rollout writes one JSON artifact (<session_id>.json) with the
# full token-level trajectory + per-turn shape, for the offline trajectory-
# concatenation and rollout-as-unit assertions. One file per session avoids
# concurrent-append interleaving across parallel rollouts.
SWE_TRAJECTORY_LOG_DIR = os.environ.get("SWE_TRAJECTORY_LOG_DIR", "")

# Turn-matching strategy for the shim: "message_hash" (default, drift-tolerant,
# captures harness intent) or "token_prefix" (strict token-prefix matching). See
# training.utils.rl.rollout.turn_matching.
SWE_TURN_MATCHING = os.environ.get("SWE_TURN_MATCHING", "message_hash")


@dataclasses.dataclass(frozen=True)
class AgentRunResult:
    reward: float
    solved: bool
    applied: bool


async def _run_agent_and_grade(
    *,
    metadata: dict,
    session_id: str,
    middleware_url: str,
) -> AgentRunResult:
    async with asyncio.timeout(SWE_GENERATE_GUARD_SEC):
        async with sbx.boot_agent_sandbox(metadata["image"]) as sandbox:
            await sbx.run_claude_code(
                sandbox,
                workdir=metadata["workdir"],
                session_id=session_id,
                middleware_url=middleware_url,
                time_budget_sec=SWE_TIME_BUDGET_SEC,
                problem_statement=metadata["problem_statement"],
                pre_commands=metadata["pre_commands"],
            )
            diff_text = await sbx.git_diff(sandbox, metadata["workdir"])
        reward, solved, applied = await sbx.evaluate(
            image=metadata["image"],
            workdir=metadata["workdir"],
            diff_text=diff_text,
            swebench_instance=metadata["swebench_instance"],
            exclude_patterns=metadata["exclude_patterns"],
            pre_commands=metadata["pre_commands"],
            timeout_sec=SWE_EVAL_TIMEOUT_SEC,
        )
    return AgentRunResult(reward=reward, solved=solved, applied=applied)


def _segments_to_samples(
    segments: list[TokenSegment],
    reward: float,
) -> list[tuple[TokenSegment, RolloutSample]]:
    kept: list[tuple[TokenSegment, RolloutSample]] = []
    for segment in segments:
        if not any(segment.loss_mask):
            continue
        prompt_tokens = len(segment.prompt_ids)
        kept.append(
            (
                segment,
                RolloutSample(
                    tokens=list(segment.prompt_ids) + list(segment.response_ids),
                    logprobs=[0.0] * prompt_tokens + list(segment.rollout_log_probs),
                    loss_mask=[0] * prompt_tokens + list(segment.loss_mask),
                    reward=float(reward),
                    finish_reason=str(segment.metadata.get("finish_reason") or "stop"),
                ),
            )
        )
    return kept


def _dump_trajectory(
    sid: str,
    row_index: int,
    sample_index: int,
    md: dict,
    reward: float,
    solved: Any,
    applied: Any,
    kept: list[tuple[Any, RolloutSample]],
    num_turns: int,
    trainable: int,
) -> None:
    if not SWE_TRAJECTORY_LOG_DIR:
        return
    try:
        os.makedirs(SWE_TRAJECTORY_LOG_DIR, exist_ok=True)
        rec = {
            "session_id": sid,
            "user_field": sid,  # forwarded as completions `user` for sticky routing
            "row_index": row_index,
            "sample_index": sample_index,
            "instance_id": md.get("instance_id"),
            "reward": float(reward),
            "solved": bool(solved),
            "applied": bool(applied),
            "num_turns": num_turns,
            "num_segments": len(kept),  # >1 iff the run compacted / used a sub-agent
            "trainable_tokens": int(trainable),
            # One entry per trained segment; all share the run reward above and
            # (in the loop) the run's broadcast GRPO advantage.
            "segments": [
                {
                    "segment_idx": i,
                    "segment_kind": seg.metadata.get("segment_kind"),
                    "prompt_len": len(seg.prompt_ids),
                    "response_len": len(seg.response_ids),
                    "total_len": len(s.tokens),
                    "finish_reason": s.finish_reason,
                    "turns": seg.metadata.get("turns"),
                    "tokens": list(s.tokens),
                    "loss_mask": list(s.loss_mask),
                }
                for i, (seg, s) in enumerate(kept)
            ],
        }
        path = os.path.join(SWE_TRAJECTORY_LOG_DIR, f"{sid}.json")
        with open(path, "w") as f:
            json.dump(rec, f)
    except OSError as e:
        logger.warning("[coding_agent] failed to write trajectory artifact for %s: %s", sid, e)


def _metadata(row: dict) -> dict:
    """Normalize a dataset row into the fields the agent + grader need."""
    md = dict(row.get("metadata") or {})
    instance = md.get("swebench_instance") or md.get("instance")
    instance_id = md.get("instance_id") or (
        str(instance.get("instance_id")) if isinstance(instance, dict) else None
    )
    problem = md.get("problem_statement") or prompt_text(row)
    if not isinstance(instance, dict):
        raise ValueError("coding_agent rows must include metadata.instance or metadata.swebench_instance")
    image = md.get("image", "")
    if not image and isinstance(instance_id, str) and "__" in instance_id:
        image = registry_image_for_instance_id(instance_id)
    return {
        "image": image,
        "workdir": md.get("workdir", "/testbed"),
        "problem_statement": problem,
        "swebench_instance": instance,
        "exclude_patterns": md.get("exclude_patterns"),
        "pre_commands": md.get("pre_commands"),
        "instance_id": instance_id or str(row.get("id") or "instance"),
    }


def make_rollout_fn(setup: "RolloutSetup") -> "RolloutFn":
    if not HEAD_HOST:
        raise ValueError("AGENT_HEAD_HOST must be set to a host the Docker runtime can reach.")

    sampler = build_deployment_sampler(setup)
    renderer = build_renderer(setup.tokenizer, setup.tokenizer_id)
    stop = shim.build_stop_strings(renderer, setup.tokenizer)

    app, store = shim.start(
        tokenizer=setup.tokenizer,
        renderer=renderer,
        sampler=sampler,
        sample_kwargs=setup.sample_kwargs,
        stop=stop,
        turn_matching=SWE_TURN_MATCHING,
    )
    handle = run_app_in_thread(
        app,
        host=SHIM_BIND_HOST,
        port=SHIM_PORT,
        thread_name="anthropic-shim",
        runner_kwargs={"handler_cancellation": True},
    )
    middleware_url = f"http://{HEAD_HOST}:{handle.port}"
    max_context_tokens = int(setup.sample_kwargs.get("max_seq_len") or 0)
    sampling_defaults = {
        k: setup.sample_kwargs[k] for k in ("temperature", "top_p", "top_k") if k in setup.sample_kwargs
    }
    logger.info("[coding_agent] shim listening on %s (sandboxes dial %s)", handle.url, middleware_url)

    async def rollout_fn(
        sample_prompt: dict,
        *,
        row_index: int = 0,
        sample_index: int = 0,
        **_: Any,
    ) -> RolloutRun | None:
        try:
            md = _metadata(sample_prompt)
        except ValueError as exc:
            logger.warning("[coding_agent] invalid SWE-Gym row; dropping sample: %s", exc)
            return None
        if not md["image"]:
            logger.warning("[coding_agent] row missing metadata.image; dropping sample")
            return None
        sid = f"cagent-{row_index}-{sample_index}-{secrets.token_hex(4)}"
        shim.open_session(store, sid, sampling_defaults=sampling_defaults, max_context_tokens=max_context_tokens)
        try:
            agent_result = await _run_agent_and_grade(metadata=md, session_id=sid, middleware_url=middleware_url)
        except asyncio.TimeoutError:
            logger.warning("[coding_agent] %s aborted: rollout timeout", sid)
            shim.pop_session_split(store, sid)  # discard partial state
            shim.close_session(sid)
            return None
        except Exception as exc:  # noqa: BLE001 - rollout boundary drops failed samples
            logger.warning("[coding_agent] %s aborted: %s: %s", sid, type(exc).__name__, str(exc)[:200])
            shim.pop_session_split(store, sid)  # discard partial state
            shim.close_session(sid)
            return None

        segments = shim.pop_session_split(store, sid)
        shim.close_session(sid)
        if not segments:
            logger.info("[coding_agent] %s produced no trainable segment; dropping", sid)
            return None

        # One agent run -> >=1 disjoint segments (compaction wipes / sub-agent
        # excursions; usually just one ``final``).  Each becomes one
        # RolloutSample carrying the SAME run reward, returned as one RolloutRun:
        # the loop counts it as a SINGLE GRPO group member and broadcasts the
        # run's advantage across all segments (rollout, not segment, is the unit).
        kept = _segments_to_samples(segments, agent_result.reward)
        if not kept:
            logger.info("[coding_agent] %s produced no trainable segment; dropping", sid)
            return None

        samples = [s for _, s in kept]
        trainable = sum(sum(s.loss_mask) for s in samples)
        num_turns = sum(int(seg.metadata.get("num_turns") or 0) for seg, _ in kept)
        resp_tokens = sum(len(seg.response_ids) for seg, _ in kept)
        total = sum(len(s.tokens) for s in samples)
        logger.info(
            "[coding_agent] %s reward=%.1f solved=%s applied=%s segments=%d turns=%d "
            "trainable=%d resp_tokens=%d total=%d user=%s",
            sid, agent_result.reward, agent_result.solved, agent_result.applied, len(samples), num_turns, trainable,
            resp_tokens, total, sid,
        )
        _dump_trajectory(
            sid, row_index, sample_index, md, agent_result.reward, agent_result.solved, agent_result.applied,
            kept, num_turns, trainable,
        )
        return RolloutRun(
            segments=samples,
            run_id=sid,
            metadata={
                "instance_id": md.get("instance_id"),
                "solved": bool(agent_result.solved),
                "applied": bool(agent_result.applied),
                "num_turns": num_turns,
                "num_segments": len(samples),
                "trainable_tokens": int(trainable),
            },
        )

    return rollout_fn
