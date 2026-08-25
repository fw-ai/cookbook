"""Multi-turn visual tool-use rollout for VisualToolBench (per-run).

Each rollout is an agentic episode on one VisualToolBench task:

1. The model sees the task prompt plus the task image(s) and a set of image
   tools (crop / zoom / rotate / adjust).
2. Every ``<tool_call>`` is executed locally with PIL; the transformed image
   is appended to the conversation as a tool response, and the model
   continues -- the "think with images" loop the benchmark is built around.
3. When the model answers without a tool call (or the turn budget runs out),
   the final answer is graded by a rubric LLM judge
   (:class:`training.examples.rl.visual_toolbench.reward.RubricJudge`).

Every assistant turn becomes one trainable segment: the segment's
``prompt_model_input`` is the renderer-built conversation prefix (text chunks
interleaved with image chunks, so the trainer's forward pass sees the real
pixels) and its completion is that turn's sampled tokens.  All segments in an
episode share the episode's scalar answer reward -- the GRPO advantage then
compares whole trajectories. Tool-call count remains a diagnostic only, so
the policy cannot earn reward by calling tools without improving its answer.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import tinker

from training.examples.rl.vanilla_sampler import build_deployment_sampler
from training.examples.rl.visual_toolbench.image_tools import (
    TOOL_SPECS,
    execute_tool_call,
)
from training.examples.rl.visual_toolbench.reward import (
    DEFAULT_CRITICAL_REWARD_WEIGHT,
    DEFAULT_JUDGE_MAX_CONCURRENCY,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TIMEOUT_S,
    RubricJudge,
)
from training.utils.rl.rollout import RolloutRun, RolloutSample
from training.utils.rl.rollout.renderer import (
    _completion_logprobs_from_sampled_completion,
    _image_placeholder_token_id,
    build_multimodal_completions_prompt_token_ids,
)
from training.utils.supervised import build_renderer

if TYPE_CHECKING:
    from training.recipes.async_rl_loop import RolloutFn, RolloutSetup

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a visual analysis assistant. The user's images may hide the "
    "detail you need at full resolution: crop into the relevant region, zoom, "
    "rotate, or adjust brightness/contrast/sharpness with the provided tools, "
    "inspect the transformed image, and only then answer. When you are "
    "confident, give the final answer directly without calling a tool."
)

DEFAULT_MAX_TURNS = 4
DEFAULT_MAX_WORKSPACE_IMAGES = 6
DEFAULT_MAX_PROMPT_TOKENS = 28672


def _resolve_image_pad_id(tokenizer: Any, renderer: Any | None = None) -> int:
    image_pad_id = _image_placeholder_token_id(tokenizer, renderer=renderer)
    if image_pad_id is not None:
        return image_pad_id
    raise ValueError(
        "Could not resolve a single-token image placeholder for this "
        "tokenizer; visual_toolbench requires a vision-language tokenizer "
        "(e.g. Qwen/Qwen3-VL-8B-Instruct)."
    )


def flatten_multimodal_model_input(
    model_input: tinker.ModelInput,
    image_pad_id: int,
) -> Tuple[List[int], List[int], List[str]]:
    """Flatten a renderer ``ModelInput`` for token-in vision completions.

    Returns ``(prompt_token_ids, text_only_token_ids, images)`` where
    ``prompt_token_ids`` carries one unexpanded image-pad token per image (the
    server expands pads), ``text_only_token_ids`` skips image positions (the
    text parallels stored on :class:`RolloutSample`), and ``images`` is the
    ordered base64 data URL list for the completions ``images`` field.
    """
    prompt_ids: List[int] = []
    text_ids: List[int] = []
    images: List[str] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            prompt_ids.extend(int(t) for t in chunk.tokens)
            text_ids.extend(int(t) for t in chunk.tokens)
        elif isinstance(chunk, tinker.types.ImageChunk):
            prompt_ids.append(image_pad_id)
            encoded = base64.b64encode(chunk.data).decode("ascii")
            images.append(f"data:image/{chunk.format};base64,{encoded}")
        elif isinstance(chunk, tinker.types.ImageAssetPointerChunk):
            prompt_ids.append(image_pad_id)
            images.append(str(chunk.location))
        else:
            raise ValueError(
                f"Unsupported ModelInput chunk for visual rollout: {type(chunk).__name__}"
            )
    return prompt_ids, text_ids, images


def assert_sampler_prompt_matches_model_input(
    model_input: tinker.ModelInput,
    sampler_full_tokens: List[int],
    prompt_len: int,
) -> None:
    """Verify every observable sampler prompt token against trainer input.

    Image positions are opaque to the client and are skipped according to the
    renderer-declared expanded length. Text positions must match exactly.
    """
    if prompt_len > len(sampler_full_tokens):
        raise RuntimeError(
            "Sampler prompt length exceeds returned token count "
            f"({prompt_len} prompt positions vs {len(sampler_full_tokens)} total)."
        )

    sampler_prompt = sampler_full_tokens[:prompt_len]
    offset = 0
    for chunk_index, chunk in enumerate(model_input.chunks):
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            expected = [int(token) for token in chunk.tokens]
            actual = sampler_prompt[offset : offset + len(expected)]
            if actual != expected:
                mismatch = next(
                    (
                        index
                        for index, (want, got) in enumerate(
                            zip(expected, actual, strict=False)
                        )
                        if want != got
                    ),
                    min(len(expected), len(actual)),
                )
                want = expected[mismatch] if mismatch < len(expected) else None
                got = actual[mismatch] if mismatch < len(actual) else None
                raise RuntimeError(
                    "Sampler/trainer prompt token mismatch at expanded position "
                    f"{offset + mismatch} (chunk {chunk_index}: sampler={got}, "
                    f"renderer={want}). Check tokenizer and renderer parity."
                )
            offset += len(expected)
        elif isinstance(
            chunk,
            (tinker.types.ImageChunk, tinker.types.ImageAssetPointerChunk),
        ):
            expected_tokens = getattr(chunk, "expected_tokens", None)
            if expected_tokens is None:
                raise RuntimeError(
                    "Renderer image chunk has no expected_tokens; cannot verify "
                    "sampler/trainer prompt alignment."
                )
            offset += int(expected_tokens)
        else:
            raise RuntimeError(
                "Unsupported trainer prompt chunk while checking sampler parity: "
                f"{type(chunk).__name__}"
            )

    if offset != prompt_len:
        raise RuntimeError(
            "Sampler/trainer multimodal prompt length mismatch "
            f"({prompt_len} sampler-expanded positions vs {offset} "
            "renderer/trainer positions). Check tokenizer and vision "
            "preprocessing parity."
        )


def _message_text(message: Dict[str, Any]) -> str:
    """Join the text parts of a parsed assistant message."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _build_user_message(row: Dict[str, Any]) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = [
        {"type": "image", "image": str(url)} for url in row.get("images") or []
    ]
    parts.append({"type": "text", "text": str(row.get("prompt", ""))})
    return {"role": "user", "content": parts}


def make_rollout_fn(setup: "RolloutSetup") -> "RolloutFn":
    # Current recipes inject one borrowed sampler for dedicated and serverless
    # runs. Keep endpoint reconstruction only for legacy/manual setups.
    sampler = setup.sampler
    if sampler is None:
        sampler = build_deployment_sampler(setup)
    tokenizer = setup.tokenizer
    renderer = build_renderer(
        tokenizer,
        setup.tokenizer_id,
        str(setup.extras.get("renderer_name", "") or ""),
    )
    image_pad_id = _resolve_image_pad_id(tokenizer, renderer)

    sample_kwargs = dict(setup.sample_kwargs)
    max_tokens = int(sample_kwargs.pop("max_tokens", 1024))
    temperature = float(sample_kwargs.pop("temperature", 1.0))
    # Set explicitly below; drop recipe-provided copies to avoid kwarg clashes.
    sample_kwargs.pop("logprobs", None)
    sample_kwargs.pop("return_token_ids", None)
    router_replay_requested = bool(sample_kwargs.get("include_routing_matrix", False))
    router_replay_completion_only = router_replay_requested and not bool(
        sample_kwargs.get("echo", False)
    )
    filter_truncated_rollouts = bool(
        setup.extras.get("filter_truncated_rollouts", True)
    )
    max_turns = int(setup.extras.get("max_turns", DEFAULT_MAX_TURNS))
    if max_turns < 1:
        raise ValueError(f"max_turns must be >= 1, got {max_turns}")
    max_workspace_images = int(
        setup.extras.get("max_workspace_images", DEFAULT_MAX_WORKSPACE_IMAGES)
    )
    max_prompt_tokens = int(
        setup.extras.get("max_prompt_tokens", DEFAULT_MAX_PROMPT_TOKENS)
    )
    tool_image_dim = int(setup.extras.get("tool_image_dim", 1024))
    system_prompt = str(setup.extras.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
    event_sink = setup.extras.get("event_sink")
    if event_sink is not None and not callable(event_sink):
        raise TypeError("RolloutSetup.extras['event_sink'] must be callable")

    def _emit_event(event: str, **fields: Any) -> None:
        if event_sink is None:
            return
        try:
            event_sink({"event": event, **fields})
        except Exception as exc:  # observability must not change the rollout
            logger.warning("VisualToolBench event sink failed: %s", exc)

    judge = RubricJudge(
        api_key=setup.api_key,
        model=str(setup.extras.get("judge_model", DEFAULT_JUDGE_MODEL)),
        max_tokens=int(setup.extras.get("judge_max_tokens", DEFAULT_JUDGE_MAX_TOKENS)),
        max_concurrency=int(
            setup.extras.get("judge_max_concurrency", DEFAULT_JUDGE_MAX_CONCURRENCY)
        ),
        timeout_s=float(setup.extras.get("judge_timeout_s", DEFAULT_JUDGE_TIMEOUT_S)),
        critical_reward_weight=float(
            setup.extras.get("critical_reward_weight", DEFAULT_CRITICAL_REWARD_WEIGHT)
        ),
    )

    stop_ids = renderer.get_stop_sequences()
    stop_strings = [
        tokenizer.decode([s], skip_special_tokens=False)
        if isinstance(s, int)
        else str(s)
        for s in stop_ids
    ]

    def _turns_from_prompt(sample_prompt: dict) -> List[Dict[str, Any]]:
        """Return the list of user turns, tolerating the legacy single-turn shape.

        New rows carry a ``turns`` list (one entry per user turn, each with its
        own ``prompt`` / ``golden_answer`` / ``rubrics`` / ``images``).  Legacy
        single-turn rows only have the top-level fields; wrap them as one turn.
        """
        turns = sample_prompt.get("turns")
        if isinstance(turns, list) and turns:
            return turns
        return [
            {
                "prompt": sample_prompt.get("prompt", ""),
                "golden_answer": sample_prompt.get("golden_answer", ""),
                "rubrics": sample_prompt.get("rubrics") or [],
                "images": sample_prompt.get("images") or [],
            }
        ]

    async def rollout_fn(sample_prompt: dict) -> RolloutRun | None:
        turns = _turns_from_prompt(sample_prompt)
        if not any(t.get("rubrics") for t in turns):
            return None
        # A task must show the model at least one image somewhere in the
        # conversation; a follow-up turn may reuse earlier images (no new ones).
        if not any(t.get("images") for t in turns):
            return None

        # One growing conversation for the whole (possibly multi-turn) task.
        # Each assistant response across every user turn becomes one trainable
        # segment; the renderer re-renders the full history each time, so prior
        # user turns, prior answers, and tool responses always land in the
        # loss-masked prefix and only the current response is trained.
        messages: List[Dict[str, Any]] = list(
            renderer.create_conversation_prefix_with_tools(
                list(TOOL_SPECS), system_prompt
            )
        )
        workspace_images: List[str] = []

        segments: List[Dict[str, Any]] = []
        per_turn_rewards: List[float] = []
        per_turn_official_scores: List[float] = []
        per_turn_critical_fractions: List[float] = []
        per_turn_passed: List[bool] = []
        num_tool_calls = 0
        last_answer = ""
        budget_exhausted = False

        for turn_dict in turns:
            # Introduce this user turn: its images enter the workspace and the
            # user message carries them plus the turn prompt.
            workspace_images.extend(str(u) for u in (turn_dict.get("images") or []))
            messages.append(_build_user_message(turn_dict))

            turn_answer = ""
            # Inner tool loop for this single user turn.
            for tool_turn in range(max_turns):
                model_input = renderer.build_generation_prompt(messages)
                prompt_ids, images = build_multimodal_completions_prompt_token_ids(
                    messages,
                    model_input,
                    tokenizer,
                    renderer=renderer,
                )
                _, text_prefix_ids, flattened_images = flatten_multimodal_model_input(
                    model_input,
                    image_pad_id,
                )
                if images != flattened_images:
                    raise RuntimeError(
                        "Renderer sampling images differ from trainer ModelInput images."
                    )
                estimated_prompt_tokens = len(prompt_ids) + len(images) * 1500
                if estimated_prompt_tokens > max_prompt_tokens:
                    # Context budget exhausted (rough expanded-vision estimate);
                    # never train on the resulting partial trajectory when
                    # truncation filtering is enabled.
                    _emit_event(
                        "prompt_budget_exhausted",
                        estimated_prompt_tokens=estimated_prompt_tokens,
                        max_prompt_tokens=max_prompt_tokens,
                    )
                    budget_exhausted = True
                    break

                try:
                    completions = await sampler.sample_with_prompt_tokens(
                        prompt_ids,
                        n=1,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop_strings,
                        images=images,
                        logprobs=True,
                        return_token_ids=True,
                        **sample_kwargs,
                    )
                except Exception as exc:
                    _emit_event("sampling_error", error=type(exc).__name__)
                    logger.warning("Rollout sampling failed: %s", str(exc)[:200])
                    return None
                if not completions:
                    return None

                completion = completions[0]
                prompt_len = int(completion.prompt_len)
                trainer_prompt_len = int(model_input.length)
                if prompt_len != trainer_prompt_len:
                    raise RuntimeError(
                        "Sampler/trainer multimodal prompt length mismatch "
                        f"({prompt_len} sampler-expanded positions vs "
                        f"{trainer_prompt_len} renderer/trainer positions). "
                        "Check tokenizer and vision preprocessing parity."
                    )
                assert_sampler_prompt_matches_model_input(
                    model_input,
                    [int(token) for token in completion.full_tokens],
                    prompt_len,
                )
                out_tokens = [int(t) for t in completion.full_tokens[prompt_len:]]
                if not out_tokens:
                    return None
                sampling_logprobs = _completion_logprobs_from_sampled_completion(
                    completion,
                    prompt_len=prompt_len,
                    completion_len=len(out_tokens),
                    attr="sampling_logprobs",
                    source="sampling_logprobs",
                    required=True,
                )
                if sampling_logprobs is None:
                    return None
                raw_logprobs = _completion_logprobs_from_sampled_completion(
                    completion,
                    prompt_len=prompt_len,
                    completion_len=len(out_tokens),
                    attr="inference_logprobs",
                    source="raw inference logprobs",
                    required=True,
                )
                if raw_logprobs is None:
                    return None

                routing_matrices = getattr(completion, "routing_matrices", None)
                if router_replay_requested:
                    if not routing_matrices:
                        raise RuntimeError(
                            "Router Replay was requested but the sampler returned no "
                            "routing_matrices."
                        )
                    expected_routes = (
                        len(out_tokens)
                        if router_replay_completion_only
                        else prompt_len + len(out_tokens) - 1
                    )
                    if len(routing_matrices) != expected_routes:
                        mode = (
                            "completion-only"
                            if router_replay_completion_only
                            else "full-sequence"
                        )
                        raise RuntimeError(
                            f"{mode} Router Replay returned misaligned routing "
                            f"matrices ({len(routing_matrices)} routes; expected "
                            f"{expected_routes})."
                        )
                    if any(not route for route in routing_matrices):
                        raise RuntimeError(
                            "Router Replay returned empty routing matrices for "
                            "replayed positions."
                        )

                finish_reason = str(getattr(completion, "finish_reason", "stop"))
                _emit_event(
                    "sample",
                    completion_tokens=len(out_tokens),
                    finish_reason=finish_reason,
                )
                if filter_truncated_rollouts and finish_reason.lower() == "length":
                    return None
                segments.append(
                    {
                        "model_input": model_input,
                        "text_prefix_ids": text_prefix_ids,
                        "out_tokens": out_tokens,
                        "sampling_logprobs": sampling_logprobs,
                        "raw_logprobs": raw_logprobs,
                        "routing_matrices": routing_matrices,
                        "finish_reason": finish_reason,
                    }
                )

                parsed_message, _termination = renderer.parse_response(out_tokens)
                tool_calls = list(parsed_message.get("tool_calls") or [])
                turn_answer = _message_text(parsed_message)
                # Keep the assistant response in the conversation so subsequent
                # tool responses AND the next user turn see it (masked prefix).
                messages.append(parsed_message)

                if not tool_calls or tool_turn + 1 >= max_turns:
                    break

                for tool_call in tool_calls:
                    function = getattr(tool_call, "function", None)
                    name = str(getattr(function, "name", "") or "")
                    arguments = getattr(function, "arguments", "") or "{}"
                    num_tool_calls += 1
                    if len(workspace_images) >= max_workspace_images:
                        new_image, result_text = (
                            None,
                            (
                                "Error: image workspace is full; answer with what you have."
                            ),
                        )
                    else:
                        new_image, result_text = execute_tool_call(
                            name, arguments, workspace_images, max_dim=tool_image_dim
                        )
                    content: List[Dict[str, Any]] = [
                        {"type": "text", "text": result_text}
                    ]
                    if new_image is not None:
                        workspace_images.append(new_image)
                        # Kimi K3 accepts image parts only in user messages.
                        # Keep the tool result attached to its call ID, then
                        # expose the transformed image in a follow-up user turn.
                        messages.append(
                            {
                                "role": "tool",
                                "content": content,
                                "tool_call_id": str(getattr(tool_call, "id", "") or ""),
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": new_image},
                                    {
                                        "type": "text",
                                        "text": "(tool output image above)",
                                    },
                                ],
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "tool",
                                "content": content,
                                "tool_call_id": str(getattr(tool_call, "id", "") or ""),
                            }
                        )

            if budget_exhausted and filter_truncated_rollouts:
                return None

            # Grade this turn against its own rubrics.
            judge_started = time.monotonic()
            result = await judge.grade(
                {
                    "prompt": turn_dict.get("prompt", ""),
                    "golden_answer": turn_dict.get("golden_answer", ""),
                    "rubrics": turn_dict.get("rubrics") or [],
                },
                turn_answer,
            )
            _emit_event(
                "judge",
                latency_s=time.monotonic() - judge_started,
                success=result is not None,
            )
            if result is None:
                logger.warning("Dropping rollout: rubric judge failed to grade a turn")
                return None
            per_turn_rewards.append(float(result.reward))
            per_turn_official_scores.append(float(result.score))
            per_turn_critical_fractions.append(float(result.critical_fraction))
            per_turn_passed.append(bool(result.passed))
            last_answer = turn_answer

            if budget_exhausted:
                # No room to continue the conversation; stop after grading.
                break

        if not segments or not per_turn_rewards:
            return None

        # Trajectory reward = mean of per-turn dense rewards; broadcast to every
        # segment (GRPO compares whole trajectories via one scalar per sample).
        reward = sum(per_turn_rewards) / len(per_turn_rewards)
        official_score = sum(per_turn_official_scores) / len(per_turn_official_scores)
        critical_fraction = sum(per_turn_critical_fractions) / len(
            per_turn_critical_fractions
        )

        rollout_samples = [
            RolloutSample(
                tokens=seg["text_prefix_ids"] + seg["out_tokens"],
                logprobs=(
                    [0.0] * len(seg["text_prefix_ids"]) + seg["sampling_logprobs"]
                ),
                loss_mask=[0] * len(seg["text_prefix_ids"])
                + [1] * len(seg["out_tokens"]),
                reward=float(reward),
                finish_reason=seg["finish_reason"],
                text=last_answer,
                prompt_model_input=seg["model_input"],
                routing_matrices=seg["routing_matrices"],
                raw_logprobs=(
                    [0.0] * len(seg["text_prefix_ids"]) + seg["raw_logprobs"]
                ),
            )
            for seg in segments
        ]
        return RolloutRun(
            segments=rollout_samples,
            metadata={
                "num_turns": len(turns),
                "num_segments": len(segments),
                "num_tool_calls": num_tool_calls,
                # Benchmark-style task pass = every user turn's critical rubrics met.
                "judge_passed": bool(all(per_turn_passed))
                if per_turn_passed
                else False,
                # Backward-compatible name from the original recipe. This is
                # the official benchmark score, not the shaped train reward.
                "mean_turn_score": float(official_score),
                "mean_turn_reward": float(reward),
                "mean_official_score": float(official_score),
                "mean_critical_fraction": float(critical_fraction),
                "metrics": {
                    "rollout/reward": float(reward),
                    "rollout/official_score": float(official_score),
                    "rollout/critical_fraction": float(critical_fraction),
                    "rollout/judge_pass": float(
                        bool(all(per_turn_passed)) if per_turn_passed else False
                    ),
                    "rollout/mean_tool_calls": float(num_tool_calls),
                },
            },
        )

    # The factory owns the judge client. The shared async loop invokes this
    # hook on the same event loop after rollout production finishes.
    setattr(rollout_fn, "close", judge.close)
    return rollout_fn
