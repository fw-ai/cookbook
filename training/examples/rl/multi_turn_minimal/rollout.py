"""Reference multi-turn rollout function using ``TrajectoryAssembler``.

This is the canonical "copy this for your custom multi-turn workflow"
template.  It calls the Fireworks Completions API ``n_turns`` times,
appending a fixed follow-up prompt between turns, and returns a
:class:`RolloutPayload` ready for the trainer.

Three patterns to copy:

1. **Sacred engine tokens.**  The output token IDs and per-token
   logprobs come straight from the inference response via
   :func:`extract_completion`.  Never decode then re-tokenize -- BPE
   doesn't always round-trip across turn boundaries.
2. **Precomputed chat suffix.**  Between turns, append
   :func:`precompute_chat_suffix` (the AReaL ``multi_turn_prompt_ids``
   trick) to the engine's output tokens.  This avoids re-rendering the
   conversation through the chat template each turn.
3. **Assembler-driven assembly.**  :class:`TrajectoryAssembler` enforces
   the prefix-equality invariant on every ``add_call``.  If the rollout
   ever drifts off the engine's view of the conversation, you get a
   loud :class:`PrefixMismatch` instead of silently misaligned tokens.
"""

from __future__ import annotations

import logging
from typing import Any, List

import httpx

from training.utils.rl.rollout_helpers import extract_completion, precompute_chat_suffix
from training.utils.rl.rollout_service import RolloutPayload
from training.utils.rl.trajectory_assembler import TrajectoryAssembler


logger = logging.getLogger(__name__)


class MultiTurnService:
    """A multi-turn rollout service driving the Fireworks Completions API.

    Constructor args:
        base_url: Fireworks endpoint, e.g. ``https://api.fireworks.ai``.
        api_key: API key for the deployment.
        model: deployment-qualified model id.
        tokenizer: a HuggingFace tokenizer matching the deployed model.
        n_turns: number of assistant turns per rollout.
        followup_text: fixed user message appended after each assistant
            turn (matches AReaL's ``MultiTurnWorkflow``).
        reward_fn: ``(messages, payload) -> float``, called once at end
            of rollout.  When ``None``, the trainer must supply
            ``reward_fn=`` to ``make_text_rollout_fn``.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        tokenizer: Any,
        n_turns: int = 2,
        followup_text: str = "Continue.",
        reward_fn: Any = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._tokenizer = tokenizer
        self._n_turns = n_turns
        self._reward_fn = reward_fn
        # Precompute once: the template tokens that follow an assistant turn.
        self._followup_suffix = precompute_chat_suffix(
            tokenizer,
            follow_up_role="user",
            follow_up_content=followup_text,
        )

    async def rollout(
        self,
        messages: List[dict],
        *,
        n: int,
        sample_kwargs: dict[str, Any],
        row: dict,
    ) -> List[RolloutPayload]:
        """Produce ``n`` token-native rollouts for one dataset row."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)) as client:
            return [
                await self._one_rollout(client, messages, sample_kwargs, row)
                for _ in range(n)
            ]

    async def _one_rollout(
        self,
        client: httpx.AsyncClient,
        messages: List[dict],
        sample_kwargs: dict[str, Any],
        row: dict,
    ) -> RolloutPayload:
        prompt_ids = list(
            self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
            ),
        )

        asm = TrajectoryAssembler()
        for turn in range(self._n_turns):
            choice = await self._sample(client, prompt_ids, sample_kwargs)
            call = extract_completion(choice, input_tokens=prompt_ids)
            asm.add_call(call)
            # Build the next prompt by token concatenation -- never re-tokenize.
            if turn + 1 < self._n_turns:
                prompt_ids = (
                    list(prompt_ids)
                    + list(call.output_tokens)
                    + list(self._followup_suffix)
                )

        reward = None
        if self._reward_fn is not None:
            reward = float(await self._reward_fn(messages, asm))
        return asm.to_payload(total_reward=reward)

    async def _sample(
        self,
        client: httpx.AsyncClient,
        prompt_ids: List[int],
        sample_kwargs: dict[str, Any],
    ) -> dict:
        body = {
            "model": self._model,
            "prompt_token_ids": prompt_ids,
            "max_tokens": int(sample_kwargs.get("max_tokens", 256)),
            "temperature": float(sample_kwargs.get("temperature", 1.0)),
            "logprobs": True,
        }
        resp = await client.post(
            f"{self._base_url}/inference/v1/completions",
            json=body,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        resp.raise_for_status()
        body = resp.json()
        return (body.get("choices") or [{}])[0]
