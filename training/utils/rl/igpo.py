"""IGPO utilities: Information Gain-based Policy Optimization.

Shared utilities for IGPO training across single-turn (igpo_loop recipe)
and multi-turn agentic (FrozenLake-style) settings.

``IGPOTurnScorer`` is the customer-facing callback class for interleaved
IG scoring during multi-turn rollouts.  It has no eval-protocol dependency
and works with any rollout loop that calls ``on_turn_complete`` after each
turn.

IG scoring goes through the **inference deployment** (completions API),
keeping the trainer exclusively for ``forward_backward_custom``.  This
avoids request-type coalescing issues on the trainer's GPU batching layer.

Reference: Wang et al., "Information Gain-based Policy Optimization"
(arXiv:2510.14967, ICLR 2026).
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import tinker
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IG scoring primitives
# ---------------------------------------------------------------------------


def build_score_datum(
    prefix_tokens: List[int], answer_tokens: List[int]
) -> tinker.Datum:
    """Build a datum for scoring log P(answer | prefix) via cross-entropy forward."""
    full = prefix_tokens + answer_tokens
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=full[1:], dtype="int64", shape=[len(full) - 1]
            ),
        },
    )


def score_prefix_via_inference(
    inference_url: str,
    model_id: str,
    api_key: str,
    tokenizer: Any,
    prefix_tokens: List[int],
    answer_tokens: List[int],
) -> float:
    """Compute mean log P(answer | prefix) via the inference deployment.

    Uses the completions API with ``echo=True`` and ``logprobs=1`` to retrieve
    per-token logprobs, then averages over answer-token positions.  This routes
    through the serving path, avoiding any interference with the trainer's
    ``forward_backward`` pipeline.
    """
    import httpx

    full_tokens = prefix_tokens + answer_tokens
    prompt_text = tokenizer.decode(full_tokens, skip_special_tokens=False)

    resp = httpx.post(
        f"{inference_url}/v1/completions",
        json={
            "model": model_id,
            "prompt": prompt_text,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    token_logprobs = data["choices"][0].get("logprobs", {}).get("token_logprobs", [])

    n_prefix = len(prefix_tokens)
    n_answer = len(answer_tokens)

    if len(token_logprobs) < n_prefix + n_answer:
        available = token_logprobs[n_prefix:] if len(token_logprobs) > n_prefix else []
    else:
        available = token_logprobs[n_prefix : n_prefix + n_answer]

    valid = [lp for lp in available if lp is not None]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def score_prefix(
    policy_client: Any,
    prefix_tokens: List[int],
    answer_tokens: List[int],
) -> float:
    """Blocking: compute mean log P(answer | prefix) on the policy trainer.

    .. deprecated::
        Use :func:`score_prefix_via_inference` instead to avoid trainer
        request coalescing issues.
    """
    datum = build_score_datum(prefix_tokens, answer_tokens)
    fwd_future = policy_client.forward([datum], "cross_entropy")
    fwd = fwd_future.result() if hasattr(fwd_future, "result") else fwd_future
    logprobs = fwd.loss_fn_outputs[0]["logprobs"].data
    start = len(prefix_tokens) - 1
    end = start + len(answer_tokens)
    n = max(end - start, 1)
    return sum(logprobs[start:end]) / n


# ---------------------------------------------------------------------------
# Turn-level advantage computation
# ---------------------------------------------------------------------------


def compute_turn_advantages(
    rewards: List[List[float]], gamma: float = 0.99, eps: float = 1e-6
) -> List[List[float]]:
    """Per-turn group-normalized advantages from turn-level rewards."""
    G = len(rewards)
    if G == 0:
        return []

    returns: List[List[float]] = []
    for i in range(G):
        T = len(rewards[i])
        ret = [0.0] * T
        if T > 0:
            ret[-1] = rewards[i][-1]
            for t in range(T - 2, -1, -1):
                ret[t] = rewards[i][t] + gamma * ret[t + 1]
        returns.append(ret)

    max_T = max(len(r) for r in returns)
    advantages: List[List[float]] = [[] for _ in range(G)]
    for t in range(max_T):
        vals = [returns[i][t] for i in range(G) if t < len(returns[i])]
        mu = sum(vals) / len(vals) if vals else 0.0
        sigma = (
            (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
            if len(vals) > 1
            else 1.0
        )
        for i in range(G):
            if t < len(returns[i]):
                advantages[i].append((returns[i][t] - mu) / (sigma + eps))
    return advantages


# ---------------------------------------------------------------------------
# Per-token advantage expansion
# ---------------------------------------------------------------------------


def expand_turn_advantages(
    turn_adv: List[float],
    turn_boundaries: List[Tuple[int, int]],
    prompt_len: int,
    total_logprobs: int,
) -> List[float]:
    """Expand per-turn advantages to per-token using regex-detected boundaries."""
    response_start = max(0, prompt_len - 1)
    per_token = [0.0] * total_logprobs
    for t_idx, (t_start, t_end) in enumerate(turn_boundaries):
        if t_idx >= len(turn_adv):
            break
        adv = turn_adv[t_idx]
        lp_start = max(t_start - 1, response_start)
        lp_end = min(t_end - 1, total_logprobs)
        for j in range(lp_start, lp_end):
            per_token[j] = adv
    return per_token


def expand_turn_advantages_from_spans(
    turn_adv: List[float],
    spans: List[Tuple[int, int, int]],
    model_input_len: int,
) -> List[float]:
    """Expand per-turn advantages to per-token using exact model output spans.

    ``spans`` comes from :func:`compute_model_output_spans` — each entry is
    ``(token_start, length, turn_index)`` where ``turn_index`` is 1-based.
    """
    per_token = [0.0] * model_input_len
    for token_start, length, turn_idx in spans:
        adv = turn_adv[turn_idx - 1] if turn_idx - 1 < len(turn_adv) else 0.0
        for j in range(length):
            pos = token_start - 1 + j
            if 0 <= pos < model_input_len:
                per_token[pos] = adv
    return per_token


# ---------------------------------------------------------------------------
# IGPO loss function (per-token advantages, forward_backward_custom)
# ---------------------------------------------------------------------------

_SAFETY_CLAMP = 20.0


def make_igpo_loss_fn(
    per_token_advantages: List[List[float]],
    ref_logprobs: List[List[float]],
    prompt_lens: List[int],
    inf_logprobs: List[List[float]],
    prox_logprobs: List[List[float]] | None = None,
    kl_beta: float = 0.001,
    eps_clip: float = 0.2,
):
    """GRPO-style clipped surrogate loss with per-token advantages.

    Compatible with ``policy.forward_backward_custom(data, loss_fn)``.

    When ``prox_logprobs`` is ``None``, the forward-pass logprobs are used as
    the proximity baseline (ratio=1, clipping is a no-op).
    """
    from training.utils.rl.common import _get_loss_mask

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, requires_grad=True)
        total_kl = 0.0
        num_tokens = 0
        total_resp_tokens = 0

        for i, pi_lp in enumerate(logprobs_list):
            plen = prompt_lens[i] if i < len(prompt_lens) else prompt_lens[0]
            response_start = max(0, plen - 1)
            resp_pi = pi_lp[response_start:]
            resp_len = len(resp_pi)
            if resp_len == 0:
                continue

            resp_mask = _get_loss_mask(
                data[i], response_start, resp_len, resp_pi.dtype, resp_pi.device,
            ) if i < len(data) else torch.ones(
                resp_len, dtype=resp_pi.dtype, device=resp_pi.device,
            )
            active = resp_mask > 0.5
            active_count = int(active.sum().item())
            if active_count == 0:
                continue

            ref_lp = ref_logprobs[i] if ref_logprobs and i < len(ref_logprobs) else []
            resp_ref = torch.tensor(
                [
                    ref_lp[response_start + j]
                    if (response_start + j) < len(ref_lp)
                    else 0.0
                    for j in range(resp_len)
                ],
                dtype=resp_pi.dtype,
                device=resp_pi.device,
            )

            if prox_logprobs is not None:
                prox_lp = prox_logprobs[i] if i < len(prox_logprobs) else []
                resp_prox = torch.tensor(
                    [
                        prox_lp[response_start + j]
                        if (response_start + j) < len(prox_lp)
                        else 0.0
                        for j in range(resp_len)
                    ],
                    dtype=resp_pi.dtype,
                    device=resp_pi.device,
                )
            else:
                resp_prox = resp_pi.detach()

            inf_lp = inf_logprobs[i] if i < len(inf_logprobs) else []
            resp_inf = torch.tensor(
                inf_lp[response_start : response_start + resp_len]
                if len(inf_lp) > response_start
                else [0.0] * resp_len,
                dtype=resp_pi.dtype,
                device=resp_pi.device,
            )

            token_adv = (
                per_token_advantages[i] if i < len(per_token_advantages) else []
            )
            resp_adv = torch.tensor(
                [
                    token_adv[response_start + j]
                    if (response_start + j) < len(token_adv)
                    else 0.0
                    for j in range(resp_len)
                ],
                dtype=resp_pi.dtype,
                device=resp_pi.device,
            )

            log_ratio = torch.clamp(
                resp_pi - resp_prox, min=-_SAFETY_CLAMP, max=_SAFETY_CLAMP
            )
            ratio = torch.exp(log_ratio)
            clipped = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip)

            tis_log = torch.clamp(
                resp_prox.detach() - resp_inf, min=-_SAFETY_CLAMP, max=_SAFETY_CLAMP
            )
            tis_weight = torch.exp(tis_log)

            surr1 = -ratio * resp_adv
            surr2 = -clipped * resp_adv
            kl_penalty = kl_beta * (resp_pi.detach() - resp_ref)
            per_token_loss = (
                torch.maximum(surr1, surr2) * tis_weight + kl_penalty
            ) * resp_mask

            total_loss = total_loss + per_token_loss.sum()
            total_kl += ((resp_pi.detach() - resp_ref) * resp_mask).sum().item()
            num_tokens += active_count
            total_resp_tokens += resp_len

        metrics = {
            "mean_kl": total_kl / num_tokens if num_tokens > 0 else 0.0,
            "active_tokens": num_tokens,
            "total_resp_tokens": total_resp_tokens,
            "mask_ratio": num_tokens / total_resp_tokens if total_resp_tokens > 0 else 0.0,
        }
        return total_loss, metrics

    return loss_fn


# ---------------------------------------------------------------------------
# IGPOTurnScorer — callback class for interleaved IG scoring
# ---------------------------------------------------------------------------


class IGPOTurnScorer:
    """Interleaved IG scorer for multi-turn agentic rollouts.

    IG scoring runs through the **inference deployment** (completions API)
    so that no ``/forward`` requests reach the trainer during sampling.
    This avoids the request-type coalescing issue where pipelined forward
    requests on the trainer's GPU batch layer conflict with subsequent
    ``forward_backward`` calls.

    Pass ``inference_url``, ``model_id``, ``api_key``, and ``tokenizer``
    to use the inference path.  Falls back to the trainer's ``forward()``
    if ``inference_url`` is not provided (legacy/testing only).
    """

    def __init__(
        self,
        answer_tokens: List[int],
        executor: ThreadPoolExecutor,
        ig_weight: float = 0.1,
        skip_ig_last_turn: bool = False,
        inference_url: str = "",
        model_id: str = "",
        api_key: str = "",
        tokenizer: Any = None,
        policy_client: Any = None,
    ):
        if ig_weight >= 0.5 and ig_weight != 0.0:
            logger.warning(
                "ig_weight=%.2f is high — IG rewards are log-probability "
                "differences that can easily dominate the environment reward "
                "and destabilize training. Recommended range: 0.01–0.2.",
                ig_weight,
            )
        self.answer_tokens = answer_tokens
        self.executor = executor
        self.ig_weight = ig_weight
        self.skip_ig_last_turn = skip_ig_last_turn
        self._baselines: Dict[str, Future] = {}
        self._turn_futs: Dict[str, List[Future]] = {}

        self._inference_url = inference_url
        self._model_id = model_id
        self._api_key = api_key
        self._tokenizer = tokenizer
        self.policy = policy_client

    def _score(self, prefix_tokens: List[int]) -> float:
        if self._inference_url and self._model_id and self._tokenizer:
            return score_prefix_via_inference(
                inference_url=self._inference_url,
                model_id=self._model_id,
                api_key=self._api_key,
                tokenizer=self._tokenizer,
                prefix_tokens=prefix_tokens,
                answer_tokens=self.answer_tokens,
            )
        return score_prefix(self.policy, prefix_tokens, self.answer_tokens)

    def on_rollout_start(self, row_id: str, prompt_tokens: List[int]) -> None:
        """Fire baseline scoring ``log P(answer | prompt)``.

        No-ops when ``ig_weight == 0``.
        """
        if self.ig_weight == 0.0:
            if row_id not in self._turn_futs:
                self._turn_futs[row_id] = []
            return
        self._baselines[row_id] = self.executor.submit(
            self._score, prompt_tokens
        )
        if row_id not in self._turn_futs:
            self._turn_futs[row_id] = []

    async def on_turn_complete(
        self,
        row_id: str,
        prefix_tokens: List[int],
        step_index: int,
        done: bool,
    ) -> None:
        """Async callback: submit IG scoring for this turn via inference API."""
        if self.ig_weight == 0.0:
            return
        fut = self.executor.submit(self._score, prefix_tokens)
        self._turn_futs[row_id].append(fut)

    def collect_rewards(
        self, row_id: str, step_rewards: List[float]
    ) -> List[float]:
        """Collect scoring futures and compute combined IG + env rewards."""
        if self.ig_weight == 0.0:
            self._turn_futs.pop(row_id, None)
            return list(step_rewards)

        baseline_logp = self._baselines[row_id].result()
        prev_logp = baseline_logp
        n_turns = len(self._turn_futs[row_id])
        combined: List[float] = []
        for k, fut in enumerate(self._turn_futs[row_id]):
            score_logp = fut.result()
            ig_k = score_logp - prev_logp
            env_r = step_rewards[k] if k < len(step_rewards) else 0.0
            is_last = k == n_turns - 1
            if is_last and self.skip_ig_last_turn and env_r != 0.0:
                combined.append(env_r)
            else:
                combined.append(env_r + self.ig_weight * ig_k)
            prev_logp = score_logp

        del self._baselines[row_id]
        del self._turn_futs[row_id]
        return combined

    @property
    def pending_rollouts(self) -> int:
        """Number of rollouts with outstanding scoring futures."""
        return len(self._baselines)
