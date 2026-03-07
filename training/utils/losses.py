"""Loss functions for DPO, ORPO, and SFT training.

RL losses (GRPO, DAPO, GSPO) live in ``utils.rl``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Callable

import torch
import tinker
import torch.nn.functional as F


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(x)) for x < 0.

    Uses two branches for stability:
      - x < -log(2): log1p(-exp(x))      (exp(x) is small)
      - x >= -log(2): log(-expm1(x))     (1 - exp(x) is small)
    """
    mask = x < -0.6931471805599453  # -log(2)
    return torch.where(
        mask,
        torch.log1p(-torch.exp(x)),
        torch.log(-torch.expm1(x)),
    )


def make_orpo_loss_fn(
    response_start: int,
    orpo_lambda: float = 1.0,
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """ORPO loss: L_SFT(chosen) + lambda * L_OR.

    Odds Ratio Preference Optimization — no reference model required.

    L_SFT   = -mean(logprobs_chosen[response:])
    L_OR    = -log(sigmoid(log_odds_ratio))
    log_odds_ratio = log_odds(chosen) - log_odds(rejected)
    log_odds(x) = avg_lp - log(1 - exp(avg_lp))

    Args:
        response_start: Token index where the response begins (prompt boundary).
        orpo_lambda: Weight for the odds-ratio term (default 1.0).
    """

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert len(logprobs_list) == 2, "ORPO expects exactly 2 datums: [chosen, rejected]"

        chosen_lp = logprobs_list[0][response_start:]
        rejected_lp = logprobs_list[1][response_start:]

        n_chosen = max(len(chosen_lp), 1)
        n_rejected = max(len(rejected_lp), 1)

        # SFT loss: cross-entropy on chosen response tokens
        sft_loss = -chosen_lp.sum() / n_chosen

        # Average log-probabilities for odds computation
        avg_chosen_lp = chosen_lp.sum() / n_chosen
        avg_rejected_lp = rejected_lp.sum() / n_rejected

        # Clamp for numerical stability (matching train-py implementation)
        avg_chosen_lp = torch.clamp(avg_chosen_lp, min=-50.0, max=-1e-6)
        avg_rejected_lp = torch.clamp(avg_rejected_lp, min=-50.0, max=-1e-6)

        # log(odds) = log(P) - log(1 - P) = avg_lp - log1mexp(avg_lp)
        log_odds_chosen = avg_chosen_lp - _log1mexp(avg_chosen_lp)
        log_odds_rejected = avg_rejected_lp - _log1mexp(avg_rejected_lp)
        log_odds_ratio = log_odds_chosen - log_odds_rejected

        # OR loss = -log(sigmoid(log_odds_ratio))
        or_loss = -F.logsigmoid(log_odds_ratio)

        # Combined ORPO loss
        orpo_loss = sft_loss + orpo_lambda * or_loss

        with torch.no_grad():
            metrics = {
                "orpo_loss": orpo_loss.item(),
                "sft_loss": sft_loss.item(),
                "or_loss": or_loss.item(),
                "log_odds_ratio": log_odds_ratio.item(),
                "orpo_lambda": orpo_lambda,
                "chosen_avg_lp": avg_chosen_lp.item(),
                "rejected_avg_lp": avg_rejected_lp.item(),
                "accuracy": 1.0 if log_odds_ratio.item() > 0 else 0.0,
            }
        return orpo_loss, metrics

    return loss_fn


def make_batch_dpo_loss_fn(
    ref_chosen_list: List[List[float]],
    ref_rejected_list: List[List[float]],
    response_starts: List[int],
    beta: float,
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """Batched DPO loss over multiple preference pairs.

    Expects ``2 * N`` datums arranged as
    ``[chosen_0, rejected_0, chosen_1, rejected_1, ...]`` and computes the
    mean DPO loss across all pairs.

    Args:
        ref_chosen_list: Per-pair reference logprobs for chosen sequences.
        ref_rejected_list: Per-pair reference logprobs for rejected sequences.
        response_starts: Per-pair token index where the response begins.
        beta: DPO temperature parameter.
    """
    n_pairs = len(ref_chosen_list)
    assert len(ref_rejected_list) == n_pairs
    assert len(response_starts) == n_pairs

    ref_chosen_ts = [torch.tensor(r, dtype=torch.float32) for r in ref_chosen_list]
    ref_rejected_ts = [torch.tensor(r, dtype=torch.float32) for r in ref_rejected_list]

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert len(logprobs_list) == 2 * n_pairs, (
            f"Expected {2 * n_pairs} logprobs (2 per pair), got {len(logprobs_list)}"
        )

        total_loss = torch.tensor(0.0)
        total_margin = 0.0
        total_accuracy = 0.0
        total_chosen_reward = 0.0
        total_rejected_reward = 0.0

        for i in range(n_pairs):
            rs = response_starts[i]
            pi_chosen = logprobs_list[2 * i][rs:].sum()
            pi_rejected = logprobs_list[2 * i + 1][rs:].sum()
            rc = ref_chosen_ts[i][rs:].sum()
            rr = ref_rejected_ts[i][rs:].sum()

            margin = (pi_chosen - rc) - (pi_rejected - rr)
            pair_loss = -F.logsigmoid(beta * margin)
            total_loss = total_loss + pair_loss

            with torch.no_grad():
                total_margin += margin.item()
                total_accuracy += 1.0 if margin.item() > 0 else 0.0
                total_chosen_reward += beta * (pi_chosen.item() - rc.item())
                total_rejected_reward += beta * (pi_rejected.item() - rr.item())

        avg_loss = total_loss / n_pairs

        with torch.no_grad():
            metrics = {
                "dpo_loss": avg_loss.item(),
                "margin": total_margin / n_pairs,
                "accuracy": total_accuracy / n_pairs,
                "chosen_reward": total_chosen_reward / n_pairs,
                "rejected_reward": total_rejected_reward / n_pairs,
                "batch_pairs": n_pairs,
            }
        return avg_loss, metrics

    return loss_fn


def make_sft_loss_fn(
    response_start: int,
    target_tokens: List[int],
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """Cross-entropy loss over response tokens only (single-sample)."""
    targets = torch.tensor(target_tokens, dtype=torch.long)

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        lp = logprobs_list[0]
        resp_lp = lp[response_start:]
        resp_t = targets[response_start:]
        n = max(len(resp_t), 1)
        ce = -resp_lp.sum() / n
        with torch.no_grad():
            ppl = torch.exp(ce).item()
        return ce, {"ce_loss": ce.item(), "ppl": ppl, "response_tokens": len(resp_t)}

    return loss_fn


def make_batch_sft_loss_fn(
    prompt_token_counts: List[int],
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """Cross-entropy loss over response tokens for a *batch* of samples.

    Each sample may have a different prompt length. The loss is averaged across
    all response tokens in the batch (token-level mean), matching the behaviour
    of ``train_sft_tinker_sdk.py``.

    Args:
        prompt_token_counts: Per-sample prompt token counts. Tokens before this
            boundary are masked (no gradient contribution).
    """

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert len(data) == len(logprobs_list)
        assert len(prompt_token_counts) == len(logprobs_list)

        total_loss = torch.tensor(0.0)
        total_response_tokens = 0
        total_nll = 0.0

        for i, logprobs in enumerate(logprobs_list):
            response_start = max(0, prompt_token_counts[i] - 1)
            response_logprobs = logprobs[response_start:]
            n = len(response_logprobs)
            if n == 0:
                continue
            sample_nll = -response_logprobs.sum()
            total_loss = total_loss + sample_nll
            total_response_tokens += n
            with torch.no_grad():
                total_nll += sample_nll.item()

        if total_response_tokens > 0:
            avg_loss = total_loss / total_response_tokens
        else:
            avg_loss = total_loss

        with torch.no_grad():
            avg_nll = total_nll / total_response_tokens if total_response_tokens > 0 else 0.0
            ppl = torch.exp(torch.tensor(avg_nll)).item()

        return avg_loss, {
            "ce_loss": avg_nll,
            "ce_loss_sum": total_nll,
            "ppl": ppl,
            "response_tokens": total_response_tokens,
            "batch_size": len(logprobs_list),
        }

    return loss_fn


def make_batch_weighted_sft_loss_fn(
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """Cross-entropy loss over per-token supervised weights stored on each datum.

    This is the renderer-safe path for multi-turn SFT. Each datum must include
    ``loss_fn_inputs["weights"]`` aligned with ``target_tokens``.
    """

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert len(data) == len(logprobs_list)

        total_loss = torch.tensor(0.0)
        total_weight = 0.0
        total_nll = 0.0

        for datum, logprobs in zip(data, logprobs_list, strict=True):
            weights_td = datum.loss_fn_inputs.get("weights")
            if weights_td is None:
                raise ValueError("Weighted SFT expects each datum to include loss_fn_inputs['weights'].")

            weights = weights_td.to_torch().to(device=logprobs.device, dtype=logprobs.dtype)
            if len(weights) != len(logprobs):
                raise ValueError(
                    f"weights/logprobs length mismatch: {len(weights)} != {len(logprobs)}"
                )

            sample_weight = float(weights.sum().item())
            if sample_weight <= 0:
                continue

            sample_nll = -(logprobs * weights).sum()
            total_loss = total_loss + sample_nll
            total_weight += sample_weight
            with torch.no_grad():
                total_nll += float(sample_nll.item())

        avg_loss = total_loss / total_weight if total_weight > 0 else total_loss
        with torch.no_grad():
            avg_nll = total_nll / total_weight if total_weight > 0 else 0.0
            ppl = torch.exp(torch.tensor(avg_nll)).item()

        return avg_loss, {
            "ce_loss": avg_nll,
            "ce_loss_sum": total_nll,
            "ppl": ppl,
            "response_tokens": total_weight,
            "weighted_tokens": total_weight,
            "batch_size": len(logprobs_list),
        }

    return loss_fn
