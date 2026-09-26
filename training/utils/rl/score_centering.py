"""Top-k score centering for off-policy policy-gradient training.

Implements Equation 6/11 of "Score Centering Stabilizes Off-policy
Reinforcement Learning" (arXiv:2609.20807). Sampler top-k token ids are packed
as additional targets so ``forward_backward_custom`` returns differentiable
trainer logprobs for the same vocabulary head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import tinker
import torch

# Public inference defaults to 5 (``FIREWORKS_MAX_LOGPROBS``). Dedicated
# deployments can raise that with ``--max-logprobs``. 8 is the cookbook cap
# for this approximation; the default stays at the public limit.
PUBLIC_DEFAULT_TOP_LOGPROBS = 5
MAX_SCORE_CENTERING_TOP_K = 8


@dataclass(frozen=True)
class ScoreCenteringConfig:
    """Top-k sampler-distribution approximation settings."""

    top_k: int = PUBLIC_DEFAULT_TOP_LOGPROBS
    tail_mass_epsilon: float = 1e-6


def validate_score_centering_config(config: ScoreCenteringConfig) -> None:
    if config.top_k < 1:
        raise ValueError("score centering top_k must be positive.")
    if config.top_k > MAX_SCORE_CENTERING_TOP_K:
        raise ValueError(
            "score centering top_k exceeds the supported sampler "
            f"top_logprobs cap ({MAX_SCORE_CENTERING_TOP_K}). "
            "Public inference defaults to 5; set the deployment "
            "--max-logprobs at least as high as top_k."
        )
    if config.tail_mass_epsilon <= 0:
        raise ValueError("score centering tail_mass_epsilon must be positive.")


def build_score_centering_datums(
    data: List[tinker.Datum],
    sampler_topk_token_ids: List[List[List[int]]],
    sampler_topk_logprobs: List[List[List[float]]],
    config: ScoreCenteringConfig,
) -> tuple[List[tinker.Datum], List[List[List[float]]]]:
    """Expand sampled targets to ``[sampled, sampler top-k...]`` per position."""
    validate_score_centering_config(config)
    if not (
        len(data)
        == len(sampler_topk_token_ids)
        == len(sampler_topk_logprobs)
    ):
        raise ValueError("score centering requires one top-k distribution per datum.")

    expanded: List[tinker.Datum] = []
    aligned_sampler_logprobs: List[List[List[float]]] = []
    for datum_index, (datum, ids_by_pos, logprobs_by_pos) in enumerate(
        zip(
            data,
            sampler_topk_token_ids,
            sampler_topk_logprobs,
            strict=True,
        )
    ):
        target = datum.loss_fn_inputs["target_tokens"]
        if len(target.shape) != 1:
            raise ValueError("score centering requires one-dimensional source targets.")
        target_tokens = [int(value) for value in target.data]
        target_len = len(target_tokens)
        mask_data = datum.loss_fn_inputs.get("weights") or datum.loss_fn_inputs.get(
            "loss_mask"
        )
        if mask_data is None:
            mask = [1.0] * target_len
        else:
            mask = [float(value) for value in mask_data.data]
        if len(mask) != target_len:
            raise ValueError(
                f"score centering datum {datum_index} mask length mismatch."
            )
        if len(ids_by_pos) != target_len or len(logprobs_by_pos) != target_len:
            raise ValueError(
                f"score centering datum {datum_index} top-k positions do not "
                f"match target length {target_len}."
            )

        target_rows: List[List[int]] = []
        weight_rows: List[List[float]] = []
        sampler_rows: List[List[float]] = []
        for position, (sampled_token, active, token_ids, logprobs) in enumerate(
            zip(
                target_tokens,
                mask,
                ids_by_pos,
                logprobs_by_pos,
                strict=True,
            )
        ):
            if active > 0:
                if len(token_ids) < config.top_k or len(logprobs) < config.top_k:
                    raise ValueError(
                        f"score centering datum {datum_index} position {position} "
                        f"has fewer than top_k={config.top_k} candidates."
                    )
                head_ids = [int(value) for value in token_ids[: config.top_k]]
                head_logprobs = [
                    float(value) for value in logprobs[: config.top_k]
                ]
            else:
                head_ids = [0] * config.top_k
                head_logprobs = [float("-inf")] * config.top_k
            target_rows.append([sampled_token, *head_ids])
            weight_rows.append([active, *([0.0] * config.top_k)])
            sampler_rows.append(head_logprobs)

        width = config.top_k + 1
        expanded.append(
            tinker.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=[value for row in target_rows for value in row],
                        dtype="int64",
                        shape=[target_len, width],
                    ),
                    "weights": tinker.TensorData(
                        data=[value for row in weight_rows for value in row],
                        dtype="float32",
                        shape=[target_len, width],
                    ),
                },
            )
        )
        aligned_sampler_logprobs.append(sampler_rows)
    return expanded, aligned_sampler_logprobs


def make_score_centering_loss_fn(
    advantages: List[float],
    sampler_topk_logprobs: List[List[List[float]]],
    config: ScoreCenteringConfig,
) -> ...:
    """Build the vanilla top-k score-centering loss from paper Equation 6."""
    validate_score_centering_config(config)
    if len(advantages) != len(sampler_topk_logprobs):
        raise ValueError("score centering advantages and top-k rows must align.")

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not (
            len(data) == len(logprobs_list) == len(advantages)
        ):
            raise ValueError("score centering loss inputs must align by datum.")

        total_loss = torch.tensor(0.0, requires_grad=True)
        active_positions = 0
        correction_abs_sum = 0.0
        sampler_head_mass_sum = 0.0
        trainer_head_mass_sum = 0.0
        tail_ratio_sum = 0.0
        width = config.top_k + 1

        for datum_index, (datum, raw_logprobs, advantage, sampler_rows) in enumerate(
            zip(
                data,
                logprobs_list,
                advantages,
                sampler_topk_logprobs,
                strict=True,
            )
        ):
            target = datum.loss_fn_inputs["target_tokens"]
            target_len = int(target.shape[0])
            trainer_logprobs = raw_logprobs.reshape(target_len, width)
            if len(sampler_rows) != target_len:
                raise ValueError(
                    f"score centering datum {datum_index} sampler rows are misaligned."
                )
            sampler_head_logprobs = torch.tensor(
                sampler_rows,
                dtype=trainer_logprobs.dtype,
                device=trainer_logprobs.device,
            )
            weights = datum.loss_fn_inputs["weights"]
            loss_mask = torch.tensor(
                weights.data,
                dtype=trainer_logprobs.dtype,
                device=trainer_logprobs.device,
            ).reshape(target_len, width)[:, 0]
            active = loss_mask > 0
            if not active.any():
                continue

            sampled_logprobs = trainer_logprobs[:, 0]
            trainer_head_logprobs = trainer_logprobs[:, 1:]
            sampler_head_probs = sampler_head_logprobs.exp()
            trainer_head_probs = trainer_head_logprobs.detach().exp()
            sampler_tail_mass = (1.0 - sampler_head_probs.sum(dim=-1)).clamp_min(
                config.tail_mass_epsilon
            )
            trainer_tail_mass = (1.0 - trainer_head_probs.sum(dim=-1)).clamp_min(
                config.tail_mass_epsilon
            )
            tail_mass_ratio = sampler_tail_mass / trainer_tail_mass
            head_residual = (
                sampler_head_probs
                - tail_mass_ratio.unsqueeze(-1) * trainer_head_probs
            ).detach()
            correction = (head_residual * trainer_head_logprobs).sum(dim=-1)
            centered_logprob = sampled_logprobs - correction
            advantage_tensor = torch.as_tensor(
                advantage,
                dtype=trainer_logprobs.dtype,
                device=trainer_logprobs.device,
            )
            total_loss = total_loss + (
                -advantage_tensor * centered_logprob * loss_mask
            ).sum()

            count = int(active.sum().item())
            active_positions += count
            correction_abs_sum += correction.detach()[active].abs().sum().item()
            sampler_head_mass_sum += (
                sampler_head_probs.detach()[active].sum(dim=-1).sum().item()
            )
            trainer_head_mass_sum += (
                trainer_head_probs[active].sum(dim=-1).sum().item()
            )
            tail_ratio_sum += tail_mass_ratio.detach()[active].sum().item()

        denominator = max(active_positions, 1)
        return total_loss, {
            "score_centering/active_positions": float(active_positions),
            "score_centering/correction_abs_mean": correction_abs_sum / denominator,
            "score_centering/sampler_head_mass_mean": (
                sampler_head_mass_sum / denominator
            ),
            "score_centering/trainer_head_mass_mean": (
                trainer_head_mass_sum / denominator
            ),
            "score_centering/tail_mass_ratio_mean": tail_ratio_sum / denominator,
        }

    return loss_fn
