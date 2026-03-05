"""Shared helpers used by RL loss variants."""

from __future__ import annotations

from typing import List, Union

import torch
import tinker


def _normalize_prompt_lens(prompt_len: Union[int, List[int]], n: int) -> List[int]:
    """Accept ``int`` (single prompt_len for all datums) or ``List[int]``."""
    if isinstance(prompt_len, int):
        return [prompt_len] * n
    prompt_lens = list(prompt_len)
    if len(prompt_lens) != n:
        raise ValueError(f"Expected {n} prompt lengths, got {len(prompt_lens)}.")
    return prompt_lens


def _get_loss_mask(
    datum: tinker.Datum,
    response_start: int,
    resp_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Extract per-position loss mask from ``loss_fn_inputs["loss_mask"]``.

    Returns a tensor of shape ``[resp_len]`` sliced from ``response_start``.
    Falls back to all-ones (no masking) when the datum has no ``loss_mask``.
    """
    mask_td = datum.loss_fn_inputs.get("loss_mask")
    if mask_td is not None:
        mask_vals = mask_td.data[response_start : response_start + resp_len]
        if len(mask_vals) < resp_len:
            mask_vals = list(mask_vals) + [0.0] * (resp_len - len(mask_vals))
        return torch.tensor(mask_vals, dtype=dtype, device=device)
    return torch.ones(resp_len, dtype=dtype, device=device)
