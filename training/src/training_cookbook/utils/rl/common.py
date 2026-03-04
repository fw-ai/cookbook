"""Shared helpers used by RL loss variants."""

from __future__ import annotations

from typing import List, Union


def _normalize_prompt_lens(prompt_len: Union[int, List[int]], n: int) -> List[int]:
    """Accept ``int`` (single prompt_len for all datums) or ``List[int]``."""
    if isinstance(prompt_len, int):
        return [prompt_len] * n
    prompt_lens = list(prompt_len)
    if len(prompt_lens) != n:
        raise ValueError(f"Expected {n} prompt lengths, got {len(prompt_lens)}.")
    return prompt_lens
