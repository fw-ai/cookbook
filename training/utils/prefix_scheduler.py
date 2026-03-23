"""Prefix-aware prompt scheduling for RL rollouts.

Groups prompts by shared token prefix so that consecutive requests to
the inference deployment maximise KV-cache reuse on the server.

This is a minimal / stub implementation for the initial PR.  Future
work will add weight-version-based eviction and tighter integration
with the radix-tree router in Miles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence

logger = logging.getLogger(__name__)


@dataclass
class _Node:
    """Internal trie node."""

    token: int | None = None  # First token of the edge (None for root).
    children: dict[int, "_Node"] = field(default_factory=dict)
    prompt_indices: list[int] = field(default_factory=list)


class PrefixTree:
    """Lightweight token-based trie for prefix-aware scheduling.

    Insert prompt token sequences, then call :meth:`get_prefix_ordered_indices`
    to retrieve prompt indices ordered so that prompts sharing a prefix are
    adjacent (DFS order).

    This is a **stub** -- no eviction, no weight versioning, no thread safety.
    It exists to establish the interface for future extension.
    """

    def __init__(self) -> None:
        self._root = _Node()
        self._size = 0

    @property
    def size(self) -> int:
        """Number of prompts inserted."""
        return self._size

    def insert(self, token_ids: Sequence[int], prompt_idx: int) -> None:
        """Insert a prompt's token sequence into the trie."""
        node = self._root
        for tok in token_ids:
            if tok not in node.children:
                node.children[tok] = _Node(token=tok)
            node = node.children[tok]
        node.prompt_indices.append(prompt_idx)
        self._size += 1

    def get_prefix_ordered_indices(self) -> List[int]:
        """Return prompt indices in DFS order (shared-prefix-adjacent)."""
        result: list[int] = []
        self._dfs(self._root, result)
        return result

    def _dfs(self, node: _Node, result: list[int]) -> None:
        result.extend(node.prompt_indices)
        for child in node.children.values():
            self._dfs(child, result)

    def clear(self) -> None:
        """Reset the trie."""
        self._root = _Node()
        self._size = 0


def reorder_by_prefix(
    prompt_token_lists: List[List[int]],
) -> List[int]:
    """Given a list of prompt token sequences, return reordered indices.

    Prompts that share a common prefix will be grouped together in the
    returned ordering, maximising KV-cache locality on the server.

    Args:
        prompt_token_lists: One token-id list per prompt.

    Returns:
        A permutation of ``range(len(prompt_token_lists))``.
    """
    if len(prompt_token_lists) <= 1:
        return list(range(len(prompt_token_lists)))

    tree = PrefixTree()
    for idx, tokens in enumerate(prompt_token_lists):
        tree.insert(tokens, idx)
    return tree.get_prefix_ordered_indices()
