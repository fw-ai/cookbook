"""Cursor-based dataloader utilities."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CursorItem(Generic[T]):
    index: int
    value: T


class CursorDataLoader(Generic[T]):
    def __init__(
        self,
        items: list[T],
        start_cursor: int = 0,
        *,
        epochs: int = 1,
        shuffle: bool = False,
        seed: int = 0,
    ):
        if start_cursor < 0:
            raise ValueError("start_cursor must be >= 0")
        if epochs < 0:
            raise ValueError("epochs must be >= 0")
        self.items = items
        self.epochs = epochs
        self.shuffle = shuffle
        self.seed = seed
        self.cursor = start_cursor
        self.next_index = start_cursor
        self._resolved: set[int] = set()
        self._permutations: dict[int, list[int]] = {}

    def __iter__(self):
        return self

    def __next__(self) -> CursorItem[T]:
        if self.next_index >= self.total_items:
            raise StopIteration
        idx = self.next_index
        self.next_index += 1
        return CursorItem(index=idx, value=copy.deepcopy(self.items[self._row_index(idx)]))

    @property
    def data_consumed(self) -> int:
        return self.cursor

    @property
    def total_items(self) -> int:
        return len(self.items) * self.epochs

    @property
    def epoch_id(self) -> int:
        return self.cursor // len(self.items) if self.items else 0

    @property
    def sample_offset(self) -> int:
        return self.cursor % len(self.items) if self.items else 0

    def mark_resolved(self, index: int) -> None:
        if index < self.cursor:
            return
        if index >= self.total_items:
            raise ValueError("resolved index out of range")
        self._resolved.add(index)
        while self.cursor in self._resolved:
            self._resolved.remove(self.cursor)
            self.cursor += 1

    def _row_index(self, index: int) -> int:
        if not self.items:
            raise IndexError("empty dataloader")
        epoch = index // len(self.items)
        offset = index % len(self.items)
        if not self.shuffle:
            return offset
        if epoch not in self._permutations:
            perm = list(range(len(self.items)))
            random.Random(self.seed + epoch).shuffle(perm)
            self._permutations[epoch] = perm
        return self._permutations[epoch][offset]
