"""Caching utilities for repeated numerical kernels and tensors."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


@dataclass
class LRUCache(Generic[K, V]):
    """Minimal LRU cache with explicit max-size control."""

    max_size: int = 16

    def __post_init__(self) -> None:
        if self.max_size <= 0:
            raise ValueError("max_size must be positive.")
        self._store: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        value = self._store.get(key)
        if value is None:
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: K, value: V) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
