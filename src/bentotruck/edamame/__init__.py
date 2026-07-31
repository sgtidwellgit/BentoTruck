"""edamame — memory: conversation, working, and vector-backed stores behind one interface."""

from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class MemoryItem:
    """One stored entry: who said it, what it was, and any caller-supplied metadata."""

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Memory(ABC):
    """Base interface every memory type implements: add, recall, clear."""

    @abstractmethod
    def add(self, role: str, content: str, **metadata: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def recall(self, query: str | None = None, *, k: int = 5) -> list[MemoryItem]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


class ConversationMemory(Memory):
    """A rolling window of the last `max_items` turns — recall() ignores `query` and returns recent history in order."""

    def __init__(self, max_items: int = 50) -> None:
        self._items: deque[MemoryItem] = deque(maxlen=max_items)

    def add(self, role: str, content: str, **metadata: Any) -> None:
        self._items.append(MemoryItem(role=role, content=content, metadata=metadata))

    def recall(self, query: str | None = None, *, k: int = 5) -> list[MemoryItem]:
        return list(self._items)[-k:]

    def clear(self) -> None:
        self._items.clear()


class WorkingMemory(Memory):
    """Key/value scratchpad for facts an agent needs during a run — not a transcript. Keys default to `role`."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def add(self, role: str, content: str, **metadata: Any) -> None:
        key = metadata.get("key", role)
        self._values[key] = content

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def recall(self, query: str | None = None, *, k: int = 5) -> list[MemoryItem]:
        items = [MemoryItem(role=key, content=str(value)) for key, value in self._values.items()]
        return items[:k]

    def clear(self) -> None:
        self._values.clear()


def _default_embed(text: str, dims: int = 256) -> list[float]:
    vector = [0.0] * dims
    for token in text.lower().split():
        vector[hash(token) % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorMemory(Memory):
    """
    Semantic recall over stored items via cosine similarity.

    Uses a simple offline hashing embedder by default (no API calls, good enough
    for demos and tests) — pass `embed_fn` to plug in a real embedding model.
    """

    def __init__(self, embed_fn: Callable[[str], list[float]] | None = None) -> None:
        self.embed_fn = embed_fn or _default_embed
        self._items: list[tuple[MemoryItem, list[float]]] = []

    def add(self, role: str, content: str, **metadata: Any) -> None:
        item = MemoryItem(role=role, content=content, metadata=metadata)
        self._items.append((item, self.embed_fn(content)))

    def recall(self, query: str | None = None, *, k: int = 5) -> list[MemoryItem]:
        if not self._items:
            return []
        if query is None:
            return [item for item, _ in self._items[-k:]]
        query_vec = self.embed_fn(query)
        ranked = sorted(self._items, key=lambda pair: _cosine(query_vec, pair[1]), reverse=True)
        return [item for item, _ in ranked[:k]]

    def items(self) -> list[MemoryItem]:
        return [item for item, _ in self._items]

    def clear(self) -> None:
        self._items.clear()


class LongTermMemory(Memory):
    """Vector memory persisted to a JSON file — same recall semantics as VectorMemory, survives process restarts."""

    def __init__(self, path: str | Path, *, embed_fn: Callable[[str], list[float]] | None = None) -> None:
        self.path = Path(path)
        self._inner = VectorMemory(embed_fn=embed_fn)
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for entry in data:
                self._inner.add(entry["role"], entry["content"], **entry.get("metadata", {}))

    def add(self, role: str, content: str, **metadata: Any) -> None:
        self._inner.add(role, content, **metadata)
        self._save()

    def recall(self, query: str | None = None, *, k: int = 5) -> list[MemoryItem]:
        return self._inner.recall(query, k=k)

    def clear(self) -> None:
        self._inner.clear()
        self._save()

    def _save(self) -> None:
        data = [{"role": item.role, "content": item.content, "metadata": item.metadata} for item in self._inner.items()]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")


__all__ = ["MemoryItem", "Memory", "ConversationMemory", "WorkingMemory", "VectorMemory", "LongTermMemory"]
