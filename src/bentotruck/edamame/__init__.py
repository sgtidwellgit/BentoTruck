"""edamame — memory: conversation, working, vector, long-term, and graph stores behind one interface."""

from __future__ import annotations

import json
import math
import re
import time
import zlib
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


STOPWORDS = frozenset(
    "a an the and or but if then of to in on at by for with from as is are was were be been being it its this that "
    "these those there their they them he she we you i our your his her not no so do does did done have has had "
    "can could would should will may might must also than which who whom what when where why how about into over".split()
)


def _terms(text: str) -> list[str]:
    """Content words for the default embedder: lowercased, stopwords dropped, simple plurals folded."""

    tokens = re.findall(r"[a-z0-9']+", text.lower())
    content = [t for t in tokens if t not in STOPWORDS] or tokens
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in content]


def _default_embed(text: str, dims: int = 256) -> list[float]:
    # crc32, not hash(): str hashing is randomized per process, which would make
    # rankings (and LongTermMemory reloads) differ from run to run
    vector = [0.0] * dims
    for token in _terms(text):
        vector[zlib.crc32(token.encode("utf-8")) % dims] += 1.0
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


@dataclass(frozen=True)
class Fact:
    """One edge in a GraphMemory: subject —relation→ object."""

    subject: str
    relation: str
    object: str

    def __str__(self) -> str:
        return f"{self.subject} {self.relation} {self.object}"


# multi-word relations come first so "is located in" wins over plain "is"
_FACT_PATTERN = re.compile(
    r"^\s*(?P<subject>.+?)\s+(?P<relation>is located in|works at|works for|lives in|belongs to|depends on|"
    r"reports to|is|are|was|were|has|have|owns|likes|prefers|uses)\s+(?P<object>.+?)\s*[.!]?\s*$",
    re.IGNORECASE,
)
_ENTITY = re.compile(r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*")


class GraphMemory(Memory):
    """
    Knowledge-graph memory: stores facts as (subject, relation, object) edges and
    recalls by walking the graph out from entities mentioned in the query.

    Add facts explicitly with `add_fact(...)`, or pass plain sentences to `add()` —
    simple "X is Y" / "X works at Y" style statements are parsed into facts, and
    anything else is linked to the capitalized entities it mentions.
    """

    def __init__(self, *, depth: int = 1) -> None:
        self.depth = depth
        self._facts: list[Fact] = []

    def add_fact(self, subject: str, relation: str, obj: str) -> Fact:
        fact = Fact(subject.strip(), relation.strip(), obj.strip())
        if fact not in self._facts:
            self._facts.append(fact)
        return fact

    def add(self, role: str, content: str, **metadata: Any) -> None:
        if {"subject", "relation", "object"} <= metadata.keys():
            self.add_fact(metadata["subject"], metadata["relation"], metadata["object"])
            return
        match = _FACT_PATTERN.match(content)
        if match:
            self.add_fact(match["subject"], match["relation"].lower(), match["object"])
            return
        for entity in dict.fromkeys(_ENTITY.findall(content)):
            self.add_fact(entity, "mentioned in", content)

    def facts(self, subject: str | None = None, relation: str | None = None, obj: str | None = None) -> list[Fact]:
        """Facts matching every given field (case-insensitive)."""

        def ok(value: str, wanted: str | None) -> bool:
            return wanted is None or value.lower() == wanted.lower()

        return [f for f in self._facts if ok(f.subject, subject) and ok(f.relation, relation) and ok(f.object, obj)]

    def entities(self) -> list[str]:
        return list(dict.fromkeys(name for f in self._facts for name in (f.subject, f.object)))

    def neighbors(self, entity: str) -> list[Fact]:
        key = entity.lower()
        return [f for f in self._facts if f.subject.lower() == key or f.object.lower() == key]

    def recall(self, query: str | None = None, *, k: int = 5) -> list[MemoryItem]:
        if query is None:
            return [MemoryItem(role="fact", content=str(f)) for f in self._facts[-k:]]

        lowered = query.lower()
        frontier = {e.lower() for e in self.entities() if re.search(rf"\b{re.escape(e.lower())}\b", lowered)}
        seen_entities = set(frontier)
        found: list[Fact] = []
        for _ in range(self.depth + 1):
            next_frontier: set[str] = set()
            for entity in frontier:
                for fact in self.neighbors(entity):
                    if fact not in found:
                        found.append(fact)
                    for name in (fact.subject.lower(), fact.object.lower()):
                        if name not in seen_entities:
                            next_frontier.add(name)
            seen_entities |= next_frontier
            frontier = next_frontier
        return [MemoryItem(role="fact", content=str(f)) for f in found[:k]]

    def clear(self) -> None:
        self._facts.clear()


__all__ = [
    "MemoryItem",
    "Memory",
    "ConversationMemory",
    "WorkingMemory",
    "VectorMemory",
    "LongTermMemory",
    "GraphMemory",
    "Fact",
]
