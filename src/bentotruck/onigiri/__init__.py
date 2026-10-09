"""
onigiri — workflows: deterministic orchestration around one or more agents.

A `Workflow` is a graph of named `Step`s joined by edges, optionally guarded
by `Condition`s, so it can branch and loop back:

    flow = onigiri.Workflow("publish")
    flow.add(onigiri.Step("draft", writer), onigiri.Step("review", editor), onigiri.Step("ship", publish_fn))
    flow.connect("draft", "review")
    flow.connect("review", "draft", when=onigiri.Condition.contains("REVISE"))
    flow.connect("review", "ship")          # unconditional edges are the fallback
    flow.run("Write the launch post.")

Each step receives the previous step's output by default; pass
`input=lambda state: ...` to feed it anything from the shared `State`.

Distinct from `udon` (pipelines): onigiri workflows are graph-shaped and
condition/loop-aware; udon pipelines are simple linear/parallel stages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bentotruck.rice import invoke

END = "__end__"


@dataclass
class State:
    """Shared state for one workflow run: the original input, every step's output, and free-form data."""

    input: Any
    last: Any = None
    results: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    path: list[str] = field(default_factory=list)
    visits: dict[str, int] = field(default_factory=dict)

    def __getitem__(self, step_name: str) -> Any:
        return self.results[step_name]


class Condition:
    """A named predicate over the workflow State. Combine with `&`, `|`, and `~`."""

    def __init__(self, predicate: Callable[[State], bool], name: str | None = None) -> None:
        self.predicate = predicate
        self.name = name or getattr(predicate, "__name__", "condition")

    def __call__(self, state: State) -> bool:
        return bool(self.predicate(state))

    def __and__(self, other: "Condition") -> "Condition":
        return Condition(lambda s: self(s) and other(s), f"({self.name} and {other.name})")

    def __or__(self, other: "Condition") -> "Condition":
        return Condition(lambda s: self(s) or other(s), f"({self.name} or {other.name})")

    def __invert__(self) -> "Condition":
        return Condition(lambda s: not self(s), f"not {self.name}")

    @classmethod
    def contains(cls, text: str, *, case_sensitive: bool = False) -> "Condition":
        """True when the last output contains `text`."""

        def check(state: State) -> bool:
            last = str(state.last)
            return text in last if case_sensitive else text.lower() in last.lower()

        return cls(check, f"contains {text!r}")

    @classmethod
    def visited(cls, step_name: str, times: int) -> "Condition":
        """True once `step_name` has run at least `times` times — the usual loop breaker."""

        return cls(lambda s: s.visits.get(step_name, 0) >= times, f"{step_name} visited {times}x")

    def __repr__(self) -> str:
        return f"Condition({self.name})"


class Step:
    """A single unit of work — an agent run, a tool call, a sub-workflow, or a plain function."""

    def __init__(self, name: str, action: Any, *, input: Callable[[State], Any] | None = None) -> None:
        self.name = name
        self.action = action
        self.input = input

    def execute(self, state: State) -> Any:
        value = self.input(state) if self.input is not None else state.last
        return invoke(self.action, value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class Loop(Step):
    """
    Repeats `body` (fed its own previous output) until `until(state)` holds or
    `max_iterations` is reached. Only the loop's final output is recorded.
    """

    def __init__(
        self,
        name: str,
        body: Any,
        *,
        until: Callable[[State], bool],
        max_iterations: int = 5,
        input: Callable[[State], Any] | None = None,
    ) -> None:
        super().__init__(name, body, input=input)
        self.until = until
        self.max_iterations = max_iterations

    def execute(self, state: State) -> Any:
        value = self.input(state) if self.input is not None else state.last
        for _ in range(self.max_iterations):
            value = invoke(self.action, value)
            state.last = value
            if self.until(state):
                break
        return value


@dataclass
class Edge:
    source: str
    target: str
    when: Callable[[State], bool] | None = None


@dataclass
class WorkflowResult:
    output: Any
    state: State

    @property
    def path(self) -> list[str]:
        return self.state.path

    @property
    def results(self) -> dict[str, Any]:
        return self.state.results


class Workflow:
    """A named, runnable graph of Steps. Starts at the first step added unless `start(...)` says otherwise."""

    def __init__(self, name: str = "workflow", *, max_steps: int = 50) -> None:
        self.name = name
        self.max_steps = max_steps
        self.steps: dict[str, Step] = {}
        self.edges: list[Edge] = []
        self._start: str | None = None

    def add(self, *steps: Step) -> "Workflow":
        for step in steps:
            if step.name in self.steps or step.name == END:
                raise ValueError(f"Duplicate or reserved step name {step.name!r}.")
            self.steps[step.name] = step
            if self._start is None:
                self._start = step.name
        return self

    def start(self, name: str) -> "Workflow":
        self._require(name)
        self._start = name
        return self

    def connect(self, source: str, target: str, *, when: Callable[[State], bool] | None = None) -> "Workflow":
        """
        Add an edge. After `source` runs, conditional edges are checked in the order
        added and the first that holds is followed; if none hold, the first
        unconditional edge is taken; with no edge to follow, the workflow ends.
        """

        self._require(source)
        if target != END:
            self._require(target)
        self.edges.append(Edge(source, target, when))
        return self

    def chain(self, *names: str) -> "Workflow":
        """Connect steps in a straight line: chain("a", "b", "c") == a→b, b→c."""

        for source, target in zip(names, names[1:]):
            self.connect(source, target)
        return self

    def _require(self, name: str) -> None:
        if name not in self.steps:
            raise KeyError(f"Workflow {self.name!r} has no step named {name!r}.")

    def _next(self, current: str, state: State) -> str | None:
        outgoing = [e for e in self.edges if e.source == current]
        for edge in outgoing:
            if edge.when is not None and edge.when(state):
                return edge.target
        for edge in outgoing:
            if edge.when is None:
                return edge.target
        return None

    def execute(self, input: Any = None, *, data: dict[str, Any] | None = None) -> WorkflowResult:
        if self._start is None:
            raise RuntimeError(f"Workflow {self.name!r} has no steps.")
        state = State(input=input, last=input, data=dict(data or {}))
        current: str | None = self._start
        for _ in range(self.max_steps):
            if current is None or current == END:
                return WorkflowResult(output=state.last, state=state)
            step = self.steps[current]
            output = step.execute(state)
            state.last = output
            state.results[current] = output
            state.path.append(current)
            state.visits[current] = state.visits.get(current, 0) + 1
            current = self._next(current, state)
        if current is None or current == END:
            return WorkflowResult(output=state.last, state=state)
        raise RuntimeError(f"Workflow {self.name!r} exceeded max_steps={self.max_steps} (path: {state.path[-5:]}...).")

    def run(self, input: Any = None) -> Any:
        return self.execute(input).output

    def __repr__(self) -> str:
        return f"Workflow({self.name!r}, steps={list(self.steps)})"


__all__ = ["Workflow", "WorkflowResult", "Step", "Loop", "Condition", "State", "Edge", "END"]
