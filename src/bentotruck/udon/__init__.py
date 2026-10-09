"""
udon — pipelines: sequential (and parallel) execution of plain processing stages.

Each stage's output feeds the next. A stage can be a function, anything
Runnable (an agent, a team, another pipeline), or one of the control stages
below. Compose with `Pipeline(...)` or the `|` operator:

    clean = udon.Pipeline(str.strip, str.lower) | summarizer_agent
    clean.run("  SOME LONG TEXT ...  ")

Distinct from `onigiri` (workflows): udon is for straight-line data/task
pipelines without onigiri's graph-shaped branching.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from bentotruck.rice import invoke


def _name(target: Any) -> str:
    return getattr(target, "name", None) or getattr(target, "__name__", None) or type(target).__name__


class Step:
    """A single named stage wrapping a function or Runnable."""

    def __init__(self, action: Any, *, name: str | None = None) -> None:
        self.action = action
        self.name = name or _name(action)

    def run(self, value: Any) -> Any:
        return invoke(self.action, value)

    def __or__(self, other: Any) -> "Pipeline":
        return Pipeline(self, other)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class Condition(Step):
    """Runs `step` only when `predicate(value)` holds; otherwise passes the value through unchanged."""

    def __init__(self, predicate: Callable[[Any], bool], step: Any, *, name: str | None = None) -> None:
        super().__init__(step, name=name or f"if:{_name(step)}")
        self.predicate = predicate

    def run(self, value: Any) -> Any:
        return invoke(self.action, value) if self.predicate(value) else value


class Loop(Step):
    """
    Repeats `step`, feeding each output back in — a fixed number of `times`, or
    until `until(value)` holds (checked after each pass), capped at `max_iterations`.
    """

    def __init__(
        self,
        step: Any,
        *,
        times: int | None = None,
        until: Callable[[Any], bool] | None = None,
        max_iterations: int = 10,
        name: str | None = None,
    ) -> None:
        if times is None and until is None:
            raise ValueError("Loop needs times= or until=.")
        super().__init__(step, name=name or f"loop:{_name(step)}")
        self.times = times
        self.until = until
        self.max_iterations = max_iterations

    def run(self, value: Any) -> Any:
        limit = self.times if self.times is not None else self.max_iterations
        for _ in range(limit):
            value = invoke(self.action, value)
            if self.until is not None and self.until(value):
                break
        return value


class Parallel(Step):
    """
    Runs several stages concurrently on the same input and joins their outputs.
    `merge` receives the list of outputs (default: return the list as-is).
    """

    def __init__(
        self,
        *steps: Any,
        merge: Callable[[list[Any]], Any] | None = None,
        max_workers: int | None = None,
        name: str | None = None,
    ) -> None:
        if not steps:
            raise ValueError("Parallel needs at least one step.")
        super().__init__(steps, name=name or "parallel")
        self.steps = list(steps)
        self.merge = merge
        self.max_workers = max_workers

    def run(self, value: Any) -> Any:
        with ThreadPoolExecutor(max_workers=self.max_workers or len(self.steps)) as pool:
            outputs = list(pool.map(lambda step: invoke(step, value), self.steps))
        return self.merge(outputs) if self.merge else outputs


@dataclass
class PipelineResult:
    output: Any
    trace: list[tuple[str, Any]] = field(default_factory=list)


class Pipeline:
    """An ordered list of stages run in sequence, each stage's output feeding the next."""

    def __init__(self, *stages: Any, name: str = "pipeline") -> None:
        self.name = name
        self.stages: list[Step] = []
        for stage in stages:
            self.then(stage)

    def then(self, stage: Any) -> "Pipeline":
        """Append a stage (a nested Pipeline is flattened in). Returns self for chaining."""

        if isinstance(stage, Pipeline):
            self.stages.extend(stage.stages)
        else:
            self.stages.append(stage if isinstance(stage, Step) else Step(stage))
        return self

    def __or__(self, other: Any) -> "Pipeline":
        return Pipeline(*self.stages, other, name=self.name)

    def execute(self, value: Any) -> PipelineResult:
        """Run every stage and return the output plus each stage's intermediate result."""

        trace = []
        for stage in self.stages:
            value = stage.run(value)
            trace.append((stage.name, value))
        return PipelineResult(output=value, trace=trace)

    def run(self, value: Any) -> Any:
        return self.execute(value).output

    __call__ = run

    def __len__(self) -> int:
        return len(self.stages)

    def __repr__(self) -> str:
        return f"Pipeline({' | '.join(s.name for s in self.stages)})"


__all__ = ["Pipeline", "PipelineResult", "Step", "Condition", "Loop", "Parallel"]
