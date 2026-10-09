"""
teriyaki — planning: how agents decide what to do next.

Every planner implements one method, `execute(agent, goal) -> str`, and plugs
into an agent with `Agent.with_planner(...)`. Planners drive the agent through
its public loop primitives — `agent.react(prompt)` (tool-calling loop),
`agent.ask(prompt)` (one tool-free question), and `agent.session.snapshot()` /
`restore()` (to explore and discard branches).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from typing import Any

from bentotruck.nigiri import Message
from bentotruck.rice import EventType


class Planner(ABC):
    """Base class every planning strategy implements."""

    @abstractmethod
    def execute(self, agent: Any, goal: str) -> str:
        raise NotImplementedError


def _parse_list(text: str) -> list[str]:
    """Pull items out of a numbered / bulleted / one-per-line list the model wrote."""

    items = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip()
        if cleaned:
            items.append(cleaned)
    return items


class ReActPlanner(Planner):
    """
    Interleaved reason/act loop — call the model, run the tools it asks for, feed
    the results back, repeat until it answers. This is the default strategy.
    """

    def __init__(self, max_steps: int | None = None) -> None:
        self.max_steps = max_steps

    def execute(self, agent: Any, goal: str) -> str:
        return agent.react(goal, max_steps=self.max_steps)


class StepPlanner(Planner):
    """
    A fixed, user-authored sequence of steps — no model-driven branching. Each step
    runs as its own ReAct turn with the goal and step number in view, so later
    steps can build on earlier answers. Returns the last step's answer.
    """

    def __init__(self, steps: Sequence[str]) -> None:
        if not steps:
            raise ValueError("StepPlanner needs at least one step.")
        self.steps = list(steps)

    def execute(self, agent: Any, goal: str) -> str:
        result = ""
        total = len(self.steps)
        for index, step in enumerate(self.steps, start=1):
            agent._publish(EventType.STEP, planner="step", index=index, step=step)
            result = agent.react(f"Overall goal: {goal}\n\nStep {index} of {total}: {step}")
        return result


class GoalPlanner(Planner):
    """
    Decomposes a goal into subgoals, works each one with the tool loop, then
    synthesizes a final answer from the results.

    By default the model does the decomposition; pass `decompose(goal) -> list[str]`
    to do it yourself.
    """

    def __init__(
        self, *, max_subgoals: int = 5, decompose: Callable[[str], Sequence[str]] | None = None
    ) -> None:
        self.max_subgoals = max_subgoals
        self.decompose = decompose

    def subgoals(self, agent: Any, goal: str) -> list[str]:
        if self.decompose is not None:
            return list(self.decompose(goal))[: self.max_subgoals]
        reply = agent.ask(
            f"Break this goal into at most {self.max_subgoals} concrete subgoals, one per line, "
            f"in the order they should be done. Reply with the list only.\n\nGoal: {goal}"
        )
        return _parse_list(reply)[: self.max_subgoals] or [goal]

    def execute(self, agent: Any, goal: str) -> str:
        subgoals = self.subgoals(agent, goal)
        for index, subgoal in enumerate(subgoals, start=1):
            agent._publish(EventType.STEP, planner="goal", index=index, step=subgoal)
            agent.react(f"Subgoal {index} of {len(subgoals)} (toward: {goal}): {subgoal}")
        return agent.react(f"Using the work above, give the final answer to the original goal: {goal}")


class TreePlanner(Planner):
    """
    Explores several candidate answers (each a full ReAct branch from the same
    starting point), keeps the best one, and discards the others from the session.

    Pass `scorer(goal, candidate) -> float` to choose; otherwise the model is
    asked to pick the best candidate.
    """

    def __init__(self, branches: int = 3, *, scorer: Callable[[str, str], float] | None = None) -> None:
        if branches < 1:
            raise ValueError("TreePlanner needs at least one branch.")
        self.branches = branches
        self.scorer = scorer

    def execute(self, agent: Any, goal: str) -> str:
        mark = agent.session.snapshot()
        candidates: list[str] = []
        for index in range(1, self.branches + 1):
            agent._publish(EventType.STEP, planner="tree", index=index, step=f"branch {index}")
            prompt = goal if self.branches == 1 else f"{goal}\n\n(Approach {index} of {self.branches}: try a distinct angle.)"
            candidates.append(agent.react(prompt))
            agent.session.restore(mark)

        best = self._choose(agent, goal, candidates)
        agent.session.add(Message(role="user", content=goal))
        agent.session.add(Message(role="assistant", content=best))
        return best

    def _choose(self, agent: Any, goal: str, candidates: list[str]) -> str:
        if self.scorer is not None:
            return max(candidates, key=lambda c: self.scorer(goal, c))
        listing = "\n\n".join(f"Candidate {i}:\n{c}" for i, c in enumerate(candidates, start=1))
        reply = agent.ask(
            f"Goal: {goal}\n\n{listing}\n\nWhich candidate best achieves the goal? Reply with its number only."
        )
        match = re.search(r"\d+", reply)
        index = int(match.group()) - 1 if match else 0
        return candidates[index] if 0 <= index < len(candidates) else candidates[0]


@dataclass
class Task:
    """One node in a GraphPlanner: a prompt plus the names of tasks it depends on."""

    prompt: str
    depends_on: list[str] = field(default_factory=list)


class GraphPlanner(Planner):
    """
    Plans over a dependency graph of subgoals rather than a linear chain. Tasks run
    in topological order; each sees the results of the tasks it depends on. The
    result of the last task in that order (or `final=` if given) is returned.

    >>> GraphPlanner({
    ...     "facts": Task("Collect the facts."),
    ...     "risks": Task("List the risks.", depends_on=["facts"]),
    ...     "memo":  Task("Write the memo.", depends_on=["facts", "risks"]),
    ... })  # doctest: +ELLIPSIS
    <...GraphPlanner object at ...>
    """

    def __init__(self, tasks: Mapping[str, Task | str], *, final: str | None = None) -> None:
        self.tasks = {name: task if isinstance(task, Task) else Task(task) for name, task in tasks.items()}
        for name, task in self.tasks.items():
            missing = [d for d in task.depends_on if d not in self.tasks]
            if missing:
                raise ValueError(f"Task {name!r} depends on unknown task(s): {missing}")
        if final is not None and final not in self.tasks:
            raise ValueError(f"final={final!r} is not a task name.")
        self.order = list(TopologicalSorter({n: t.depends_on for n, t in self.tasks.items()}).static_order())
        self.final = final or self.order[-1]
        self.results: dict[str, str] = {}

    def execute(self, agent: Any, goal: str) -> str:
        self.results = {}
        for index, name in enumerate(self.order, start=1):
            task = self.tasks[name]
            agent._publish(EventType.STEP, planner="graph", index=index, step=name)
            inputs = "\n".join(f"[{dep}]\n{self.results[dep]}" for dep in task.depends_on)
            prompt = f"Overall goal: {goal}\n\nTask '{name}': {task.prompt}"
            if inputs:
                prompt += f"\n\nResults from prerequisite tasks:\n{inputs}"
            self.results[name] = agent.react(prompt)
        return self.results[self.final]


__all__ = ["Planner", "ReActPlanner", "StepPlanner", "GoalPlanner", "TreePlanner", "GraphPlanner", "Task"]
