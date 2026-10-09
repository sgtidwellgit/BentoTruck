"""
bento — multi-agent: team orchestration (the flagship module the whole fleet is named after).

    team = (
        bento.Bento("launch-team", strategy=bento.Sequential())
        .add(researcher, role="find the facts")
        .add(writer, role="turn facts into copy")
    )
    team.run("Announce our new menu.")

Members are anything Runnable with a `.name` — agents, or whole other teams.
Every handoff is a `sake.Message` on the team's `Hub`, so `team.transcript`
shows exactly who said what to whom. Strategies:

- `Sequential` — a relay: each member builds on the previous member's output
- `Parallel` — everyone works the goal at once; results merged (optionally by a synthesizer)
- `Coordinator` — a lead agent delegates to members through `ask_<member>` tools
- `Vote` — everyone answers independently; a judge (or majority) picks the winner
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from bentotruck.gyoza import FunctionTool
from bentotruck.nigiri import ModelProvider
from bentotruck.rice import invoke
from bentotruck.sake import Hub, Message

ORCHESTRATOR = "bento"


@dataclass
class Member:
    agent: Any
    role: str

    @property
    def name(self) -> str:
        return self.agent.name


class Strategy(ABC):
    """How a team turns one goal into one answer."""

    @abstractmethod
    def run(self, team: "Bento", goal: str) -> str:
        raise NotImplementedError


class Sequential(Strategy):
    """A relay: each member gets the goal plus the previous member's output. `rounds` > 1 loops the relay."""

    def __init__(self, rounds: int = 1) -> None:
        self.rounds = rounds

    def run(self, team: "Bento", goal: str) -> str:
        work, previous = "", None
        for _ in range(self.rounds):
            for member in team.members:
                if previous is None:
                    prompt = f"{goal}\n\nYour role: {member.role}"
                else:
                    prompt = (
                        f"Team goal: {goal}\n\nWork so far (from {previous}):\n{work}\n\n"
                        f"Your role: {member.role}. Build on the work so far and reply with the updated result."
                    )
                work = team.ask(member, prompt)
                previous = member.name
        return work


class Parallel(Strategy):
    """
    Every member works the goal concurrently. With a `synthesizer` (an agent or
    any Runnable) the contributions are merged into one answer; without one they
    are returned as labeled sections.
    """

    def __init__(self, synthesizer: Any = None, *, max_workers: int | None = None) -> None:
        self.synthesizer = synthesizer
        self.max_workers = max_workers

    def gather(self, team: "Bento", goal: str) -> dict[str, str]:
        def work(member: Member) -> str:
            return team.ask(member, f"{goal}\n\nYour role: {member.role}")

        with ThreadPoolExecutor(max_workers=self.max_workers or len(team.members)) as pool:
            answers = list(pool.map(work, team.members))
        return {member.name: answer for member, answer in zip(team.members, answers)}

    def run(self, team: "Bento", goal: str) -> str:
        contributions = self.gather(team, goal)
        sections = "\n\n".join(f"[{name}]\n{answer}" for name, answer in contributions.items())
        if self.synthesizer is None:
            return sections
        return str(
            invoke(
                self.synthesizer,
                f"Goal: {goal}\n\nTeam contributions:\n\n{sections}\n\n"
                "Merge these into one final answer. Resolve any disagreements and keep what's best from each.",
            )
        )


class Coordinator(Strategy):
    """
    A lead agent runs the goal with one `ask_<member>` tool per team member, and
    decides itself who to delegate which subtask to. The tools are removed again
    after the run.
    """

    def __init__(self, lead: Any) -> None:
        self.lead = lead

    def run(self, team: "Bento", goal: str) -> str:
        team.hub.register(self.lead.name)
        tools = [self._delegate_tool(team, member) for member in team.members]
        roster = "\n".join(f"- ask_{_slug(m.name)}: {m.name} — {m.role}" for m in team.members)
        added = [t.name for t in tools if t.name not in {x.name for x in self.lead.tools}]
        self.lead.with_tools(*tools)
        try:
            return self.lead.run(
                f"{goal}\n\nYou lead a team. Delegate subtasks with these tools, then combine the results "
                f"into the final answer:\n{roster}"
            )
        finally:
            for name in added:
                self.lead._tools.pop(name, None)

    def _delegate_tool(self, team: "Bento", member: Member) -> FunctionTool:
        lead_name = self.lead.name

        def delegate(task: str) -> str:
            return team.ask(member, task, sender=lead_name)

        return FunctionTool(
            delegate,
            name=f"ask_{_slug(member.name)}",
            description=f"Delegate a subtask to {member.name} ({member.role}) and get their answer.",
            parameters={
                "type": "object",
                "properties": {"task": {"type": "string", "description": "The subtask, fully self-contained."}},
                "required": ["task"],
            },
        )


class Vote(Strategy):
    """
    Every member answers independently (in parallel). A `judge` — an agent, any
    Runnable, or a ModelProvider — picks the best answer; without a judge the
    most common answer (normalized) wins, ties going to the earliest member.
    """

    def __init__(self, judge: Any = None) -> None:
        self.judge = judge
        self.ballots: dict[str, str] = {}

    def run(self, team: "Bento", goal: str) -> str:
        self.ballots = Parallel().gather(team, goal)
        answers = list(self.ballots.values())
        if self.judge is None:
            counts = Counter(_normalize(a) for a in answers)
            winner = max(answers, key=lambda a: counts[_normalize(a)])
            return winner
        listing = "\n\n".join(f"Answer {i}:\n{a}" for i, a in enumerate(answers, start=1))
        prompt = f"Goal: {goal}\n\n{listing}\n\nWhich answer best achieves the goal? Reply with its number only."
        reply = self.judge.complete(prompt) if isinstance(self.judge, ModelProvider) else str(invoke(self.judge, prompt))
        match = re.search(r"\d+", reply)
        index = int(match.group()) - 1 if match else 0
        return answers[index] if 0 <= index < len(answers) else answers[0]


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower() or "member"


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


_STRATEGIES = {"sequential": Sequential, "parallel": Parallel, "vote": Vote}


class Bento:
    """
    A team of agents working toward one goal. `strategy` is a Strategy instance or
    one of "sequential", "parallel", "vote" (a Coordinator needs a lead, so pass
    `bento.Coordinator(lead_agent)`).
    """

    def __init__(self, name: str = "bento", *, strategy: Strategy | str = "sequential") -> None:
        self.name = name
        if isinstance(strategy, str):
            if strategy not in _STRATEGIES:
                raise ValueError(f"Unknown strategy {strategy!r}; choose from {sorted(_STRATEGIES)} or pass a Strategy.")
            strategy = _STRATEGIES[strategy]()
        self.strategy = strategy
        self.members: list[Member] = []
        self.hub = Hub()
        self.hub.register(ORCHESTRATOR)

    def add(self, agent: Any, *, role: str | None = None) -> "Bento":
        """Add a member (anything Runnable with a unique `.name`). `role` defaults to its description."""

        if not hasattr(agent, "name") or not hasattr(agent, "run"):
            raise TypeError("Team members need a .name and a .run(goal) method.")
        if any(m.name == agent.name for m in self.members):
            raise ValueError(f"This team already has a member named {agent.name!r}.")
        self.members.append(Member(agent, role or getattr(agent, "description", "") or agent.name))
        self.hub.register_agent(agent)
        return self

    @property
    def agents(self) -> list[Any]:
        """Every member, flattened through nested teams — handy for `wasabi.Tracer().attach(*team.agents)`."""

        flat: list[Any] = []
        for member in self.members:
            flat.extend(member.agent.agents if isinstance(member.agent, Bento) else [member.agent])
        return flat

    def ask(self, member: Member, prompt: str, *, sender: str = ORCHESTRATOR) -> str:
        """Send one task to one member over the hub and return their reply."""

        return str(self.hub.request(sender, member.name, prompt))

    def run(self, goal: Any) -> str:
        if not self.members:
            raise RuntimeError(f"Team {self.name!r} has no members — call .add(agent) first.")
        return self.strategy.run(self, str(goal))

    @property
    def transcript(self) -> list[Message]:
        return list(self.hub.log)

    def __repr__(self) -> str:
        return f"Bento({self.name!r}, members={[m.name for m in self.members]})"


__all__ = ["Bento", "Member", "Strategy", "Sequential", "Parallel", "Coordinator", "Vote"]
