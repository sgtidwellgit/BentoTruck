"""
tempura — skills: reusable, named behaviors an agent can learn and invoke.

A skill bundles instructions (a system prompt), the tools it needs, and an
output contract. Unlike a raw `gyoza.Tool`, a skill runs its own focused
sub-agent — so it can take several model/tool steps to do its job.

    agent.learn(tempura.SummarizeSkill())   # the model can now call "summarize"
    agent.use("summarize", long_text)       # or invoke it directly
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from bentotruck.gyoza import FunctionTool, Tool
from bentotruck.nigiri import ModelProvider
from bentotruck.teriyaki import GoalPlanner, _parse_list


class Skill:
    """
    A reusable behavior: `instructions` become the sub-agent's system prompt,
    `template` turns the caller's input into its first message, and `tools` are
    what it may call. Subclass and override `postprocess` to shape the output.
    """

    name = "skill"
    description = "A reusable skill."
    instructions = "You are a focused assistant. Complete the task you are given."
    template = "{input}"

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        *,
        tools: Sequence[Tool] = (),
        template: str | None = None,
        planner: Any = None,
        provider: ModelProvider | None = None,
        max_steps: int = 6,
    ) -> None:
        self.name = name or self.name
        self.description = description or self.description
        self.instructions = instructions or self.instructions
        self.template = template or self.template
        self.tools = list(tools)
        self.planner = planner
        self.provider = provider
        self.max_steps = max_steps

    def prompt(self, input: str) -> str:
        return self.template.format(input=input)

    def postprocess(self, output: str) -> str:
        return output.strip()

    def run(self, input: Any, *, agent: Any = None, provider: ModelProvider | None = None) -> str:
        """Run the skill on `input` using `provider`, the skill's bound provider, or `agent`'s provider."""

        from bentotruck.rice import Agent

        model = provider or self.provider or (agent.provider if agent is not None else None)
        if model is None:
            raise RuntimeError(f"Skill {self.name!r} needs a model — pass provider=, bind one, or run it via an agent.")
        sub = Agent(
            f"{agent.name}.{self.name}" if agent is not None else self.name,
            system_prompt=self.instructions,
            max_steps=self.max_steps,
        ).using(model)
        sub.with_tools(*self.tools)
        if self.planner is not None:
            sub.with_planner(self.planner)
        if agent is not None:
            sub.events.subscribe(agent.events.publish)
        return self.postprocess(sub.run(self.prompt(str(input))))

    def as_tool(self, agent: Any = None) -> Tool:
        """Expose this skill as a tool the model can call (what `Agent.learn` does)."""

        skill = self

        def call(input: str) -> str:
            return skill.run(input, agent=agent)

        return FunctionTool(
            call,
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {"input": {"type": "string", "description": "What the skill should work on."}},
                "required": ["input"],
            },
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class SummarizeSkill(Skill):
    """Condenses text to its key points."""

    name = "summarize"
    description = "Summarize a piece of text into its key points."
    template = "Summarize the following text.\n\n---\n{input}\n---"

    def __init__(self, *, max_words: int = 120, style: str = "bullets", **kwargs: Any) -> None:
        shape = "as short bullet points" if style == "bullets" else "as one tight paragraph"
        kwargs.setdefault(
            "instructions",
            f"You summarize text faithfully {shape}, in at most {max_words} words. "
            "Never add information that is not in the text.",
        )
        super().__init__(**kwargs)
        self.max_words = max_words
        self.style = style


def _search_tool(source: Any, k: int) -> Tool:
    """Adapt a Memory (has .recall), a Tool, or a plain `fn(query) -> list[str]` into a 'search' tool."""

    if isinstance(source, Tool):
        return source
    if hasattr(source, "recall"):

        def search(query: str) -> list[str]:
            """Search the knowledge base for passages relevant to the query."""
            return [item.content for item in source.recall(query, k=k)]

    elif callable(source):

        def search(query: str) -> list[str]:
            """Search for passages relevant to the query."""
            return list(source(query))[:k]

    else:
        raise TypeError("search source must be a Memory, a gyoza Tool, or a callable(query) -> list[str].")
    return FunctionTool(search, name="search")


class SearchSkill(Skill):
    """
    Answers a question from a searchable source — an `edamame` memory, any search
    `gyoza.Tool`, or a `fn(query) -> list[str]` — citing what it found.
    """

    name = "search"
    description = "Search the knowledge base and answer a question from what is found."
    instructions = (
        "You answer questions by searching first. Call the search tool with focused queries, "
        "then answer only from the results. If the results don't contain the answer, say so."
    )

    def __init__(self, source: Any, *, k: int = 5, **kwargs: Any) -> None:
        self.search_tool = _search_tool(source, k)
        kwargs.setdefault("tools", [self.search_tool])
        super().__init__(**kwargs)

    def search(self, query: str) -> Any:
        """Run the underlying search directly — no model involved."""

        return self.search_tool.run(query=query)


class ResearchSkill(SearchSkill):
    """
    Multi-step research: breaks the topic into questions, searches each one, and
    writes a synthesis citing sources as [1], [2], ...
    """

    name = "research"
    description = "Research a topic across several searches and write a cited synthesis."
    instructions = (
        "You are a careful researcher. Search for each question you are given, keep track of what "
        "each result says, and when asked for the final answer write a concise synthesis that cites "
        "results as [1], [2], ... Never state anything the results don't support."
    )

    def __init__(self, source: Any, *, questions: int = 3, **kwargs: Any) -> None:
        kwargs.setdefault("planner", GoalPlanner(max_subgoals=questions))
        kwargs.setdefault("max_steps", 8)
        super().__init__(source, **kwargs)


_CODE_BLOCK = re.compile(r"```[\w+-]*\n(.*?)```", re.DOTALL)


class CodingSkill(Skill):
    """
    Writes code for a task and returns just the code (fences stripped). Give it a
    `gyoza.DockerTool` via `tools=` if it should run and test what it writes.
    """

    name = "code"
    description = "Write code that accomplishes a programming task."
    template = "Task: {input}"

    def __init__(self, *, language: str = "python", **kwargs: Any) -> None:
        kwargs.setdefault(
            "instructions",
            f"You are an expert {language} programmer. Write correct, idiomatic, minimal {language} code "
            "for the task. Reply with a single fenced code block and nothing else.",
        )
        super().__init__(**kwargs)
        self.language = language

    def postprocess(self, output: str) -> str:
        match = _CODE_BLOCK.search(output)
        return (match.group(1) if match else output).strip()


class PlanningSkill(Skill):
    """Turns a goal into a short, ordered, actionable plan (returned as numbered lines)."""

    name = "plan"
    description = "Turn a goal into a short, ordered, actionable plan."
    template = "Goal: {input}"

    def __init__(self, *, max_steps_in_plan: int = 7, **kwargs: Any) -> None:
        kwargs.setdefault(
            "instructions",
            f"You are a pragmatic planner. Produce at most {max_steps_in_plan} concrete steps, one per line, "
            "in the order they should be done. Reply with the list only.",
        )
        super().__init__(**kwargs)
        self.max_steps_in_plan = max_steps_in_plan

    def postprocess(self, output: str) -> str:
        steps = _parse_list(output)[: self.max_steps_in_plan]
        return "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))

    def plan(self, goal: str, **kwargs: Any) -> list[str]:
        """Like `run`, but returns the steps as a list."""

        return _parse_list(self.run(goal, **kwargs))


def skill(
    fn: Callable[[str], str] | None = None, *, name: str | None = None, description: str | None = None
) -> Any:
    """
    Decorator: turn a plain `fn(input) -> str` into a model-free Skill, for
    deterministic behaviors you still want to teach an agent by name.
    """

    def wrap(f: Callable[[str], str]) -> Skill:
        class _FunctionSkill(Skill):
            def run(self, input: Any, *, agent: Any = None, provider: ModelProvider | None = None) -> str:
                return self.postprocess(str(f(str(input))))

        doc = (f.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
        return _FunctionSkill(name=name or f.__name__, description=description or doc or f.__name__)

    return wrap(fn) if fn is not None else wrap


__all__ = [
    "Skill",
    "SummarizeSkill",
    "SearchSkill",
    "ResearchSkill",
    "CodingSkill",
    "PlanningSkill",
    "skill",
]
