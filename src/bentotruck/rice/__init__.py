"""rice — core agent: Agent, Session, Context, State, Events."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from bentotruck.nigiri import Message, ModelProvider, ModelResponse


@runtime_checkable
class Runnable(Protocol):
    """
    Anything with `run(input)`. Agents, teams (`bento.Bento`), workflows,
    pipelines, routers, and reflection wrappers all satisfy it — which is
    what lets every compartment nest inside every other.
    """

    def run(self, goal: Any) -> Any: ...


def invoke(target: Any, value: Any) -> Any:
    """Run `target` on `value`: call `.run(value)` if it's Runnable, otherwise call it directly."""

    if isinstance(target, Runnable):
        return target.run(value)
    if callable(target):
        return target(value)
    raise TypeError(f"{target!r} is neither Runnable (has .run) nor callable.")


class EventType(str, Enum):
    AGENT_START = "agent_start"
    MESSAGE = "message"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STEP = "step"
    ROUTE = "route"
    GUARD = "guard"
    AGENT_DONE = "agent_done"
    ERROR = "error"


@dataclass(frozen=True)
class Event:
    """One entry in an agent's lifecycle — published to any subscriber on the agent's EventBus."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str | None = None


class EventBus:
    """Publish/subscribe hub for agent lifecycle events."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[Event], None]] = []

    def subscribe(self, handler: Callable[[Event], None]) -> None:
        self._subscribers.append(handler)

    def unsubscribe(self, handler: Callable[[Event], None]) -> None:
        if handler in self._subscribers:
            self._subscribers.remove(handler)

    def publish(self, event: Event) -> None:
        for handler in list(self._subscribers):
            handler(event)


@dataclass
class State:
    """Free-form key/value scratch state carried across an agent run."""

    values: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def update(self, **kwargs: Any) -> None:
        self.values.update(kwargs)


@dataclass
class Context:
    """Everything the agent knows about the current run: system prompt, state, metadata."""

    system_prompt: str | None = None
    state: State = field(default_factory=State)
    metadata: dict[str, Any] = field(default_factory=dict)


class Session:
    """Ongoing conversation history for one agent run (or one user thread)."""

    def __init__(self, session_id: str | None = None) -> None:
        self.id = session_id or str(uuid.uuid4())
        self.messages: list[Message] = []

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def history(self) -> list[Message]:
        return list(self.messages)

    def snapshot(self) -> int:
        """Mark the current point in history so it can be rolled back to with `restore()`."""

        return len(self.messages)

    def restore(self, mark: int) -> None:
        del self.messages[mark:]

    def clear(self) -> None:
        self.messages.clear()


class Agent:
    """
    Fluent, vendor-neutral agent: pick a model, bolt on memory/tools/router, run it.

    Example
    -------
    >>> from bentotruck import nigiri, gyoza
    >>> agent = Agent("assistant").using(nigiri.Mock(reply="4")).with_tools(gyoza.PythonTool())
    >>> agent.run("What's 2 + 2?")
    '4'
    """

    def __init__(
        self,
        name: str = "agent",
        *,
        system_prompt: Any = None,
        description: str = "",
        max_steps: int = 8,
    ) -> None:
        self.name = name
        self.description = description
        self.max_steps = max_steps
        self.context = Context()
        self.session = Session()
        self.events = EventBus()
        self.skills: dict[str, Any] = {}
        self._provider: ModelProvider | None = None
        self._memory: Any = None
        self._recall_k = 0
        self._router: Any = None
        self._planner: Any = None
        self._tools: dict[str, Any] = {}
        self._input_guards: list[Any] = []
        self._output_guards: list[Any] = []
        self._tool_guards: list[Any] = []
        self._recalled: list[Any] = []
        if system_prompt is not None:
            self.with_system_prompt(system_prompt)

    # --- configuration ------------------------------------------------------

    def using(self, provider: ModelProvider) -> "Agent":
        self._provider = provider
        return self

    @property
    def provider(self) -> ModelProvider | None:
        return self._provider

    def with_memory(self, memory: Any, *, recall: int = 5) -> "Agent":
        """
        Attach a memory. Every run stores the goal and final answer in it, and the
        `recall` most relevant prior items are shown to the model as context
        (anything already in the live session is skipped). `recall=0` disables that.
        """

        self._memory = memory
        self._recall_k = recall
        return self

    @property
    def memory(self) -> Any:
        return self._memory

    def with_router(self, router: Any) -> "Agent":
        """
        Attach a router (anything with `route(text)` returning an object with a
        `.target`, or None). If the chosen target is a ModelProvider, this run uses
        it; if it's Runnable (another agent, team, workflow...), the run is
        delegated to it entirely.
        """

        self._router = router
        return self

    def with_planner(self, planner: Any) -> "Agent":
        """Swap the planning strategy (any `teriyaki.Planner`). Default is a ReAct tool-calling loop."""

        self._planner = planner
        return self

    def with_tools(self, *tools: Any) -> "Agent":
        for tool in tools:
            self._tools[tool.name] = tool
        return self

    def learn(self, *skills: Any) -> "Agent":
        """Teach the agent `tempura` skills — each becomes a tool the model can call by name."""

        for skill in skills:
            self.skills[skill.name] = skill
            self._tools[skill.name] = skill.as_tool(self)
        return self

    def use(self, skill_name: str, input: Any) -> str:
        """Invoke a learned skill directly, without going through the model."""

        if skill_name not in self.skills:
            raise KeyError(f"Agent {self.name!r} has not learned a skill named {skill_name!r}.")
        return self.skills[skill_name].run(input, agent=self)

    def with_guards(self, *, input: Any = (), output: Any = (), tools: Any = ()) -> "Agent":
        """
        Attach `katsu` guards (anything with `enforce(text) -> text`). Input guards
        see the goal, output guards the final answer, tool guards every tool result.
        A blocking guard raises on input/output; on a tool result the blocked
        content is replaced with an error the model can see.
        """

        self._input_guards.extend(input)
        self._output_guards.extend(output)
        self._tool_guards.extend(tools)
        return self

    def with_system_prompt(self, prompt: Any) -> "Agent":
        """Set the system prompt from a string, or anything with `render()` (e.g. a `dango.SystemPrompt`)."""

        self.context.system_prompt = prompt.render() if hasattr(prompt, "render") else prompt
        return self

    @property
    def tools(self) -> list[Any]:
        return list(self._tools.values())

    # --- running ------------------------------------------------------------

    def run(self, goal: str) -> str:
        """Run the agent to completion on `goal`, executing any tool calls along the way, and return the final text."""

        goal = str(goal)
        if self._router is not None:
            delegated = self._dispatch_route(goal)
            if delegated is not None:
                return delegated[0]

        if self._provider is None:
            raise RuntimeError(f"Agent {self.name!r} has no model provider — call .using(...) first.")

        self._publish(EventType.AGENT_START, goal=goal)
        try:
            for guard in self._input_guards:
                goal = self._enforce(guard, goal, "input")

            self._recalled = self._recall(goal)
            if self._memory is not None:
                self._memory.add("user", goal)

            if self._planner is None:
                from bentotruck.teriyaki import ReActPlanner

                self._planner = ReActPlanner()
            result = self._planner.execute(self, goal)

            for guard in self._output_guards:
                result = self._enforce(guard, result, "output")
        except Exception as exc:
            self._publish(EventType.ERROR, reason=str(exc), error=type(exc).__name__)
            raise
        finally:
            self._recalled = []

        if self._memory is not None:
            self._memory.add("assistant", result)
        self._publish(EventType.AGENT_DONE, result=result)
        return result

    def react(self, prompt: str | None = None, *, max_steps: int | None = None, tools: bool = True) -> str:
        """
        The core tool-calling loop: add `prompt` as a user turn (if given), then call
        the model, run any tools it asks for, and repeat until it answers in plain
        text. Planners build on this; most callers want `run()` instead.
        """

        if prompt is not None:
            self.session.add(Message(role="user", content=prompt))
            self._publish(EventType.MESSAGE, role="user", content=prompt)

        steps = max_steps or self.max_steps
        for _ in range(steps):
            response = self.generate(tools=tools)
            if not response.tool_calls:
                self.session.add(Message(role="assistant", content=response.content))
                self._publish(EventType.MESSAGE, role="assistant", content=response.content)
                return response.content

            self.session.add(Message(role="assistant", content=response.content, tool_calls=response.tool_calls))
            for call in response.tool_calls:
                self._publish(EventType.TOOL_CALL, name=call.name, arguments=call.arguments, id=call.id)
                started = time.perf_counter()
                result = self._invoke_tool(call.name, call.arguments)
                self._publish(
                    EventType.TOOL_RESULT,
                    name=call.name,
                    result=result,
                    id=call.id,
                    duration=time.perf_counter() - started,
                )
                self.session.add(Message(role="tool", content=str(result), tool_call_id=call.id, name=call.name))

        raise RuntimeError(f"Agent {self.name!r} did not converge within {steps} steps.")

    def generate(self, *, tools: bool = True, extra: list[Message] | None = None) -> ModelResponse:
        """Make one model call over the current session (plus optional `extra` messages), publishing a MODEL_CALL event."""

        if self._provider is None:
            raise RuntimeError(f"Agent {self.name!r} has no model provider — call .using(...) first.")
        tool_specs = [tool.to_schema() for tool in self._tools.values()] if tools and self._tools else None
        started = time.perf_counter()
        response = self._provider.generate(self._build_messages() + list(extra or []), tools=tool_specs)
        self._publish(
            EventType.MODEL_CALL,
            model=response.model,
            usage=dict(response.usage),
            duration=time.perf_counter() - started,
            tool_calls=len(response.tool_calls),
        )
        return response

    def ask(self, prompt: str) -> str:
        """One model call with no tools and no session side effects — for internal questions (planning, critique)."""

        return self.generate(tools=False, extra=[Message(role="user", content=prompt)]).content

    # --- internals ----------------------------------------------------------

    def _publish(self, event_type: EventType, **data: Any) -> None:
        self.events.publish(Event(event_type, data, source=self.name))

    def _dispatch_route(self, goal: str) -> tuple[str] | None:
        route = self._router.route(goal)
        if route is None:
            return None
        target = getattr(route, "target", route)
        self._publish(EventType.ROUTE, route=getattr(route, "name", None), score=getattr(route, "score", None))
        if target is self or target is None:
            return None
        if isinstance(target, ModelProvider):
            previous, self._provider = self._provider, target
            router, self._router = self._router, None
            try:
                return (self.run(goal),)
            finally:
                self._provider, self._router = previous, router
        return (str(invoke(target, goal)),)

    def _enforce(self, guard: Any, text: str, stage: str) -> str:
        try:
            return guard.enforce(text)
        except Exception as exc:
            self._publish(EventType.GUARD, stage=stage, guard=type(guard).__name__, reason=str(exc))
            raise

    def _recall(self, goal: str) -> list[Any]:
        if self._memory is None or self._recall_k <= 0:
            return []
        live = {(m.role, m.content) for m in self.session.messages}
        return [item for item in self._memory.recall(goal, k=self._recall_k) if (item.role, item.content) not in live]

    def _invoke_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: no tool named {name!r} is registered."
        try:
            result = tool.run(**arguments)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool result, not raised
            return f"Error: {exc}"
        for guard in self._tool_guards:
            try:
                result = self._enforce(guard, str(result), "tool")
            except Exception as exc:  # noqa: BLE001 - a blocked tool result is shown to the model, not raised
                return f"Error: tool output blocked by guard — {exc}"
        return result

    def _build_messages(self) -> list[Message]:
        messages: list[Message] = []
        if self.context.system_prompt:
            messages.append(Message(role="system", content=self.context.system_prompt))
        if self._recalled:
            notes = "\n".join(f"- ({item.role}) {item.content}" for item in self._recalled)
            messages.append(Message(role="system", content=f"Relevant memory:\n{notes}"))
        messages.extend(self.session.history())
        return messages

    def __repr__(self) -> str:
        return f"Agent({self.name!r})"


__all__ = [
    "Agent",
    "Session",
    "Context",
    "State",
    "Event",
    "EventType",
    "EventBus",
    "Runnable",
    "invoke",
]
