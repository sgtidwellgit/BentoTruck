"""rice — core agent: Agent, Session, Context, State, Events."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from bentotruck.nigiri import Message, ModelProvider, ModelResponse


class EventType(str, Enum):
    AGENT_START = "agent_start"
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_DONE = "agent_done"
    ERROR = "error"


@dataclass(frozen=True)
class Event:
    """One entry in an agent's lifecycle — published to any subscriber on the agent's EventBus."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


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
        for handler in self._subscribers:
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

    def __init__(self, name: str = "agent", *, system_prompt: str | None = None, max_steps: int = 8) -> None:
        self.name = name
        self.max_steps = max_steps
        self.context = Context(system_prompt=system_prompt)
        self.session = Session()
        self.events = EventBus()
        self._provider: ModelProvider | None = None
        self._memory: Any = None
        self._router: Any = None
        self._tools: dict[str, Any] = {}

    def using(self, provider: ModelProvider) -> "Agent":
        self._provider = provider
        return self

    def with_memory(self, memory: Any) -> "Agent":
        self._memory = memory
        return self

    def with_router(self, router: Any) -> "Agent":
        self._router = router
        return self

    def with_tools(self, *tools: Any) -> "Agent":
        for tool in tools:
            self._tools[tool.name] = tool
        return self

    def with_system_prompt(self, prompt: str) -> "Agent":
        self.context.system_prompt = prompt
        return self

    @property
    def tools(self) -> list[Any]:
        return list(self._tools.values())

    def run(self, goal: str) -> str:
        """Run the agent to completion on `goal`, executing any tool calls along the way, and return the final text."""

        if self._provider is None:
            raise RuntimeError(f"Agent {self.name!r} has no model provider — call .using(...) first.")

        self.events.publish(Event(EventType.AGENT_START, {"goal": goal}))

        if self._memory is not None:
            self._memory.add("user", goal)

        self.session.add(Message(role="user", content=goal))
        self.events.publish(Event(EventType.MESSAGE, {"role": "user", "content": goal}))

        tool_specs = [tool.to_schema() for tool in self._tools.values()] or None

        for _ in range(self.max_steps):
            response = self._provider.generate(self._build_messages(), tools=tool_specs)

            if not response.tool_calls:
                return self._finish(response)

            self.session.add(Message(role="assistant", content=response.content, tool_calls=response.tool_calls))

            for call in response.tool_calls:
                self.events.publish(Event(EventType.TOOL_CALL, {"name": call.name, "arguments": call.arguments}))
                result = self._invoke_tool(call.name, call.arguments)
                self.events.publish(Event(EventType.TOOL_RESULT, {"name": call.name, "result": result}))
                self.session.add(Message(role="tool", content=str(result), tool_call_id=call.id, name=call.name))

        self.events.publish(Event(EventType.ERROR, {"reason": "max_steps exceeded"}))
        raise RuntimeError(f"Agent {self.name!r} did not converge within {self.max_steps} steps.")

    def _finish(self, response: ModelResponse) -> str:
        self.session.add(Message(role="assistant", content=response.content))
        self.events.publish(Event(EventType.MESSAGE, {"role": "assistant", "content": response.content}))
        if self._memory is not None:
            self._memory.add("assistant", response.content)
        self.events.publish(Event(EventType.AGENT_DONE, {"result": response.content}))
        return response.content

    def _invoke_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: no tool named {name!r} is registered."
        try:
            return tool.run(**arguments)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool result, not raised
            return f"Error: {exc}"

    def _build_messages(self) -> list[Message]:
        messages: list[Message] = []
        if self.context.system_prompt:
            messages.append(Message(role="system", content=self.context.system_prompt))
        messages.extend(self.session.history())
        return messages


__all__ = ["Agent", "Session", "Context", "State", "Event", "EventType", "EventBus"]
