"""
sake — communication: actor-model-inspired messaging between agents.

- `Message` is an addressed payload; `Broadcast` goes to everyone; `Event` is
  fire-and-forget (no reply expected).
- A `Hub` is the post office: it registers addresses, each with a `Mailbox`
  (a queue you poll) and/or a handler (an actor that reacts on delivery and
  may reply). `hub.request(...)` sends and waits for the reply synchronously.
- `Channel` is a named pub/sub route; `Topic` groups channels hierarchically.

Note: distinct from `nigiri.Message`, which is one turn of an LLM
conversation, not inter-agent communication.
"""

from __future__ import annotations

import fnmatch
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

BROADCAST = "*"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Message:
    """An addressed payload from one agent/component to another."""

    sender: str
    recipient: str
    content: Any
    topic: str | None = None
    reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=_now)

    def reply(self, content: Any, **metadata: Any) -> "Message":
        """Build the reply to this message, addressed back to its sender."""

        return Message(sender=self.recipient, recipient=self.sender, content=content, topic=self.topic,
                       reply_to=self.id, metadata=metadata)


class Broadcast(Message):
    """A Message with no single addressee — delivered to every registered address except the sender."""

    def __init__(self, sender: str, content: Any, **kwargs: Any) -> None:
        super().__init__(sender=sender, recipient=BROADCAST, content=content, **kwargs)


@dataclass
class Event:
    """A fire-and-forget notification, distinct from a Message expecting a reply."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    timestamp: datetime = field(default_factory=_now)


class Mailbox:
    """A thread-safe FIFO queue of Messages for one address."""

    def __init__(self, address: str) -> None:
        self.address = address
        self._queue: deque[Message] = deque()
        self._lock = threading.Lock()

    def put(self, message: Message) -> None:
        with self._lock:
            self._queue.append(message)

    def receive(self) -> Message | None:
        """Pop the oldest message, or None if the mailbox is empty."""

        with self._lock:
            return self._queue.popleft() if self._queue else None

    def drain(self) -> list[Message]:
        with self._lock:
            messages = list(self._queue)
            self._queue.clear()
        return messages

    def __len__(self) -> int:
        return len(self._queue)


Handler = Callable[[Message], Any]


class Hub:
    """
    The post office for a set of agents. `register(address, handler=None)` gives
    each address a Mailbox; with a handler it's also an actor — the handler runs
    on delivery, and a non-None return value is sent back as a reply.
    Every delivered message is recorded in `.log`.
    """

    def __init__(self) -> None:
        self.mailboxes: dict[str, Mailbox] = {}
        self.handlers: dict[str, Handler] = {}
        self.log: list[Message] = []
        self._lock = threading.Lock()

    def register(self, address: str, handler: Handler | None = None) -> Mailbox:
        mailbox = self.mailboxes.setdefault(address, Mailbox(address))
        if handler is not None:
            self.handlers[address] = handler
        return mailbox

    def register_agent(self, agent: Any, address: str | None = None) -> Mailbox:
        """Register an agent as an actor: incoming message content is run through it, and the answer is the reply."""

        return self.register(address or agent.name, lambda message: agent.run(str(message.content)))

    def unregister(self, address: str) -> None:
        self.mailboxes.pop(address, None)
        self.handlers.pop(address, None)

    def send(self, message: Message) -> list[Message]:
        """Deliver a message (or broadcast). Returns any replies the receiving actors produced."""

        if message.recipient == BROADCAST:
            targets = [a for a in self.mailboxes if a != message.sender]
        elif message.recipient in self.mailboxes:
            targets = [message.recipient]
        else:
            raise KeyError(f"No address {message.recipient!r} is registered on this hub.")

        with self._lock:
            self.log.append(message)
        replies = []
        for address in targets:
            self.mailboxes[address].put(message)
            handler = self.handlers.get(address)
            if handler is None:
                continue
            outcome = handler(message)
            if outcome is None:
                continue
            reply = outcome if isinstance(outcome, Message) else message.reply(outcome)
            reply.sender = address
            replies.append(reply)
            with self._lock:
                self.log.append(reply)
            if reply.recipient in self.mailboxes:
                self.mailboxes[reply.recipient].put(reply)
        return replies

    def tell(self, sender: str, recipient: str, content: Any, **kwargs: Any) -> list[Message]:
        """Shorthand for `send(Message(sender, recipient, content))`."""

        return self.send(Message(sender=sender, recipient=recipient, content=content, **kwargs))

    def request(self, sender: str, recipient: str, content: Any, **kwargs: Any) -> Any:
        """Send to an actor and return its reply's content (synchronous request/response)."""

        if recipient not in self.handlers:
            raise KeyError(f"{recipient!r} has no handler, so it can't reply.")
        replies = self.tell(sender, recipient, content, **kwargs)
        if not replies:
            raise RuntimeError(f"{recipient!r} did not reply.")
        if sender in self.mailboxes:
            self._discard(self.mailboxes[sender], replies[0])
        return replies[0].content

    def broadcast(self, sender: str, content: Any, **kwargs: Any) -> list[Message]:
        return self.send(Broadcast(sender, content, **kwargs))

    def conversation(self, a: str, b: str) -> list[Message]:
        """Every logged message between two addresses, in order."""

        return [m for m in self.log if {m.sender, m.recipient} == {a, b}]

    @staticmethod
    def _discard(mailbox: Mailbox, message: Message) -> None:
        with mailbox._lock:
            try:
                mailbox._queue.remove(message)
            except ValueError:
                pass


class Channel:
    """A named delivery route: publishers push, every subscriber receives, history is kept."""

    def __init__(self, name: str, *, history: int = 1000) -> None:
        self.name = name
        self.subscribers: list[Callable[[Any], Any]] = []
        self.history: deque[Any] = deque(maxlen=history)

    def subscribe(self, subscriber: Callable[[Any], Any] | Mailbox) -> Callable[[], None]:
        """Add a callback (or a Mailbox). Returns a function that unsubscribes it."""

        callback = subscriber.put if isinstance(subscriber, Mailbox) else subscriber
        self.subscribers.append(callback)
        return lambda: self.subscribers.remove(callback) if callback in self.subscribers else None

    def publish(self, item: Message | Event | Any) -> int:
        """Deliver to every subscriber; returns how many received it."""

        self.history.append(item)
        for subscriber in list(self.subscribers):
            subscriber(item)
        return len(self.subscribers)

    def emit(self, name: str, source: str | None = None, **data: Any) -> int:
        """Publish a fire-and-forget Event."""

        return self.publish(Event(name=name, data=data, source=source))

    def __repr__(self) -> str:
        return f"Channel({self.name!r}, subscribers={len(self.subscribers)})"


class Topic:
    """
    A logical grouping of Channels for pub/sub routing. Channels are named
    `topic.sub` (e.g. "orders.created"); publishing accepts glob patterns, so
    `publish("orders.*", ...)` reaches every orders channel.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.channels: dict[str, Channel] = {}

    def channel(self, sub: str) -> Channel:
        """Get (creating if needed) the channel `<topic>.<sub>`."""

        full = f"{self.name}.{sub}"
        return self.channels.setdefault(full, Channel(full))

    def subscribe(self, pattern: str, subscriber: Callable[[Any], Any] | Mailbox) -> None:
        """Subscribe to every existing channel matching `pattern` (relative or full name)."""

        for channel in self.match(pattern):
            channel.subscribe(subscriber)

    def match(self, pattern: str) -> list[Channel]:
        full = pattern if pattern.startswith(self.name + ".") else f"{self.name}.{pattern}"
        return [c for name, c in self.channels.items() if fnmatch.fnmatchcase(name, full)]

    def publish(self, pattern: str, item: Any) -> int:
        """Publish to every channel matching `pattern`; returns total deliveries."""

        return sum(channel.publish(item) for channel in self.match(pattern))


__all__ = ["Message", "Broadcast", "Event", "Mailbox", "Hub", "Channel", "Topic", "BROADCAST"]
