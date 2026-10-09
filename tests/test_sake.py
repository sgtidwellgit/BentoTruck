from __future__ import annotations

import pytest

from bentotruck.nigiri import Mock
from bentotruck.rice import Agent
from bentotruck.sake import Broadcast, Channel, Event, Hub, Mailbox, Message, Topic


def test_mailbox_fifo():
    box = Mailbox("a")
    box.put(Message("x", "a", 1))
    box.put(Message("x", "a", 2))
    assert len(box) == 2
    assert box.receive().content == 1
    assert [m.content for m in box.drain()] == [2]
    assert box.receive() is None


def test_hub_delivers_to_mailbox_and_logs():
    hub = Hub()
    inbox = hub.register("alice")
    hub.register("bob")
    hub.tell("bob", "alice", "hi")
    assert inbox.receive().content == "hi"
    assert len(hub.log) == 1
    with pytest.raises(KeyError):
        hub.tell("bob", "nobody", "hi")


def test_actor_replies_and_request():
    hub = Hub()
    hub.register("client")
    hub.register("echo", lambda m: f"echo: {m.content}")
    assert hub.request("client", "echo", "ping") == "echo: ping"
    assert len(hub.mailboxes["client"]) == 0  # request consumes its own reply
    reply = hub.log[-1]
    assert reply.sender == "echo" and reply.recipient == "client" and reply.reply_to == hub.log[0].id
    assert hub.conversation("client", "echo") == hub.log


def test_request_requires_handler():
    hub = Hub()
    hub.register("mute")
    with pytest.raises(KeyError):
        hub.request("x", "mute", "hi")


def test_register_agent_as_actor():
    hub = Hub()
    hub.register_agent(Agent("helper").using(Mock(reply="on it")))
    assert hub.request("boss", "helper", "do the thing") == "on it"


def test_broadcast_skips_sender():
    hub = Hub()
    boxes = {name: hub.register(name) for name in ["a", "b", "c"]}
    hub.broadcast("a", "news")
    assert len(boxes["a"]) == 0 and len(boxes["b"]) == 1 and len(boxes["c"]) == 1
    assert isinstance(hub.log[0], Broadcast)


def test_channel_pubsub_and_unsubscribe():
    channel = Channel("alerts")
    got, box = [], Mailbox("m")
    off = channel.subscribe(got.append)
    channel.subscribe(box)
    assert channel.emit("low_stock", source="kitchen", item="rice") == 2
    assert isinstance(got[0], Event) and got[0].data == {"item": "rice"}
    off()
    channel.publish("x")
    assert len(got) == 1 and len(box) == 2 and len(channel.history) == 2


def test_topic_glob_routing():
    orders = Topic("orders")
    created, shipped = orders.channel("created"), orders.channel("shipped")
    seen = []
    orders.subscribe("*", seen.append)
    assert orders.publish("created", "o1") == 1
    assert orders.publish("orders.*", "all") == 2
    assert seen == ["o1", "all", "all"]
    assert created.history[-1] == "all" and shipped.history[-1] == "all"
