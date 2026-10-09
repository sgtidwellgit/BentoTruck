from __future__ import annotations

import pytest

from bentotruck.edamame import ConversationMemory
from bentotruck.gyoza import PythonTool
from bentotruck.nigiri import ModelResponse, Mock, ToolCall
from bentotruck.rice import Agent, EventType


def test_run_requires_a_provider():
    agent = Agent("no-provider")
    with pytest.raises(RuntimeError, match="no model provider"):
        agent.run("hi")


def test_simple_run_returns_text_and_records_session():
    agent = Agent("assistant").using(Mock(reply="hello there"))
    result = agent.run("hi")

    assert result == "hello there"
    roles = [m.role for m in agent.session.history()]
    assert roles == ["user", "assistant"]


def test_run_executes_tool_call_then_returns_final_answer():
    calls = {"n": 0}

    def responder(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(
                content="",
                tool_calls=[ToolCall(id="call_1", name="python", arguments={"expression": "2 + 2"})],
            )
        # second call: the tool result should now be in the conversation
        tool_messages = [m for m in messages if m.role == "tool"]
        assert tool_messages and tool_messages[-1].content == "4"
        return ModelResponse(content="The answer is 4")

    agent = Agent("assistant").using(Mock(responder=responder)).with_tools(PythonTool())
    result = agent.run("what's 2 + 2?")

    assert result == "The answer is 4"
    assert calls["n"] == 2


def test_run_reports_error_for_unregistered_tool():
    def responder(messages, tools):
        if not any(m.role == "tool" for m in messages):
            return ModelResponse(content="", tool_calls=[ToolCall(id="c1", name="ghost", arguments={})])
        tool_message = next(m for m in messages if m.role == "tool")
        return ModelResponse(content=tool_message.content)

    agent = Agent("assistant").using(Mock(responder=responder))
    result = agent.run("do something")
    assert "no tool named" in result


def test_run_raises_when_max_steps_exceeded():
    provider = Mock(
        responder=lambda messages, tools: ModelResponse(
            content="", tool_calls=[ToolCall(id="c", name="python", arguments={"expression": "1"})]
        )
    )
    agent = Agent("loopy", max_steps=2).using(provider).with_tools(PythonTool())
    with pytest.raises(RuntimeError, match="did not converge"):
        agent.run("loop forever")


def test_with_memory_records_user_and_assistant_turns():
    memory = ConversationMemory()
    agent = Agent("assistant").using(Mock(reply="hi back")).with_memory(memory)
    agent.run("hello")

    recalled = memory.recall(k=10)
    assert [(item.role, item.content) for item in recalled] == [("user", "hello"), ("assistant", "hi back")]


def test_events_are_published_in_order():
    events = []
    agent = Agent("assistant").using(Mock(reply="done"))
    agent.events.subscribe(lambda event: events.append(event.type))

    agent.run("go")

    assert events == [
        EventType.AGENT_START,
        EventType.MESSAGE,
        EventType.MODEL_CALL,
        EventType.MESSAGE,
        EventType.AGENT_DONE,
    ]


def test_system_prompt_is_included_in_generated_messages():
    provider = Mock(reply="ok")
    agent = Agent("assistant", system_prompt="Be terse.").using(provider)
    agent.run("hi")

    sent = provider.calls[0]
    assert sent[0].role == "system"
    assert sent[0].content == "Be terse."


def test_with_tools_accepts_toolbox_unpacked():
    from bentotruck.gyoza import RESTTool, Toolbox

    box = Toolbox(PythonTool(), RESTTool("api", "d", url="https://example.com"))
    agent = Agent("assistant").using(Mock()).with_tools(*box)
    assert {t.name for t in agent.tools} == {"python", "api"}


def test_memory_recall_is_shown_to_the_model():
    from bentotruck.edamame import VectorMemory

    memory = VectorMemory()
    memory.add("user", "My favorite dish is gyoza")
    memory.add("user", "The truck is blue")
    provider = Mock(reply="gyoza!")
    agent = Agent("a").using(provider).with_memory(memory, recall=1)
    agent.run("what is my favorite dish?")

    notes = [m for m in provider.calls[0] if m.role == "system"]
    assert len(notes) == 1 and "My favorite dish is gyoza" in notes[0].content
    assert "blue" not in notes[0].content


def test_memory_recall_skips_live_session_and_can_be_disabled():
    memory = ConversationMemory()
    provider = Mock(reply="ok")
    agent = Agent("a").using(provider).with_memory(memory)
    agent.run("first")
    agent.run("second")
    assert not any(m.role == "system" for m in provider.calls[1])  # everything recalled is already in the session

    off = Mock(reply="ok")
    memory.add("user", "old fact")
    Agent("b").using(off).with_memory(memory, recall=0).run("q")
    assert not any(m.role == "system" for m in off.calls[0])


def test_model_call_event_carries_usage_and_duration():
    events = []
    agent = Agent("a").using(Mock(replies=[ModelResponse("hi", usage={"input_tokens": 3, "output_tokens": 1})]))
    agent.events.subscribe(lambda e: events.append(e) if e.type == EventType.MODEL_CALL else None)
    agent.run("x")
    assert events[0].data["usage"] == {"input_tokens": 3, "output_tokens": 1}
    assert events[0].data["duration"] >= 0 and events[0].source == "a"


def test_ask_has_no_session_side_effects_and_no_tools():
    seen = {}

    def responder(messages, tools):
        seen["tools"] = tools
        return ModelResponse(content="answer")

    agent = Agent("a").using(Mock(responder=responder)).with_tools(PythonTool())
    assert agent.ask("quick question") == "answer"
    assert seen["tools"] is None
    assert agent.session.history() == []


def test_session_snapshot_restore():
    from bentotruck.nigiri import Message
    from bentotruck.rice import Session

    session = Session()
    session.add(Message(role="user", content="keep"))
    mark = session.snapshot()
    session.add(Message(role="user", content="drop"))
    session.restore(mark)
    assert [m.content for m in session.history()] == ["keep"]


def test_errors_publish_error_event():
    events = []
    agent = Agent("a").using(Mock(responder=lambda m, t: (_ for _ in ()).throw(ValueError("api down"))))
    agent.events.subscribe(lambda e: events.append(e))
    with pytest.raises(ValueError):
        agent.run("x")
    assert events[-1].type == EventType.ERROR and events[-1].data["reason"] == "api down"


def test_invoke_handles_runnables_and_callables():
    from bentotruck.rice import invoke

    assert invoke(Agent("a").using(Mock(reply="r")), "x") == "r"
    assert invoke(str.upper, "x") == "X"
    with pytest.raises(TypeError):
        invoke(42, "x")
