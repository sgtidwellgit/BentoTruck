from __future__ import annotations

import pytest

from bentotruck.edamame import VectorMemory
from bentotruck.nigiri import Mock, ModelResponse
from bentotruck.rice import Agent, EventType
from bentotruck.tempura import CodingSkill, PlanningSkill, ResearchSkill, SearchSkill, Skill, SummarizeSkill, skill


def test_skill_runs_sub_agent_with_its_instructions():
    provider = Mock(reply="  summary  ")
    out = SummarizeSkill(max_words=50).run("long text", provider=provider)
    assert out == "summary"
    system, user = provider.calls[0][0], provider.calls[0][-1]
    assert system.role == "system" and "50 words" in system.content
    assert "long text" in user.content


def test_skill_needs_a_model():
    with pytest.raises(RuntimeError, match="needs a model"):
        Skill().run("x")


def test_learned_skill_is_a_tool_the_model_can_call():
    provider = Mock(replies=[ModelResponse.calling("summarize", input="the doc"), "- point", "Here: - point"])
    agent = Agent("a").using(provider).learn(SummarizeSkill())
    assert "summarize" in {t.name for t in agent.tools}
    assert agent.run("summarize the doc") == "Here: - point"


def test_skill_events_flow_to_parent_agent():
    sources = set()
    agent = Agent("parent").using(Mock(reply="x")).learn(SummarizeSkill())
    agent.events.subscribe(lambda e: sources.add(e.source) if e.type == EventType.AGENT_START else None)
    agent.use("summarize", "text")
    assert sources == {"parent.summarize"}


def test_use_unknown_skill_raises():
    with pytest.raises(KeyError):
        Agent("a").use("nope", "x")


def test_search_skill_over_memory_and_callable():
    memory = VectorMemory()
    memory.add("doc", "The truck opens at 11am")
    memory.add("doc", "Gyoza costs 6 dollars")
    assert SearchSkill(memory, k=1).search("when does the truck open") == ["The truck opens at 11am"]
    assert SearchSkill(lambda q: ["a", "b", "c"], k=2).search("q") == ["a", "b"]
    with pytest.raises(TypeError):
        SearchSkill(42)


def test_search_skill_answers_from_search_results():
    provider = Mock(replies=[ModelResponse.calling("search", query="hours"), "Opens at 11am."])
    out = SearchSkill(lambda q: ["Opens at 11am"]).run("When do you open?", provider=provider)
    assert out == "Opens at 11am."
    tool_msg = [m for m in provider.calls[1] if m.role == "tool"][0]
    assert "Opens at 11am" in tool_msg.content


def test_research_skill_uses_goal_planner():
    provider = Mock(replies=["1. q1\n2. q2", "a1", "a2", "synthesis [1]"])
    out = ResearchSkill(lambda q: ["fact"], questions=2).run("topic", provider=provider)
    assert out == "synthesis [1]"


def test_coding_skill_strips_fences():
    provider = Mock(reply="Sure!\n```python\nprint('hi')\n```")
    assert CodingSkill().run("say hi", provider=provider) == "print('hi')"


def test_planning_skill_numbers_steps():
    provider = Mock(reply="- buy truck\n- get permit\n- open")
    skill_ = PlanningSkill(max_steps_in_plan=2)
    assert skill_.run("start", provider=provider) == "1. buy truck\n2. get permit"
    assert skill_.plan("start", provider=provider) == ["buy truck", "get permit"]


def test_function_skill_decorator_needs_no_model():
    @skill
    def shout(text):
        """Uppercase the text."""
        return text.upper()

    assert shout.name == "shout" and shout.description == "Uppercase the text."
    agent = Agent("a").learn(shout)
    assert agent.use("shout", "hey") == "HEY"
