from __future__ import annotations

import pytest

from bentotruck.gyoza import PythonTool
from bentotruck.nigiri import Mock, ModelResponse
from bentotruck.rice import Agent, EventType
from bentotruck.teriyaki import GoalPlanner, GraphPlanner, ReActPlanner, StepPlanner, Task, TreePlanner


def last_user(messages):
    return [m for m in messages if m.role == "user"][-1].content


def test_default_planner_is_react_and_runs_tools():
    provider = Mock(replies=[ModelResponse.calling("python", expression="2 + 3"), "5"])
    agent = Agent("a").using(provider).with_tools(PythonTool())
    assert agent.run("2 + 3?") == "5"
    assert isinstance(agent._planner, ReActPlanner)


def test_step_planner_runs_each_step_in_order():
    seen = []

    def responder(messages, tools):
        seen.append(last_user(messages))
        return ModelResponse(content=f"done {len(seen)}")

    agent = Agent("a").using(Mock(responder=responder)).with_planner(StepPlanner(["gather", "write"]))
    assert agent.run("launch") == "done 2"
    assert "Step 1 of 2: gather" in seen[0] and "Step 2 of 2: write" in seen[1]
    assert "Overall goal: launch" in seen[1]


def test_step_planner_publishes_step_events():
    steps = []
    agent = Agent("a").using(Mock(reply="ok")).with_planner(StepPlanner(["x", "y"]))
    agent.events.subscribe(lambda e: steps.append(e.data["step"]) if e.type == EventType.STEP else None)
    agent.run("go")
    assert steps == ["x", "y"]


def test_step_planner_requires_steps():
    with pytest.raises(ValueError):
        StepPlanner([])


def test_goal_planner_decomposes_with_model_then_synthesizes():
    provider = Mock(replies=["1. find menu\n2. price it", "menu found", "priced", "final plan"])
    agent = Agent("a").using(provider).with_planner(GoalPlanner())
    assert agent.run("open a truck") == "final plan"
    assert "Subgoal 1 of 2" in last_user(provider.calls[1])
    assert "final answer" in last_user(provider.calls[3])


def test_goal_planner_custom_decompose_skips_model_decomposition():
    provider = Mock(replies=["a", "b", "done"])
    agent = Agent("a").using(provider).with_planner(GoalPlanner(decompose=lambda g: ["one", "two"]))
    assert agent.run("goal") == "done"
    assert len(provider.calls) == 3


def test_tree_planner_uses_scorer_and_discards_losing_branches():
    provider = Mock(replies=["short", "a much longer answer", "mid"])
    agent = Agent("a").using(provider).with_planner(TreePlanner(3, scorer=lambda goal, c: len(c)))
    assert agent.run("goal") == "a much longer answer"
    assert [m.content for m in agent.session.history()] == ["goal", "a much longer answer"]


def test_tree_planner_asks_model_to_choose_without_scorer():
    provider = Mock(replies=["first", "second", "Candidate 2"])
    agent = Agent("a").using(provider).with_planner(TreePlanner(2))
    assert agent.run("goal") == "second"


def test_graph_planner_runs_in_dependency_order_with_inputs():
    prompts = []

    def responder(messages, tools):
        prompts.append(last_user(messages))
        return ModelResponse(content=f"result{len(prompts)}")

    planner = GraphPlanner(
        {"memo": Task("write memo", depends_on=["facts", "risks"]), "facts": "collect facts", "risks": Task("risks", ["facts"])}
    )
    agent = Agent("a").using(Mock(responder=responder)).with_planner(planner)
    assert agent.run("goal") == "result3"
    assert planner.order == ["facts", "risks", "memo"]
    assert "[facts]\nresult1" in prompts[2] and "[risks]\nresult2" in prompts[2]


def test_graph_planner_validates_dependencies():
    with pytest.raises(ValueError, match="unknown"):
        GraphPlanner({"a": Task("x", depends_on=["missing"])})
