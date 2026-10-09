from __future__ import annotations

import pytest

from bentotruck.bento import Bento, Coordinator, Parallel, Sequential, Vote
from bentotruck.nigiri import Mock, ModelResponse
from bentotruck.rice import Agent


def agent(name, reply, **kwargs):
    return Agent(name, **kwargs).using(Mock(reply=reply))


def test_sequential_relay_passes_work_forward():
    writer_model = Mock(reply="final copy")
    team = Bento("t").add(agent("researcher", "facts"), role="research").add(
        Agent("writer").using(writer_model), role="write"
    )
    assert team.run("launch") == "final copy"
    prompt = writer_model.calls[0][-1].content
    assert "Work so far (from researcher):\nfacts" in prompt and "Your role: write" in prompt
    assert [(m.sender, m.recipient) for m in team.transcript] == [
        ("bento", "researcher"), ("researcher", "bento"), ("bento", "writer"), ("writer", "bento"),
    ]


def test_sequential_rounds():
    team = Bento("t", strategy=Sequential(rounds=2)).add(agent("a", "x")).add(agent("b", "y"))
    team.run("go")
    assert len(team.transcript) == 8


def test_parallel_with_and_without_synthesizer():
    team = Bento("t", strategy="parallel").add(agent("a", "alpha")).add(agent("b", "beta"))
    assert team.run("go") == "[a]\nalpha\n\n[b]\nbeta"

    synth_model = Mock(reply="merged")
    team2 = Bento("t2", strategy=Parallel(synthesizer=Agent("s").using(synth_model)))
    team2.add(agent("a", "alpha")).add(agent("b", "beta"))
    assert team2.run("go") == "merged"
    assert "[a]\nalpha" in synth_model.calls[0][-1].content


def test_coordinator_delegates_via_tools_and_cleans_up():
    lead_model = Mock(replies=[ModelResponse.calling("ask_chef", task="price the gyoza"), "Gyoza: $6"])
    lead = Agent("lead").using(lead_model)
    team = Bento("t", strategy=Coordinator(lead)).add(agent("chef", "$6"), role="pricing")
    assert team.run("make a menu") == "Gyoza: $6"
    tool_result = [m for m in lead_model.calls[1] if m.role == "tool"][0]
    assert tool_result.content == "$6"
    assert lead.tools == []
    assert ("lead", "chef") in [(m.sender, m.recipient) for m in team.transcript]


def test_vote_majority_and_judge():
    team = Bento("t", strategy="vote")
    for name, reply in [("a", "Paris"), ("b", "paris."), ("c", "Lyon")]:
        team.add(agent(name, reply))
    assert team.run("capital?") == "Paris"

    judged = Bento("j", strategy=Vote(judge=Mock(reply="Answer 2")))
    judged.add(agent("a", "one")).add(agent("b", "two"))
    assert judged.run("pick") == "two"


def test_nested_teams_and_agents_flattening():
    inner = Bento("inner").add(agent("x", "inner result"))
    outer = Bento("outer").add(inner).add(agent("y", "outer result"))
    assert outer.run("go") == "outer result"
    assert [a.name for a in outer.agents] == ["x", "y"]


def test_validation():
    with pytest.raises(RuntimeError):
        Bento().run("go")
    team = Bento().add(agent("a", "x"))
    with pytest.raises(ValueError):
        team.add(agent("a", "y"))
    with pytest.raises(TypeError):
        team.add(object())
    with pytest.raises(ValueError):
        Bento(strategy="chaos")


def test_role_defaults_to_description():
    team = Bento().add(agent("a", "x", description="finds facts"))
    assert team.members[0].role == "finds facts"
