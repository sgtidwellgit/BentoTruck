from __future__ import annotations

import pytest

from bentotruck.dango import FewShot, PromptRepository, SystemPrompt, Template, Var, Variables
from bentotruck.nigiri import Mock
from bentotruck.rice import Agent


def test_template_render_slots_and_errors():
    t = Template("Hi {name}, {name}! Order {order[id]}. Literal {{braces}}.", name="greet")
    assert t.slots == ["name", "order"]
    assert t.render(name="Ada", order={"id": 7}) == "Hi Ada, Ada! Order 7. Literal {braces}."
    with pytest.raises(KeyError, match="name"):
        t.render(order={})


def test_partial_and_to_message():
    t = Template("{greeting}, {name}").partial(greeting="Hello")
    assert t.render(name="Bo") == "Hello, Bo"
    message = t.to_message(name="Bo")
    assert message.role == "user" and message.content == "Hello, Bo"


def test_variables_coerce_default_and_choices():
    v = Variables(topic=str, n=Var(int, default=3), tone=Var(str, choices=["formal", "casual"]))
    assert v.validate({"topic": "rice", "n": "5", "tone": "casual"}) == {"topic": "rice", "n": 5, "tone": "casual"}
    assert v.validate({"topic": "rice", "tone": "formal"})["n"] == 3
    with pytest.raises(ValueError) as info:
        v.validate({"n": "many", "tone": "rude"})
    message = str(info.value)
    assert "'topic' is required" in message and "'n' must be int" in message and "'tone' must be one of" in message

    t = Template("List {n} facts about {topic}.", variables=Variables(topic=str, n=Var(int, default=3)))
    assert t.render(topic="miso") == "List 3 facts about miso."


def test_system_prompt_builds_sections_and_plugs_into_agent():
    sp = SystemPrompt(
        "You are {persona}.",
        instructions=["Answer menu questions."],
        constraints=["Never invent prices."],
        output_format="One sentence.",
        defaults={"persona": "a food truck assistant"},
    )
    text = sp.render()
    assert text.startswith("You are a food truck assistant.")
    assert "Instructions:\n- Answer menu questions." in text and "Rules:\n- Never invent prices." in text
    assert sp.to_message().role == "system"

    provider = Mock()
    Agent("a", system_prompt=sp).using(provider).run("hi")
    assert provider.calls[0][0].content == text


def test_few_shot():
    fs = FewShot(
        "Input: {input}\nOutput:",
        examples=[{"input": "2+2", "output": "4"}, {"input": "3+3", "output": "6"}],
        prefix="Do arithmetic.",
    )
    assert fs.render(input="5+5") == "Do arithmetic.\n\nInput: 2+2\nOutput: 4\n\nInput: 3+3\nOutput: 6\n\nInput: 5+5\nOutput:"


def test_repository_versions_persistence_and_ab(tmp_path):
    path = tmp_path / "prompts.json"
    repo = PromptRepository(path)
    repo.add("welcome", "Hi {name}")
    repo.add("welcome", "Hello there, {name}")
    repo.register(Template("Yo {name}", name="welcome", version="casual"))
    assert repo.versions("welcome") == ["1", "2", "casual"]
    assert repo.get("welcome").version == "casual"
    assert repo.render("welcome", "1", name="Ada") == "Hi Ada"
    with pytest.raises(KeyError):
        repo.get("welcome", "9")
    with pytest.raises(ValueError):
        repo.register(Template("unnamed"))

    reloaded = PromptRepository(path)
    assert reloaded.names() == ["welcome"] and reloaded.versions("welcome") == ["1", "2", "casual"]

    picks = {repo.ab("welcome", f"user{i}", versions=["1", "2"]).version for i in range(50)}
    assert picks == {"1", "2"}
    assert repo.ab("welcome", "user7").version == repo.ab("welcome", "user7").version
    assert repo.ab("welcome", "anyone", versions=["1", "2"], weights=[0, 1]).version == "2"
