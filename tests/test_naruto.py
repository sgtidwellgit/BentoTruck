from __future__ import annotations

import pytest

from bentotruck.gyoza import PythonTool, SQLTool
from bentotruck.naruto import (
    CapabilityRouter,
    HybridRouter,
    IntentRouter,
    NoRouteError,
    Rule,
    RuleRouter,
    SemanticRouter,
)
from bentotruck.nigiri import Mock
from bentotruck.rice import Agent, EventType


def test_rule_router_predicate_regex_keywords_and_default():
    router = RuleRouter(
        [Rule(lambda t: t.startswith("!"), "admin", "bang"), (r"\brefund\b", "billing"), Rule(["menu", "vegan"], "menu", "food")],
        default="general",
    )
    assert router.route("!reset").name == "bang"
    assert router.route("I want a REFUND").target == "billing"
    assert router.route("anything vegan?").target == "menu"
    assert router.route("hello").target == "general"
    assert RuleRouter().when("x", "X").route("y") is None


def test_semantic_router_scores_and_threshold():
    router = SemanticRouter(
        {"billing": (["refund my order", "I was charged twice"], "B"), "menu": (["vegan options", "gluten free menu"], "M")},
        threshold=0.3,
    )
    route = router.route("can I get a refund for my order")
    assert route.name == "billing" and route.target == "B" and route.score > 0.3
    assert router.route("do you have vegan options").target == "M"
    assert router.route("weather tomorrow") is None


def test_intent_router_with_model_and_function():
    router = IntentRouter(Mock(reply="Refund."), {"refund": ("B", "wants money back"), "hours": "H"}, default="D")
    assert router.route("money back please").target == "B"
    assert IntentRouter(lambda t: "nonsense", {"refund": "B"}, default="D").route("x").target == "D"
    assert IntentRouter(lambda t: "hours", {"hours": "H"}).route("x").name == "hours"


def test_capability_router_uses_tools_and_declared_capabilities():
    calc = Agent("calc").with_tools(PythonTool())
    db = Agent("db").with_tools(SQLTool(connection=None))

    class Searcher:
        name = "searcher"
        capabilities = ["search", "browse"]

    router = CapabilityRouter([calc, db, Searcher()], aliases={"python": ["calculate", "math"]})
    assert router.route("please calculate this").target is calc
    assert router.route("run the sql query").target is db
    assert router.route("search the web").target.name == "searcher"
    assert router.route("x", required=["sql"]).target is db
    assert router.route("nothing relevant") is None


def test_hybrid_router_falls_through():
    rules = RuleRouter([("^!", "admin")], default="should-be-ignored")
    semantic = SemanticRouter({"menu": ("vegan menu options", "M")}, threshold=0.3)
    router = HybridRouter(rules, semantic, default="fallback")
    assert router.route("!x").target == "admin"
    assert router.route("vegan menu").target == "M"
    assert router.route("zzz").target == "fallback"
    assert rules.default == "should-be-ignored"


def test_router_run_dispatches_to_runnable_provider_or_callable():
    agent = Agent("menu").using(Mock(reply="we have gyoza"))
    router = RuleRouter([("menu", agent), ("model", Mock(reply="model says hi")), ("fn", str.upper)])
    assert router.run("menu please") == "we have gyoza"
    assert router.run("model?") == "model says hi"
    assert router.run("fn") == "FN"
    with pytest.raises(NoRouteError):
        router.run("nothing")


def test_agent_with_router_delegates_or_switches_model():
    specialist = Agent("specialist").using(Mock(reply="specialist answer"))
    cheap = Mock(reply="cheap model answer")
    router = RuleRouter([("hard", specialist), ("easy", cheap)])
    front = Agent("front").using(Mock(reply="front answer")).with_router(router)

    routes = []
    front.events.subscribe(lambda e: routes.append(e.data["route"]) if e.type == EventType.ROUTE else None)
    assert front.run("a hard question") == "specialist answer"
    assert front.run("an easy question") == "cheap model answer"
    assert front.run("anything else") == "front answer"
    assert front.provider is not cheap  # model switch is per-run only
    assert routes == ["rule0", "rule1"]
