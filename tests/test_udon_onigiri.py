from __future__ import annotations

import time

import pytest

from bentotruck import onigiri, udon
from bentotruck.nigiri import Mock
from bentotruck.rice import Agent


# --- udon ---------------------------------------------------------------------


def test_pipeline_runs_stages_in_order_with_trace():
    pipe = udon.Pipeline(str.strip, str.upper, udon.Step(lambda s: s + "!", name="bang"))
    result = pipe.execute("  hi ")
    assert result.output == "HI!"
    assert [name for name, _ in result.trace] == ["strip", "upper", "bang"]
    assert pipe("  yo ") == "YO!"


def test_pipe_operator_and_agents_as_stages():
    pipe = udon.Step(str.strip) | Agent("echo").using(Mock(reply="agent says hi"))
    assert isinstance(pipe, udon.Pipeline) and len(pipe) == 2
    assert pipe.run("  x ") == "agent says hi"
    assert len(udon.Pipeline(str.strip) | udon.Pipeline(str.upper, str.lower)) == 3


def test_condition_and_loop():
    double_if_small = udon.Condition(lambda n: n < 10, lambda n: n * 2)
    assert udon.Pipeline(double_if_small).run(3) == 6
    assert udon.Pipeline(double_if_small).run(30) == 30
    assert udon.Loop(lambda n: n + 1, times=3).run(0) == 3
    assert udon.Loop(lambda n: n * 2, until=lambda n: n > 20).run(1) == 32
    with pytest.raises(ValueError):
        udon.Loop(lambda n: n)


def test_parallel_runs_concurrently_and_merges():
    def slow(tag):
        def inner(x):
            time.sleep(0.2)
            return f"{tag}{x}"
        return inner

    started = time.perf_counter()
    out = udon.Parallel(slow("a"), slow("b"), slow("c"), merge=",".join).run(1)
    assert out == "a1,b1,c1"
    assert time.perf_counter() - started < 0.5
    assert udon.Parallel(str.upper, str.lower).run("Hi") == ["HI", "hi"]


# --- onigiri ------------------------------------------------------------------


def test_workflow_linear_chain():
    flow = onigiri.Workflow("w").add(
        onigiri.Step("strip", str.strip), onigiri.Step("upper", str.upper)
    ).chain("strip", "upper")
    result = flow.execute("  hi ")
    assert result.output == "HI" and result.path == ["strip", "upper"]
    assert result.results["strip"] == "hi"


def test_workflow_branches_and_loops_back():
    drafts = iter(["draft REVISE", "draft v2 REVISE", "draft v3"])
    flow = onigiri.Workflow("publish")
    flow.add(
        onigiri.Step("draft", lambda _: next(drafts)),
        onigiri.Step("ship", lambda d: f"shipped: {d}"),
    )
    flow.connect("draft", "draft", when=onigiri.Condition.contains("REVISE"))
    flow.connect("draft", "ship")
    result = flow.execute("brief")
    assert result.output == "shipped: draft v3"
    assert result.path == ["draft", "draft", "draft", "ship"]


def test_step_input_reads_state():
    flow = onigiri.Workflow().add(
        onigiri.Step("a", lambda x: x * 2),
        onigiri.Step("b", lambda pair: pair[0] + pair[1], input=lambda s: (s.input, s["a"])),
    ).chain("a", "b")
    assert flow.run(5) == 15


def test_condition_combinators_and_visited():
    yes = onigiri.Condition(lambda s: True, "yes")
    no = onigiri.Condition(lambda s: False, "no")
    state = onigiri.State(input=None, visits={"x": 2})
    assert (yes & ~no)(state) and (no | yes)(state) and not (yes & no)(state)
    assert onigiri.Condition.visited("x", 2)(state)


def test_loop_step_and_end_edge():
    flow = onigiri.Workflow().add(
        onigiri.Loop("grow", lambda n: n * 2, until=lambda s: s.last > 50, max_iterations=10),
        onigiri.Step("never", lambda n: -1),
    )
    flow.connect("grow", onigiri.END)
    flow.connect("grow", "never", when=lambda s: False)
    assert flow.run(1) == 64
    capped = onigiri.Workflow().add(onigiri.Loop("grow", lambda n: n * 2, until=lambda s: False, max_iterations=3))
    assert capped.run(1) == 8


def test_workflow_guards():
    flow = onigiri.Workflow(max_steps=3).add(onigiri.Step("spin", lambda x: x))
    flow.connect("spin", "spin")
    with pytest.raises(RuntimeError, match="max_steps"):
        flow.run(1)
    with pytest.raises(KeyError):
        flow.connect("spin", "missing")
    with pytest.raises(ValueError):
        flow.add(onigiri.Step("spin", str))
    with pytest.raises(RuntimeError):
        onigiri.Workflow().run(1)


def test_workflow_step_can_be_an_agent():
    flow = onigiri.Workflow().add(onigiri.Step("ask", Agent("a").using(Mock(reply="answer"))))
    assert flow.run("question") == "answer"
