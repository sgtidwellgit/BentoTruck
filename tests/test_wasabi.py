from __future__ import annotations

import io
import json
import logging

import pytest

from bentotruck.gyoza import PythonTool
from bentotruck.nigiri import Mock, ModelResponse
from bentotruck.rice import Agent
from bentotruck.tempura import SummarizeSkill
from bentotruck.wasabi import ConsoleExporter, JSONLExporter, Logger, Metrics, Profiler, Tracer


def tool_agent(name="calc"):
    replies = [
        ModelResponse.calling("python", expression="6 * 7"),
        ModelResponse(content="42", usage={"input_tokens": 5, "output_tokens": 1}),
    ]
    return Agent(name).using(Mock(replies=replies)).with_tools(PythonTool())


def test_tracer_builds_nested_spans():
    agent = tool_agent()
    tracer = Tracer().attach(agent)
    agent.run("6 * 7?")

    trace = tracer.last
    assert [(s.kind, s.name) for s in trace.spans] == [("agent", "calc"), ("model", "mock"), ("tool", "python"), ("model", "mock")]
    assert all(s.parent_id == trace.root.id for s in trace.spans[1:])
    assert trace.root.attributes["result"] == "42"
    assert trace.find(kind="tool")[0].attributes["result"] == 42
    assert "tool:python" in trace.tree()


def test_tracer_marks_errors():
    agent = Agent("loop", max_steps=1).using(Mock(replies=[ModelResponse.calling("python", expression="1")])).with_tools(PythonTool())
    tracer = Tracer().attach(agent)
    with pytest.raises(RuntimeError):
        agent.run("go")
    assert tracer.last.status == "error"


def test_tracer_nests_skill_runs():
    agent = Agent("parent").using(Mock(reply="x")).learn(SummarizeSkill())
    tracer = Tracer().attach(agent)
    agent.use("summarize", "text")
    agent.run("hi")
    # the direct skill call is its own trace; the run is another
    assert [t.root.name for t in tracer.traces] == ["parent.summarize", "parent"]


def test_exporters(tmp_path):
    stream = io.StringIO()
    path = tmp_path / "traces.jsonl"
    agent = tool_agent()
    Tracer(exporters=[ConsoleExporter(stream), JSONLExporter(path)]).attach(agent)
    agent.run("go")
    assert "agent:calc" in stream.getvalue()
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert len(record["spans"]) == 4 and record["status"] == "ok"


def test_detach_stops_tracing():
    agent = tool_agent()
    tracer = Tracer().attach(agent)
    tracer.detach()
    agent.run("go")
    assert tracer.traces == []


def test_logger(caplog):
    agent = tool_agent()
    Logger().attach(agent)
    with caplog.at_level(logging.INFO, logger="bentotruck"):
        agent.run("go")
    messages = [r.getMessage() for r in caplog.records]
    assert any("calc tool_call name=python" in m for m in messages)
    assert caplog.records[0].bentotruck_event["type"] == "agent_start"


def test_metrics():
    agent = tool_agent()
    metrics = Metrics().attach(agent)
    agent.run("go")
    snap = metrics.snapshot()
    assert snap["runs"] == 1 and snap["model_calls"] == 2 and snap["tool_calls"] == 1
    assert snap["tokens"] == {"input_tokens": 5, "output_tokens": 1}
    assert snap["tool_usage"] == {"python": 1}
    assert snap["latency"]["model"]["count"] == 2
    metrics.reset()
    assert metrics.snapshot()["runs"] == 0


def test_profiler():
    agent = tool_agent()
    tracer = Tracer().attach(agent)
    agent.run("go")
    agent.run("go")
    profile = Profiler.profile(tracer)
    # the scripted Mock repeats its last reply, so run 2 answers directly
    assert profile.by_name["model:mock"][0] == 3
    assert profile.by_name["tool:python"][0] == 1
    assert profile.tokens == {"input_tokens": 10, "output_tokens": 2}
    assert "tool:python" in profile.table()


def test_opentelemetry_exporter():
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    export = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from bentotruck.wasabi import OpenTelemetryExporter

    memory = export.InMemorySpanExporter()
    provider = sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    agent = tool_agent()
    Tracer(exporters=[OpenTelemetryExporter(provider.get_tracer("test"))]).attach(agent)
    agent.run("go")
    names = sorted(s.name for s in memory.get_finished_spans())
    assert names == ["agent:calc", "model:mock", "model:mock", "tool:python"]
