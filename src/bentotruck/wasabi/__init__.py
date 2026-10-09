"""
wasabi — observability: tracing, logging, and metrics for agent runs.

Everything here attaches to an agent's `rice.EventBus` — no agent code
changes needed:

    tracer = wasabi.Tracer(exporters=[wasabi.ConsoleExporter()]).attach(agent)
    agent.run("...")
    print(tracer.last.tree())

`OpenTelemetryExporter` forwards traces to any OTLP backend (Jaeger, Grafana
Tempo, Honeycomb, Arize Phoenix, LangSmith, MLflow, ...) when the optional
`opentelemetry-api` / `opentelemetry-sdk` packages are installed.
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

from bentotruck.rice import Event, EventType


def _id() -> str:
    return uuid.uuid4().hex[:16]


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


@dataclass
class Span:
    """A single timed unit of work: an agent run, a model call, a tool call, or a planner step."""

    name: str
    kind: str
    trace_id: str
    start: datetime
    end: datetime | None = None
    parent_id: str | None = None
    id: str = field(default_factory=_id)
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    @property
    def duration(self) -> float:
        return (self.end - self.start).total_seconds() if self.end else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start": self.start.isoformat(),
            "end": self.end.isoformat() if self.end else None,
            "duration": self.duration,
            "status": self.status,
            "attributes": {k: _jsonable(v) for k, v in self.attributes.items()},
        }


@dataclass
class Trace:
    """The full record of one top-level agent run, as a tree of Spans."""

    id: str
    spans: list[Span] = field(default_factory=list)

    @property
    def root(self) -> Span:
        return self.spans[0]

    @property
    def duration(self) -> float:
        return self.root.duration

    @property
    def status(self) -> str:
        return self.root.status

    def children(self, span: Span) -> list[Span]:
        return [s for s in self.spans if s.parent_id == span.id]

    def find(self, kind: str | None = None, name: str | None = None) -> list[Span]:
        return [s for s in self.spans if (kind is None or s.kind == kind) and (name is None or s.name == name)]

    def tree(self) -> str:
        """Human-readable indented view of the trace."""

        lines: list[str] = []

        def walk(span: Span, depth: int) -> None:
            flag = "" if span.status == "ok" else f"  [{span.status}]"
            lines.append(f"{'  ' * depth}{span.kind}:{span.name}  {span.duration * 1000:.1f}ms{flag}")
            for child in self.children(span):
                walk(child, depth + 1)

        walk(self.root, 0)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "duration": self.duration, "status": self.status, "spans": [s.to_dict() for s in self.spans]}


class Tracer:
    """
    Builds a Trace for every top-level run of the agents it's attached to.
    Nested runs (skills, team members sharing a bus) become child spans.
    Finished traces are kept in `.traces` and handed to each exporter.
    """

    def __init__(self, *, exporters: Iterable[Callable[[Trace], Any]] = (), keep: int = 100) -> None:
        self.exporters = list(exporters)
        self.keep = keep
        self.traces: list[Trace] = []
        self._agents: list[Any] = []
        self._local = threading.local()  # in-flight state is per thread, so parallel runs trace separately
        self._lock = threading.Lock()

    @property
    def _trace(self) -> Trace | None:
        return getattr(self._local, "trace", None)

    @_trace.setter
    def _trace(self, value: Trace | None) -> None:
        self._local.trace = value

    @property
    def _stack(self) -> list[Span]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    @property
    def _tools(self) -> dict[str, Span]:
        if not hasattr(self._local, "tools"):
            self._local.tools = {}
        return self._local.tools

    def attach(self, *agents: Any) -> "Tracer":
        for agent in agents:
            agent.events.subscribe(self.handle)
            self._agents.append(agent)
        return self

    def detach(self) -> None:
        for agent in self._agents:
            agent.events.unsubscribe(self.handle)
        self._agents.clear()

    @property
    def last(self) -> Trace | None:
        return self.traces[-1] if self.traces else None

    def _open(self, name: str, kind: str, start: datetime, **attributes: Any) -> Span:
        assert self._trace is not None
        parent = self._stack[-1].id if self._stack else None
        span = Span(name=name, kind=kind, trace_id=self._trace.id, start=start, parent_id=parent, attributes=attributes)
        self._trace.spans.append(span)
        return span

    def handle(self, event: Event) -> None:
        data, source, at = event.data, event.source or "agent", event.timestamp

        if event.type == EventType.AGENT_START:
            if self._trace is None:
                self._trace = Trace(id=_id())
            self._stack.append(self._open(source, "agent", at, goal=data.get("goal")))
            return
        if self._trace is None or not self._stack:
            return

        if event.type == EventType.MODEL_CALL:
            start = at - timedelta(seconds=data.get("duration", 0.0))
            span = self._open(data.get("model") or "model", "model", start, **data)
            span.end = at
        elif event.type == EventType.TOOL_CALL:
            self._tools[data.get("id") or data["name"]] = self._open(data["name"], "tool", at, arguments=data.get("arguments"))
        elif event.type == EventType.TOOL_RESULT:
            span = self._tools.pop(data.get("id") or data["name"], None)
            if span is not None:
                span.end = at
                span.attributes["result"] = data.get("result")
                if str(data.get("result", "")).startswith("Error:"):
                    span.status = "error"
        elif event.type in (EventType.STEP, EventType.ROUTE, EventType.GUARD):
            span = self._open(str(data.get("step") or data.get("route") or data.get("guard")), event.type.value, at, **data)
            span.end = at
            if event.type == EventType.GUARD:
                span.status = "blocked"
        elif event.type in (EventType.AGENT_DONE, EventType.ERROR):
            span = self._stack.pop()
            span.end = at
            if event.type == EventType.ERROR:
                span.status = "error"
                span.attributes["error"] = data.get("reason")
            else:
                span.attributes["result"] = data.get("result")
            if not self._stack:
                self._finish()

    def _finish(self) -> None:
        trace, self._trace = self._trace, None
        self._tools.clear()
        assert trace is not None
        with self._lock:
            self.traces.append(trace)
            del self.traces[: -self.keep]
        for exporter in self.exporters:
            exporter(trace)


class ConsoleExporter:
    """Prints each finished trace as an indented tree."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream

    def __call__(self, trace: Trace) -> None:
        print(trace.tree(), file=self.stream or sys.stdout)


class JSONLExporter:
    """Appends each finished trace to a JSON-lines file, one trace per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __call__(self, trace: Trace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict()) + "\n")


class OpenTelemetryExporter:
    """
    Re-emits each trace as OpenTelemetry spans (preserving timing and nesting) so
    it reaches whatever OTLP backend your OpenTelemetry SDK is configured for.
    Requires `pip install opentelemetry-api opentelemetry-sdk`.
    """

    def __init__(self, tracer: Any = None, *, instrumentation_name: str = "bentotruck") -> None:
        try:
            from opentelemetry import trace as otel_trace
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError("OpenTelemetryExporter needs `pip install opentelemetry-api opentelemetry-sdk`.") from exc
        self._otel = otel_trace
        self.tracer = tracer or otel_trace.get_tracer(instrumentation_name)

    def __call__(self, trace: Trace) -> None:
        emitted: dict[str, Any] = {}
        for span in trace.spans:
            parent = emitted.get(span.parent_id) if span.parent_id else None
            context = self._otel.set_span_in_context(parent) if parent is not None else None
            attributes = {f"bentotruck.{k}": v if isinstance(v, (str, bool, int, float)) else json.dumps(_jsonable(v))
                          for k, v in span.attributes.items() if v is not None}
            attributes["bentotruck.kind"] = span.kind
            otel_span = self.tracer.start_span(
                f"{span.kind}:{span.name}",
                context=context,
                start_time=int(span.start.timestamp() * 1e9),
                attributes=attributes,
            )
            if span.status != "ok":
                otel_span.set_status(self._otel.Status(self._otel.StatusCode.ERROR, span.status))
            emitted[span.id] = otel_span
        for span in reversed(trace.spans):
            end = span.end or span.start
            emitted[span.id].end(end_time=int(end.timestamp() * 1e9))


class Logger:
    """Structured logging of every agent event through the standard `logging` module."""

    def __init__(self, logger: logging.Logger | None = None, *, level: int = logging.INFO) -> None:
        self.logger = logger or logging.getLogger("bentotruck")
        self.level = level
        self._agents: list[Any] = []

    def attach(self, *agents: Any) -> "Logger":
        for agent in agents:
            agent.events.subscribe(self.handle)
            self._agents.append(agent)
        return self

    def detach(self) -> None:
        for agent in self._agents:
            agent.events.unsubscribe(self.handle)
        self._agents.clear()

    def handle(self, event: Event) -> None:
        level = logging.ERROR if event.type == EventType.ERROR else self.level
        fields = " ".join(f"{k}={_short(v)}" for k, v in event.data.items())
        self.logger.log(
            level,
            "%s %s %s",
            event.source or "agent",
            event.type.value,
            fields,
            extra={"bentotruck_event": {"type": event.type.value, "source": event.source, **event.data}},
        )


def _short(value: Any, limit: int = 120) -> str:
    text = repr(value) if not isinstance(value, str) else value
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


class Metrics:
    """Aggregated counters, token totals, and latency stats across every run it's attached to."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.tokens: dict[str, int] = {}
        self.timings: dict[str, list[float]] = {"model": [], "tool": []}
        self.tool_usage: dict[str, int] = {}
        self._agents: list[Any] = []

    def attach(self, *agents: Any) -> "Metrics":
        for agent in agents:
            agent.events.subscribe(self.handle)
            self._agents.append(agent)
        return self

    def detach(self) -> None:
        for agent in self._agents:
            agent.events.unsubscribe(self.handle)
        self._agents.clear()

    def _count(self, name: str) -> None:
        self.counters[name] = self.counters.get(name, 0) + 1

    def handle(self, event: Event) -> None:
        self._count(event.type.value)
        if event.type == EventType.MODEL_CALL:
            self.timings["model"].append(event.data.get("duration", 0.0))
            for key, value in event.data.get("usage", {}).items():
                self.tokens[key] = self.tokens.get(key, 0) + value
        elif event.type == EventType.TOOL_RESULT:
            self.timings["tool"].append(event.data.get("duration", 0.0))
            name = event.data.get("name", "?")
            self.tool_usage[name] = self.tool_usage.get(name, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        latency = {
            kind: {
                "count": len(values),
                "mean": statistics.fmean(values) if values else 0.0,
                "p50": _percentile(values, 50),
                "p95": _percentile(values, 95),
            }
            for kind, values in self.timings.items()
        }
        return {
            "runs": self.counters.get(EventType.AGENT_START.value, 0),
            "errors": self.counters.get(EventType.ERROR.value, 0),
            "model_calls": self.counters.get(EventType.MODEL_CALL.value, 0),
            "tool_calls": self.counters.get(EventType.TOOL_CALL.value, 0),
            "tokens": dict(self.tokens),
            "tool_usage": dict(self.tool_usage),
            "latency": latency,
        }

    def reset(self) -> None:
        self.counters.clear()
        self.tokens.clear()
        self.tool_usage.clear()
        self.timings = {"model": [], "tool": []}


@dataclass
class Profile:
    """Where the time and tokens went across one or more traces."""

    total_seconds: float
    by_kind: dict[str, float]
    by_name: dict[str, tuple[int, float]]
    tokens: dict[str, int]

    def table(self) -> str:
        lines = [f"total {self.total_seconds * 1000:.1f}ms"]
        for (name, (count, seconds)) in sorted(self.by_name.items(), key=lambda kv: -kv[1][1]):
            share = seconds / self.total_seconds if self.total_seconds else 0.0
            lines.append(f"  {name:<28} x{count:<4} {seconds * 1000:9.1f}ms  {share:6.1%}")
        if self.tokens:
            lines.append("  tokens: " + ", ".join(f"{k}={v}" for k, v in self.tokens.items()))
        return "\n".join(lines)


class Profiler:
    """Wall-clock and token breakdowns per span — from a Trace, a list of traces, or a Tracer."""

    @staticmethod
    def profile(source: Trace | Tracer | Iterable[Trace]) -> Profile:
        if isinstance(source, Tracer):
            traces = list(source.traces)
        elif isinstance(source, Trace):
            traces = [source]
        else:
            traces = list(source)

        by_kind: dict[str, float] = {}
        by_name: dict[str, tuple[int, float]] = {}
        tokens: dict[str, int] = {}
        total = 0.0
        for trace in traces:
            total += trace.duration
            for span in trace.spans:
                if span.kind == "agent" and span.parent_id is None:
                    continue
                by_kind[span.kind] = by_kind.get(span.kind, 0.0) + span.duration
                key = f"{span.kind}:{span.name}"
                count, seconds = by_name.get(key, (0, 0.0))
                by_name[key] = (count + 1, seconds + span.duration)
                for k, v in (span.attributes.get("usage") or {}).items():
                    tokens[k] = tokens.get(k, 0) + v
        return Profile(total_seconds=total, by_kind=by_kind, by_name=by_name, tokens=tokens)


__all__ = [
    "Span",
    "Trace",
    "Tracer",
    "ConsoleExporter",
    "JSONLExporter",
    "OpenTelemetryExporter",
    "Logger",
    "Metrics",
    "Profile",
    "Profiler",
]
