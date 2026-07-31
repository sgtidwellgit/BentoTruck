"""
wasabi — observability: tracing, logging, and metrics for agent runs.

Planned classes
---------------
Trace
    A full record of one agent run, composed of Spans.
Span
    A single timed unit of work within a Trace (a model call, a tool call, a step).
Logger
    Structured event logging, subscribable to `rice.EventBus`.
Metrics
    Aggregated counters/timers/histograms over Traces.
Profiler
    Wall-clock and token-cost breakdowns per Span.

Planned integrations: LangSmith, OpenTelemetry, Phoenix, MLflow exporters.

Status: design stage — not yet implemented. `rice.EventBus` already
provides the publish/subscribe hook these exporters will attach to.
"""
