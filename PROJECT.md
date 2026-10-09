# BentoTruck — Project Document

> **Current version:** 2026.10.9 | **Python:** ≥ 3.9 (tested on 3.9, 3.12, 3.13) | **Status:** All sixteen compartments implemented

---

## Table of Contents

1. [What BentoTruck Is](#what-bentotruck-is)
2. [The Philosophy](#the-philosophy)
3. [Current State](#current-state)
4. [The Compartments](#the-compartments)
5. [Design Principles](#design-principles)
6. [Inter-Op with the Fleet](#inter-op-with-the-fleet)
7. [The Food Truck Fleet](#the-food-truck-fleet)
8. [Build & Publish Plan](#build--publish-plan)

---

## What BentoTruck Is

BentoTruck is the **agent engineering** toolkit for the food truck fleet. Where SushiTruck ingests data, ThaiTruck cleans it, and RamenTruck trains models on it, BentoTruck covers what comes after a model exists: wiring it into an agent that can hold state, call tools, remember things, and act.

It is not tied to one model vendor and does not wrap an existing agent framework — every provider (OpenAI, Anthropic, Gemini, Azure, Ollama, vLLM, LM Studio, any OpenAI-compatible server) speaks through the same `nigiri.ModelProvider.generate()` interface, so switching vendors is a one-line change.

```bash
pip install bentotruck
```

## The Philosophy

**Why a bento box?** Each compartment holds one distinct, purposeful thing, and together they make a complete meal. That's the module boundary discipline: `rice` doesn't know how to call a model's HTTP API, `nigiri` doesn't know what a tool is, `gyoza` doesn't know what memory is — each compartment does one job and composes with the others through small, explicit interfaces.

The guiding design values:

- **Vendor-neutral at the core** — `nigiri.ModelProvider` is the only seam an LLM vendor touches; `rice.Agent` never imports a vendor SDK
- **Duck-typed composition over inheritance** — `Agent.with_memory(...)`, `.with_tools(...)`, `.with_router(...)` accept anything with the right shape, not a registered subclass
- **Security-conscious defaults** — `gyoza.PythonTool` is a restricted-AST sandbox, not `eval()`; `gyoza.SQLTool` requires parameter binding; `gyoza.FilesystemTool` rejects path traversal out of its root
- **Offline-testable** — `nigiri.Mock` and `edamame`'s default hashing embedder mean the whole agent loop is unit-testable with zero network calls or API keys
- **Consistent with the fleet** — same `src/` layout, hatchling build, MIT license, typed (`py.typed`) as ThaiTruck, SushiTruck, and RamenTruck

## Current State

| Item | Status |
|---|---|
| PyPI name `bentotruck` | Published (0.1.0 placeholder 2026-06-24; 0.2.0 / 0.2.1 MVP core) |
| Versioning | Date-based `YYYY.M.D` from 2026.10.9 on (replaces semver) |
| Version | 2026.10.9 — all compartments implemented |
| Runtime dependencies | `requests` only; optional `otel` extra for OpenTelemetry export |
| Tests | 200+ tests, all offline (`nigiri.Mock`, `responses`, a fake stdio MCP server); every `README.md` example is executed by `tests/test_readme.py` |
| `README.md` | Copy-pasteable, tested examples for every compartment |

## The Compartments

| Module | Role | What's implemented |
|--------|------|--------------------|
| `rice` | Core agent | `Agent` (fluent: `using`, `with_tools`, `with_memory`, `with_planner`, `with_router`, `with_guards`, `learn`/`use`), `Session` (with snapshot/restore), `Context`, `State`, `EventBus`, `Event`/`EventType`, `Runnable` protocol + `invoke()` |
| `nigiri` | Models | `ModelProvider` (+ `complete()`), `Mock` (fixed / queued / responder), `OpenAI`, `Azure`, `VLLM`, `LMStudio`, `OpenAICompatible`, `Anthropic`, `Gemini`, `Ollama`; normalized token `usage` on every `ModelResponse` |
| `gyoza` | Tools | `Tool`, `FunctionTool` + `@tool` (schema inferred from type hints), `PythonTool`, `RESTTool`, `SQLTool`, `FilesystemTool`, `BrowserTool` (SSRF-safe), `DockerTool` (locked-down container), `MCPClient`/`MCPTool` (stdio MCP), `Toolbox` |
| `edamame` | Memory | `ConversationMemory`, `WorkingMemory`, `VectorMemory`, `LongTermMemory`, `GraphMemory` (+ `Fact`); deterministic offline embedder; recall injected into agent context |
| `teriyaki` | Planning | `Planner`, `ReActPlanner` (default), `StepPlanner`, `GoalPlanner`, `TreePlanner`, `GraphPlanner` + `Task` |
| `tempura` | Skills | `Skill`, `SummarizeSkill`, `SearchSkill`, `ResearchSkill`, `CodingSkill`, `PlanningSkill`, `@skill` |
| `miso` | Reflection | `Critique`, `Reflect`, `Improve`, `Retry`, `Verify` (+ ready-made checks) |
| `yuzu` | Evaluation | `Sample`, `capture()`, `Latency`, `Cost`, `Grounding`, `Hallucination`, `Confidence`, `Quality`, `Evaluator` (+ `run_suite`) |
| `onigiri` | Workflows | `Workflow` (conditional edges, loops, `END`), `Step`, `Loop`, `Condition` (`&`/`|`/`~`, `contains`, `visited`), `State` |
| `bento` | Multi-agent | `Bento` team with `Sequential`, `Parallel`, `Coordinator`, `Vote` strategies; nested teams; transcript over `sake` |
| `sake` | Messaging | `Hub` (actors, request/reply, broadcast), `Mailbox`, `Message`, `Broadcast`, `Event`, `Channel`, `Topic` (glob routing) |
| `udon` | Pipelines | `Pipeline` (`|` composition, per-stage trace), `Step`, `Condition`, `Loop`, `Parallel` |
| `dango` | Prompts | `Template`, `Variables`/`Var`, `SystemPrompt`, `FewShot`, `PromptRepository` (versions, JSON persistence, stable A/B) |
| `naruto` | Routing | `RuleRouter`, `SemanticRouter`, `IntentRouter`, `CapabilityRouter`, `HybridRouter`; `Route`, `NoRouteError` |
| `katsu` | Safety | `PIIFilter`, `InjectionGuard`, `Moderation`, `OutputValidator` (JSON Schema subset), `PolicyEngine`; `GuardViolation` |
| `wasabi` | Observability | `Tracer` (nested spans, thread-safe), `ConsoleExporter`, `JSONLExporter`, `OpenTelemetryExporter`, `Logger`, `Metrics`, `Profiler` |

**The unifying idea:** anything with `.run(input)` is `rice.Runnable` — agents, teams, workflows, pipelines, routers, and `miso` wrappers. `rice.invoke(target, value)` runs a Runnable or a plain callable, so every compartment can nest inside every other.

## Design Principles

1. **One interface per concern.** `ModelProvider.generate()`, `Tool.run()`, `Memory.add()`/`recall()` — every implementation of a concern is interchangeable because the interface is minimal and concern-specific.
2. **No vendor lock-in leaks upward.** Only `nigiri` knows vendor wire formats. `rice.Agent` only ever sees `nigiri.Message` / `nigiri.ModelResponse`, and no compartment imports a vendor SDK.
3. **Tools are sandboxed by default.** Anything that executes code, hits the filesystem or the web, or runs SQL ships with a restrictive default (AST-whitelist eval, root-jailed paths, parameterized queries, private-address-blocking fetches, network-less read-only containers) rather than a raw `exec`/`open`/`requests.get`/string-formatted query.
4. **Everything works offline.** `nigiri.Mock` plus `edamame.VectorMemory`'s default embedder mean `rice.Agent`'s full tool-calling loop is exercised in tests with no network access and no API keys.
5. **Everything composes.** One `Runnable` shape (`.run(input)`) across agents, teams, workflows, pipelines, routers, and reflection wrappers; one `EventBus` that planners, skills, and teams all publish to, so `wasabi` sees everything.

## Inter-Op with the Fleet

BentoTruck's tools are a natural place to plug in the rest of the fleet: a `gyoza.Tool` subclass wrapping `sushitruck.stream(...)` or `thaitruck`'s cleaning pipeline lets an agent pull and process fleet data mid-run. No such adapter exists yet — noted here as the obvious next integration point, not a current dependency (BentoTruck has zero fleet-package dependencies today).

## The Food Truck Fleet

- **ThaiTruck** — batch DataFrame cleaning & processing (Data Engineering)
- **SushiTruck** — streaming ingestion & API connectors (Data Acquisition)
- **RamenTruck** — ML/AI toolkit (Machine Learning)
- **BentoTruck** — agent engineering framework (Agent Engineering) — this package

## Build & Publish Plan

1. MVP core (`rice`, `nigiri`, `gyoza`, `edamame`) — **done** (0.2.x).
2. Remaining twelve compartments, plus the vision's missing pieces (Azure/vLLM/LM Studio providers, Browser/Docker/MCP tools, GraphMemory) — **done** (2026.10.9).
3. Next: fleet adapters — `gyoza` tools wrapping SushiTruck streams and ThaiTruck cleaning pipelines (see Inter-Op above).
4. Next: async variants (`Agent.arun`, async providers) and streaming responses, driven by real usage.
