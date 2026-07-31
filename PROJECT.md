# BentoTruck — Project Document

> **Current version:** 0.1.0 | **Python:** ≥ 3.9 | **Status:** MVP core implemented, remaining compartments design-stage

---

## Table of Contents

1. [What BentoTruck Is](#what-bentotruck-is)
2. [The Philosophy](#the-philosophy)
3. [Current State](#current-state)
4. [The Compartments — All Planned Modules](#the-compartments--all-planned-modules)
5. [Design Principles](#design-principles)
6. [Inter-Op with the Fleet](#inter-op-with-the-fleet)
7. [The Food Truck Fleet](#the-food-truck-fleet)
8. [Build & Publish Plan](#build--publish-plan)

---

## What BentoTruck Is

BentoTruck is the **agent engineering** toolkit for the food truck fleet. Where SushiTruck ingests data, ThaiTruck cleans it, and RamenTruck trains models on it, BentoTruck covers what comes after a model exists: wiring it into an agent that can hold state, call tools, remember things, and act.

It is not tied to one model vendor and does not wrap an existing agent framework — every provider (OpenAI, Anthropic, Gemini, Ollama, ...) speaks through the same `nigiri.ModelProvider.generate()` interface, so switching vendors is a one-line change.

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
| PyPI name `bentotruck` | Reserved |
| Version | 0.1.0 |
| `rice` (core agent) | Implemented — `Agent`, `Session`, `Context`, `State`, `Event`, `EventBus` |
| `nigiri` (model providers) | Implemented — `ModelProvider` base, `Mock`, `OpenAI`, `Anthropic`, `Gemini`, `Ollama` |
| `gyoza` (tools) | Implemented — `Tool` base, `PythonTool`, `RESTTool`, `SQLTool`, `FilesystemTool`, `Toolbox` |
| `edamame` (memory) | Implemented — `Memory` base, `ConversationMemory`, `WorkingMemory`, `VectorMemory`, `LongTermMemory` |
| `teriyaki`, `tempura`, `miso`, `yuzu`, `onigiri`, `bento`, `sake`, `udon`, `dango`, `naruto`, `katsu`, `wasabi` | Design-stage — docstring-only stubs describing planned API |
| `pyproject.toml` | Exists — hatchling build, Python ≥ 3.9, MIT license, `requests` dependency, `dev` extra |
| `README.md` | Exists — install, quick example, fleet context |
| Tests | `tests/test_{rice,nigiri,gyoza,edamame}.py` covering the implemented compartments |

## The Compartments — All Planned Modules

| Module | Role | Status |
|--------|------|--------|
| `rice` | Core Agent — Agent, Session, Context, State, Events | **Implemented** |
| `nigiri` | Models — vendor-neutral providers | **Implemented** |
| `gyoza` | Tools — one calling interface | **Implemented** |
| `edamame` | Memory — conversation, working, vector, long-term | **Implemented** |
| `teriyaki` | Planning — ReAct, Tree, Graph, Step, Goal planners | Planned |
| `tempura` | Skills — reusable named behaviors | Planned |
| `miso` | Reflection — self-critique and retry | Planned |
| `yuzu` | Evaluation — confidence, grounding, cost, quality metrics | Planned |
| `onigiri` | Workflows — graph-shaped orchestration with branching | Planned |
| `bento` | Multi-Agent — team orchestration (flagship) | Planned |
| `sake` | Communication — actor-model inter-agent messaging | Planned |
| `udon` | Pipelines — linear/parallel stage execution | Planned |
| `dango` | Prompts — templated, versioned prompt management | Planned |
| `naruto` | Routing — capability/semantic/rule/intent routers | Planned |
| `katsu` | Safety — PII, injection, moderation guardrails | Planned |
| `wasabi` | Observability — tracing, logging, metrics | Planned |

`rice.Agent` already exposes forward-compatible hooks for several of these:
`.with_router(...)` accepts anything shaped like a future `naruto` router,
`.events` (an `EventBus`) is the attachment point `wasabi` exporters will
subscribe to, and the tool-calling loop in `Agent.run` is the concrete
strategy `teriyaki.ReActPlanner` will eventually become a pluggable
alternative to.

## Design Principles

1. **One interface per concern.** `ModelProvider.generate()`, `Tool.run()`, `Memory.add()`/`recall()` — every implementation of a concern is interchangeable because the interface is minimal and concern-specific.
2. **No vendor lock-in leaks upward.** Only `nigiri` imports `requests` and knows vendor wire formats. `rice.Agent` only ever sees `nigiri.Message` / `nigiri.ModelResponse`.
3. **Tools are sandboxed by default.** Anything that executes code, hits the filesystem, or runs SQL ships with a restrictive default (AST-whitelist eval, root-jailed paths, parameterized queries) rather than a raw `exec`/`open`/string-formatted query.
4. **Everything works offline.** `nigiri.Mock` plus `edamame.VectorMemory`'s default embedder mean `rice.Agent`'s full tool-calling loop is exercised in tests with no network access and no API keys.

## Inter-Op with the Fleet

BentoTruck's tools are a natural place to plug in the rest of the fleet: a `gyoza.Tool` subclass wrapping `sushitruck.stream(...)` or `thaitruck`'s cleaning pipeline lets an agent pull and process fleet data mid-run. No such adapter exists yet — noted here as the obvious next integration point, not a current dependency (BentoTruck has zero fleet-package dependencies today).

## The Food Truck Fleet

- **ThaiTruck** — batch DataFrame cleaning & processing (Data Engineering)
- **SushiTruck** — streaming ingestion & API connectors (Data Acquisition)
- **RamenTruck** — ML/AI toolkit (Machine Learning)
- **BentoTruck** — agent engineering framework (Agent Engineering) — this package

## Build & Publish Plan

1. MVP core (`rice`, `nigiri`, `gyoza`, `edamame`) — **done**, this pass.
2. `teriyaki` (planning) — make the agent loop's strategy pluggable, starting with `ReActPlanner` extracted from `rice.Agent.run`.
3. `wasabi` (observability) — subscribe an exporter to `rice.EventBus`.
4. `bento` + `sake` — multi-agent orchestration, the flagship module.
5. Remaining compartments (`tempura`, `miso`, `yuzu`, `onigiri`, `udon`, `dango`, `naruto`, `katsu`) as needed by real usage rather than speculatively.
6. First PyPI release once `teriyaki` and `wasabi` land — a framework with one hardcoded planning strategy and no observability hook isn't yet "agent engineering," just an agent.
