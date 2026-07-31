# bentotruck

**BentoTruck** is an agent engineering framework — every compartment of the bento box is a building block for agentic systems.

> Just like a bento box: each compartment holds a distinct, purposeful piece — together they make a complete meal.

## Philosophy

BentoTruck is not "yet another AI framework." It is **Agent Engineering** — focused on the fundamental pieces that every agentic system needs, regardless of which model or provider sits underneath.

Not focused on prompts. Not focused on OpenAI. Not focused on LangChain. Focused on architecture.

## Compartments

| Module | Emoji | Role |
|--------|-------|------|
| `rice` | `rice` | Core Agent — Agent, Session, Context, State, Events |
| `teriyaki` | `meat_on_bone` | Planning — ReAct, Tree, Graph, Step, Goal planners |
| `edamame` | `broccoli` | Memory — Conversation, Vector, Graph, Working, LongTerm |
| `gyoza` | `dumpling` | Tools — Python, REST, SQL, Filesystem, Browser, MCP, Docker |
| `tempura` | `fried_shrimp` | Skills — Reusable behaviors: Search, Summarize, Code, Research |
| `nigiri` | `sushi` | Models — Vendor-neutral providers: OpenAI, Anthropic, Gemini, Ollama, Azure, vLLM |
| `miso` | `bowl_with_spoon` | Reflection — Reflect, Retry, Improve, Critique, Verify |
| `yuzu` | `tangerine` | Evaluation — Confidence, Grounding, Latency, Hallucination, Cost, Quality |
| `onigiri` | `rice_ball` | Workflows — Workflow, Step, Condition, Loop |
| `bento` | `bento_box` | Multi-Agent — Team orchestration |
| `sake` | `sake` | Communication — Message, Broadcast, Channel, Event, Topic |
| `udon` | `steaming_bowl` | Pipelines — Pipeline, Step, Condition, Loop, Parallel |
| `dango` | `dango` | Prompts — Template, SystemPrompt, FewShot, Variables, PromptRepository |
| `naruto` | `fish_cake` | Routing — Capability, Semantic, Rule, Intent, Hybrid routers |
| `katsu` | `curry` | Safety — PII, Injection, Moderation, OutputValidator, PolicyEngine |
| `wasabi` | `herb` | Observability — Trace, Span, Logger, Metrics, Profiler |

## Quick Example

```python
from bentotruck import rice, gyoza, edamame, nigiri

agent = (
    rice.Agent("Restaurant Assistant")
    .using(nigiri.Mock(reply="Order confirmed."))  # swap in nigiri.OpenAI(), nigiri.Anthropic(model=...), nigiri.Gemini(), or nigiri.Ollama()
    .with_memory(edamame.VectorMemory())
    .with_tools(
        gyoza.SQLTool(connection=my_db_connection),
        gyoza.RESTTool("orders_api", "Look up an order.", url="https://example.com/orders"),
        gyoza.PythonTool(),
    )
)

agent.run("What's 2 + 2?")
```

`naruto.SemanticRouter()` and friends are on the roadmap — see the module table above for what's implemented (`rice`, `nigiri`, `gyoza`, `edamame`) versus planned.

## Fleet

- **ThaiTruck** — batch DataFrame cleaning & processing (Data Engineering)
- **SushiTruck** — streaming ingestion & API connectors (Data Acquisition)
- **RamenTruck** — ML/AI toolkit (Machine Learning)
- **BentoTruck** — agent engineering framework (Agent Engineering) *(this package, in development)*

---

*Core compartments (`rice`, `nigiri`, `gyoza`, `edamame`) implemented. Remaining compartments are design-stage — see `PROJECT.md`.*
