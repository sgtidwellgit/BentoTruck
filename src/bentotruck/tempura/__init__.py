"""
tempura — skills: reusable, named behaviors an agent can learn and invoke.

Planned classes
---------------
SearchSkill, SummarizeSkill, CodingSkill, ResearchSkill, PlanningSkill
    Pre-built behaviors bundling a prompt, tool selection, and output contract.

Planned API: `agent.learn(SearchSkill())` registers a skill the agent can
invoke by name, distinct from a raw `gyoza.Tool` in that a skill can carry
its own multi-step prompting strategy.

Status: design stage — not yet implemented.
"""
