"""
onigiri — workflows: deterministic orchestration around one or more agents.

Planned classes
---------------
Workflow
    A named, runnable graph of Steps.
Step
    A single unit of work — an agent run, a tool call, or a plain function.
Condition
    Branches a Workflow based on a predicate over prior step output.
Loop
    Repeats a Step or sub-Workflow until a condition is met.

Distinct from `udon` (pipelines): onigiri workflows are graph-shaped and
condition/loop-aware; udon pipelines are simple linear/parallel stages.

Status: design stage — not yet implemented.
"""
