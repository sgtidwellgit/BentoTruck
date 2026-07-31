"""
teriyaki — planning: how agents decide what to do next.

Planned classes
---------------
ReActPlanner
    Interleaved reason/act loop — the default strategy `rice.Agent` runs internally today.
TreePlanner
    Explores multiple candidate action branches before committing.
GraphPlanner
    Plans over a dependency graph of subgoals rather than a linear chain.
StepPlanner
    Fixed, user-authored sequence of steps — no model-driven branching.
GoalPlanner
    Decomposes a high-level goal into subgoals, delegating each to the agent loop.

Status: design stage — not yet implemented. See `rice.Agent.run` for the
built-in ReAct-style loop these planners will eventually become pluggable
alternatives to (via a future `Agent.with_planner(...)`).
"""
