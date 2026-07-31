"""
bento — multi-agent: team orchestration (the flagship module the whole fleet is named after).

Planned API
-----------
`Bento()` — a team of `rice.Agent` instances.
    `.add(agent)` registers a member.
    `.run(goal)` coordinates members toward a shared goal — handing off
    subtasks, sharing `sake` messages between members, and merging results.

Status: design stage — not yet implemented. Depends on `rice` (implemented)
and `sake` (planned) for inter-agent messaging.
"""
