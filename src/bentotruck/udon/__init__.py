"""
udon — pipelines: sequential (and parallel) execution of plain processing stages.

Planned classes
---------------
Pipeline
    An ordered list of Steps run in sequence, each stage's output feeding the next.
Step
    A single callable stage.
Condition
    Skips a Step unless a predicate over prior output holds.
Loop
    Repeats a Step a fixed number of times or until a condition holds.
Parallel
    Runs a group of Steps concurrently and joins their results.

Distinct from `onigiri` (workflows): udon is for straight-line data/task
pipelines without onigiri's graph-shaped branching.

Status: design stage — not yet implemented.
"""
