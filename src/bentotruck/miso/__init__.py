"""
miso — reflection: self-evaluation and iterative improvement of an agent's own output.

Planned classes
---------------
Reflect
    Prompts the model to critique its own last response before returning it.
Retry
    Re-runs a failed step with the failure fed back into context.
Improve
    Iteratively revises a response against a scoring function until it converges.
Critique
    Produces a structured critique of a response without necessarily revising it.
Verify
    Checks a response against an external ground truth or constraint set.

Status: design stage — not yet implemented.
"""
