"""
katsu — safety: guardrails around agent input and output.

Planned classes
---------------
PIIFilter
    Detects and redacts personally identifiable information.
InjectionGuard
    Detects prompt-injection attempts in tool output or retrieved content.
Moderation
    Flags disallowed content categories in input or output.
OutputValidator
    Validates a response against a schema or set of constraints before it's returned.
PolicyEngine
    Evaluates a response or action against a configurable rule set, allow/deny/transform.

Status: design stage — not yet implemented.
"""
