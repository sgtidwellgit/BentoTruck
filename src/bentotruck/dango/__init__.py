"""
dango — prompts: templated, versioned prompt management.

Planned classes
---------------
Template
    A parameterized prompt string with named slots.
SystemPrompt
    A Template specialized for the system role, composable with agent context.
FewShot
    Wraps a Template with example input/output pairs prepended at render time.
Variables
    Typed slot-filling for a Template, validated before render.
PromptRepository
    Named, versioned storage/retrieval of Templates (e.g. for A/B testing prompts).

Status: design stage — not yet implemented.
"""
