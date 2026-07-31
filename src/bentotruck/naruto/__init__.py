"""
naruto — routing: directing a request to the right agent, skill, or model.

Planned classes
---------------
CapabilityRouter
    Routes based on declared tool/skill capabilities of candidate agents.
SemanticRouter
    Routes based on embedding similarity between the request and route descriptions.
RuleRouter
    Routes based on explicit, ordered predicate rules.
IntentRouter
    Routes based on a classified user intent.
HybridRouter
    Combines multiple routers, falling back in order.

Status: design stage — not yet implemented. `rice.Agent.with_router(...)`
already accepts any router-shaped object for forward compatibility.
"""
