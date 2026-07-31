"""
sake — communication: actor-model-inspired messaging between agents.

Planned classes
---------------
Message
    An addressed payload sent from one agent/component to another.
Broadcast
    A Message with no single addressee — delivered to every subscriber of a Channel.
Channel
    A named delivery route agents publish to and subscribe from.
Event
    A fire-and-forget notification, distinct from a Message expecting a reply.
Topic
    A logical grouping of Channels for pub/sub-style routing.

Status: design stage — not yet implemented. Note: distinct from
`nigiri.Message`, which represents one LLM conversation turn, not
inter-agent communication.
"""
