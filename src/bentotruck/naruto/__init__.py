"""
naruto — routing: directing a request to the right agent, skill, or model.

Every router has `route(text) -> Route | None`, where a Route names the chosen
target (any object — an agent, a team, a model provider...) and a score.
Routers are Runnable too: `router.run(text)` routes and then runs the target.

Plug one into an agent with `Agent.with_router(...)`: a ModelProvider target
switches the model for that run; a Runnable target takes the run over entirely.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bentotruck.edamame import _cosine, _default_embed
from bentotruck.nigiri import ModelProvider
from bentotruck.rice import invoke


@dataclass
class Route:
    """The outcome of routing: which named route won, its target, and how confident the match was (0–1)."""

    name: str
    target: Any
    score: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


class NoRouteError(LookupError):
    pass


class Router(ABC):
    """Base class: implement `route(text)`; `run(text)` routes and executes the target."""

    default: Any = None

    @abstractmethod
    def route(self, text: str) -> Route | None:
        raise NotImplementedError

    def _fallback(self) -> Route | None:
        return Route("default", self.default, 0.0) if self.default is not None else None

    def run(self, text: Any) -> Any:
        route = self.route(str(text))
        if route is None:
            raise NoRouteError(f"No route matched {str(text)[:80]!r} and no default is set.")
        if isinstance(route.target, ModelProvider):
            return route.target.complete(str(text))
        return invoke(route.target, text)


@dataclass
class Rule:
    """One RuleRouter rule: a match (predicate, regex, or keyword list), a target, and a name."""

    match: Callable[[str], bool] | str | Sequence[str]
    target: Any
    name: str | None = None

    def matches(self, text: str) -> bool:
        if callable(self.match):
            return bool(self.match(text))
        if isinstance(self.match, str):
            return re.search(self.match, text, re.IGNORECASE) is not None
        lowered = text.lower()
        return any(re.search(rf"\b{re.escape(k.lower())}\b", lowered) for k in self.match)


class RuleRouter(Router):
    """
    Routes by explicit, ordered rules — the first rule that matches wins. A rule's
    match can be a predicate `fn(text) -> bool`, a regex string, or a list of keywords.
    """

    def __init__(self, rules: Iterable[Rule | tuple] = (), *, default: Any = None) -> None:
        self.rules: list[Rule] = []
        self.default = default
        for rule in rules:
            self.rules.append(rule if isinstance(rule, Rule) else Rule(*rule))

    def when(self, match: Callable[[str], bool] | str | Sequence[str], target: Any, *, name: str | None = None) -> "RuleRouter":
        self.rules.append(Rule(match, target, name))
        return self

    def route(self, text: str) -> Route | None:
        for index, rule in enumerate(self.rules):
            if rule.matches(text):
                return Route(rule.name or f"rule{index}", rule.target, 1.0)
        return self._fallback()


class SemanticRouter(Router):
    """
    Routes by embedding similarity between the request and each route's
    description/example utterances. Uses edamame's offline hashing embedder unless
    you pass `embed_fn` (recommended for production: a real embedding model).

    SemanticRouter({
        "billing": (["refund my order", "I was charged twice"], billing_agent),
        "menu":    (["what's vegan?", "do you have gluten-free"], menu_agent),
    }, threshold=0.2)
    """

    def __init__(
        self,
        routes: Mapping[str, tuple[str | Sequence[str], Any]] | None = None,
        *,
        threshold: float = 0.2,
        embed_fn: Callable[[str], list[float]] | None = None,
        default: Any = None,
    ) -> None:
        self.threshold = threshold
        self.embed_fn = embed_fn or _default_embed
        self.default = default
        self._routes: dict[str, tuple[list[list[float]], Any]] = {}
        for name, (examples, target) in (routes or {}).items():
            self.add(name, examples, target)

    def add(self, name: str, examples: str | Sequence[str], target: Any) -> "SemanticRouter":
        utterances = [examples] if isinstance(examples, str) else list(examples)
        self._routes[name] = ([self.embed_fn(u) for u in utterances], target)
        return self

    def scores(self, text: str) -> dict[str, float]:
        query = self.embed_fn(text)
        return {name: max(_cosine(query, v) for v in vectors) for name, (vectors, _) in self._routes.items()}

    def route(self, text: str) -> Route | None:
        if not self._routes:
            return self._fallback()
        scores = self.scores(text)
        best = max(scores, key=scores.__getitem__)
        if scores[best] < self.threshold:
            return self._fallback()
        return Route(best, self._routes[best][1], scores[best], {"scores": scores})


class IntentRouter(Router):
    """
    Routes by a classified intent. `classifier` is a ModelProvider (asked to pick
    one label) or any `fn(text) -> label`. `intents` maps each label to a target;
    the label can come with a description to help the model classify.

    IntentRouter(model, {"refund": (billing_agent, "wants money back"), "hours": hours_agent})
    """

    def __init__(
        self,
        classifier: ModelProvider | Callable[[str], str],
        intents: Mapping[str, Any],
        *,
        default: Any = None,
    ) -> None:
        self.classifier = classifier
        self.default = default
        self.intents: dict[str, tuple[Any, str]] = {
            label: value if isinstance(value, tuple) else (value, "") for label, value in intents.items()
        }

    def classify(self, text: str) -> str:
        if isinstance(self.classifier, ModelProvider):
            options = "\n".join(f"- {label}" + (f": {desc}" if desc else "") for label, (_, desc) in self.intents.items())
            reply = self.classifier.complete(
                f"Classify the request into exactly one intent.\n\nIntents:\n{options}\n- other: none of the above\n\n"
                f"Request: {text}\n\nReply with the intent label only."
            )
        else:
            reply = self.classifier(text)
        cleaned = reply.strip().strip(".'\"`").lower()
        for label in self.intents:
            if cleaned == label.lower():
                return label
        for label in self.intents:
            if re.search(rf"\b{re.escape(label.lower())}\b", cleaned):
                return label
        return "other"

    def route(self, text: str) -> Route | None:
        label = self.classify(text)
        if label in self.intents:
            return Route(label, self.intents[label][0], 1.0)
        return self._fallback()


def _capabilities(candidate: Any) -> set[str]:
    declared = getattr(candidate, "capabilities", None)
    if declared:
        return {c.lower() for c in declared}
    names = set()
    for tool in getattr(candidate, "tools", []) or []:
        names.add(tool.name.lower())
    names |= {s.lower() for s in getattr(candidate, "skills", {}) or {}}
    return names


class CapabilityRouter(Router):
    """
    Routes to the candidate whose capabilities best cover what the request needs.
    A candidate's capabilities are its `.capabilities` if declared, else its tool
    and skill names. Required capabilities are passed explicitly, or detected as
    capability names (and `aliases`) mentioned in the request.
    """

    def __init__(
        self,
        candidates: Iterable[Any],
        *,
        aliases: Mapping[str, Sequence[str]] | None = None,
        default: Any = None,
    ) -> None:
        self.candidates = list(candidates)
        self.aliases = {k.lower(): [a.lower() for a in v] for k, v in (aliases or {}).items()}
        self.default = default

    def required_for(self, text: str) -> set[str]:
        lowered = text.lower()
        known = set().union(*(_capabilities(c) for c in self.candidates)) if self.candidates else set()
        needed = set()
        for capability in known:
            words = [capability, capability.replace("_", " ")] + self.aliases.get(capability, [])
            if any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in words):
                needed.add(capability)
        return needed

    def route(self, text: str, *, required: Iterable[str] | None = None) -> Route | None:
        needed = {r.lower() for r in required} if required is not None else self.required_for(text)
        if not needed:
            return self._fallback()
        best, best_score = None, 0.0
        for candidate in self.candidates:
            score = len(needed & _capabilities(candidate)) / len(needed)
            if score > best_score:
                best, best_score = candidate, score
        if best is None:
            return self._fallback()
        return Route(getattr(best, "name", type(best).__name__), best, best_score, {"required": sorted(needed)})


class HybridRouter(Router):
    """Tries routers in order and returns the first route found; falls back to `default`."""

    def __init__(self, *routers: Router, default: Any = None) -> None:
        self.routers = list(routers)
        self.default = default

    def route(self, text: str) -> Route | None:
        for router in self.routers:
            saved, router.default = router.default, None  # only accept real matches from inner routers
            try:
                route = router.route(text)
            finally:
                router.default = saved
            if route is not None:
                return route
        return self._fallback()


__all__ = [
    "Route",
    "Router",
    "NoRouteError",
    "Rule",
    "RuleRouter",
    "SemanticRouter",
    "IntentRouter",
    "CapabilityRouter",
    "HybridRouter",
]
