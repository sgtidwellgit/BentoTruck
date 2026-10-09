"""
katsu — safety: guardrails around agent input and output.

Every guard has `check(text) -> GuardResult` (allowed?, the possibly-transformed
text, and the violations found) and `enforce(text) -> text`, which raises
`GuardViolation` when blocked. Attach them to an agent:

    agent.with_guards(
        input=[katsu.InjectionGuard()],
        tools=[katsu.InjectionGuard()],          # screen what tools bring back
        output=[katsu.PIIFilter()],              # redact before returning
    )

These are pattern-based first lines of defense — fast, offline, predictable.
For higher-stakes use, pair them with a model-based check (`Moderation` takes
a `provider=`).
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bentotruck.nigiri import ModelProvider


@dataclass
class GuardResult:
    allowed: bool
    text: str
    violations: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.allowed


class GuardViolation(ValueError):
    """Raised by `enforce` when a guard blocks content. `.result` holds the details."""

    def __init__(self, guard: str, result: GuardResult) -> None:
        super().__init__(f"{guard} blocked content: {'; '.join(result.violations)}")
        self.guard = guard
        self.result = result


class Guard(ABC):
    """Base class every guard implements."""

    @abstractmethod
    def check(self, text: str) -> GuardResult:
        raise NotImplementedError

    def enforce(self, text: str) -> str:
        result = self.check(text)
        if not result.allowed:
            raise GuardViolation(type(self).__name__, result)
        return result.text

    def __call__(self, text: str) -> GuardResult:
        return self.check(text)


# --- PII ---------------------------------------------------------------------------


def _luhn(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit *= 2
            digit -= 9 if digit > 9 else 0
        total += digit
    return total % 10 == 0


_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "ssn": re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    "phone": re.compile(r"(?<!\w)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\w)"),
    "ip_address": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
}


class PIIFilter(Guard):
    """
    Detects personally identifiable information — emails, credit card numbers
    (Luhn-checked), US SSNs, phone numbers, IP addresses — and redacts it
    (`mode="redact"`, the default) or blocks the text outright (`mode="block"`).
    Add your own with `extra={"employee_id": r"EMP-\\d{6}"}`.
    """

    def __init__(
        self,
        *,
        kinds: Iterable[str] | None = None,
        mode: str = "redact",
        extra: Mapping[str, str] | None = None,
        mask: str = "[{kind}]",
    ) -> None:
        if mode not in ("redact", "block"):
            raise ValueError("mode must be 'redact' or 'block'.")
        selected = list(kinds) if kinds is not None else list(_PII_PATTERNS)
        unknown = [k for k in selected if k not in _PII_PATTERNS]
        if unknown:
            raise ValueError(f"Unknown PII kinds {unknown}; known: {list(_PII_PATTERNS)}")
        self.patterns = {k: _PII_PATTERNS[k] for k in selected}
        self.patterns.update({k: re.compile(v) for k, v in (extra or {}).items()})
        self.mode = mode
        self.mask = mask

    def find(self, text: str) -> list[tuple[str, str]]:
        """Every (kind, matched text) found."""

        found = []
        for kind, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                if kind == "credit_card" and not _luhn(match.group()):
                    continue
                found.append((kind, match.group()))
        return found

    def check(self, text: str) -> GuardResult:
        found = self.find(text)
        if not found:
            return GuardResult(True, text)
        violations = sorted({f"pii:{kind}" for kind, _ in found})
        if self.mode == "block":
            return GuardResult(False, text, violations)
        redacted = text
        for kind, value in sorted(found, key=lambda kv: -len(kv[1])):
            redacted = redacted.replace(value, self.mask.format(kind=kind.upper()))
        return GuardResult(True, redacted, violations)


# --- Prompt injection -------------------------------------------------------------------

_INJECTION_PATTERNS = [
    (r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your)\b.{0,20}"
     r"\b(instructions?|prompts?|rules?|directions?|context)\b", 0.9),
    (r"\b(reveal|show|print|repeat|output|leak)\b.{0,40}\b(system prompt|hidden (instructions?|prompt)|"
     r"initial instructions?|your instructions)\b", 0.8),
    (r"\byou are now\b|\bfrom now on,? you (are|will)\b|\bact as\b.{0,30}\b(unrestricted|jailbroken|dan)\b", 0.6),
    (r"\b(developer|god|admin|debug) mode\b", 0.5),
    (r"\bnew instructions?\s*:", 0.6),
    (r"<\|?(im_start|im_end|system|endoftext)\|?>|\[/?INST\]|<</?SYS>>", 0.9),
    (r"^\s*(system|assistant)\s*:", 0.5),
    (r"\bdo not (tell|inform|alert) the user\b", 0.7),
    (r"\b(send|post|exfiltrate|upload)\b.{0,40}\b(api[_ ]?key|password|credentials?|secrets?|tokens?)\b", 0.8),
]


class InjectionGuard(Guard):
    """
    Detects prompt-injection attempts — most useful on tool output and retrieved
    documents, where untrusted text enters the conversation. Each matched pattern
    has a weight; text scoring at or above `threshold` is blocked.
    """

    def __init__(self, *, threshold: float = 0.5, extra_patterns: Sequence[tuple[str, float]] = ()) -> None:
        self.threshold = threshold
        self.patterns = [(re.compile(p, re.IGNORECASE | re.MULTILINE), w) for p, w in [*_INJECTION_PATTERNS, *extra_patterns]]

    def score(self, text: str) -> tuple[float, list[str]]:
        hits = [(match.group(0).strip(), weight) for pattern, weight in self.patterns if (match := pattern.search(text))]
        if not hits:
            return 0.0, []
        # combine like independent probabilities: 1 - Π(1 - w)
        remaining = 1.0
        for _, weight in hits:
            remaining *= 1 - weight
        return 1 - remaining, [f"injection:{snippet[:60]}" for snippet, _ in hits]

    def check(self, text: str) -> GuardResult:
        score, violations = self.score(text)
        return GuardResult(score < self.threshold, text, violations)


# --- Moderation ---------------------------------------------------------------------------

DEFAULT_CATEGORIES = ("hate", "harassment", "violence", "self-harm", "sexual", "illegal-activity")


class Moderation(Guard):
    """
    Flags disallowed content categories. Supply any combination of:

    - `blocklist={"category": ["term", r"regex", ...]}` — deterministic term matching
    - `provider=` — a model asked to label the text against `categories`
    - `classifier=fn(text) -> list[str]` — your own moderation API/model

    With none of them, there's nothing to moderate against, so construction fails.
    """

    def __init__(
        self,
        *,
        blocklist: Mapping[str, Sequence[str]] | None = None,
        provider: ModelProvider | None = None,
        classifier: Callable[[str], Sequence[str]] | None = None,
        categories: Sequence[str] = DEFAULT_CATEGORIES,
    ) -> None:
        if not (blocklist or provider or classifier):
            raise ValueError("Moderation needs a blocklist=, provider=, or classifier= to judge against.")
        self.blocklist = {
            category: [re.compile(rf"\b{t}\b" if re.fullmatch(r"[\w\s'-]+", t) else t, re.IGNORECASE) for t in terms]
            for category, terms in (blocklist or {}).items()
        }
        self.provider = provider
        self.classifier = classifier
        self.categories = list(categories)

    def flagged(self, text: str) -> list[str]:
        flags = [category for category, patterns in self.blocklist.items() if any(p.search(text) for p in patterns)]
        if self.classifier is not None:
            flags += list(self.classifier(text))
        if self.provider is not None:
            reply = self.provider.complete(
                "You are a content moderator. Which of these categories does the text clearly violate? "
                f"Categories: {', '.join(self.categories)}.\n\nText:\n\"\"\"\n{text}\n\"\"\"\n\n"
                "Reply with a comma-separated list of violated categories, or 'none'."
            )
            lowered = reply.lower()
            flags += [c for c in self.categories if re.search(rf"\b{re.escape(c)}\b", lowered)]
        return list(dict.fromkeys(flags))

    def check(self, text: str) -> GuardResult:
        flags = self.flagged(text)
        return GuardResult(not flags, text, [f"moderation:{f}" for f in flags])


# --- Output validation ----------------------------------------------------------------------

_JSON_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def validate_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Validate against a practical subset of JSON Schema: type, enum, const, required, properties,
    additionalProperties, items, min/maxLength, minimum/maximum, min/maxItems, pattern."""

    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        ok = any(
            isinstance(value, _JSON_TYPES[t]) and not (t in ("integer", "number") and isinstance(value, bool))
            for t in types
        )
        if not ok:
            return [f"{path}: expected {expected}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must equal {schema['const']!r}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match {schema['pattern']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        properties = schema.get("properties", {})
        for key, sub in properties.items():
            if key in value:
                errors += validate_schema(value[key], sub, f"{path}.{key}")
        if schema.get("additionalProperties") is False:
            errors += [f"{path}: unexpected key {k!r}" for k in value if k not in properties]
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for index, item in enumerate(value):
                errors += validate_schema(item, schema["items"], f"{path}[{index}]")
    return errors


_FENCED_JSON = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


class OutputValidator(Guard):
    """
    Validates a response before it's returned: length bounds, required/forbidden
    patterns, JSON parseability and a JSON Schema, plus any custom checks
    (`fn(text) -> True/None` to pass, or False / a reason string to fail).

    With `schema=` the text must be JSON (a ```json fenced block is accepted and
    unwrapped); the validated text returned is the bare JSON.
    """

    def __init__(
        self,
        *,
        schema: Mapping[str, Any] | None = None,
        require_json: bool = False,
        min_length: int = 0,
        max_length: int | None = None,
        must_match: Sequence[str] = (),
        must_not_match: Sequence[str] = (),
        checks: Sequence[Callable[[str], Any]] = (),
    ) -> None:
        self.schema = schema
        self.require_json = require_json or schema is not None
        self.min_length = min_length
        self.max_length = max_length
        self.must_match = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in must_match]
        self.must_not_match = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in must_not_match]
        self.checks = list(checks)

    def check(self, text: str) -> GuardResult:
        errors: list[str] = []
        output = text
        if len(text) < self.min_length:
            errors.append(f"shorter than {self.min_length} chars")
        if self.max_length is not None and len(text) > self.max_length:
            errors.append(f"longer than {self.max_length} chars")
        errors += [f"missing pattern {p.pattern!r}" for p in self.must_match if not p.search(text)]
        errors += [f"forbidden pattern {p.pattern!r}" for p in self.must_not_match if p.search(text)]
        if self.require_json:
            fenced = _FENCED_JSON.search(text)
            candidate = fenced.group(1) if fenced else text
            try:
                data = json.loads(candidate)
                output = candidate.strip()
                if self.schema is not None:
                    errors += validate_schema(data, self.schema)
            except ValueError as exc:
                errors.append(f"not valid JSON ({exc.args[0] if exc.args else exc})")
        for check in self.checks:
            outcome = check(text)
            if outcome is False:
                errors.append(f"failed check {getattr(check, '__name__', 'check')!r}")
            elif isinstance(outcome, str):
                errors.append(outcome)
        return GuardResult(not errors, output, [f"validation:{e}" for e in errors])


# --- Policy engine ----------------------------------------------------------------------------


@dataclass
class Rule:
    """
    One policy rule. `when(text)` decides if it applies; `action` is "deny",
    "allow" (stop evaluating, let it through), or "transform" (apply `transform`
    and continue). A rule can also wrap a whole Guard via `Rule.from_guard`.
    """

    name: str
    when: Callable[[str], bool]
    action: str = "deny"
    transform: Callable[[str], str] | None = None
    reason: str = ""

    @classmethod
    def from_guard(cls, guard: Guard, name: str | None = None) -> "Rule":
        rule = cls(name or type(guard).__name__, when=lambda text: True, action="guard")
        rule._guard = guard  # type: ignore[attr-defined]
        return rule


class PolicyEngine(Guard):
    """
    Evaluates text against an ordered rule set — allow / deny / transform — and
    can embed other guards as steps. First "allow" or "deny" decides; transforms
    accumulate. Rules can be added fluently:

    PolicyEngine().deny("no-urls", lambda t: "http" in t).transform("trim", str.strip).guard(PIIFilter())
    """

    def __init__(self, rules: Iterable[Rule | Guard] = ()) -> None:
        self.rules: list[Rule] = [r if isinstance(r, Rule) else Rule.from_guard(r) for r in rules]

    def deny(self, name: str, when: Callable[[str], bool], *, reason: str = "") -> "PolicyEngine":
        self.rules.append(Rule(name, when, "deny", reason=reason))
        return self

    def allow(self, name: str, when: Callable[[str], bool]) -> "PolicyEngine":
        self.rules.append(Rule(name, when, "allow"))
        return self

    def transform(self, name: str, fn: Callable[[str], str], *, when: Callable[[str], bool] = lambda t: True) -> "PolicyEngine":
        self.rules.append(Rule(name, when, "transform", transform=fn))
        return self

    def guard(self, guard: Guard, name: str | None = None) -> "PolicyEngine":
        self.rules.append(Rule.from_guard(guard, name))
        return self

    def check(self, text: str) -> GuardResult:
        violations: list[str] = []
        for rule in self.rules:
            if not rule.when(text):
                continue
            if rule.action == "allow":
                return GuardResult(True, text, violations)
            if rule.action == "deny":
                violations.append(f"policy:{rule.name}" + (f" ({rule.reason})" if rule.reason else ""))
                return GuardResult(False, text, violations)
            if rule.action == "transform" and rule.transform is not None:
                text = rule.transform(text)
            elif rule.action == "guard":
                result = rule._guard.check(text)  # type: ignore[attr-defined]
                violations += result.violations
                if not result.allowed:
                    return GuardResult(False, text, violations)
                text = result.text
        return GuardResult(True, text, violations)


__all__ = [
    "Guard",
    "GuardResult",
    "GuardViolation",
    "PIIFilter",
    "InjectionGuard",
    "Moderation",
    "OutputValidator",
    "PolicyEngine",
    "Rule",
    "validate_schema",
]
