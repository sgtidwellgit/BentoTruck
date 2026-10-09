"""
dango — prompts: templated, versioned prompt management.

Templates use Python `{slot}` syntax (write `{{` / `}}` for literal braces).

    greet = dango.Template("Hello {name}, welcome to {place}.", name="greet")
    greet.render(name="Ada", place="the truck")

`SystemPrompt` builds structured system prompts, `FewShot` prepends worked
examples, `Variables` validates and coerces slot values, and
`PromptRepository` stores named, versioned templates (with stable A/B bucketing).
"""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bentotruck.nigiri import Message

_MISSING = object()


@dataclass
class Var:
    """One typed slot: its type, an optional default, allowed choices, and a description."""

    type: type = str
    default: Any = _MISSING
    choices: Sequence[Any] | None = None
    description: str = ""

    @property
    def required(self) -> bool:
        return self.default is _MISSING


class Variables:
    """
    Typed slot-filling for a Template. Values are coerced to the declared type
    (so "3" becomes 3 for an int slot), defaults are applied, and choices enforced.

    Variables(topic=str, n=Var(int, default=3), tone=Var(str, choices=["formal", "casual"]))
    """

    def __init__(self, **specs: type | Var) -> None:
        self.specs: dict[str, Var] = {k: v if isinstance(v, Var) else Var(type=v) for k, v in specs.items()}

    def validate(self, values: Mapping[str, Any]) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        errors: list[str] = []
        for name, spec in self.specs.items():
            if name in values:
                raw = values[name]
            elif not spec.required:
                raw = spec.default
            else:
                errors.append(f"{name!r} is required")
                continue
            try:
                value = raw if isinstance(raw, spec.type) else spec.type(raw)
            except (TypeError, ValueError):
                errors.append(f"{name!r} must be {spec.type.__name__}, got {raw!r}")
                continue
            if spec.choices is not None and value not in spec.choices:
                errors.append(f"{name!r} must be one of {list(spec.choices)}, got {value!r}")
                continue
            resolved[name] = value
        if errors:
            raise ValueError("Invalid prompt variables: " + "; ".join(errors))
        extras = {k: v for k, v in values.items() if k not in self.specs}
        return {**extras, **resolved}


class Template:
    """A parameterized prompt string with named slots, an optional name/version, and optional Variables."""

    def __init__(
        self,
        text: str,
        *,
        name: str | None = None,
        version: str = "1",
        variables: Variables | None = None,
        defaults: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.name = name
        self.version = str(version)
        self.variables = variables
        self.defaults = dict(defaults or {})
        self.metadata = dict(metadata or {})

    @property
    def slots(self) -> list[str]:
        """The slot names the template expects, in order of first appearance."""

        names = [field_name for _, field_name, _, _ in string.Formatter().parse(self.text) if field_name]
        return list(dict.fromkeys(n.split(".")[0].split("[")[0] for n in names))

    def render(self, **values: Any) -> str:
        merged = {**self.defaults, **values}
        if self.variables is not None:
            merged = self.variables.validate(merged)
        missing = [s for s in self.slots if s not in merged]
        if missing:
            raise KeyError(f"Template {self.name or ''!r} is missing values for: {missing}")
        return self.text.format(**merged)

    def partial(self, **values: Any) -> "Template":
        """A copy with some slots pre-filled (they become defaults)."""

        return Template(self.text, name=self.name, version=self.version, variables=self.variables,
                        defaults={**self.defaults, **values}, metadata=self.metadata)

    def to_message(self, role: str = "user", **values: Any) -> Message:
        return Message(role=role, content=self.render(**values))

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "name": self.name, "version": self.version, "defaults": self.defaults,
                "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Template":
        return cls(data["text"], name=data.get("name"), version=data.get("version", "1"),
                   defaults=data.get("defaults"), metadata=data.get("metadata"))

    def __repr__(self) -> str:
        return f"Template(name={self.name!r}, version={self.version!r}, slots={self.slots})"


class SystemPrompt(Template):
    """
    A Template specialized for the system role, assembled from parts — who the
    agent is, what it should do, the rules it must follow, and the output format.
    Pass it straight to `Agent(system_prompt=...)` or `.with_system_prompt(...)`.
    """

    def __init__(
        self,
        role: str,
        *,
        instructions: Sequence[str] = (),
        constraints: Sequence[str] = (),
        output_format: str | None = None,
        name: str | None = None,
        version: str = "1",
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        parts = [role.strip()]
        if instructions:
            parts.append("Instructions:\n" + "\n".join(f"- {i}" for i in instructions))
        if constraints:
            parts.append("Rules:\n" + "\n".join(f"- {c}" for c in constraints))
        if output_format:
            parts.append(f"Output format:\n{output_format}")
        super().__init__("\n\n".join(parts), name=name, version=version, defaults=defaults)

    def to_message(self, role: str = "system", **values: Any) -> Message:
        return super().to_message(role, **values)


class FewShot:
    """
    Wraps a Template with worked examples prepended at render time.

    FewShot(Template("Input: {input}\\nOutput:"), examples=[{"input": "2+2", "output": "4"}])
    """

    def __init__(
        self,
        template: Template | str,
        examples: Sequence[Mapping[str, Any]],
        *,
        example_template: str = "Input: {input}\nOutput: {output}",
        prefix: str = "",
        separator: str = "\n\n",
    ) -> None:
        self.template = template if isinstance(template, Template) else Template(template)
        self.examples = list(examples)
        self.example_template = example_template
        self.prefix = prefix
        self.separator = separator

    @property
    def name(self) -> str | None:
        return self.template.name

    def render(self, **values: Any) -> str:
        shots = [self.example_template.format(**example) for example in self.examples]
        parts = ([self.prefix] if self.prefix else []) + shots + [self.template.render(**values)]
        return self.separator.join(parts)

    def to_message(self, role: str = "user", **values: Any) -> Message:
        return Message(role=role, content=self.render(**values))


class PromptRepository:
    """
    Named, versioned storage of Templates. `get(name)` returns the latest version
    (by registration order) unless a version is given. Optionally persisted to a
    JSON file. `ab(name, key)` picks a version deterministically per key — the same
    user always gets the same variant.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._store: dict[str, dict[str, Template]] = {}
        if self.path is not None and self.path.exists():
            for entry in json.loads(self.path.read_text(encoding="utf-8")):
                self.register(Template.from_dict(entry), save=False)

    def register(self, template: Template, *, save: bool = True) -> Template:
        if not template.name:
            raise ValueError("Only named templates can be registered.")
        self._store.setdefault(template.name, {})[template.version] = template
        if save:
            self.save()
        return template

    def add(self, name: str, text: str, *, version: str | None = None, **kwargs: Any) -> Template:
        """Register text as a new version of `name` (auto-numbered if no version is given)."""

        version = version or str(len(self._store.get(name, {})) + 1)
        return self.register(Template(text, name=name, version=version, **kwargs))

    def get(self, name: str, version: str | None = None) -> Template:
        versions = self._store.get(name)
        if not versions:
            raise KeyError(f"No prompt named {name!r}.")
        if version is None:
            return list(versions.values())[-1]
        if str(version) not in versions:
            raise KeyError(f"Prompt {name!r} has no version {version!r}; have {list(versions)}.")
        return versions[str(version)]

    def versions(self, name: str) -> list[str]:
        return list(self._store.get(name, {}))

    def names(self) -> list[str]:
        return list(self._store)

    def render(self, name: str, version: str | None = None, /, **values: Any) -> str:
        return self.get(name, version).render(**values)

    def ab(self, name: str, key: str, *, versions: Sequence[str] | None = None, weights: Sequence[float] | None = None) -> Template:
        """Pick a version for `key` by stable hashing — consistent across processes and runs."""

        candidates = list(versions or self.versions(name))
        if not candidates:
            raise KeyError(f"No prompt named {name!r}.")
        weights = list(weights or [1.0] * len(candidates))
        if len(weights) != len(candidates):
            raise ValueError("weights must match versions one-to-one.")
        point = int(hashlib.sha256(f"{name}:{key}".encode()).hexdigest()[:12], 16) / 16**12 * sum(weights)
        running = 0.0
        for version, weight in zip(candidates, weights):
            running += weight
            if point < running:
                return self.get(name, version)
        return self.get(name, candidates[-1])

    def save(self) -> None:
        if self.path is None:
            return
        data = [t.to_dict() for versions in self._store.values() for t in versions.values()]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")


__all__ = ["Template", "SystemPrompt", "FewShot", "Variables", "Var", "PromptRepository"]
