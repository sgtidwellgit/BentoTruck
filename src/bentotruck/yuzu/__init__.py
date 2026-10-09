"""
yuzu — evaluation: metrics for judging agent runs and model outputs.

Every metric scores a `Sample` (goal, answer, retrieved context, timing, token
usage) on a 0–1 scale where higher is always better, with a written rationale.
`capture(agent, goal)` runs an agent and records a Sample from its events;
`Evaluator` applies a set of metrics with pass/fail thresholds, to one sample or
a whole test suite.

Metrics are also plain `scorer(goal, answer) -> float` callables, so any of
them can drive `miso.Improve`.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from bentotruck.edamame import STOPWORDS as _STOPWORDS
from bentotruck.nigiri import ModelProvider
from bentotruck.rice import Event, EventType


@dataclass
class Sample:
    """Everything a metric might look at for one run."""

    goal: str
    answer: str
    context: list[str] = field(default_factory=list)
    reference: str | None = None
    latency: float | None = None
    usage: dict[str, int] = field(default_factory=dict)
    model_calls: int = 0
    tool_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def capture(agent: Any, goal: str, *, reference: str | None = None, context: Iterable[str] = ()) -> Sample:
    """
    Run `agent` on `goal` and record a Sample: wall-clock latency, summed token
    usage, model/tool call counts, and every tool result as retrieved context.
    """

    usage: dict[str, int] = {}
    counts = {"model": 0, "tool": 0}
    gathered = list(context)

    def listen(event: Event) -> None:
        if event.type == EventType.MODEL_CALL:
            counts["model"] += 1
            for key, value in event.data.get("usage", {}).items():
                usage[key] = usage.get(key, 0) + value
        elif event.type == EventType.TOOL_RESULT:
            counts["tool"] += 1
            gathered.append(str(event.data.get("result", "")))

    agent.events.subscribe(listen)
    started = time.perf_counter()
    try:
        answer = agent.run(goal)
    finally:
        agent.events.unsubscribe(listen)
    return Sample(
        goal=goal,
        answer=answer,
        context=gathered,
        reference=reference,
        latency=time.perf_counter() - started,
        usage=usage,
        model_calls=counts["model"],
        tool_calls=counts["tool"],
    )


@dataclass
class MetricResult:
    name: str
    value: float
    rationale: str
    details: dict[str, Any] = field(default_factory=dict)


class Metric(ABC):
    """Base class: `score(sample) -> MetricResult`, values in [0, 1], higher is better."""

    name = "metric"

    @abstractmethod
    def score(self, sample: Sample) -> MetricResult:
        raise NotImplementedError

    def __call__(self, goal: str, answer: str) -> float:
        return self.score(Sample(goal=goal, answer=answer)).value


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


class Latency(Metric):
    """1.0 at or under `target` seconds, decaying as target / latency above it."""

    name = "latency"

    def __init__(self, target: float = 5.0) -> None:
        self.target = target

    def score(self, sample: Sample) -> MetricResult:
        if sample.latency is None:
            raise ValueError("Latency needs a measured sample — use yuzu.capture(agent, goal).")
        value = 1.0 if sample.latency <= self.target else self.target / sample.latency
        return MetricResult(
            self.name, value, f"{sample.latency:.2f}s against a {self.target:.2f}s target", {"seconds": sample.latency}
        )


class Cost(Metric):
    """
    Estimated spend from token usage and your per-1k-token prices. 1.0 within
    `budget`, decaying as budget / cost above it. When the provider reported no
    usage, tokens are estimated at ~4 characters each.
    """

    name = "cost"

    def __init__(self, *, input_per_1k: float, output_per_1k: float, budget: float) -> None:
        self.input_per_1k = input_per_1k
        self.output_per_1k = output_per_1k
        self.budget = budget

    def score(self, sample: Sample) -> MetricResult:
        estimated = not sample.usage
        input_tokens = sample.usage.get("input_tokens", (len(sample.goal) + sum(map(len, sample.context))) // 4)
        output_tokens = sample.usage.get("output_tokens", len(sample.answer) // 4)
        cost = input_tokens / 1000 * self.input_per_1k + output_tokens / 1000 * self.output_per_1k
        value = 1.0 if cost <= self.budget else self.budget / cost
        note = " (estimated)" if estimated else ""
        return MetricResult(
            self.name,
            value,
            f"{cost:.6f} spent{note} against a {self.budget:.6f} budget",
            {"cost": cost, "input_tokens": input_tokens, "output_tokens": output_tokens, "estimated": estimated},
        )


class Grounding(Metric):
    """
    Share of answer sentences supported by the context: a sentence counts as
    supported when at least `threshold` of its content words appear in the
    context (retrieved passages and tool results).
    """

    name = "grounding"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def score(self, sample: Sample) -> MetricResult:
        if not sample.context:
            return MetricResult(self.name, 0.0, "no context to ground against")
        vocabulary = set(_tokens(" ".join(sample.context)))
        sentences = [s for s in _sentences(sample.answer) if _tokens(s)]
        if not sentences:
            return MetricResult(self.name, 1.0, "answer makes no checkable statements")
        unsupported = []
        for sentence in sentences:
            words = _tokens(sentence)
            if sum(w in vocabulary for w in words) / len(words) < self.threshold:
                unsupported.append(sentence)
        value = 1 - len(unsupported) / len(sentences)
        return MetricResult(
            self.name,
            value,
            f"{len(sentences) - len(unsupported)}/{len(sentences)} sentences supported by context",
            {"unsupported": unsupported},
        )


_CLAIM = re.compile(r"\b\d[\d,.:%/-]*\b|\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


class Hallucination(Metric):
    """
    Checks specific claims — numbers, dates, and proper nouns — against the goal
    and context. 1.0 means every specific claim is backed; each unbacked claim
    lowers the score.
    """

    name = "hallucination"

    def score(self, sample: Sample) -> MetricResult:
        source = " ".join([sample.goal, *sample.context]).lower()
        claims = []
        for index, match in enumerate(_CLAIM.finditer(sample.answer)):
            token = match.group().rstrip(".,")
            sentence_start = match.start() == 0 or sample.answer[: match.start()].rstrip()[-1:] in ".!?\n"
            if token and not (sentence_start and token.isalpha() and token.lower() in _STOPWORDS | {"yes", "no"}):
                claims.append(token)
        claims = list(dict.fromkeys(claims))
        if not claims:
            return MetricResult(self.name, 1.0, "no specific claims to check")
        unbacked = [c for c in claims if c.lower() not in source]
        value = 1 - len(unbacked) / len(claims)
        return MetricResult(
            self.name, value, f"{len(claims) - len(unbacked)}/{len(claims)} specific claims backed", {"unbacked": unbacked}
        )


_HEDGES = (
    "i think", "i believe", "maybe", "perhaps", "possibly", "probably", "might be", "may be", "not sure",
    "i'm not certain", "it seems", "it appears", "unclear", "i don't know", "hard to say", "likely",
)


class Confidence(Metric):
    """How assertive the answer is: starts at 1.0 and loses `penalty` per hedging phrase."""

    name = "confidence"

    def __init__(self, penalty: float = 0.2) -> None:
        self.penalty = penalty

    def score(self, sample: Sample) -> MetricResult:
        lowered = sample.answer.lower()
        found = [h for h in _HEDGES if re.search(rf"\b{re.escape(h)}\b", lowered)]
        value = max(0.0, 1.0 - self.penalty * len(found))
        return MetricResult(self.name, value, f"{len(found)} hedging phrase(s)", {"hedges": found})


class Quality(Metric):
    """
    Overall answer quality. With a reference answer on the sample, it's token-level
    F1 against that reference (deterministic). Otherwise a `judge` model scores it
    against `rubric` via `miso.Critique`.
    """

    name = "quality"

    def __init__(self, judge: ModelProvider | None = None, *, rubric: str = "correctness, completeness, and clarity") -> None:
        self.judge = judge
        self.rubric = rubric

    def score(self, sample: Sample) -> MetricResult:
        if sample.reference is not None:
            predicted, expected = _tokens(sample.answer), _tokens(sample.reference)
            common = sum(min(predicted.count(t), expected.count(t)) for t in set(predicted))
            if not predicted or not expected or not common:
                return MetricResult(self.name, 0.0, "no overlap with the reference answer")
            precision, recall = common / len(predicted), common / len(expected)
            f1 = 2 * precision * recall / (precision + recall)
            return MetricResult(self.name, f1, f"token F1 {f1:.2f} vs reference", {"precision": precision, "recall": recall})
        if self.judge is None:
            raise ValueError("Quality needs either sample.reference or a judge= model provider.")
        from bentotruck.miso import Critique

        verdict = Critique(self.judge, rubric=self.rubric).critique(sample.goal, sample.answer)
        return MetricResult(self.name, verdict.score, verdict.feedback, {"judge": verdict.raw})


@dataclass
class Report:
    """Metric results for one sample, plus which ones fell below their threshold."""

    sample: Sample
    results: dict[str, MetricResult]
    failed: list[str]

    @property
    def passed(self) -> bool:
        return not self.failed

    @property
    def scores(self) -> dict[str, float]:
        return {name: result.value for name, result in self.results.items()}

    def summary(self) -> str:
        lines = []
        for name, result in self.results.items():
            mark = "FAIL" if name in self.failed else "ok"
            lines.append(f"{name:<14} {result.value:5.2f}  {mark:<4}  {result.rationale}")
        return "\n".join(lines)


@dataclass
class SuiteReport:
    reports: list[Report]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.reports)

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.reports) / len(self.reports) if self.reports else 0.0

    @property
    def averages(self) -> dict[str, float]:
        names = {name for r in self.reports for name in r.results}
        return {
            name: sum(r.results[name].value for r in self.reports if name in r.results)
            / sum(1 for r in self.reports if name in r.results)
            for name in sorted(names)
        }

    def summary(self) -> str:
        lines = [f"{len(self.reports)} cases, pass rate {self.pass_rate:.0%}"]
        lines += [f"  {name:<14} {value:5.2f}" for name, value in self.averages.items()]
        return "\n".join(lines)


class Evaluator:
    """
    Applies metrics with thresholds. `thresholds` is a single float for every
    metric or a {name: float} mapping; a metric below its threshold fails the sample.
    """

    def __init__(self, *metrics: Metric, thresholds: float | Mapping[str, float] = 0.7) -> None:
        if not metrics:
            raise ValueError("Evaluator needs at least one metric.")
        self.metrics = list(metrics)
        self.thresholds = thresholds

    def _threshold(self, name: str) -> float:
        if isinstance(self.thresholds, Mapping):
            return self.thresholds.get(name, 0.0)
        return float(self.thresholds)

    def evaluate(self, sample: Sample) -> Report:
        results = {metric.name: metric.score(sample) for metric in self.metrics}
        failed = [name for name, result in results.items() if result.value < self._threshold(name)]
        return Report(sample=sample, results=results, failed=failed)

    def run_suite(self, agent: Any, cases: Iterable[Mapping[str, Any] | str]) -> SuiteReport:
        """
        Run `agent` over test cases and evaluate each. A case is a goal string or a
        dict with "goal" and optional "reference" / "context".
        """

        reports = []
        for case in cases:
            case = {"goal": case} if isinstance(case, str) else case
            sample = capture(agent, case["goal"], reference=case.get("reference"), context=case.get("context", ()))
            reports.append(self.evaluate(sample))
        return SuiteReport(reports)


__all__ = [
    "Sample",
    "capture",
    "Metric",
    "MetricResult",
    "Latency",
    "Cost",
    "Grounding",
    "Hallucination",
    "Confidence",
    "Quality",
    "Evaluator",
    "Report",
    "SuiteReport",
]
