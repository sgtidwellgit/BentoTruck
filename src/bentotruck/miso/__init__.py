"""
miso — reflection: self-evaluation and iterative improvement of an agent's own output.

`Reflect`, `Improve`, and `Retry` wrap any Runnable (an agent, a team, a
workflow, or a plain function) and are Runnable themselves, so they stack:

    reliable = miso.Retry(miso.Reflect(agent), attempts=3, validator=miso.Verify.max_length(500))
    reliable.run("Draft the release note.")

`Critique` and `Verify` are the judging halves — usable standalone too.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bentotruck.nigiri import ModelProvider
from bentotruck.rice import invoke


def _provider_of(target: Any, provider: ModelProvider | None) -> ModelProvider | None:
    return provider or getattr(target, "provider", None)


@dataclass
class CritiqueResult:
    """A judgement of one answer: a 0–1 score, written feedback, and whether it clears the bar."""

    score: float
    feedback: str
    passed: bool
    raw: str = ""


_SCORE = re.compile(r"SCORE:\s*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+(?:\.\d+)?))?", re.IGNORECASE)
_FEEDBACK = re.compile(r"FEEDBACK:\s*(.*)", re.IGNORECASE | re.DOTALL)


class Critique:
    """
    Produces a structured critique of an answer — a score and feedback — without
    revising it. Uses a model as the judge.
    """

    def __init__(
        self,
        provider: ModelProvider | None = None,
        *,
        rubric: str = "correctness, completeness, and clarity",
        threshold: float = 0.8,
    ) -> None:
        self.provider = provider
        self.rubric = rubric
        self.threshold = threshold

    def prompt(self, goal: str, answer: str) -> str:
        return (
            f"You are a strict reviewer. Judge the answer to the task on {self.rubric}.\n\n"
            f"Task:\n{goal}\n\nAnswer:\n{answer}\n\n"
            "Reply in exactly this format:\nSCORE: <0-10>/10\nFEEDBACK: <specific, actionable problems, or 'none'>"
        )

    def critique(self, goal: str, answer: str, *, provider: ModelProvider | None = None) -> CritiqueResult:
        judge = provider or self.provider
        if judge is None:
            raise RuntimeError("Critique needs a model provider to act as the judge.")
        reply = judge.complete(self.prompt(goal, answer))
        return self.parse(reply)

    def parse(self, reply: str) -> CritiqueResult:
        match = _SCORE.search(reply)
        if match:
            value = float(match.group(1))
            scale = float(match.group(2)) if match.group(2) else (10.0 if value > 1 else 1.0)
            score = max(0.0, min(1.0, value / scale))
        else:
            score = 0.0
        feedback_match = _FEEDBACK.search(reply)
        feedback = feedback_match.group(1).strip() if feedback_match else reply.strip()
        return CritiqueResult(score=score, feedback=feedback, passed=score >= self.threshold, raw=reply)

    __call__ = critique


class Reflect:
    """
    Runs the target, has a critic review the answer, and revises it until the
    critique passes or `rounds` revisions have been made.

    The critic defaults to a `Critique` using the target agent's own model.
    """

    def __init__(
        self,
        target: Any,
        *,
        critic: Critique | None = None,
        rounds: int = 1,
        provider: ModelProvider | None = None,
    ) -> None:
        self.target = target
        self.critic = critic or Critique()
        self.rounds = rounds
        self.provider = provider
        self.history: list[tuple[str, CritiqueResult]] = []

    def run(self, goal: Any) -> str:
        goal = str(goal)
        model = _provider_of(self.target, self.provider)
        if model is None and self.critic.provider is None:
            raise RuntimeError("Reflect needs a model: wrap an Agent, or pass provider= or a critic with a provider.")
        self.history = []
        answer = str(invoke(self.target, goal))
        for _ in range(self.rounds):
            verdict = self.critic.critique(goal, answer, provider=self.critic.provider or model)
            self.history.append((answer, verdict))
            if verdict.passed:
                return answer
            answer = self._revise(goal, answer, verdict, model)
        return answer

    def _revise(self, goal: str, answer: str, verdict: CritiqueResult, model: ModelProvider | None) -> str:
        prompt = (
            f"Revise your answer to: {goal}\n\nYour previous answer:\n{answer}\n\n"
            f"A reviewer gave it {verdict.score:.0%} with this feedback:\n{verdict.feedback}\n\n"
            "Reply with the improved answer only."
        )
        if hasattr(self.target, "react"):
            return self.target.react(prompt)
        if model is not None:
            return model.complete(prompt)
        return str(invoke(self.target, prompt))


class Improve:
    """
    Iteratively revises an answer against a scoring function until it reaches
    `target_score` or `max_rounds` is spent. Returns the best answer seen.

    `scorer(goal, answer) -> float` is yours — a `yuzu` metric, a test suite, a
    length check, anything that returns a number where higher is better.
    """

    def __init__(
        self,
        target: Any,
        scorer: Callable[[str, str], float],
        *,
        target_score: float = 1.0,
        max_rounds: int = 3,
    ) -> None:
        self.target = target
        self.scorer = scorer
        self.target_score = target_score
        self.max_rounds = max_rounds
        self.history: list[tuple[str, float]] = []

    def run(self, goal: Any) -> str:
        goal = str(goal)
        self.history = []
        answer = str(invoke(self.target, goal))
        score = self.scorer(goal, answer)
        self.history.append((answer, score))
        for _ in range(self.max_rounds):
            if score >= self.target_score:
                break
            prompt = (
                f"{goal}\n\nYour previous answer scored {score:.2f} (target {self.target_score:.2f}):\n"
                f"{answer}\n\nImprove it. Reply with the improved answer only."
            )
            answer = str(invoke(self.target, prompt))
            score = self.scorer(goal, answer)
            self.history.append((answer, score))
        return max(self.history, key=lambda pair: pair[1])[0]


class RetryError(RuntimeError):
    """Raised when every Retry attempt failed. `.failures` holds the reason for each attempt."""

    def __init__(self, message: str, failures: list[str]) -> None:
        super().__init__(message)
        self.failures = failures


def _validate(validator: Callable[[str], Any] | None, answer: str) -> str | None:
    """Run a validator; return a failure reason, or None if the answer passed."""

    if validator is None:
        return None
    outcome = validator(answer)
    if outcome is True or outcome is None:
        return None
    if outcome is False:
        return "the answer failed validation"
    return str(outcome)


class Retry:
    """
    Re-runs a failed step with the failure fed back into the prompt. A step fails
    if it raises one of `retry_on`, or if `validator(answer)` returns False or a
    failure-reason string (True / None mean it passed).
    """

    def __init__(
        self,
        target: Any,
        *,
        attempts: int = 3,
        validator: Callable[[str], Any] | None = None,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
        backoff: float = 0.0,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least 1.")
        self.target = target
        self.attempts = attempts
        self.validator = validator
        self.retry_on = retry_on
        self.backoff = backoff
        self.failures: list[str] = []

    def run(self, goal: Any) -> Any:
        goal = str(goal)
        self.failures = []
        prompt = goal
        for attempt in range(1, self.attempts + 1):
            try:
                answer = invoke(self.target, prompt)
                reason = _validate(self.validator, str(answer))
            except self.retry_on as exc:  # type: ignore[misc]
                reason = f"{type(exc).__name__}: {exc}"
            if reason is None:
                return answer
            self.failures.append(reason)
            prompt = f"{goal}\n\nYour previous attempt failed: {reason}\nFix that and try again."
            if self.backoff and attempt < self.attempts:
                time.sleep(self.backoff * attempt)
        raise RetryError(f"All {self.attempts} attempts failed; last: {self.failures[-1]}", self.failures)


@dataclass
class VerificationResult:
    passed: bool
    failures: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


class VerificationError(ValueError):
    def __init__(self, result: VerificationResult) -> None:
        super().__init__("; ".join(result.failures))
        self.result = result


class Verify:
    """
    Checks an answer against a set of constraints or ground truth. Each check is
    `fn(answer)` returning True/None (pass), False, or a failure-reason string.

    Use standalone (`verify.check(answer)`), as a `Retry` validator (it's
    callable), or wrap a target so `run()` raises VerificationError on failure.
    """

    def __init__(self, *checks: Callable[[str], Any], target: Any = None) -> None:
        self.checks = list(checks)
        self.target = target

    def check(self, answer: Any) -> VerificationResult:
        failures = []
        for check in self.checks:
            reason = _validate(check, str(answer))
            if reason is not None:
                name = getattr(check, "__name__", type(check).__name__)
                failures.append(reason if reason != "the answer failed validation" else f"failed check {name!r}")
        return VerificationResult(passed=not failures, failures=failures)

    def __call__(self, answer: Any) -> Any:
        result = self.check(answer)
        return True if result.passed else "; ".join(result.failures)

    def run(self, goal: Any) -> Any:
        if self.target is None:
            raise RuntimeError("Verify.run needs a target — pass target= to wrap something.")
        answer = invoke(self.target, goal)
        result = self.check(answer)
        if not result.passed:
            raise VerificationError(result)
        return answer

    # --- ready-made checks ---------------------------------------------------

    @staticmethod
    def contains(*needles: str, case_sensitive: bool = False) -> Callable[[str], Any]:
        def contains(answer: str) -> Any:
            hay = answer if case_sensitive else answer.lower()
            missing = [n for n in needles if (n if case_sensitive else n.lower()) not in hay]
            return f"missing required text: {missing}" if missing else True

        return contains

    @staticmethod
    def excludes(*needles: str) -> Callable[[str], Any]:
        def excludes(answer: str) -> Any:
            found = [n for n in needles if n.lower() in answer.lower()]
            return f"contains forbidden text: {found}" if found else True

        return excludes

    @staticmethod
    def matches(pattern: str) -> Callable[[str], Any]:
        regex = re.compile(pattern, re.DOTALL)

        def matches(answer: str) -> Any:
            return True if regex.search(answer) else f"does not match pattern {pattern!r}"

        return matches

    @staticmethod
    def max_length(limit: int) -> Callable[[str], Any]:
        def max_length(answer: str) -> Any:
            return True if len(answer) <= limit else f"is {len(answer)} chars, limit is {limit}"

        return max_length

    @staticmethod
    def is_json() -> Callable[[str], Any]:
        def is_json(answer: str) -> Any:
            try:
                json.loads(answer)
            except ValueError as exc:
                return f"is not valid JSON ({exc})"
            return True

        return is_json

    @staticmethod
    def equals(expected: Any, *, normalize: Callable[[str], str] = lambda s: " ".join(s.lower().split())) -> Callable[[str], Any]:
        def equals(answer: str) -> Any:
            return True if normalize(str(answer)) == normalize(str(expected)) else f"expected {expected!r}"

        return equals


__all__ = [
    "Critique",
    "CritiqueResult",
    "Reflect",
    "Improve",
    "Retry",
    "RetryError",
    "Verify",
    "VerificationResult",
    "VerificationError",
]
