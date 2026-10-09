from __future__ import annotations

import pytest

from bentotruck.miso import Critique, Improve, Reflect, Retry, RetryError, VerificationError, Verify
from bentotruck.nigiri import Mock
from bentotruck.rice import Agent


@pytest.mark.parametrize(
    "reply,score",
    [("SCORE: 7/10\nFEEDBACK: tighten it", 0.7), ("score: 9\nfeedback: none", 0.9), ("SCORE: 0.5", 0.5), ("no idea", 0.0)],
)
def test_critique_parses_scores(reply, score):
    result = Critique(Mock(reply=reply), threshold=0.8).critique("goal", "answer")
    assert result.score == pytest.approx(score)
    assert result.passed is (score >= 0.8)


def test_critique_requires_provider():
    with pytest.raises(RuntimeError):
        Critique().critique("g", "a")


def test_reflect_revises_until_critique_passes():
    provider = Mock(replies=["draft", "SCORE: 4/10\nFEEDBACK: add detail", "better draft", "SCORE: 9/10\nFEEDBACK: none"])
    reflect = Reflect(Agent("w").using(provider), rounds=3)
    assert reflect.run("write") == "better draft"
    assert [answer for answer, _ in reflect.history] == ["draft", "better draft"]


def test_reflect_returns_immediately_when_first_answer_passes():
    provider = Mock(replies=["great", "SCORE: 10/10\nFEEDBACK: none"])
    assert Reflect(Agent("w").using(provider)).run("go") == "great"
    assert len(provider.calls) == 2


def test_reflect_wraps_plain_function_with_provider():
    judge = Mock(replies=["SCORE: 2/10\nFEEDBACK: wrong", "fixed"])
    assert Reflect(lambda g: "bad", provider=judge).run("go") == "fixed"


def test_improve_keeps_best_answer():
    answers = iter(["a", "abc", "ab"])
    improve = Improve(lambda prompt: next(answers), scorer=lambda g, a: len(a) / 3, max_rounds=2)
    assert improve.run("goal") == "abc"
    assert [a for a, _ in improve.history] == ["a", "abc"]


def test_retry_feeds_failure_back_and_succeeds():
    prompts = []

    def flaky(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise ValueError("boom")
        return "ok"

    retry = Retry(flaky, attempts=3)
    assert retry.run("task") == "ok"
    assert "ValueError: boom" in prompts[1]


def test_retry_uses_validator_and_raises_after_attempts():
    retry = Retry(lambda p: "nope", attempts=2, validator=Verify.contains("yes"))
    with pytest.raises(RetryError) as info:
        retry.run("task")
    assert len(info.value.failures) == 2


def test_verify_checks_and_wraps():
    verify = Verify(Verify.max_length(10), Verify.matches(r"^\d+$"), Verify.excludes("secret"))
    assert verify.check("12345").passed
    result = verify.check("a secret value that is long")
    assert not result.passed and len(result.failures) == 3
    assert Verify(Verify.is_json()).check('{"a": 1}').passed
    assert Verify(Verify.equals("Hello  World")).check("hello world").passed
    with pytest.raises(VerificationError):
        Verify(Verify.contains("x"), target=lambda g: "y").run("go")
    assert Verify(lambda a: a == "y", target=lambda g: "y").run("go") == "y"


def test_wrappers_compose():
    provider = Mock(replies=["draft", "SCORE: 9/10\nFEEDBACK: none"])
    stack = Retry(Reflect(Agent("w").using(provider)), validator=Verify.contains("draft"))
    assert stack.run("go") == "draft"
