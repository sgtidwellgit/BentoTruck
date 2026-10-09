from __future__ import annotations

import pytest

from bentotruck.gyoza import PythonTool
from bentotruck.nigiri import Mock, ModelResponse
from bentotruck.rice import Agent
from bentotruck.yuzu import (
    Confidence,
    Cost,
    Evaluator,
    Grounding,
    Hallucination,
    Latency,
    Quality,
    Sample,
    capture,
)


def test_capture_records_usage_tools_and_latency():
    replies = [
        ModelResponse.calling("python", expression="6 * 7"),
        ModelResponse(content="42", usage={"input_tokens": 10, "output_tokens": 2}),
    ]
    agent = Agent("a").using(Mock(replies=replies)).with_tools(PythonTool())
    sample = capture(agent, "6 * 7?", reference="42")
    assert sample.answer == "42"
    assert sample.context == ["42"]
    assert sample.usage == {"input_tokens": 10, "output_tokens": 2}
    assert sample.model_calls == 2 and sample.tool_calls == 1
    assert sample.latency is not None and sample.reference == "42"


def test_latency():
    assert Latency(target=2).score(Sample("g", "a", latency=1)).value == 1.0
    assert Latency(target=2).score(Sample("g", "a", latency=4)).value == 0.5
    with pytest.raises(ValueError):
        Latency().score(Sample("g", "a"))


def test_cost_uses_usage_or_estimates():
    metric = Cost(input_per_1k=1.0, output_per_1k=2.0, budget=0.01)
    result = metric.score(Sample("g", "a", usage={"input_tokens": 1000, "output_tokens": 1000}))
    assert result.details["cost"] == pytest.approx(3.0)
    assert result.value == pytest.approx(0.01 / 3.0)
    estimated = metric.score(Sample("g" * 40, "a" * 40))
    assert estimated.details["estimated"]
    assert (estimated.details["input_tokens"], estimated.details["output_tokens"]) == (10, 10)
    assert estimated.details["cost"] == pytest.approx(0.03)


def test_grounding():
    sample = Sample("q", "The truck opens at noon. Dragons guard the menu.", context=["The truck opens at noon daily."])
    result = Grounding().score(sample)
    assert result.value == 0.5
    assert result.details["unsupported"] == ["Dragons guard the menu."]
    assert Grounding().score(Sample("q", "x")).value == 0.0


def test_hallucination_checks_specific_claims():
    sample = Sample("When does Bento open?", "Bento opens at 11 in Austin.", context=["Bento opens at 11."])
    result = Hallucination().score(sample)
    assert result.details["unbacked"] == ["Austin"]
    assert 0 < result.value < 1
    assert Hallucination().score(Sample("q", "it is fine")).value == 1.0


def test_confidence_penalizes_hedges():
    assert Confidence().score(Sample("q", "It is 4.")).value == 1.0
    assert Confidence(penalty=0.25).score(Sample("q", "I think it might be 4, maybe.")).value == 0.25


def test_quality_reference_f1_and_judge():
    assert Quality().score(Sample("q", "the truck opens at noon", reference="truck opens noon")).value == pytest.approx(1.0)
    assert Quality().score(Sample("q", "banana", reference="truck")).value == 0.0
    judged = Quality(judge=Mock(reply="SCORE: 8/10\nFEEDBACK: fine")).score(Sample("q", "answer"))
    assert judged.value == pytest.approx(0.8)
    with pytest.raises(ValueError):
        Quality().score(Sample("q", "a"))


def test_metric_is_a_scorer():
    assert Confidence()("goal", "Certainly 4.") == 1.0


def test_evaluator_thresholds_and_suite():
    evaluator = Evaluator(Confidence(), Quality(), thresholds={"confidence": 0.9, "quality": 0.5})
    report = evaluator.evaluate(Sample("q", "maybe noon", reference="noon"))
    assert report.failed == ["confidence"] and not report.passed
    assert "confidence" in report.summary()

    agent = Agent("a").using(Mock(replies=["noon", "maybe noon"]))
    suite = evaluator.run_suite(agent, [{"goal": "when?", "reference": "noon"}, {"goal": "when?", "reference": "noon"}])
    assert suite.pass_rate == 0.5
    assert set(suite.averages) == {"confidence", "quality"}
    assert "pass rate 50%" in suite.summary()
