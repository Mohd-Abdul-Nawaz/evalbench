"""Tests for evalbench.runner (evaluation flow, scoring, anomalies)."""

import pytest

from evalbench.llm import MockModel
from evalbench.runner import Evaluator, Scenario, load_scenarios, score_output, summarize
from evalbench.checks import CheckResult


def _scenario(sid="s1", checks=(), category="general"):
    return Scenario(
        id=sid,
        task="task",
        system_prompt="",
        user_input="hello",
        checks=tuple(checks),
        category=category,
    )


def test_score_output_weighted():
    results = [
        CheckResult("contains", True, 2.0, "d"),
        CheckResult("regex", False, 1.0, "d"),
    ]
    assert score_output(results) == pytest.approx(2.0 / 3.0)


def test_score_output_no_checks_is_one():
    assert score_output([]) == 1.0


def test_evaluator_pass_and_fail_with_mock():
    passing = _scenario(
        "pass-case",
        checks=[{"type": "json_object"}, {"type": "key_exists", "key": "status"}],
    )
    failing = _scenario(
        "fail-case",
        checks=[{"type": "contains", "value": "unicorn-zookeeper"}],
    )
    evaluator = Evaluator(MockModel(mode="pass", latency_s=0))
    r_pass = evaluator.run_scenario(passing)
    r_fail = evaluator.run_scenario(failing)
    assert r_pass.passed and r_pass.weighted_score == 1.0
    assert not r_fail.passed and r_fail.weighted_score == 0.0
    assert not r_pass.anomaly and not r_fail.anomaly


def test_model_failure_becomes_anomaly():
    class Boom:
        def complete(self, system_prompt, user_input):
            raise ConnectionError("network down")

    result = Evaluator(Boom()).run_scenario(_scenario(checks=[{"type": "contains", "value": "x"}]))
    assert result.anomaly
    assert not result.passed
    assert "network down" in result.anomaly_note
    assert result.raw_output == ""


def test_empty_output_is_anomaly():
    class Empty:
        def complete(self, system_prompt, user_input):
            return "   "

    result = Evaluator(Empty()).run_scenario(_scenario(checks=[{"type": "contains", "value": "x"}]))
    assert result.anomaly
    assert "empty" in result.anomaly_note.lower()


def test_malformed_check_recorded_not_fatal():
    evaluator = Evaluator(MockModel(mode="mixed", latency_s=0))
    result = evaluator.run_scenario(_scenario(checks=[{"type": "bogus_check"}]))
    assert result.anomaly
    assert "check spec error" in result.anomaly_note


def test_progress_callback_fires_per_scenario():
    seen = []
    evaluator = Evaluator(
        MockModel(mode="mixed", latency_s=0),
        on_progress=lambda done, total, sid: seen.append((done, total, sid)),
    )
    evaluator.run([_scenario("a"), _scenario("b")])
    assert seen == [(1, 2, "a"), (2, 2, "b")]


def test_summarize_by_category():
    evaluator = Evaluator(MockModel(mode="pass", latency_s=0))
    results = evaluator.run([_scenario("a", [{"type": "json_object"}], "cat1"),
                             _scenario("b", [{"type": "json_object"}], "cat2")])
    summary = summarize(results)
    assert summary.total == 2
    assert summary.passed == 2
    assert summary.anomalies == 0
    assert set(summary.by_category) == {"cat1", "cat2"}


def test_load_scenarios_sample_file():
    scenarios = load_scenarios("scenarios/sample_scenarios.json")
    assert len(scenarios) >= 5
    ids = [s.id for s in scenarios]
    assert len(ids) == len(set(ids)), "scenario ids must be unique"
    for s in scenarios:
        assert s.checks, f"scenario {s.id} should have at least one check"


def test_scenario_from_dict_validation():
    with pytest.raises(ValueError):
        Scenario.from_dict({"id": "x", "task": "t"})  # missing user_input
