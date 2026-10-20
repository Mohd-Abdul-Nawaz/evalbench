"""Tests for evalbench.report exporters."""

import csv
import io
import json

from evalbench.llm import MockModel
from evalbench.report import run_report_to_csv, run_report_to_json, run_report_to_markdown
from evalbench.runner import Evaluator, Scenario


def _run_batch():
    scenarios = [
        Scenario(
            id="ok-case",
            task="t",
            system_prompt="",
            user_input="u",
            checks=({"type": "json_object"},),
            category="format",
        ),
        Scenario(
            id="bad-case",
            task="t",
            system_prompt="",
            user_input="u",
            checks=({"type": "contains", "value": "never-in-mock-output"},),
            category="format",
        ),
    ]
    return Evaluator(MockModel(mode="pass", latency_s=0)).run(scenarios)


def test_json_report_structure():
    payload = json.loads(run_report_to_json(_run_batch(), model_label="mock:test", notes="n1"))
    assert payload["meta"]["model"] == "mock:test"
    assert payload["meta"]["notes"] == "n1"
    assert payload["summary"]["total"] == 2
    assert len(payload["results"]) == 2
    assert payload["results"][0]["scenario_id"] == "ok-case"
    assert payload["results"][0]["passed"] is True


def test_csv_report_round_trip():
    text = run_report_to_csv(_run_batch())
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 2
    assert rows[0]["scenario_id"] == "ok-case"
    assert rows[0]["passed"] == "True"
    assert {"weighted_score", "anomaly", "raw_output"} <= set(rows[0].keys())


def test_markdown_report_contains_sections():
    md = run_report_to_markdown(_run_batch(), model_label="mock:test", notes="run after reset")
    assert "# Evaluation Report" in md
    assert "## Scores by category" in md
    assert "## Anomalies" in md
    assert "## Scenario details" in md
    assert "run after reset" in md
    assert "ok-case" in md and "bad-case" in md