"""Structured report exporters for evaluation runs.

Produces the written artefacts an evaluation engagement requires:
JSON (machine-readable dataset), CSV (spreadsheet-friendly), and
Markdown (human-readable findings report).
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Iterable

from .runner import EvaluationResult, RunSummary, summarize


def _build_payload(
    results: list[EvaluationResult],
    model_label: str,
    notes: str,
) -> dict:
    summary = summarize(results)
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model_label,
            "notes": notes,
        },
        "summary": summary.to_dict(),
        "results": [r.to_dict() for r in results],
    }


def run_report_to_json(
    results: Iterable[EvaluationResult],
    model_label: str = "unknown",
    notes: str = "",
    indent: int = 2,
) -> str:
    """Full structured report as a JSON string."""
    payload = _build_payload(list(results), model_label, notes)
    return json.dumps(payload, indent=indent, ensure_ascii=False)


def run_report_to_csv(results: Iterable[EvaluationResult]) -> str:
    """One row per scenario with pass/fail, score, anomaly flag and output."""
    results = list(results)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "scenario_id",
            "category",
            "passed",
            "weighted_score",
            "anomaly",
            "anomaly_note",
            "latency_s",
            "checks_total",
            "checks_passed",
            "raw_output",
        ]
    )
    for r in results:
        writer.writerow(
            [
                r.scenario_id,
                r.category,
                r.passed,
                round(r.weighted_score, 4),
                r.anomaly,
                r.anomaly_note,
                round(r.latency_s, 3),
                len(r.check_results),
                sum(1 for c in r.check_results if c.passed),
                r.raw_output,
            ]
        )
    return buffer.getvalue()


def run_report_to_markdown(
    results: Iterable[EvaluationResult],
    model_label: str = "unknown",
    notes: str = "",
) -> str:
    """Human-readable findings report with summary, anomalies, and detail."""
    results = list(results)
    summary: RunSummary = summarize(results)
    lines = [
        "# Evaluation Report",
        "",
        f"- **Model:** {model_label}",
        f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **Scenarios:** {summary.total} | **Passed:** {summary.passed} | "
        f"**Failed:** {summary.failed} | **Anomalies:** {summary.anomalies}",
        f"- **Mean weighted score:** {summary.mean_score:.1%}",
    ]
    if notes:
        lines += ["", f"> {notes}"]

    lines += ["", "## Scores by category", "", "| Category | Scenarios | Passed | Mean score |", "|---|---|---|---|"]
    for cat, stats in summary.by_category.items():
        lines.append(
            f"| {cat} | {int(stats['total'])} | {int(stats['passed'])} | {stats['mean_score']:.1%} |"
        )

    anomalies = [r for r in results if r.anomaly]
    lines += ["", "## Anomalies", ""]
    if anomalies:
        for r in anomalies:
            lines.append(f"- **{r.scenario_id}** ({r.category}): {r.anomaly_note}")
    else:
        lines.append("_No anomalies detected._")

    lines += ["", "## Scenario details", ""]
    for r in results:
        badge = "✅ PASS" if r.passed else ("⚠️ ANOMALY" if r.anomaly else "❌ FAIL")
        lines += [
            f"### {r.scenario_id} — {badge}",
            "",
            f"- Category: {r.category}",
            f"- Weighted score: {r.weighted_score:.1%}",
            f"- Latency: {r.latency_s:.2f}s",
        ]
        if r.anomaly_note:
            lines.append(f"- Anomaly: {r.anomaly_note}")
        if r.check_results:
            lines += ["", "| Check | Result | Detail |", "|---|---|---|"]
            for c in r.check_results:
                mark = "pass" if c.passed else "FAIL"
                lines.append(f"| {c.description} | {mark} | {c.details} |")
        lines += ["", "<details><summary>Raw output</summary>", "", "```", r.raw_output, "```", "", "</details>", ""]
    return "\n".join(lines)