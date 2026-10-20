"""Evaluation runner: executes scenarios against a model and scores outputs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .checks import CheckResult, run_check
from .llm import ModelClient
from .scenarios import Scenario


@dataclass(frozen=True)
class EvaluationResult:
    """Result of running a single scenario.

    Attributes:
        scenario_id: The scenario's stable id.
        category: Scenario category (for grouped reports).
        raw_output: Exactly what the model returned ("" if the call failed).
        check_results: Per-check outcomes.
        weighted_score: Fraction of available check weight passed (0..1).
        passed: True when every check passed.
        anomaly: True when the run failed or output was empty.
        anomaly_note: Human-readable description of any anomaly.
        latency_s: Wall-clock model call duration in seconds.
        error: Exception text when the model call itself failed.
    """

    scenario_id: str
    category: str
    raw_output: str
    check_results: tuple[CheckResult, ...]
    weighted_score: float
    passed: bool
    anomaly: bool
    anomaly_note: str
    latency_s: float
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "raw_output": self.raw_output,
            "checks": [
                {
                    "type": c.check_type,
                    "passed": c.passed,
                    "weight": c.weight,
                    "description": c.description,
                    "details": c.details,
                }
                for c in self.check_results
            ],
            "weighted_score": round(self.weighted_score, 4),
            "passed": self.passed,
            "anomaly": self.anomaly,
            "anomaly_note": self.anomaly_note,
            "latency_s": round(self.latency_s, 3),
            "error": self.error,
        }


@dataclass(frozen=True)
class RunSummary:
    """Aggregate statistics for a batch of evaluation results."""

    total: int
    passed: int
    failed: int
    anomalies: int
    mean_score: float
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "anomalies": self.anomalies,
            "mean_score": round(self.mean_score, 4),
            "by_category": self.by_category,
        }


def score_output(check_results: Iterable[CheckResult]) -> float:
    """Weighted fraction of check weight passed; 1.0 when there are no checks."""
    results = list(check_results)
    total_weight = sum(r.weight for r in results)
    if total_weight == 0:
        return 1.0
    earned = sum(r.weight for r in results if r.passed)
    return earned / total_weight


class Evaluator:
    """Runs scenarios against a model client and collects scored results.

    Args:
        client: Any object with ``complete(system_prompt, user_input)``.
        on_progress: Optional callback ``(done, total, scenario_id)`` fired
            after each scenario, for UI progress display.
    """

    def __init__(
        self,
        client: ModelClient,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> None:
        self.client = client
        self.on_progress = on_progress

    def run_scenario(self, scenario: Scenario) -> EvaluationResult:
        """Run one scenario, score it, and flag anomalies."""
        start = time.perf_counter()
        raw_output = ""
        error = ""
        try:
            raw_output = self.client.complete(scenario.system_prompt, scenario.user_input)
        except Exception as exc:  # noqa: BLE001 - model errors become anomalies
            error = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - start

        check_results: list[CheckResult] = []
        check_errors: list[str] = []
        for spec in scenario.checks:
            try:
                check_results.append(run_check(spec, raw_output))
            except Exception as exc:  # malformed specs shouldn't kill a batch
                check_errors.append(f"check spec error ({spec.get('type')}): {exc}")

        score = score_output(check_results)
        all_passed = bool(check_results) and all(c.passed for c in check_results)

        anomaly_note = ""
        if error:
            anomaly_note = f"Model call failed: {error}"
        elif not raw_output.strip():
            anomaly_note = "Model returned empty output."
        elif check_errors:
            anomaly_note = "; ".join(check_errors)

        return EvaluationResult(
            scenario_id=scenario.id,
            category=scenario.category,
            raw_output=raw_output,
            check_results=tuple(check_results),
            weighted_score=score,
            passed=all_passed and not anomaly_note,
            anomaly=bool(anomaly_note),
            anomaly_note=anomaly_note,
            latency_s=latency,
            error=error,
        )

    def run(self, scenarios: Iterable[Scenario]) -> list[EvaluationResult]:
        """Run every scenario in order, invoking the progress callback."""
        scenarios = list(scenarios)
        results = []
        total = len(scenarios)
        for i, scenario in enumerate(scenarios, start=1):
            results.append(self.run_scenario(scenario))
            if self.on_progress:
                self.on_progress(i, total, scenario.id)
        return results


def summarize(results: Iterable[EvaluationResult]) -> RunSummary:
    """Aggregate results into a RunSummary with per-category breakdowns."""
    results = list(results)
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    anomalies = sum(1 for r in results if r.anomaly)
    mean_score = sum(r.weighted_score for r in results) / total if total else 0.0

    cats: dict[str, list[EvaluationResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    by_category = {
        cat: {
            "total": float(len(items)),
            "passed": float(sum(1 for r in items if r.passed)),
            "mean_score": round(sum(r.weighted_score for r in items) / len(items), 4),
        }
        for cat, items in sorted(cats.items())
    }
    return RunSummary(
        total=total,
        passed=passed,
        failed=total - passed,
        anomalies=anomalies,
        mean_score=mean_score,
        by_category=by_category,
    )


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load and validate scenarios from a JSON file (list or {"scenarios": [...]})."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("scenarios")
        if data is None:
            raise ValueError("Scenario file object must contain a 'scenarios' list")
    if not isinstance(data, list):
        raise ValueError("Scenario file must contain a list of scenarios")
    return [Scenario.from_dict(item) for item in data]
