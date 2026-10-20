"""EvalBench: a lightweight harness for evaluating AI models and agents.

Run predefined scenarios against a model, score outputs with rule-based
checks, flag anomalies, and export structured evaluation reports.
"""

from .checks import CheckResult, run_check
from .llm import MockModel, OpenAICompatibleClient, ModelClient
from .report import run_report_to_csv, run_report_to_json, run_report_to_markdown
from .runner import EvaluationResult, Evaluator, RunSummary, load_scenarios
from .scenarios import Scenario

__version__ = "1.0.0"

__all__ = [
    "CheckResult",
    "run_check",
    "MockModel",
    "OpenAICompatibleClient",
    "ModelClient",
    "run_report_to_csv",
    "run_report_to_json",
    "run_report_to_markdown",
    "EvaluationResult",
    "Evaluator",
    "RunSummary",
    "load_scenarios",
    "Scenario",
    "__version__",
]