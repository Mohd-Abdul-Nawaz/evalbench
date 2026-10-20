"""Command-line interface for EvalBench.

Usage:
    python -m evalbench.cli --scenarios scenarios/sample_scenarios.json --mock mixed
    python -m evalbench.cli --scenarios scenarios/sample_scenarios.json \
        --model gpt-4o-mini --base-url https://api.openai.com/v1

Writes eval-report.md, eval-dataset.json and eval-results.csv into --outdir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .llm import MockModel, OpenAICompatibleClient
from .report import run_report_to_csv, run_report_to_json, run_report_to_markdown
from .runner import Evaluator, load_scenarios, summarize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalbench", description="Run an evaluation batch.")
    parser.add_argument("--scenarios", required=True, help="Path to scenarios JSON file.")
    parser.add_argument("--outdir", default=".", help="Directory for report files.")
    parser.add_argument("--notes", default="", help="Evaluator notes to embed in reports.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mock", choices=["mixed", "pass", "fail"], help="Offline mock model.")
    source.add_argument("--model", help="Model name for an OpenAI-compatible API.")

    parser.add_argument("--base-url", default=None, help="API base URL (default: env or OpenAI).")
    parser.add_argument("--api-key", default=None, help="API key (default: OPENAI_API_KEY env var).")
    parser.add_argument("--fail-on-anomaly", action="store_true",
                        help="Exit with code 2 if any anomaly was flagged.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        scenarios = load_scenarios(args.scenarios)
    except (OSError, ValueError) as exc:
        print(f"error: could not load scenarios: {exc}", file=sys.stderr)
        return 1

    if args.mock:
        client = MockModel(mode=args.mock, latency_s=0)
        model_label = f"mock:{args.mock}"
    else:
        client = OpenAICompatibleClient(model=args.model, api_key=args.api_key,
                                        base_url=args.base_url)
        model_label = args.model

    evaluator = Evaluator(
        client, on_progress=lambda d, t, sid: print(f"[{d}/{t}] {sid}", flush=True)
    )
    results = evaluator.run(scenarios)
    summary = summarize(results)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "eval-report.md").write_text(
        run_report_to_markdown(results, model_label, args.notes), encoding="utf-8"
    )
    (outdir / "eval-dataset.json").write_text(
        run_report_to_json(results, model_label, args.notes), encoding="utf-8"
    )
    (outdir / "eval-results.csv").write_text(run_report_to_csv(results), encoding="utf-8")

    print(
        f"\nDone: {summary.passed}/{summary.total} passed, "
        f"{summary.anomalies} anomalies, mean score {summary.mean_score:.0%}. "
        f"Reports written to {outdir.resolve()}"
    )
    if args.fail_on_anomaly and summary.anomalies:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())