"""EvalBench Streamlit dashboard.

Run predefined evaluation scenarios against an AI model, inspect scored
results and anomalies, and export structured reports (JSON / CSV / Markdown).

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (see README.md)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from evalbench import (
    Evaluator,
    MockModel,
    OpenAICompatibleClient,
    Scenario,
    __version__,
    load_scenarios,
    run_report_to_csv,
    run_report_to_json,
    run_report_to_markdown,
)
from evalbench.runner import summarize

SAMPLE_SCENARIOS_PATH = Path(__file__).parent / "scenarios" / "sample_scenarios.json"

st.set_page_config(
    page_title="EvalBench — AI Model Evaluation",
    page_icon="🧪",
    layout="wide",
)


def load_sample_scenarios() -> list[Scenario]:
    return load_scenarios(SAMPLE_SCENARIOS_PATH)


def scenario_editor_payload(scenarios: list[Scenario]) -> str:
    return json.dumps([s.to_dict() for s in scenarios], indent=2)


def parse_scenario_editor(text: str) -> list[Scenario]:
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("scenarios", [])
    return [Scenario.from_dict(item) for item in data]


def build_client(mode: str, api_config: dict):
    if mode == "Demo (offline mock model)":
        return MockModel(mode=api_config["mock_mode"]), f"mock:{api_config['mock_mode']}"
    if not api_config["api_key"]:
        st.error("API key required. Paste it in the sidebar, or switch to Demo mode.")
        st.stop()
    client = OpenAICompatibleClient(
        model=api_config["model"],
        api_key=api_config["api_key"],
        base_url=api_config["base_url"] or None,
        temperature=api_config["temperature"],
    )
    return client, f"{api_config['model']} @ {api_config['base_url'] or 'api.openai.com'}"


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("🧪 EvalBench v" + __version__)
    st.header("Configuration")

    mode = st.radio(
        "Model source",
        ["Demo (offline mock model)", "OpenAI-compatible API"],
        help="Demo mode needs no API key — useful for trying the workflow.",
    )

    mock_mode = "mixed"
    api_key = ""
    base_url = ""
    model_name = "gpt-4o-mini"
    temperature = 0.0

    if mode == "Demo (offline mock model)":
        mock_mode = st.selectbox(
            "Mock behaviour",
            ["mixed", "pass", "fail"],
            help="'mixed' deterministically mixes passing and failing answers.",
        )
    else:
        api_key = st.text_input(
            "API key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            help="Defaults to the OPENAI_API_KEY environment variable.",
        )
        base_url = st.text_input(
            "Base URL",
            value=os.environ.get("OPENAI_BASE_URL", ""),
            placeholder="https://api.openai.com/v1",
            help="Any OpenAI-compatible endpoint (Together, Groq, Ollama, vLLM...).",
        )
        model_name = st.text_input("Model", value=model_name)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)

    st.divider()
    st.header("Scenarios")
    scenario_source = st.radio("Source", ["Built-in sample set", "Edit / paste JSON"])
    editor_text = st.session_state.get(
        "scenario_editor", scenario_editor_payload(load_sample_scenarios())
    )
    if scenario_source == "Edit / paste JSON":
        editor_text = st.text_area(
            "Scenarios (JSON)",
            value=editor_text,
            height=320,
            help="List of scenario objects: id, task, user_input, checks[], category.",
        )
        st.session_state["scenario_editor"] = editor_text

    st.divider()
    run_notes = st.text_area(
        "Evaluator notes (goes into the report)",
        value=st.session_state.get("run_notes_input", ""),
        placeholder="e.g. Run 3 after usage-limit reset; testing refusal behaviour.",
    )
    st.session_state["run_notes_input"] = run_notes

    run_clicked = st.button("▶  Run evaluation", type="primary", width="stretch")

# ------------------------------------------------------------------- main ---
st.title("🧪 EvalBench")
st.caption(
    "Evaluate AI models and agents against pre-defined scenarios — scored checks, "
    "anomaly flags, and structured export-ready reports."
)

try:
    scenarios = (
        load_sample_scenarios()
        if scenario_source == "Built-in sample set"
        else parse_scenario_editor(editor_text)
    )
    st.session_state["scenario_editor"] = scenario_editor_payload(scenarios)
except (json.JSONDecodeError, ValueError) as exc:
    st.error(f"Scenario configuration is invalid: {exc}")
    st.stop()

col_a, col_b = st.columns([3, 1])
with col_a:
    st.subheader(f"Scenario set ({len(scenarios)} scenarios)")
    with st.expander("Preview scenarios and their checks"):
        for s in scenarios:
            st.markdown(f"**`{s.id}`** · _{s.category}_ — {s.task}")
            st.code(f"USER: {s.user_input}", language=None)
            if s.checks:
                st.markdown("Checks: " + ", ".join(f"`{c.get('type')}`" for c in s.checks))
with col_b:
    st.metric("Scenarios", len(scenarios))
    st.metric("Model mode", "mock" if mode.startswith("Demo") else model_name)

if run_clicked and scenarios:
    client, model_label = build_client(
        mode,
        {
            "mock_mode": mock_mode,
            "api_key": api_key,
            "base_url": base_url,
            "model": model_name,
            "temperature": temperature,
        },
    )
    progress = st.progress(0.0, text="Running scenarios…")
    evaluator = Evaluator(
        client,
        on_progress=lambda done, total, sid: progress.progress(
            done / total, text=f"Running {done}/{total}: {sid}"
        ),
    )
    results = evaluator.run(scenarios)
    progress.progress(1.0, text="Done.")
    st.session_state["results"] = results
    st.session_state["model_label"] = model_label
    st.session_state["run_notes"] = run_notes
    st.session_state["ran_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

results = st.session_state.get("results")

if not results:
    st.info("Configure the run in the sidebar, then press **Run evaluation**.")
    st.stop()

summary = summarize(results)

st.divider()
st.subheader("Results")
st.caption(f"Model: `{st.session_state['model_label']}` · Run at {st.session_state['ran_at']}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Scenarios", summary.total)
m2.metric("Passed", summary.passed)
m3.metric("Failed", summary.failed)
m4.metric("Anomalies", summary.anomalies)
m5.metric("Mean score", f"{summary.mean_score:.0%}")

tab_details, tab_anomalies, tab_export = st.tabs(
    ["📋 Scenario details", "⚠️ Anomalies", "📦 Export report"]
)

with tab_details:
    for r in results:
        if r.passed:
            icon, label = "✅", "PASS"
        elif r.anomaly:
            icon, label = "⚠️", "ANOMALY"
        else:
            icon, label = "❌", "FAIL"
        with st.expander(f"{icon} `{r.scenario_id}` — {label} · score {r.weighted_score:.0%}"):
            st.markdown(f"**Category:** {r.category} · **Latency:** {r.latency_s:.2f}s")
            if r.anomaly_note:
                st.warning(r.anomaly_note)
            if r.check_results:
                rows = [
                    {
                        "Check": c.description,
                        "Result": "pass" if c.passed else "FAIL",
                        "Detail": c.details,
                    }
                    for c in r.check_results
                ]
                st.dataframe(rows, hide_index=True)
            st.markdown("**Raw model output**")
            st.code(r.raw_output, language=None)

with tab_anomalies:
    anomalies = [r for r in results if r.anomaly]
    if not anomalies:
        st.success("No anomalies detected in this run.")
    else:
        for r in anomalies:
            st.error(f"**`{r.scenario_id}`** ({r.category}): {r.anomaly_note}")

with tab_export:
    st.markdown(
        "Download the structured artefacts of this run. "
        "The Markdown report is the written findings document; "
        "JSON/CSV are the evaluation dataset submissions."
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "📝 Markdown report",
        run_report_to_markdown(
            results, st.session_state["model_label"], st.session_state["run_notes"]
        ),
        file_name=f"eval-report-{stamp}.md",
        mime="text/markdown",
        width="stretch",
    )
    c2.download_button(
        "🧾 JSON dataset",
        run_report_to_json(
            results, st.session_state["model_label"], st.session_state["run_notes"]
        ),
        file_name=f"eval-dataset-{stamp}.json",
        mime="application/json",
        width="stretch",
    )
    c3.download_button(
        "📊 CSV table",
        run_report_to_csv(results),
        file_name=f"eval-results-{stamp}.csv",
        mime="text/csv",
        width="stretch",
    )
    if st.session_state["run_notes"]:
        st.caption(f"Notes included in report: {st.session_state['run_notes']}")
