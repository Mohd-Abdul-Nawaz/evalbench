# 🧪 EvalBench

**Evaluate AI models and agents against pre-defined scenarios — with scored checks, anomaly flags, and structured report exports.**

EvalBench is a lightweight evaluation harness built for AI evaluation workflows: you describe scenarios (prompt + pass/fail checks), point it at any OpenAI-compatible model endpoint, and it produces the three artefacts an evaluation engagement needs:

| Artefact | Format | Purpose |
|---|---|---|
| Findings report | Markdown | Human-readable written report with anomalies |
| Evaluation dataset | JSON | Machine-readable, submission-ready results |
| Results table | CSV | Spreadsheet-friendly score tracking |

It ships with a **demo mode** (deterministic offline mock model) so the full workflow can be exercised without an API key.

## Features

- **Pre-defined scenarios** — each scenario carries a system prompt, user input, category, and a list of declarative checks
- **16 rule-based check types** — `contains`, `not_contains`, `contains_any`, `contains_all`, `regex`, `not_regex`, `starts_with`, `ends_with`, `word_count`, `char_count`, `json_object`, `key_exists`, `required_headings`, `max_exclamations`, `placeholder_free`, plus weighted scoring
- **Anomaly detection** — failed model calls, empty outputs, and malformed check specs are flagged rather than crashing a batch
- **Works with any OpenAI-compatible API** — OpenAI, Together, Groq, Ollama, vLLM, or a local mock
- **Streamlit dashboard** — run evaluations, inspect per-check results, download reports
- **CLI-friendly core** — the `evalbench` package is pure Python and scriptable

## Quick start

```bash
git clone https://github.com/Mohd-Abdul-Nawaz/evalbench.git
cd evalbench
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Launch the dashboard in demo mode (no API key needed)
streamlit run app.py
```

## Using a real model

Set the environment variable and switch the sidebar to "OpenAI-compatible API":

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional, any compatible endpoint
streamlit run app.py
```

## Scripting the core

```python
from evalbench import Evaluator, MockModel, load_scenarios
from evalbench.report import run_report_to_markdown

scenarios = load_scenarios("scenarios/sample_scenarios.json")
results = Evaluator(MockModel(mode="mixed")).run(scenarios)
print(run_report_to_markdown(results, model_label="mock:mixed"))
```

## Scenario format

```json
{
  "id": "format-compliance-json-01",
  "task": "Model must return a strict JSON object with keys status and answer.",
  "system_prompt": "Always respond with a single valid JSON object and nothing else.",
  "user_input": "What is the capital of France? Respond ONLY as JSON.",
  "category": "format-compliance",
  "checks": [
    {"type": "json_object", "weight": 1.0},
    {"type": "key_exists", "key": "answer", "weight": 1.0}
  ]
}
```

## Tests

```bash
pytest -q
```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **New app**.
3. Pick the repo, branch `main`, and main file path `app.py`.
4. (Optional) Under *Advanced settings* add `OPENAI_API_KEY` as a secret — demo mode works without it.
5. Click **Deploy** — your app is live at `https://<app-name>.streamlit.app`.

## Project structure

```
evalbench/
├── app.py                     # Streamlit dashboard
├── evalbench/                 # Core library
│   ├── checks.py              # 15+ rule-based check implementations
│   ├── llm.py                 # MockModel + OpenAI-compatible client
│   ├── scenarios.py           # Scenario dataclass & validation
│   ├── runner.py              # Evaluator, scoring, anomaly detection
│   └── report.py              # JSON / CSV / Markdown exporters
├── scenarios/
│   └── sample_scenarios.json  # 6 ready-made evaluation scenarios
└── tests/                     # pytest suite
```

## License

MIT