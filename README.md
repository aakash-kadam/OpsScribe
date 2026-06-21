# OpsScribe

OpsScribe loads Ops Manager case CSV exports with pandas and provides a basic LangChain agent for case-data analysis.

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

The project expects Python `>=3.13`.

## Data

Place CSV exports in the local `data/` directory:

```text
data/*.csv
```

Files inside `data/` are ignored by git.

## Parse CSV Data

Preview parsed CSV data:

```bash
uv run python scripts/parse_csv.py
```

Preview a specific file:

```bash
uv run python scripts/parse_csv.py "data/FY2027Q2 Ops Manager Analysis - Cases.csv" --rows 10
```

The parser returns analysis-ready pandas DataFrames via:

```python
from csv_parser import load_cases_data

df = load_cases_data()
```

Stress test the parser with generated CSV data:

```bash
uv run python scripts/stress_test_csv_parser.py
```

Run a smaller stress test:

```bash
uv run python scripts/stress_test_csv_parser.py --rows 1000 --files 2
```

## Run The Agent

Set your LLM endpoint and API key with environment variables or a local `.env` file:

```bash
export AI_ENDPOINT="https://your-custom-chat-endpoint"
export AI_KEY="your-api-key"
export AI_MODEL="gpt-5.5"
```

`AI_BASE_URL` is also accepted as an alias for `AI_ENDPOINT`. `GROVE_API_KEY` is also accepted as an alias for `AI_KEY`.

Run the LangChain agent:

```bash
uv run python scripts/analyze_cases_agent.py "Summarize the major support case trends."
```

Include Aha ideas from JSON as an additional agent data source:

```bash
uv run python scripts/analyze_cases_agent.py "Find Aha ideas related to backup pain points." --include-aha-ideas
```

When this flag is used, the app expects this file to exist:

```text
data/aha_ideas.json
```

Run a data-only check without calling the LLM:

```bash
uv run python scripts/analyze_cases_agent.py --dry-run
```

## Run The YAML Analysis Pipeline

Report analysis sections are defined in YAML under `report_specs/`.

The pipeline is:

```text
data/*.csv -> csv_parser.py -> report_specs/*.yaml -> agents.py -> stdout
```

Preview the report plan without calling the LLM:

```bash
uv run python scripts/run_report.py --dry-run
```

Run the default analysis spec and print the generated report to stdout:

```bash
uv run python scripts/run_report.py
```

Use a different YAML spec or data path:

```bash
uv run python scripts/run_report.py --spec report_specs/ops_manager_q2.yaml --data-path data
```

Include Aha ideas from `data/*.json` in the YAML analysis run:

```bash
uv run python scripts/run_report.py --spec report_specs/ops_manager_q2.yaml --data-path data --include-aha-ideas
```

When `--include-aha-ideas` is provided, the app looks for `aha_ideas.json` inside the `--data-path` directory and exposes Aha-specific tools to inspect idea counts, search idea titles/descriptions, and fetch ideas by ID.

For example, with `--data-path data`, Aha ideas are loaded from:

```text
data/aha_ideas.json
```

Optional environment variables:

```bash
export AI_ENDPOINT="https://your-custom-chat-endpoint"
export AI_BASE_URL="https://your-custom-chat-endpoint"
export AI_MODEL="gpt-5.5"
export AI_KEY_HEADER="api-key"
export AI_PAYLOAD_FORMAT="auto"
export AI_TEMPERATURE="0.2"
```

The agent uses a generic LangChain `BaseChatModel` wrapper over HTTP. It auto-detects `/responses` endpoints and sends Responses-style `input` payloads; otherwise it sends chat-style `messages` payloads.

Use `AI_KEY_HEADER` when your provider or gateway requires a different API key header. If omitted, the key is sent as `api-key: <AI_KEY>`.

Use `AI_PAYLOAD_FORMAT=responses` or `AI_PAYLOAD_FORMAT=messages` to override auto-detection.

## Project Layout

```text
agents.py                       # LangChain agent and pandas-backed tools
csv_parser.py                   # CSV parsing and DataFrame loading
json_parser.py                  # JSON parsing and DataFrame loading
report_models.py                # Report dataclasses
report_runner.py                # YAML-driven analysis orchestration
report_specs/                   # YAML report section prompts
scripts/analyze_cases_agent.py  # CLI for running one agent prompt
scripts/parse_csv.py            # CLI for inspecting parsed CSV data
scripts/parse_json.py           # CLI for inspecting parsed JSON data
scripts/run_report.py           # CLI for running YAML analysis to stdout
scripts/stress_test_csv_parser.py # Generated-data stress test for csv_parser.py
```

## Token Usage

Agent runs print token usage when the provider returns usage metadata:

```text
Token usage:
Input tokens: ...
Output tokens: ...
Total tokens: ...
```

Hide token usage with:

```bash
uv run python scripts/analyze_cases_agent.py "Your question" --hide-token-usage
```
