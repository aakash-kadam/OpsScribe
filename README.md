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

`AI_BASE_URL` is also accepted as an alias for `AI_ENDPOINT`.

Run the LangChain agent:

```bash
uv run python scripts/analyze_cases_agent.py "Summarize the major support case trends."
```

Run a data-only check without calling the LLM:

```bash
uv run python scripts/analyze_cases_agent.py --dry-run
```

## Generate A Report

Report sections are defined in YAML under `report_specs/`.

Preview the report plan without calling the LLM:

```bash
uv run python scripts/run_report.py --dry-run
```

Generate the default markdown report:

```bash
AI_KEY="your-api-key" uv run python scripts/run_report.py
```

Generate markdown, Word, and PDF outputs:

```bash
AI_KEY="your-api-key" uv run python scripts/run_report.py --format md --format docx --format pdf
```

Generated reports are written to `outputs/`, which is ignored by git.

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

Use `AI_KEY_HEADER` when your provider or gateway requires the API key in a custom header such as `api-key` or `Ocp-Apim-Subscription-Key`. If omitted, the key is sent as `Authorization: Bearer <AI_KEY>`.

Use `AI_PAYLOAD_FORMAT=responses` or `AI_PAYLOAD_FORMAT=messages` to override auto-detection.

## Project Layout

```text
agents.py                       # LangChain agent and pandas-backed tools
csv_parser.py                   # CSV parsing and DataFrame loading
document_writer.py              # Markdown, DOCX, and PDF report writers
report_models.py                # Report dataclasses
report_runner.py                # YAML-driven report orchestration
report_specs/                   # YAML report section prompts
scripts/analyze_cases_agent.py  # CLI for running one agent prompt
scripts/parse_csv.py            # CLI for inspecting parsed CSV data
scripts/run_report.py           # CLI for generating reports
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
