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

## Run The Agent

Set your LLM API key with `AI_KEY`:

```bash
export AI_KEY="your-api-key"
```

Run the LangChain agent:

```bash
uv run python scripts/analyze_cases_agent.py "Summarize the major support case trends."
```

Run a data-only check without calling the LLM:

```bash
uv run python scripts/analyze_cases_agent.py --dry-run
```

Optional environment variables:

```bash
export AI_MODEL="gpt-4o-mini"
export AI_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

## Project Layout

```text
agents.py                       # LangChain agent and pandas-backed tools
csv_parser.py                   # CSV parsing and DataFrame loading
scripts/analyze_cases_agent.py  # CLI for running the agent
scripts/parse_csv.py            # CLI for inspecting parsed CSV data
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
