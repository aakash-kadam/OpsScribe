"""LangChain agents for analyzing OpsScribe case data."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from csv_parser import DEFAULT_DATA_DIR, load_cases_data


DEFAULT_MODEL = "gpt-4o-mini"
ANALYSIS_COLUMNS = [
    "case_number",
    "severity",
    "root_cause",
    "subject",
    "ai_case_summary",
    "resolution",
    "components",
    "triage_components",
    "account_name",
    "date_time_opened",
]
GROUPABLE_COLUMNS = {
    "severity",
    "root_cause",
    "components",
    "triage_components",
    "account_name",
    "cloud_project",
}


@dataclass(frozen=True)
class TokenUsage:
    """Token usage aggregated across one agent run."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @property
    def available(self) -> bool:
        return self.total_tokens > 0


@dataclass(frozen=True)
class AgentResponse:
    """Agent answer plus token usage metadata."""

    answer: str
    token_usage: TokenUsage


def require_ai_key() -> str:
    """Read the LLM API key from AI_KEY."""
    api_key = os.getenv("AI_KEY")
    if not api_key:
        raise RuntimeError("Missing AI_KEY environment variable.")
    return api_key


def build_chat_model() -> ChatOpenAI:
    """Create the default LangChain chat model using AI_KEY."""
    return ChatOpenAI(
        model=os.getenv("AI_MODEL", DEFAULT_MODEL),
        api_key=require_ai_key(),
        base_url=os.getenv("AI_BASE_URL") or None,
        stream_usage=True,
        temperature=0,
    )


def extract_token_usage(agent_result: dict[str, Any]) -> TokenUsage:
    """Extract token usage from LangChain messages returned by an agent run."""
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for message in agent_result.get("messages", []):
        usage_metadata = getattr(message, "usage_metadata", None)
        if usage_metadata:
            input_tokens += usage_metadata.get("input_tokens", 0)
            output_tokens += usage_metadata.get("output_tokens", 0)
            total_tokens += usage_metadata.get("total_tokens", 0)
            continue

        response_metadata = getattr(message, "response_metadata", {}) or {}
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
        input_tokens += token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
        output_tokens += token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)
        total_tokens += token_usage.get("total_tokens", 0)

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def dataset_overview(df: pd.DataFrame) -> str:
    """Return a compact overview useful for both CLI output and agent context."""
    date_min = df["date_time_opened"].min()
    date_max = df["date_time_opened"].max()
    severity_counts = df["severity"].value_counts(dropna=False).to_string()
    missing_values = df.isna().sum().sort_values(ascending=False).head(8).to_string()

    return "\n".join(
        [
            f"Rows: {len(df)}",
            f"Columns: {len(df.columns)}",
            f"Date range: {date_min} to {date_max}",
            "",
            "Severity counts:",
            severity_counts,
            "",
            "Top missing-value columns:",
            missing_values,
        ]
    )


def _records_as_text(df: pd.DataFrame, limit: int) -> str:
    columns = [column for column in ANALYSIS_COLUMNS if column in df.columns]
    return df[columns].head(limit).to_json(orient="records", date_format="iso")


def build_case_analysis_tools(df: pd.DataFrame) -> list[Any]:
    """Build LangChain tools backed by the cases DataFrame."""

    @tool
    def get_dataset_overview() -> str:
        """Get row counts, date range, severity counts, and missing-value summary."""
        return dataset_overview(df)

    @tool
    def group_cases_by(column: str, limit: int = 10) -> str:
        """Group cases by a supported categorical column and return top counts."""
        if column not in GROUPABLE_COLUMNS:
            supported = ", ".join(sorted(GROUPABLE_COLUMNS))
            return f"Unsupported column: {column}. Supported columns: {supported}"

        if column in {"components", "triage_components"}:
            values = df[column].explode()
            counts = values[values.notna() & (values != "")].value_counts().head(limit)
        else:
            counts = df[column].fillna("Unknown").value_counts().head(limit)
        return counts.to_string()

    @tool
    def search_cases(query: str, limit: int = 5) -> str:
        """Search case text fields and return matching case records as JSON."""
        query = query.strip()
        if not query:
            return "Query cannot be empty."

        searchable_columns = [
            "subject",
            "ai_case_summary",
            "resolution",
            "root_cause",
            "components",
            "triage_components",
            "account_name",
        ]
        mask = pd.Series(False, index=df.index)
        for column in searchable_columns:
            values = df[column].apply(
                lambda value: "; ".join(value) if isinstance(value, list) else value
            )
            mask = mask | values.fillna("").str.contains(query, case=False, regex=False)

        matches = df[mask]
        if matches.empty:
            return f"No cases matched query: {query}"

        return _records_as_text(matches, limit)

    return [get_dataset_overview, group_cases_by, search_cases]


def build_cases_agent(data_path: Path = DEFAULT_DATA_DIR) -> Any:
    """Create a LangChain agent for analyzing case data."""
    df = load_cases_data(data_path)
    tools = build_case_analysis_tools(df)
    model = build_chat_model()

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are an operations support data analyst. Use the provided pandas-backed "
            "tools to analyze case trends, root causes, severity distribution, accounts, "
            "components, triage components, and case summaries. Cite counts from tools "
            "instead of guessing."
        ),
    )


def run_cases_agent(question: str, data_path: Path = DEFAULT_DATA_DIR) -> AgentResponse:
    """Run one question against the cases agent and return answer plus token usage."""
    agent = build_cases_agent(data_path)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return AgentResponse(
        answer=result["messages"][-1].content,
        token_usage=extract_token_usage(result),
    )


def ask_cases_agent(question: str, data_path: Path = DEFAULT_DATA_DIR) -> str:
    """Run one question against the cases agent and return the final response text."""
    return run_cases_agent(question, data_path=data_path).answer
