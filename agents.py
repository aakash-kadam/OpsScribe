"""LangChain agents for analyzing OpsScribe case data."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from csv_parser import DEFAULT_DATA_DIR, load_cases_data
from json_parser import load_aha_ideas_data


load_dotenv()


DEFAULT_MODEL = "gpt-5.5"
DEFAULT_AI_KEY_HEADER = "api-key"
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


class GenericEndpointChatModel(BaseChatModel):
    """LangChain chat model wrapper for custom HTTP chat endpoints."""

    endpoint_url: str
    model: str
    payload_format: str = "auto"
    api_key: str | None = None
    api_key_header: str = DEFAULT_AI_KEY_HEADER
    temperature: float | None = None
    timeout: float = 120
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "generic-endpoint-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"endpoint_url": self.endpoint_url, "model": self.model}

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "GenericEndpointChatModel":
        """Bind LangChain tools by passing their schemas to the endpoint payload."""
        return self.model_copy(update={"bound_tools": list(tools)})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop=stop)
        response = httpx.post(
            self.endpoint_url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise RuntimeError(f"AI endpoint request failed: {response.status_code} {response.text}") from error

        response_data = response.json()
        message = self._parse_response(response_data)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if not self.api_key:
            return headers

        if self.api_key_header.lower() == "authorization":
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers[self.api_key_header] = self.api_key
        return headers

    def _build_payload(self, messages: list[BaseMessage], stop: list[str] | None) -> dict[str, Any]:
        if self._uses_responses_payload():
            payload: dict[str, Any] = {
                "model": self.model,
                "input": serialize_responses_input(messages),
            }
        else:
            payload = {
                "model": self.model,
                "messages": [serialize_message(message) for message in messages],
            }

        if self.temperature is not None:
            payload["temperature"] = self.temperature

        if stop:
            payload["stop"] = stop
        if self.bound_tools:
            payload["tools"] = [self._serialize_tool(bound_tool) for bound_tool in self.bound_tools]
            payload["tool_choice"] = "auto"
        return payload

    def _parse_response(self, response_data: dict[str, Any]) -> AIMessage:
        message_data = extract_message_data(response_data)
        content = extract_response_content(message_data)
        tool_calls = normalize_tool_calls(
            message_data.get("tool_calls")
            or response_data.get("tool_calls")
            or extract_responses_tool_calls(response_data)
            or []
        )
        usage_metadata = normalize_usage(response_data.get("usage") or response_data.get("token_usage") or {})

        return AIMessage(
            content=content,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata or None,
            response_metadata=response_data,
        )

    def _serialize_tool(self, bound_tool: Any) -> dict[str, Any]:
        tool_schema = convert_to_openai_tool(bound_tool)
        if not self._uses_responses_payload():
            return tool_schema

        function = tool_schema.get("function", {})
        return {
            "type": "function",
            "name": function.get("name"),
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
        }

    def _uses_responses_payload(self) -> bool:
        if self.payload_format != "auto":
            return self.payload_format == "responses"
        return self.endpoint_url.rstrip("/").endswith("/responses")


def require_ai_key() -> str:
    """Read the LLM API key from AI_KEY or GROVE_API_KEY."""
    api_key = os.getenv("AI_KEY") or os.getenv("GROVE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing AI_KEY or GROVE_API_KEY environment variable.")
    return api_key


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    """Convert LangChain messages into generic chat endpoint message dictionaries."""
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, ToolMessage):
        role = "tool"
    else:
        role = getattr(message, "type", "user")

    payload: dict[str, Any] = {"role": role, "content": message.content or ""}

    if isinstance(message, AIMessage) and message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id

    return payload


def serialize_responses_input(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages into Responses API-style input items."""
    input_items = []
    function_call_ids = set()

    for message in messages:
        if isinstance(message, ToolMessage):
            call_id = message.tool_call_id
            if call_id not in function_call_ids:
                continue

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(message.content or ""),
                }
            )
            continue

        payload = serialize_message(message)
        tool_calls = payload.get("tool_calls") or []
        if tool_calls:
            for tool_call in tool_calls:
                call_id = str(tool_call.get("id"))
                function_call_ids.add(call_id)
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": tool_call.get("name"),
                        "arguments": json.dumps(tool_call.get("args", {})),
                    }
                )
            continue

        input_items.append({"role": payload["role"], "content": payload["content"]})

    return input_items


def extract_message_data(response_data: dict[str, Any]) -> dict[str, Any]:
    """Extract assistant message payload from common generic chat response shapes."""
    if isinstance(response_data.get("message"), dict):
        return response_data["message"]

    choices = response_data.get("choices")
    if choices and isinstance(choices, list):
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            if isinstance(first_choice.get("message"), dict):
                return first_choice["message"]
            return first_choice

    output = response_data.get("output")
    if output and isinstance(output, list):
        for output_item in output:
            if isinstance(output_item, dict) and output_item.get("type") == "message":
                return output_item

    return response_data


def extract_response_content(message_data: dict[str, Any]) -> str:
    """Extract text content from common chat response fields."""
    content = message_data.get("content") or message_data.get("text") or message_data.get("output") or ""
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or item.get("content") if isinstance(item, dict) else item)
            for item in content
        )
    return str(content)


def extract_responses_tool_calls(response_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract function calls from Responses API-style output arrays."""
    tool_calls = []
    output = response_data.get("output") or []
    for output_item in output:
        if not isinstance(output_item, dict) or output_item.get("type") != "function_call":
            continue

        tool_calls.append(
            {
                "id": output_item.get("call_id") or output_item.get("id"),
                "name": output_item.get("name"),
                "args": output_item.get("arguments") or {},
            }
        )
    return tool_calls


def normalize_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """Normalize endpoint tool calls to LangChain's expected shape."""
    normalized_calls = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue

        function = tool_call.get("function") or {}
        name = tool_call.get("name") or function.get("name")
        args = tool_call.get("args") or function.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}
        if not name:
            continue

        normalized_calls.append(
            {
                "id": str(tool_call.get("id") or f"tool_call_{index}"),
                "name": str(name),
                "args": args,
                "type": "tool_call",
            }
        )
    return normalized_calls


def normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    """Normalize provider token usage fields to LangChain usage metadata."""
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens) or 0

    if not total_tokens:
        return {}

    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }


def build_chat_model() -> GenericEndpointChatModel:
    """Create the LangChain chat model using a generic HTTP endpoint."""
    api_key = require_ai_key()
    endpoint_url = os.getenv("AI_ENDPOINT") or os.getenv("AI_BASE_URL")
    if not endpoint_url:
        raise RuntimeError("Missing AI_ENDPOINT or AI_BASE_URL environment variable.")

    temperature = os.getenv("AI_TEMPERATURE")

    return GenericEndpointChatModel(
        endpoint_url=endpoint_url,
        model=os.getenv("AI_MODEL", DEFAULT_MODEL),
        payload_format=os.getenv("AI_PAYLOAD_FORMAT", "auto"),
        api_key=api_key,
        api_key_header=os.getenv("AI_KEY_HEADER", DEFAULT_AI_KEY_HEADER),
        temperature=float(temperature) if temperature else None,
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


def aha_ideas_overview(df: pd.DataFrame) -> str:
    """Return a compact overview of loaded Aha ideas."""
    lines = [f"Rows: {len(df)}", f"Columns: {len(df.columns)}"]

    if "source_file" in df.columns:
        lines.extend(["", "Source files:", df["source_file"].value_counts().to_string()])

    if "idea_id" in df.columns:
        lines.extend(["", f"Unique idea IDs: {df['idea_id'].nunique()}"])

    missing_values = df.isna().sum().sort_values(ascending=False).head(8)
    if not missing_values.empty:
        lines.extend(["", "Top missing-value columns:", missing_values.to_string()])

    return "\n".join(lines)


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


def build_aha_ideas_tools(df: pd.DataFrame) -> list[Any]:
    """Build LangChain tools backed by the Aha ideas DataFrame."""

    @tool
    def get_aha_ideas_overview() -> str:
        """Get row counts, source files, idea counts, and missing-value summary for Aha ideas."""
        return aha_ideas_overview(df)

    @tool
    def search_aha_ideas(query: str, limit: int = 5) -> str:
        """Search Aha idea titles and descriptions and return matching ideas as JSON."""
        query = query.strip()
        if not query:
            return "Query cannot be empty."

        searchable_columns = [column for column in ("idea_id", "title", "description", "url") if column in df.columns]
        if not searchable_columns:
            return "No searchable Aha idea columns are available."

        mask = pd.Series(False, index=df.index)
        for column in searchable_columns:
            mask = mask | df[column].fillna("").astype(str).str.contains(query, case=False, regex=False)

        matches = df[mask]
        if matches.empty:
            return f"No Aha ideas matched query: {query}"

        output_columns = [column for column in ("idea_id", "title", "description", "url") if column in matches.columns]
        return matches[output_columns].head(limit).to_json(orient="records")

    @tool
    def get_aha_idea_by_id(idea_id: str) -> str:
        """Return one Aha idea by ID, such as FF-I-13638."""
        if "idea_id" not in df.columns:
            return "Aha ideas data does not include an idea_id column."

        matches = df[df["idea_id"].fillna("").str.casefold() == idea_id.strip().casefold()]
        if matches.empty:
            return f"No Aha idea found for ID: {idea_id}"

        output_columns = [column for column in ("idea_id", "title", "description", "url") if column in matches.columns]
        return matches[output_columns].head(1).to_json(orient="records")

    return [get_aha_ideas_overview, search_aha_ideas, get_aha_idea_by_id]


def build_cases_agent(
    df: pd.DataFrame | None = None,
    ideas_df: pd.DataFrame | None = None,
    data_path: Path = DEFAULT_DATA_DIR,
    ideas_path: Path = DEFAULT_DATA_DIR,
) -> Any:
    """Create a LangChain agent for analyzing case data and optional Aha ideas."""
    if df is None:
        df = load_cases_data(data_path)

    tools = build_case_analysis_tools(df)
    if ideas_df is not None:
        tools += build_aha_ideas_tools(ideas_df)

    model = build_chat_model()

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are an operations support data analyst. Use the provided pandas-backed "
            "tools to analyze case trends, root causes, severity distribution, accounts, "
            "components, triage components, and case summaries. If Aha idea tools are "
            "available, use them to connect support pain points to product feedback. "
            "Cite counts and idea IDs from tools instead of guessing."
        ),
    )


def run_cases_agent(
    question: str,
    df: pd.DataFrame | None = None,
    ideas_df: pd.DataFrame | None = None,
    data_path: Path = DEFAULT_DATA_DIR,
    ideas_path: Path = DEFAULT_DATA_DIR,
) -> AgentResponse:
    """Run one question against the cases agent and return answer plus token usage."""
    agent = build_cases_agent(df=df, ideas_df=ideas_df, data_path=data_path, ideas_path=ideas_path)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return AgentResponse(
        answer=result["messages"][-1].content,
        token_usage=extract_token_usage(result),
    )


def ask_cases_agent(
    question: str,
    df: pd.DataFrame | None = None,
    ideas_df: pd.DataFrame | None = None,
    data_path: Path = DEFAULT_DATA_DIR,
    ideas_path: Path = DEFAULT_DATA_DIR,
) -> str:
    """Run one question against the cases agent and return the final response text."""
    return run_cases_agent(question, df=df, ideas_df=ideas_df, data_path=data_path, ideas_path=ideas_path).answer
