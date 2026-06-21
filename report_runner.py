"""Run YAML-defined report sections through the OpsScribe agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agents import aha_ideas_overview, dataset_overview, run_cases_agent
from csv_parser import DEFAULT_DATA_DIR, load_cases_data
from json_parser import load_aha_ideas_for_data_path
from report_models import ReportResult, ReportSectionResult, ReportSectionSpec, ReportSpec


DEFAULT_REPORT_SPEC = Path("report_specs/ops_manager_q2.yaml")


def load_report_spec(spec_path: Path = DEFAULT_REPORT_SPEC) -> ReportSpec:
    """Load a report definition from YAML."""
    with spec_path.open("r", encoding="utf-8") as spec_file:
        raw_spec = yaml.safe_load(spec_file) or {}

    sections = raw_spec.get("sections") or []
    if not sections:
        raise ValueError(f"Report spec has no sections: {spec_path}")

    return ReportSpec(
        title=str(raw_spec.get("title") or "OpsScribe Report"),
        sections=[parse_section(section) for section in sections],
    )


def parse_section(section: dict[str, Any]) -> ReportSectionSpec:
    """Parse one YAML section into a typed section spec."""
    missing_fields = [field for field in ("id", "title", "prompt") if not section.get(field)]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Report section is missing required fields: {missing}")

    return ReportSectionSpec(
        id=str(section["id"]),
        title=str(section["title"]),
        prompt=str(section["prompt"]),
    )


def render_report_markdown(report: ReportResult) -> str:
    """Render generated report sections into one markdown document."""
    lines = [f"# {report.title}", "", "## Dataset Overview", "", "```text"]
    lines.extend(report.dataset_overview.splitlines())
    lines.extend(["```", ""])

    for section in report.sections:
        lines.extend([f"## {section.title}", "", section.content.strip(), ""])

    usage = report.token_usage
    if usage.available:
        lines.extend(
            [
                "## Token Usage",
                "",
                f"Input tokens: {usage.input_tokens}",
                f"Output tokens: {usage.output_tokens}",
                f"Total tokens: {usage.total_tokens}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def run_report(
    spec_path: Path = DEFAULT_REPORT_SPEC,
    data_path: Path = DEFAULT_DATA_DIR,
    include_aha_ideas: bool = False,
) -> ReportResult:
    """Generate each report section with the agent."""
    spec = load_report_spec(spec_path)
    df = load_cases_data(data_path)
    ideas_df = load_aha_ideas_for_data_path(data_path) if include_aha_ideas else None
    section_results = []

    for section in spec.sections:
        response = run_cases_agent(section.prompt, df=df, ideas_df=ideas_df)
        section_results.append(
            ReportSectionResult(
                id=section.id,
                title=section.title,
                content=response.answer,
                token_usage=response.token_usage,
            )
        )

    return ReportResult(
        title=spec.title,
        sections=section_results,
        dataset_overview=dataset_overview(df),
    )


def describe_report_plan(
    spec_path: Path,
    data_path: Path,
    include_aha_ideas: bool = False,
) -> str:
    """Describe what a report run would do without calling the LLM."""
    spec = load_report_spec(spec_path)
    df = load_cases_data(data_path)
    lines = [f"Report: {spec.title}", "", dataset_overview(df), "", "Sections:"]

    if include_aha_ideas:
        ideas_df = load_aha_ideas_for_data_path(data_path)
        lines[3:3] = ["Aha Ideas Overview", aha_ideas_overview(ideas_df), ""]

    for index, section in enumerate(spec.sections, start=1):
        lines.append(f"{index}. {section.title} ({section.id})")

    return "\n".join(lines)
