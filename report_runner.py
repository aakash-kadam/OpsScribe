"""Run YAML-defined report sections through the OpsScribe agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agents import dataset_overview, run_cases_agent
from csv_parser import DEFAULT_DATA_DIR, load_cases_data
from document_writer import write_docx_report, write_markdown_report, write_pdf_report
from report_models import ReportResult, ReportSectionResult, ReportSectionSpec, ReportSpec


DEFAULT_REPORT_SPEC = Path("report_specs/ops_manager_q2.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs")


def load_report_spec(spec_path: Path = DEFAULT_REPORT_SPEC) -> ReportSpec:
    """Load a report definition from YAML."""
    with spec_path.open("r", encoding="utf-8") as spec_file:
        raw_spec = yaml.safe_load(spec_file) or {}

    sections = raw_spec.get("sections") or []
    if not sections:
        raise ValueError(f"Report spec has no sections: {spec_path}")

    return ReportSpec(
        title=str(raw_spec.get("title") or "OpsScribe Report"),
        output_name=str(raw_spec.get("output_name") or "report"),
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
        output_format=str(section.get("output_format") or "markdown"),
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
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> ReportResult:
    """Generate each report section with the agent."""
    spec = load_report_spec(spec_path)
    df = load_cases_data(data_path)
    section_results = []

    for section in spec.sections:
        response = run_cases_agent(section.prompt, df=df)
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
        output_dir=output_dir,
        output_name=spec.output_name,
    )


def write_report_outputs(report: ReportResult, formats: list[str]) -> list[Path]:
    """Write a report to the requested output formats."""
    report.output_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_report_markdown(report)
    output_paths = []

    for output_format in formats:
        output_format = output_format.lower()
        output_path = report.output_dir / f"{report.output_name}.{output_format}"

        if output_format == "md":
            write_markdown_report(markdown, output_path)
        elif output_format == "docx":
            write_docx_report(markdown, output_path)
        elif output_format == "pdf":
            write_pdf_report(markdown, output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        output_paths.append(output_path)

    return output_paths


def describe_report_plan(spec_path: Path, data_path: Path) -> str:
    """Describe what a report run would do without calling the LLM."""
    spec = load_report_spec(spec_path)
    df = load_cases_data(data_path)
    lines = [f"Report: {spec.title}", "", dataset_overview(df), "", "Sections:"]

    for index, section in enumerate(spec.sections, start=1):
        lines.append(f"{index}. {section.title} ({section.id})")

    return "\n".join(lines)
