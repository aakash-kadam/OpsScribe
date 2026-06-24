"""DOCX writer for generated OpsScribe reports."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt

from report_models import ReportResult


def add_markdownish_content(document: Document, content: str) -> None:
    """Add basic markdown-like text to a Word document."""
    in_code_block = False

    for raw_line in content.splitlines():
        line = raw_line.strip()

        if not line:
            continue
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            paragraph = document.add_paragraph(line)
            for run in paragraph.runs:
                run.font.name = "Courier New"
                run.font.size = Pt(9)
            continue
        if line.startswith("### "):
            document.add_heading(line.removeprefix("### "), level=3)
        elif line.startswith("## "):
            document.add_heading(line.removeprefix("## "), level=2)
        elif line.startswith("# "):
            document.add_heading(line.removeprefix("# "), level=1)
        elif line.startswith("- "):
            document.add_paragraph(line.removeprefix("- "), style="List Bullet")
        elif line.startswith(tuple(f"{index}. " for index in range(1, 10))):
            _, item = line.split(". ", 1)
            document.add_paragraph(item, style="List Number")
        else:
            document.add_paragraph(line)


def write_report_docx(report: ReportResult, output_path: Path) -> None:
    """Write a generated report to a DOCX file."""
    document = Document()
    document.add_heading(report.title, level=0)

    document.add_heading("Dataset Overview", level=1)
    add_markdownish_content(document, report.dataset_overview)

    for section in report.sections:
        document.add_heading(section.title, level=1)
        add_markdownish_content(document, section.content)

    usage = report.token_usage
    if usage.available:
        document.add_heading("Token Usage", level=1)
        document.add_paragraph(f"Input tokens: {usage.input_tokens}")
        document.add_paragraph(f"Output tokens: {usage.output_tokens}")
        document.add_paragraph(f"Total tokens: {usage.total_tokens}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)
