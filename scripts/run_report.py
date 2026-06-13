#!/usr/bin/env python3
"""Generate report sections from a YAML report spec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import dataset_overview, run_cases_agent
from csv_parser import DEFAULT_DATA_DIR, load_cases_data
from report_runner import (
    DEFAULT_REPORT_SPEC,
    describe_report_plan,
    load_report_spec,
)


def emit(text: str = "") -> None:
    print(text, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an OpsScribe report from YAML prompts.")
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_REPORT_SPEC,
        help="YAML report spec to run.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="CSV file or folder of CSV files to analyze (default: data).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the report plan without calling the LLM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        emit(describe_report_plan(args.spec, args.data_path))
        return

    spec = load_report_spec(args.spec)
    df = load_cases_data(args.data_path)
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    emit(f"# {spec.title}\n")
    emit("## Dataset Overview\n")
    emit("```text")
    emit(dataset_overview(df))
    emit("```\n")

    for section in spec.sections:
        response = run_cases_agent(section.prompt, df=df)
        emit(f"## {section.title}\n")
        emit(response.answer.strip())
        emit()

        total_input_tokens += response.token_usage.input_tokens
        total_output_tokens += response.token_usage.output_tokens
        total_tokens += response.token_usage.total_tokens

    if total_tokens:
        emit("## Token Usage\n")
        emit(f"Input tokens: {total_input_tokens}")
        emit(f"Output tokens: {total_output_tokens}")
        emit(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    main()
