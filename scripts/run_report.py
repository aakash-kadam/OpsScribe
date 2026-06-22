#!/usr/bin/env python3
"""Generate report sections from a YAML report spec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import aha_ideas_overview, dataset_overview, run_cases_agent
from csv_parser import DEFAULT_DATA_DIR, load_cases_data
from json_parser import load_aha_ideas_for_data_path
from report_runner import (
    DEFAULT_REPORT_SPEC,
    describe_report_plan,
    date_context,
    filter_cases_by_date_range,
    load_report_spec,
    render_prompt,
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
        "--include-aha-ideas",
        action="store_true",
        help="Load Aha ideas JSON data and expose it to the agent.",
    )
    parser.add_argument(
        "--start-date",
        help="Optional inclusive start date for filtering cases and templating prompts.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional inclusive end date for filtering cases and templating prompts.",
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
        try:
            emit(
                describe_report_plan(
                    args.spec,
                    args.data_path,
                    include_aha_ideas=args.include_aha_ideas,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
            )
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(str(error)) from error
        return

    spec = load_report_spec(args.spec)
    try:
        template_values = date_context(args.start_date, args.end_date)
        df = filter_cases_by_date_range(load_cases_data(args.data_path), args.start_date, args.end_date)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    try:
        ideas_df = load_aha_ideas_for_data_path(args.data_path) if args.include_aha_ideas else None
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    emit(f"# {spec.title}\n")
    emit("## Dataset Overview\n")
    emit("```text")
    emit(dataset_overview(df))
    emit("```\n")

    if template_values:
        emit("## Date Range\n")
        emit(f"start_date: {template_values['start_date']}")
        emit(f"end_date: {template_values['end_date']}")
        emit()

    if ideas_df is not None:
        emit("## Aha Ideas Overview\n")
        emit("```text")
        emit(aha_ideas_overview(ideas_df))
        emit("```\n")

    for section in spec.sections:
        response = run_cases_agent(render_prompt(section.prompt, template_values), df=df, ideas_df=ideas_df)
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
