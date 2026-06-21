#!/usr/bin/env python3
"""Run a LangChain agent against OpsScribe case CSV data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import aha_ideas_overview, dataset_overview, run_cases_agent
from csv_parser import DEFAULT_DATA_DIR, load_cases_data
from json_parser import load_aha_ideas_for_data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze case CSV data with LangChain.")
    parser.add_argument(
        "question",
        nargs="?",
        default="Summarize the major support case trends in this dataset.",
        help="Question to ask the agent.",
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
        "--dry-run",
        action="store_true",
        help="Load the data and print an overview without calling the LLM.",
    )
    parser.add_argument(
        "--hide-token-usage",
        action="store_true",
        help="Do not print token usage after the agent response.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        df = load_cases_data(args.data_path)
        print(dataset_overview(df))
        if args.include_aha_ideas:
            ideas_df = load_aha_ideas_for_data_path(args.data_path)
            print("\nAha Ideas Overview:")
            print(aha_ideas_overview(ideas_df))
        return

    try:
        ideas_df = load_aha_ideas_for_data_path(args.data_path) if args.include_aha_ideas else None
        response = run_cases_agent(args.question, ideas_df=ideas_df, data_path=args.data_path)
    except (FileNotFoundError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    print(response.answer)

    if not args.hide_token_usage:
        usage = response.token_usage
        if usage.available:
            print("\nToken usage:")
            print(f"Input tokens: {usage.input_tokens}")
            print(f"Output tokens: {usage.output_tokens}")
            print(f"Total tokens: {usage.total_tokens}")
        else:
            print("\nToken usage: unavailable from provider response")


if __name__ == "__main__":
    main()
