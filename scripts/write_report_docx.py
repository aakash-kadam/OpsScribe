#!/usr/bin/env python3
"""Generate an OpsScribe report and write it to a DOCX file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csv_parser import DEFAULT_DATA_DIR
from docx_writer import write_report_docx
from report_runner import DEFAULT_REPORT_SPEC, run_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an OpsScribe report as a DOCX file.")
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
        help="Load data/aha_ideas.json and expose it to the agent.",
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
        "--output",
        type=Path,
        default=Path("report.docx"),
        help="Output DOCX path (default: report.docx).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        report = run_report(
            spec_path=args.spec,
            data_path=args.data_path,
            include_aha_ideas=args.include_aha_ideas,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    write_report_docx(report, args.output)
    print(f"Wrote DOCX report to: {args.output}")


if __name__ == "__main__":
    main()
