#!/usr/bin/env python3
"""Generate report sections from a YAML report spec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csv_parser import DEFAULT_DATA_DIR
from report_runner import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_SPEC,
    describe_report_plan,
    run_report,
    write_report_outputs,
)


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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated reports (default: outputs).",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=["md", "docx", "pdf"],
        default=None,
        help="Output format to write. Repeat for multiple formats. Default: md.",
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
        print(describe_report_plan(args.spec, args.data_path))
        return

    report = run_report(
        spec_path=args.spec,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
    output_paths = write_report_outputs(report, args.formats or ["md"])

    print("Generated report outputs:")
    for output_path in output_paths:
        print(output_path)

    usage = report.token_usage
    if usage.available:
        print("\nToken usage:")
        print(f"Input tokens: {usage.input_tokens}")
        print(f"Output tokens: {usage.output_tokens}")
        print(f"Total tokens: {usage.total_tokens}")


if __name__ == "__main__":
    main()
