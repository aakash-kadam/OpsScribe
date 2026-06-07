#!/usr/bin/env python3
"""Stress test csv_parser.py with generated Ops Manager CSV-like data."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csv_parser import ANALYSIS_COLUMNS, load_cases_csv, load_cases_data


CSV_HEADERS = {
    "case_number": "Case\nNumber",
    "case_link": "Case Links\nTotal: 1487",
    "jira_links": "Jira\nLinks",
    "aha_links": "Aha Links\nTotal: 29",
    "root_cause": "Root Cause",
    "subject": "Subject",
    "ai_case_summary": "AI Case Summary\nTotal: 1434   #N/A: 53",
    "resolution": "Resolution",
    "components": "Components",
    "triage_components": "Triage Components",
    "account_name": "Account Name",
    "date_time_opened": "Date/Time Opened",
    "cloud_project": "Cloud Project",
    "severity": "Severity",
}

COMPONENTS = [
    "Ops Manager",
    "Ops Manager Monitoring",
    "Ops Manager Automation",
    "Headless Backups",
    "Backups: Snapshot Stores",
    "Kubernetes Operator",
    "MongoDB Enterprise Server",
]
ROOT_CAUSES = [
    "Acknowledged Product Defect",
    "Configuration",
    "Education/Usage",
    "Environment",
    "Undetermined",
]
SEVERITIES = ["S1", "S2", "S3", "S4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test csv_parser.py.")
    parser.add_argument(
        "--rows",
        type=int,
        default=50_000,
        help="Rows to generate per CSV file (default: 50000).",
    )
    parser.add_argument(
        "--files",
        type=int,
        default=3,
        help="Number of CSV files to generate (default: 3).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep generated temporary CSV files for inspection.",
    )
    return parser.parse_args()


def component_list(row_number: int, offset: int = 0) -> str:
    first = COMPONENTS[(row_number + offset) % len(COMPONENTS)]
    second = COMPONENTS[(row_number + offset + 2) % len(COMPONENTS)]
    third = COMPONENTS[(row_number + offset + 4) % len(COMPONENTS)]
    return f" {first} ; {second}; ; {third} "


def generate_csv(csv_path: Path, rows: int, file_index: int) -> None:
    """Generate one synthetic CSV with messy exported-report headers."""
    base_time = pd.Timestamp("2026-02-01 00:00")
    records = []

    for row_number in range(rows):
        case_number = file_index * 1_000_000 + row_number
        opened = base_time + pd.Timedelta(minutes=row_number + file_index)

        records.append(
            {
                CSV_HEADERS["case_number"]: case_number,
                CSV_HEADERS["case_link"]: case_number,
                CSV_HEADERS["jira_links"]: "" if row_number % 5 else f"OPS-{case_number}",
                CSV_HEADERS["aha_links"]: "" if row_number % 11 else f"AHA-{case_number}",
                CSV_HEADERS["root_cause"]: "" if row_number % 97 == 0 else ROOT_CAUSES[row_number % len(ROOT_CAUSES)],
                CSV_HEADERS["subject"]: f"Synthetic case subject {case_number}",
                CSV_HEADERS["ai_case_summary"]: "" if row_number % 29 == 0 else f"Synthetic summary for case {case_number}",
                CSV_HEADERS["resolution"]: "" if row_number % 7 else f"Synthetic resolution {case_number}",
                CSV_HEADERS["components"]: component_list(row_number),
                CSV_HEADERS["triage_components"]: "" if row_number % 3 == 0 else component_list(row_number, offset=1),
                CSV_HEADERS["account_name"]: f"Account {row_number % 250}",
                CSV_HEADERS["date_time_opened"]: opened.strftime("%m/%d/%Y %I:%M %p"),
                CSV_HEADERS["cloud_project"]: "" if row_number % 13 else f"project-{row_number % 20}",
                CSV_HEADERS["severity"]: SEVERITIES[row_number % len(SEVERITIES)],
            }
        )

    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_single_file(csv_path: Path, expected_rows: int) -> None:
    df = load_cases_csv(csv_path)

    assert_true(len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}")
    assert_true(list(df.columns) == ANALYSIS_COLUMNS, "Parsed columns do not match ANALYSIS_COLUMNS")
    assert_true("case_link" not in df.columns, "case_link should not be stored")
    assert_true(pd.api.types.is_datetime64_any_dtype(df["date_time_opened"]), "date_time_opened is not datetime")
    assert_true(df["date_time_opened"].isna().sum() == 0, "date_time_opened has parse failures")
    assert_true(df["components"].map(type).eq(list).all(), "components values are not lists")
    assert_true(df["triage_components"].map(type).eq(list).all(), "triage_components values are not lists")
    assert_true(df["triage_components"].map(len).min() == 0, "missing triage components should become empty lists")
    assert_true(df["components"].map(len).min() >= 3, "component list splitting dropped expected values")


def validate_multi_file(data_dir: Path, expected_rows: int, expected_files: int) -> None:
    df = load_cases_data(data_dir)

    assert_true(len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}")
    assert_true("source_file" in df.columns, "source_file column missing from load_cases_data")
    assert_true(df["source_file"].nunique() == expected_files, "source_file count does not match generated files")
    assert_true("case_link" not in df.columns, "case_link should not be stored")


def run_stress_test(rows: int, files: int, keep_temp: bool) -> None:
    temp_dir_context = None
    if keep_temp:
        data_dir = Path(tempfile.mkdtemp(prefix="opsscribe-csv-stress-"))
    else:
        temp_dir_context = tempfile.TemporaryDirectory(prefix="opsscribe-csv-stress-")
        data_dir = Path(temp_dir_context.name)

    try:
        start_generate = time.perf_counter()
        for file_index in range(files):
            generate_csv(data_dir / f"cases_{file_index}.csv", rows, file_index)
        generate_seconds = time.perf_counter() - start_generate

        start_parse = time.perf_counter()
        validate_single_file(data_dir / "cases_0.csv", rows)
        validate_multi_file(data_dir, rows * files, files)
        parse_seconds = time.perf_counter() - start_parse

        print("CSV parser stress test passed")
        print(f"Generated files: {files}")
        print(f"Rows per file: {rows}")
        print(f"Total parsed rows: {rows * files}")
        print(f"Generation time: {generate_seconds:.2f}s")
        print(f"Parse/validation time: {parse_seconds:.2f}s")
        print(f"Temporary data dir: {data_dir}")
    finally:
        if temp_dir_context is not None:
            temp_dir_context.cleanup()


def main() -> None:
    args = parse_args()
    run_stress_test(rows=args.rows, files=args.files, keep_temp=args.keep_temp)


if __name__ == "__main__":
    main()
