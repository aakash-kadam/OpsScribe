#!/usr/bin/env python3
"""Parse and summarize OpsScribe CSV files with pandas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csv_parser import DEFAULT_DATA_DIR, csv_files, load_cases_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse and summarize CSV files.")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=DEFAULT_DATA_DIR,
        help="CSV file or folder of CSV files to parse (default: data)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of preview rows to print (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the parsed CSV back out. For folders, this must be a folder.",
    )
    return parser.parse_args()


def output_path_for(csv_file: Path, output: Path | None, total_files: int) -> Path | None:
    if output is None:
        return None

    if total_files == 1:
        return output

    if output.exists() and not output.is_dir():
        raise SystemExit("When parsing a folder, --output must be a folder.")

    output.mkdir(parents=True, exist_ok=True)
    return output / csv_file.name


def summarize_csv(csv_file: Path, rows: int, output: Path | None) -> None:
    df = load_cases_csv(csv_file)

    print(f"File: {csv_file}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    print(", ".join(df.columns))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print(f"\nPreview ({rows} rows):")
    print(df.head(rows))

    if output:
        df.to_csv(output, index=False)
        print(f"\nWrote parsed CSV to: {output}")


def main() -> None:
    args = parse_args()
    try:
        files = csv_files(args.path)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    for index, csv_file in enumerate(files):
        if index:
            print("\n" + "=" * 80 + "\n")

        try:
            summarize_csv(
                csv_file,
                rows=args.rows,
                output=output_path_for(csv_file, args.output, len(files)),
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
