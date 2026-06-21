#!/usr/bin/env python3
"""Parse and summarize JSON files with pandas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from json_parser import DEFAULT_DATA_DIR, json_files, load_json_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse and summarize JSON files.")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=DEFAULT_DATA_DIR,
        help="JSON file or folder of JSON files to parse (default: data)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of preview rows to print (default: 5)",
    )
    return parser.parse_args()


def summarize_json(json_file: Path, rows: int) -> None:
    df = load_json_file(json_file)

    print(f"File: {json_file}")
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


def main() -> None:
    args = parse_args()
    try:
        files = json_files(args.path)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    for index, json_file in enumerate(files):
        if index:
            print("\n" + "=" * 80 + "\n")

        summarize_json(json_file, rows=args.rows)


if __name__ == "__main__":
    main()
