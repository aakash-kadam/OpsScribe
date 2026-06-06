"""CSV loading helpers for OpsScribe analysis agents."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path("data")

ANALYSIS_COLUMNS = [
    "case_number",
    "jira_links",
    "aha_links",
    "root_cause",
    "subject",
    "ai_case_summary",
    "resolution",
    "components",
    "triage_components",
    "account_name",
    "date_time_opened",
    "cloud_project",
    "severity",
]
LIST_COLUMNS = ["components", "triage_components"]

COLUMN_RENAMES = {
    "case number": "case_number",
    "case links total 1487": "case_link",
    "jira links": "jira_links",
    "aha links total 29": "aha_links",
    "root cause": "root_cause",
    "subject": "subject",
    "ai case summary total 1434 n a 53": "ai_case_summary",
    "resolution": "resolution",
    "components": "components",
    "triage components": "triage_components",
    "account name": "account_name",
    "date time opened": "date_time_opened",
    "cloud project": "cloud_project",
    "severity": "severity",
}


def normalize_column_name(column: object) -> str:
    """Convert exported report headers into stable snake_case names."""
    normalized = " ".join(str(column).replace("/", " ").replace("#", " ").split())
    normalized = "".join(char.lower() if char.isalnum() else " " for char in normalized)
    normalized = " ".join(normalized.split())
    return COLUMN_RENAMES.get(normalized, normalized.replace(" ", "_"))


def split_semicolon_list(value: object) -> list[str]:
    """Convert semicolon-delimited CSV values into clean string lists."""
    if pd.isna(value):
        return []

    return [item.strip() for item in str(value).split(";") if item.strip()]


def csv_files(path: Path = DEFAULT_DATA_DIR) -> list[Path]:
    """Return CSV files from a file path or directory path."""
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Not a CSV file: {path}")
        return [path]

    files = sorted(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {path}")
    return files


def load_cases_csv(csv_file: Path) -> pd.DataFrame:
    """Load one exported cases CSV as an analysis-ready pandas DataFrame."""
    df = pd.read_csv(csv_file)
    df = df.rename(columns=normalize_column_name)

    missing_columns = [column for column in ANALYSIS_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing expected columns in {csv_file}: {missing}")

    df = df[ANALYSIS_COLUMNS].copy()
    df["date_time_opened"] = pd.to_datetime(
        df["date_time_opened"],
        format="%m/%d/%Y %I:%M %p",
        errors="coerce",
    )

    text_columns = df.select_dtypes(include="object").columns
    df[text_columns] = df[text_columns].apply(lambda series: series.str.strip())

    for column in LIST_COLUMNS:
        df[column] = df[column].apply(split_semicolon_list)

    return df


def load_cases_data(path: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load all case CSV files from a path into one pandas DataFrame."""
    frames = []
    for csv_file in csv_files(path):
        frame = load_cases_csv(csv_file)
        frame.insert(0, "source_file", csv_file.name)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)
