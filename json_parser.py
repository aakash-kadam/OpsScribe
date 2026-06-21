"""JSON loading helpers for downstream OpsScribe agents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATA_DIR = Path("data")
DEFAULT_AHA_IDEAS_FILE = "aha_ideas.json"
RECORD_KEYS = ("ideas", "data", "records", "items", "results")


def normalize_column_name(column: object) -> str:
    """Convert JSON field names into stable snake_case names."""
    normalized = str(column).replace(".", "_").replace("/", "_").replace("-", "_")
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def json_files(path: Path = DEFAULT_DATA_DIR) -> list[Path]:
    """Return JSON files from a file path or directory path."""
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"Not a JSON file: {path}")
        return [path]

    files = sorted(path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {path}")
    return files


def aha_ideas_path_from_data_path(data_path: Path) -> Path:
    """Return the expected aha_ideas.json path for a data path."""
    if data_path.is_file():
        return data_path.parent / DEFAULT_AHA_IDEAS_FILE
    return data_path / DEFAULT_AHA_IDEAS_FILE


def records_from_json(payload: Any) -> list[dict[str, Any]]:
    """Extract records from common JSON export shapes."""
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]

    if isinstance(payload, dict):
        for key in RECORD_KEYS:
            records = payload.get(key)
            if isinstance(records, list):
                return [record for record in records if isinstance(record, dict)]
        return [payload]

    raise ValueError("JSON payload must be an object or a list of objects")


def extract_idea_id(url: object) -> str | None:
    """Extract Aha idea IDs like FF-I-123 from an idea URL."""
    if not isinstance(url, str):
        return None

    match = re.search(r"/ideas/([^/?#]+)", url)
    return match.group(1) if match else None


def load_json_file(json_file: Path) -> pd.DataFrame:
    """Load one JSON file as an analysis-ready pandas DataFrame."""
    with json_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    df = pd.json_normalize(records_from_json(payload), sep="_")
    df = df.rename(columns=normalize_column_name)

    text_columns = df.select_dtypes(include="object").columns
    df[text_columns] = df[text_columns].apply(
        lambda series: series.map(lambda value: value.strip() if isinstance(value, str) else value)
    )

    if "url" in df.columns and "idea_id" not in df.columns:
        df.insert(0, "idea_id", df["url"].map(extract_idea_id))

    return df


def load_json_data(path: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load all JSON files from a path into one pandas DataFrame."""
    frames = []
    for json_file in json_files(path):
        frame = load_json_file(json_file)
        frame.insert(0, "source_file", json_file.name)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def load_aha_ideas_data(path: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load Aha idea JSON exports into one pandas DataFrame."""
    return load_json_data(path)


def load_aha_ideas_for_data_path(data_path: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load data/aha_ideas.json for the provided data path."""
    aha_path = aha_ideas_path_from_data_path(data_path)
    if not aha_path.exists():
        raise FileNotFoundError(f"Aha ideas file not found: {aha_path}")
    return load_aha_ideas_data(aha_path)
