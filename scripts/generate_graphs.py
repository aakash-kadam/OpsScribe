#!/usr/bin/env python3
"""Generate graph images from OpsScribe case and optional Aha idea data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csv_parser import DEFAULT_DATA_DIR, load_cases_csv, load_cases_data
from json_parser import load_aha_ideas_for_data_path


DEFAULT_OUTPUT_DIR = Path("outputs/graphs")


def normalize_graph_column_name(column: object) -> str:
    return "_".join(str(column).strip().lower().replace("/", " ").split()).replace("(", "").replace(")", "")


def csv_paths(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_file():
        if path.suffix.lower() != ".csv":
            return []
        return [path]
    return sorted(path.glob("*.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OpsScribe graph images.")
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
        help="Directory for generated graph images (default: outputs/graphs).",
    )
    parser.add_argument(
        "--include-aha-ideas",
        action="store_true",
        help="Also generate graphs from data/aha_ideas.json.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Maximum categories to show in top-N charts (default: 15).",
    )
    return parser.parse_args()


def save_bar_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    output_path: Path,
    *,
    horizontal: bool = False,
) -> None:
    if data.empty:
        return

    plt.figure(figsize=(11, max(5, len(data) * 0.35) if horizontal else 6))
    if horizontal:
        sns.barplot(data=data, x=y_column, y=x_column, hue=x_column, palette="viridis", legend=False)
        plt.xlabel("Count")
        plt.ylabel("")
    else:
        sns.barplot(data=data, x=x_column, y=y_column, hue=x_column, palette="viridis", legend=False)
        plt.xlabel("")
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def value_counts_frame(series: pd.Series, name: str, top_n: int) -> pd.DataFrame:
    counts = series.fillna("Unknown").astype(str).value_counts().head(top_n)
    return counts.rename_axis(name).reset_index(name="count")


def exploded_value_counts_frame(df: pd.DataFrame, column: str, top_n: int) -> pd.DataFrame:
    values = df[column].explode() if column in df.columns else pd.Series(dtype="object")
    counts = values[values.notna() & (values != "")].astype(str).value_counts().head(top_n)
    return counts.rename_axis(column).reset_index(name="count")


def load_case_data_from_mixed_path(path: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Load case CSV exports, skipping other CSV data sources in mixed data folders."""
    if path.is_file():
        try:
            return load_cases_data(path), []
        except ValueError:
            return pd.DataFrame(), [path]

    frames = []
    skipped = []
    for csv_file in csv_paths(path):
        try:
            frame = load_cases_csv(csv_file)
        except ValueError:
            skipped.append(csv_file)
            continue

        frame.insert(0, "source_file", csv_file.name)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(), skipped
    return pd.concat(frames, ignore_index=True), skipped


def load_aha_link_csvs(path: Path) -> pd.DataFrame:
    """Load Aha linkage CSVs with Case, Idea, Date, Quarter, and Account fields."""
    frames = []
    for csv_file in csv_paths(path):
        df = pd.read_csv(csv_file)
        df = df.rename(columns=normalize_graph_column_name)

        columns = set(df.columns)
        if not {"case", "idea"}.issubset(columns):
            continue

        url_columns = [column for column in df.columns if column.startswith("unnamed")]
        rename_map = {
            "case": "case_number",
            "idea": "idea_id",
            "aha_s": "aha_id",
            "aha_w_links": "aha_link_id",
        }
        if url_columns:
            rename_map[url_columns[0]] = "url"

        df = df.rename(columns=rename_map)
        text_columns = df.select_dtypes(include=["object", "string"]).columns
        df[text_columns] = df[text_columns].apply(lambda series: series.str.strip())
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.insert(0, "source_file", csv_file.name)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def generate_case_graphs(df: pd.DataFrame, output_dir: Path, top_n: int) -> list[Path]:
    generated = []

    severity_counts = value_counts_frame(df["severity"], "severity", top_n)
    output_path = output_dir / "cases_by_severity.png"
    save_bar_chart(severity_counts, "severity", "count", "Cases by Severity", output_path)
    generated.append(output_path)

    root_cause_counts = value_counts_frame(df["root_cause"], "root_cause", top_n)
    output_path = output_dir / "cases_by_root_cause.png"
    save_bar_chart(root_cause_counts, "root_cause", "count", "Cases by Root Cause", output_path, horizontal=True)
    generated.append(output_path)

    component_counts = exploded_value_counts_frame(df, "components", top_n)
    output_path = output_dir / "cases_by_component.png"
    save_bar_chart(component_counts, "components", "count", "Cases by Component", output_path, horizontal=True)
    generated.append(output_path)

    triage_counts = exploded_value_counts_frame(df, "triage_components", top_n)
    output_path = output_dir / "cases_by_triage_component.png"
    save_bar_chart(triage_counts, "triage_components", "count", "Cases by Triage Component", output_path, horizontal=True)
    generated.append(output_path)

    account_counts = value_counts_frame(df["account_name"], "account_name", top_n)
    output_path = output_dir / "cases_by_account.png"
    save_bar_chart(account_counts, "account_name", "count", "Cases by Account", output_path, horizontal=True)
    generated.append(output_path)

    opened = df.dropna(subset=["date_time_opened"]).copy()
    if not opened.empty:
        monthly_counts = (
            opened.set_index("date_time_opened")
            .resample("ME")
            .size()
            .rename("count")
            .reset_index()
        )
        monthly_counts["month"] = monthly_counts["date_time_opened"].dt.strftime("%Y-%m")
        output_path = output_dir / "cases_opened_over_time.png"
        plt.figure(figsize=(11, 6))
        sns.lineplot(data=monthly_counts, x="month", y="count", marker="o")
        plt.title("Cases Opened Over Time")
        plt.xlabel("Month")
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        generated.append(output_path)

    if {"severity", "root_cause"}.issubset(df.columns):
        top_root_causes = root_cause_counts["root_cause"].tolist()
        filtered = df[df["root_cause"].fillna("Unknown").astype(str).isin(top_root_causes)].copy()
        filtered["root_cause"] = filtered["root_cause"].fillna("Unknown").astype(str)
        pivot = pd.crosstab(filtered["root_cause"], filtered["severity"])
        if not pivot.empty:
            output_path = output_dir / "root_cause_by_severity.png"
            pivot.plot(kind="barh", stacked=True, figsize=(11, max(5, len(pivot) * 0.4)), colormap="viridis")
            plt.title("Root Cause by Severity")
            plt.xlabel("Count")
            plt.ylabel("")
            plt.tight_layout()
            plt.savefig(output_path, dpi=160)
            plt.close()
            generated.append(output_path)

    return [path for path in generated if path.exists()]


def generate_aha_graphs(df: pd.DataFrame, output_dir: Path, top_n: int) -> list[Path]:
    generated = []
    candidate_columns = [
        "status",
        "workflow_status_name",
        "category",
        "product_name",
        "score",
    ]

    for column in candidate_columns:
        if column not in df.columns:
            continue

        counts = value_counts_frame(df[column], column, top_n)
        output_path = output_dir / f"aha_ideas_by_{column}.png"
        save_bar_chart(counts, column, "count", f"Aha Ideas by {column.replace('_', ' ').title()}", output_path, horizontal=True)
        if output_path.exists():
            generated.append(output_path)

    return generated


def generate_aha_link_graphs(df: pd.DataFrame, output_dir: Path, top_n: int) -> list[Path]:
    generated = []

    for column, title, filename in [
        ("idea_id", "Aha Links by Idea", "aha_links_by_idea.png"),
        ("account", "Aha Links by Account", "aha_links_by_account.png"),
        ("quarter", "Aha Links by Quarter", "aha_links_by_quarter.png"),
    ]:
        if column not in df.columns:
            continue

        counts = value_counts_frame(df[column], column, top_n)
        output_path = output_dir / filename
        save_bar_chart(counts, column, "count", title, output_path, horizontal=column != "quarter")
        if output_path.exists():
            generated.append(output_path)

    if "date" in df.columns:
        dated = df.dropna(subset=["date"]).copy()
        if not dated.empty:
            monthly_counts = dated.set_index("date").resample("ME").size().rename("count").reset_index()
            monthly_counts["month"] = monthly_counts["date"].dt.strftime("%Y-%m")
            output_path = output_dir / "aha_links_over_time.png"
            plt.figure(figsize=(11, 6))
            sns.lineplot(data=monthly_counts, x="month", y="count", marker="o")
            plt.title("Aha Links Over Time")
            plt.xlabel("Month")
            plt.ylabel("Count")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(output_path, dpi=160)
            plt.close()
            generated.append(output_path)

    if {"account", "idea_id"}.issubset(df.columns):
        unique_counts = (
            df.dropna(subset=["account", "idea_id"])
            .groupby("account")["idea_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(top_n)
            .rename("count")
            .rename_axis("account")
            .reset_index()
        )
        output_path = output_dir / "unique_aha_ideas_by_account.png"
        save_bar_chart(unique_counts, "account", "count", "Unique Aha Ideas by Account", output_path, horizontal=True)
        if output_path.exists():
            generated.append(output_path)

    return generated


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cases_df, skipped_case_csvs = load_case_data_from_mixed_path(args.data_path)
        aha_link_df = load_aha_link_csvs(args.data_path)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    generated = []
    if not cases_df.empty:
        generated.extend(generate_case_graphs(cases_df, args.output_dir, args.top_n))
    if not aha_link_df.empty:
        generated.extend(generate_aha_link_graphs(aha_link_df, args.output_dir, args.top_n))

    if args.include_aha_ideas:
        try:
            aha_df = load_aha_ideas_for_data_path(args.data_path)
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(str(error)) from error
        generated.extend(generate_aha_graphs(aha_df, args.output_dir, args.top_n))

    if not generated:
        skipped = ", ".join(path.name for path in skipped_case_csvs)
        detail = f" Skipped incompatible CSVs: {skipped}." if skipped else ""
        raise SystemExit(f"No supported data files found in: {args.data_path}.{detail}")

    print(f"Generated {len(generated)} graph(s):")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
