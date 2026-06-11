from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SENTIMENT_LABELS = ["positive", "neutral", "negative"]


def normalize_timestamps(df: pd.DataFrame, invalid_output_path: str | Path | None = None) -> pd.DataFrame:
    output = df.copy()

    # Timestamp handling is intentionally centralized here.
    # `utc=True` converts mixed platform timezones into one comparable timezone.
    # `errors="coerce"` turns invalid strings into NaT, so filtering never crashes.
    output["created_at_utc"] = pd.to_datetime(output["created_at"], utc=True, errors="coerce")

    invalid_mask = output["created_at"].notna() & output["created_at_utc"].isna()
    invalid_rows = output.loc[invalid_mask].copy()
    if invalid_output_path and not invalid_rows.empty:
        invalid_path = Path(invalid_output_path)
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_rows.to_csv(invalid_path, index=False)

    return output


def assign_periods(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = df.copy()
    output["period"] = pd.NA

    for period_name, period_config in config.get("periods", {}).items():
        start = pd.to_datetime(period_config["start"], utc=True)
        end = pd.to_datetime(period_config["end"], utc=True)

        # Start is inclusive and end is exclusive. This avoids double-counting
        # exact boundary times, especially 2025-06-01T00:00:00Z.
        mask = (output["created_at_utc"] >= start) & (output["created_at_utc"] < end)
        output.loc[mask, "period"] = period_name

    return output


def normalize_and_assign_periods(
    df: pd.DataFrame,
    config: dict[str, Any],
    invalid_output_path: str | Path | None = None,
) -> pd.DataFrame:
    normalized = normalize_timestamps(df, invalid_output_path=invalid_output_path)
    return assign_periods(normalized, config)


def deduplicate_comments(df: pd.DataFrame) -> pd.DataFrame:
    subset = [column for column in ["source", "username", "text", "created_at_utc"] if column in df.columns]
    if not subset:
        return df
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def sentiment_by_period(df: pd.DataFrame) -> pd.DataFrame:
    if "sentiment_label" not in df.columns:
        return pd.DataFrame(columns=["period", "sentiment_label", "count", "percentage"])

    scoped = df[df["period"].notna()].copy()
    if scoped.empty:
        return pd.DataFrame(columns=["period", "sentiment_label", "count", "percentage"])

    grouped = scoped.groupby(["period", "sentiment_label"], dropna=False).size().reset_index(name="count")
    totals = scoped.groupby("period").size().rename("total").reset_index()
    result = grouped.merge(totals, on="period", how="left")
    result["percentage"] = (result["count"] / result["total"] * 100).round(2)
    return result.drop(columns=["total"]).sort_values(["period", "sentiment_label"]).reset_index(drop=True)


def sentiment_by_platform(df: pd.DataFrame) -> pd.DataFrame:
    required = {"sentiment_label", "source"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["period", "source", "sentiment_label", "count", "percentage"])

    scoped = df[df["period"].notna()].copy()
    if scoped.empty:
        return pd.DataFrame(columns=["period", "source", "sentiment_label", "count", "percentage"])

    grouped = scoped.groupby(["period", "source", "sentiment_label"]).size().reset_index(name="count")
    totals = scoped.groupby(["period", "source"]).size().rename("total").reset_index()
    result = grouped.merge(totals, on=["period", "source"], how="left")
    result["percentage"] = (result["count"] / result["total"] * 100).round(2)
    return result.drop(columns=["total"]).sort_values(["period", "source", "sentiment_label"]).reset_index(drop=True)


def topic_distribution_by_period(df: pd.DataFrame) -> pd.DataFrame:
    if "dominant_topic" not in df.columns:
        return pd.DataFrame(columns=["period", "dominant_topic", "count", "percentage"])

    scoped = df[df["period"].notna() & df["dominant_topic"].notna()].copy()
    if scoped.empty:
        return pd.DataFrame(columns=["period", "dominant_topic", "count", "percentage"])

    grouped = scoped.groupby(["period", "dominant_topic"]).size().reset_index(name="count")
    totals = scoped.groupby("period").size().rename("total").reset_index()
    result = grouped.merge(totals, on="period", how="left")
    result["percentage"] = (result["count"] / result["total"] * 100).round(2)
    return result.drop(columns=["total"]).sort_values(["period", "dominant_topic"]).reset_index(drop=True)


def run_analysis(input_path: str | Path, config: dict[str, Any], results_dir: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    invalid_path = results_dir / "invalid_dates.csv"
    prepared = normalize_and_assign_periods(df, config, invalid_output_path=invalid_path)
    prepared = deduplicate_comments(prepared)

    prepared.to_csv(results_dir / "comments_analyzed.csv", index=False)
    sentiment_by_period(prepared).to_csv(results_dir / "sentiment_by_period.csv", index=False)
    sentiment_by_platform(prepared).to_csv(results_dir / "sentiment_by_platform.csv", index=False)
    topic_distribution_by_period(prepared).to_csv(results_dir / "topic_distribution_by_period.csv", index=False)
    return prepared
