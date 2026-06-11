from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import REQUIRED_RAW_COLUMNS


def empty_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_RAW_COLUMNS + ["platform_id", "url", "query"])


def validate_raw_columns(df: pd.DataFrame, path: str | Path) -> None:
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"CSV {path} missing required columns: {missing_text}")


def load_csv_file(path: str | Path, default_source: str | None = None) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    if default_source and "source" not in df.columns:
        df["source"] = default_source
    validate_raw_columns(df, path)
    return df


def load_manual_csvs(manual_dir: str | Path) -> pd.DataFrame:
    manual_dir = Path(manual_dir)
    if not manual_dir.exists():
        return empty_raw_frame()

    frames = []
    for path in sorted(manual_dir.glob("*.csv")):
        frames.append(load_csv_file(path))

    if not frames:
        return empty_raw_frame()
    return pd.concat(frames, ignore_index=True)


def load_existing_raw_csvs(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        path = Path(path)
        if path.exists():
            frames.append(load_csv_file(path))
    if not frames:
        return empty_raw_frame()
    return pd.concat(frames, ignore_index=True)


def save_raw_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_raw_columns(df, path)
    df.to_csv(path, index=False)
    return path
