"""Load raw feedback from a file (a real public dataset) or the synthetic generator.

Every source is mapped onto the same canonical columns:

    feedback_id, created_at, channel, plan, region, customer_segment, rating, text

Missing metadata columns are filled with ``"unknown"`` so the rest of the pipeline
never has to care where the data came from.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_generator import generate_feedback

CANONICAL_COLUMNS = [
    "feedback_id",
    "created_at",
    "channel",
    "plan",
    "region",
    "customer_segment",
    "rating",
    "text",
]


def load_file(path: str | Path, column_map: dict[str, str] | None = None, limit: int | None = None) -> pd.DataFrame:
    """Read a CSV or JSON-lines file (optionally .gz) and map it onto the canonical schema.

    ``column_map`` maps *source* column names to canonical names, e.g. for the
    Amazon Reviews 2023 dataset: ``{"timestamp": "created_at"}`` (``text`` and
    ``rating`` already have the right names). ``limit`` reads only the first rows,
    which is handy for multi-GB files.
    """
    suffixes = Path(path).suffixes
    if ".jsonl" in suffixes or ".json" in suffixes:
        df = pd.read_json(path, lines=True, nrows=limit)
    else:
        df = pd.read_csv(path, nrows=limit)
    if column_map:
        df = df.rename(columns=column_map)

    missing = {"created_at", "text"} - set(df.columns)
    if missing:
        raise ValueError(
            f"File is missing required columns after mapping: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}. Use --map SOURCE=TARGET."
        )

    if "feedback_id" not in df.columns:
        df["feedback_id"] = [f"FB{i:07d}" for i in range(len(df))]
    df["feedback_id"] = df["feedback_id"].astype(str)
    if "rating" not in df.columns:
        df["rating"] = pd.NA
    for col in ("channel", "plan", "region", "customer_segment"):
        if col not in df.columns:
            df[col] = "unknown"

    # numeric timestamps (e.g. Amazon's millisecond epochs) vs. date strings
    if pd.api.types.is_numeric_dtype(df["created_at"]):
        unit = "ms" if df["created_at"].max() > 1e11 else "s"
        df["created_at"] = pd.to_datetime(df["created_at"], unit=unit)
    else:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True, format="mixed").dt.tz_localize(
            None
        )

    return df[CANONICAL_COLUMNS]


def load_source(
    source: str = "sample",
    path: str | None = None,
    n: int = 5000,
    column_map: dict[str, str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    if source == "sample":
        return generate_feedback(n=n)
    if source in ("file", "csv"):
        if not path:
            raise ValueError("--path is required when --source file")
        return load_file(path, column_map, limit)
    raise ValueError(f"Unknown source: {source}")
