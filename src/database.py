"""Database access layer (SQLAlchemy): schema creation, loading data and named queries."""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import DATABASE_URL, SQL_DIR

FEEDBACK_COLUMNS = [
    "feedback_id",
    "created_at",
    "feedback_date",
    "feedback_week",
    "feedback_month",
    "channel",
    "plan",
    "region",
    "customer_segment",
    "rating",
    "nps_group",
    "text",
    "clean_text",
    "word_count",
]
ANALYSIS_COLUMNS = [
    "feedback_id",
    "sentiment_label",
    "sentiment_score",
    "sentiment_confidence",
    "topic_id",
    "complaint_category",
    "is_complaint",
]


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or DATABASE_URL, future=True)


def _split_statements(sql: str) -> list[str]:
    sql = re.sub(r"--[^\n]*", "", sql)
    return [s.strip() for s in sql.split(";") if s.strip()]


def init_schema(engine: Engine) -> None:
    """(Re)create all tables from sql/schema.sql."""
    statements = _split_statements((SQL_DIR / "schema.sql").read_text())
    with engine.begin() as conn:
        for stmt in statements:
            if engine.dialect.name == "postgresql" and stmt.upper().startswith("DROP TABLE"):
                stmt += " CASCADE"
            conn.execute(text(stmt))


def write_table(engine: Engine, df: pd.DataFrame, table: str, columns: list[str] | None = None) -> int:
    """Append a dataframe to an existing table (schema is owned by schema.sql)."""
    data = df[columns] if columns else df
    data.to_sql(table, engine, if_exists="append", index=False, chunksize=1000, method="multi")
    return len(data)


@lru_cache(maxsize=1)
def load_named_queries() -> dict[str, str]:
    """Parse sql/analytics_queries.sql into {name: sql}."""
    content = (SQL_DIR / "analytics_queries.sql").read_text()
    queries: dict[str, str] = {}
    for block in re.split(r"^-- name:\s*", content, flags=re.MULTILINE)[1:]:
        name, _, body = block.partition("\n")
        queries[name.strip()] = body.strip().rstrip(";")
    return queries


def run_query(engine: Engine, name: str, **params) -> pd.DataFrame:
    sql = load_named_queries()[name]
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def read_feedback_full(engine: Engine) -> pd.DataFrame:
    df = run_query(engine, "feedback_full")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["feedback_date"] = pd.to_datetime(df["feedback_date"])
    df["feedback_week"] = pd.to_datetime(df["feedback_week"])
    return df
