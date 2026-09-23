"""End-to-end pipeline:

    raw feedback -> preprocessing -> database -> sentiment + topics + categories -> KPIs

Usage:
    python -m src.pipeline                              # synthetic sample data
    python -m src.pipeline --n 20000                    # bigger sample
    python -m src.pipeline --source file --path data/Cell_Phones_and_Accessories.jsonl.gz \
        --map timestamp=created_at --limit 50000
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import pandas as pd

from src import analytics
from src.config import MODELS_DIR, REPORTS_DIR
from src.database import ANALYSIS_COLUMNS, FEEDBACK_COLUMNS, get_engine, init_schema, read_feedback_full, write_table
from src.ingest import load_source
from src.preprocessing import preprocess
from src.sentiment import SentimentClassifier, TransformerSentiment, train_and_evaluate
from src.topic_modeling import TopicModeler, categorize_series

log = logging.getLogger("pipeline")


def run(
    source: str = "sample",
    path: str | None = None,
    n: int = 5000,
    column_map: dict[str, str] | None = None,
    limit: int | None = None,
    database_url: str | None = None,
    n_topics: int = 10,
    topic_backend: str = "nmf",
    sentiment_backend: str = "tfidf",
    save_artifacts: bool = True,
) -> dict:
    engine = get_engine(database_url)

    log.info("1/6 Loading data (source=%s)", source)
    raw = load_source(source, path=path, n=n, column_map=column_map, limit=limit)

    log.info("2/6 Preprocessing %d raw rows", len(raw))
    df = preprocess(raw)
    log.info("    %d rows after cleaning/deduplication", len(df))

    log.info("3/6 Sentiment model (%s)", sentiment_backend)
    metrics = None
    n_labelled = int(df["rating_sentiment"].notna().sum())
    if sentiment_backend == "tfidf" and n_labelled < 100:
        raise ValueError(
            f"Only {n_labelled} rows have a star rating to learn sentiment from. "
            "Map a rating column with --map, or use --sentiment-backend transformer."
        )
    if sentiment_backend == "transformer":
        sentiment = TransformerSentiment().predict(df["text"])
        model_name = TransformerSentiment.name
    else:
        model, metrics = train_and_evaluate(df)
        sentiment = model.predict(df["clean_text"])
        model_name = SentimentClassifier.name
        log.info("    hold-out accuracy=%.3f macro-F1=%.3f", metrics.accuracy, metrics.macro_f1)
        if save_artifacts:
            model.save(MODELS_DIR / "sentiment_tfidf_logreg.joblib")

    log.info("4/6 Complaint categorisation and topic modeling (%s)", topic_backend)
    df = df.reset_index(drop=True)
    analysis = pd.concat([df[["feedback_id"]], sentiment.reset_index(drop=True)], axis=1)
    analysis["complaint_category"] = categorize_series(df["clean_text"]).to_numpy()
    # a complaint = negative sentiment, or a low rating even if the text sounds neutral
    analysis["is_complaint"] = ((analysis["sentiment_label"] == "negative") | (df["rating"] <= 2)).astype(int)

    # topics are discovered on complaints only: that is where "what keeps going wrong" lives,
    # and it stops positive praise from dominating the vocabulary
    complaint_mask = analysis["is_complaint"] == 1
    topic_model = TopicModeler(n_topics=n_topics, backend=topic_backend)
    analysis["topic_id"] = pd.Series(pd.NA, index=analysis.index, dtype="Int64")
    analysis.loc[complaint_mask, "topic_id"] = topic_model.fit_transform(df.loc[complaint_mask, "clean_text"])

    log.info("5/6 Writing to database (%s)", engine.url.render_as_string(hide_password=True))
    init_schema(engine)
    write_table(engine, df, "feedback", FEEDBACK_COLUMNS)
    write_table(engine, topic_model.topics_, "topics")
    write_table(engine, analysis, "feedback_analysis", ANALYSIS_COLUMNS)
    if metrics is not None:
        run_at = datetime.now(timezone.utc).replace(tzinfo=None)
        write_table(
            engine,
            pd.DataFrame(
                [
                    {
                        "run_id": run_at.strftime("%Y%m%d%H%M%S"),
                        "run_at": run_at,
                        "model_name": model_name,
                        "n_train": metrics.n_train,
                        "n_test": metrics.n_test,
                        "accuracy": metrics.accuracy,
                        "macro_f1": metrics.macro_f1,
                        "metrics_json": json.dumps(
                            {"report": metrics.report, "confusion": metrics.confusion, "labels": metrics.labels}
                        ),
                    }
                ]
            ),
            "model_runs",
        )

    log.info("6/6 KPIs and summary")
    full = read_feedback_full(engine)
    kpis = analytics.compute_kpis(full)
    summary = analytics.summarize_findings(full)
    if save_artifacts:
        (REPORTS_DIR / "summary.md").write_text("# Automated CX summary\n\n" + summary + "\n")
        (REPORTS_DIR / "kpis.json").write_text(json.dumps(kpis, indent=2))
    return {"kpis": kpis, "summary": summary, "metrics": metrics, "topics": topic_model.topics_}


def _parse_map(pairs: list[str]) -> dict[str, str]:
    return dict(p.split("=", 1) for p in pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the customer-feedback analytics pipeline.")
    parser.add_argument("--source", choices=["sample", "file", "csv"], default="sample")
    parser.add_argument("--path", help="CSV / JSON-lines file (optionally .gz) when --source file")
    parser.add_argument("--limit", type=int, help="Read only the first N rows of the file")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="SRC=DEST",
        help="Rename a source column to a canonical one, e.g. --map review_text=text",
    )
    parser.add_argument("--n", type=int, default=5000, help="Number of synthetic records")
    parser.add_argument("--topics", type=int, default=10, help="Number of complaint topics")
    parser.add_argument("--topic-backend", choices=["nmf", "bertopic"], default="nmf")
    parser.add_argument("--sentiment-backend", choices=["tfidf", "transformer"], default="tfidf")
    parser.add_argument("--database-url", help="Override DATABASE_URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(
        source=args.source,
        path=args.path,
        n=args.n,
        column_map=_parse_map(args.map) or None,
        limit=args.limit,
        database_url=args.database_url,
        n_topics=args.topics,
        topic_backend=args.topic_backend,
        sentiment_backend=args.sentiment_backend,
    )
    print("\nKPIs:", json.dumps(result["kpis"], indent=2))
    print("\nSummary:\n" + result["summary"])


if __name__ == "__main__":
    main()
