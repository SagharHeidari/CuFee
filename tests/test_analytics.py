import numpy as np
import pandas as pd

from src import analytics


def _frame(ratings, sentiments, dates=None):
    n = len(ratings)
    groups = ["promoter" if r == 5 else "passive" if r == 4 else "detractor" for r in ratings]
    return pd.DataFrame(
        {
            "feedback_id": [str(i) for i in range(n)],
            "created_at": pd.to_datetime(dates or ["2025-01-01"] * n),
            "rating": ratings,
            "nps_group": groups,
            "sentiment_label": sentiments,
            "sentiment_score": [-1 if s == "negative" else 1 if s == "positive" else 0 for s in sentiments],
            "is_complaint": [int(s == "negative") for s in sentiments],
        }
    )


def test_nps_formula():
    # 2 promoters, 1 passive, 1 detractor -> (50% - 25%) = +25
    assert analytics.nps(pd.Series(["promoter", "promoter", "passive", "detractor"])) == 25.0


def test_compute_kpis():
    k = analytics.compute_kpis(_frame([5, 4, 1, 2], ["positive", "positive", "negative", "negative"]))
    assert k["total_feedback"] == 4
    assert k["avg_rating"] == 3.0
    assert k["csat"] == 50.0
    assert k["negative_share"] == 50.0
    assert k["nps"] == -25.0


def test_detect_anomalies_finds_injected_spike():
    rng = np.random.default_rng(0)
    days = pd.date_range("2025-01-01", periods=120, freq="D")
    counts = rng.poisson(5, size=len(days))
    counts[100] = 40
    dates = np.repeat(days, counts)
    df = pd.DataFrame({"feedback_id": range(len(dates)), "created_at": dates, "is_complaint": 1})
    result = analytics.detect_anomalies(df)
    flagged = result.loc[result["is_anomaly"], "period"]
    assert days[100] in set(flagged)
    assert len(flagged) <= 3  # no alarm storm on normal noise


def test_pipeline_outputs_are_consistent(pipeline_db):
    from src.database import get_engine, read_feedback_full

    url, result = pipeline_db
    df = read_feedback_full(get_engine(url))
    assert result["kpis"]["total_feedback"] == len(df)
    summary = analytics.summarize_findings(df)
    assert "NPS" in summary and "complaint" in summary
    # the synthetic outage week must be found by the anomaly detector
    anomalies = analytics.detect_anomalies(df)
    assert anomalies.loc[anomalies["is_anomaly"], "period"].dt.strftime("%Y-%m").eq("2025-09").any()
    phrases = analytics.recurring_problems(df)
    assert not phrases.empty and phrases["mentions"].is_monotonic_decreasing
