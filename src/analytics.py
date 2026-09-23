"""KPIs, trends, segments, anomaly detection and rule-based insight summaries.

All functions take the enriched feedback frame (see ``database.read_feedback_full``)
so the dashboard can apply filters first and then recompute everything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.topic_modeling import STOP_WORDS


def nps(nps_groups: pd.Series) -> float:
    groups = nps_groups.dropna()
    if groups.empty:
        return float("nan")
    return float(100 * ((groups == "promoter").mean() - (groups == "detractor").mean()))


def compute_kpis(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            k: float("nan")
            for k in (
                "total_feedback",
                "avg_rating",
                "csat",
                "nps",
                "negative_share",
                "complaint_share",
                "avg_sentiment",
            )
        }
    return {
        "total_feedback": int(len(df)),
        "avg_rating": float(df["rating"].mean()),
        "csat": float((df["rating"] >= 4).mean() * 100),
        "nps": nps(df["nps_group"]),
        "negative_share": float((df["sentiment_label"] == "negative").mean() * 100),
        "complaint_share": float(df["is_complaint"].mean() * 100),
        "avg_sentiment": float(df["sentiment_score"].mean()),
    }


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("feedback_month")
        .agg(
            feedback_count=("feedback_id", "count"),
            avg_rating=("rating", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
            complaints=("is_complaint", "sum"),
            nps=("nps_group", nps),
        )
        .reset_index()
        .sort_values("feedback_month")
    )


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    complaints = df[df["is_complaint"] == 1]
    out = (
        complaints.groupby("complaint_category")
        .agg(complaints=("feedback_id", "count"), avg_rating=("rating", "mean"))
        .reset_index()
        .sort_values("complaints", ascending=False)
    )
    out["share"] = out["complaints"] / max(len(complaints), 1) * 100
    return out


def category_month_over_month(df: pd.DataFrame) -> pd.DataFrame:
    """Complaint counts for the last two months per category, with the % change."""
    months = sorted(df["feedback_month"].unique())
    if len(months) < 2:
        return pd.DataFrame(columns=["complaint_category", "previous", "current", "change_pct"])
    prev_m, cur_m = months[-2], months[-1]
    c = df[(df["is_complaint"] == 1) & df["feedback_month"].isin([prev_m, cur_m])]
    pivot = c.pivot_table(
        index="complaint_category", columns="feedback_month", values="feedback_id", aggfunc="count", fill_value=0
    )
    pivot = pivot.reindex(columns=[prev_m, cur_m], fill_value=0)
    pivot.columns = ["previous", "current"]
    pivot["change_pct"] = (pivot["current"] - pivot["previous"]) / pivot["previous"].replace(0, np.nan) * 100
    return pivot.reset_index().sort_values("current", ascending=False)


def segment_table(df: pd.DataFrame, by: str | list[str] = "customer_segment") -> pd.DataFrame:
    return (
        df.groupby(by)
        .agg(
            feedback_count=("feedback_id", "count"),
            avg_rating=("rating", "mean"),
            negative_share=("sentiment_label", lambda s: (s == "negative").mean() * 100),
            nps=("nps_group", nps),
        )
        .reset_index()
        .sort_values("nps")
    )


def detect_anomalies(df: pd.DataFrame, freq: str = "D", window: int = 28, z_threshold: float = 3.0) -> pd.DataFrame:
    """Flag periods where complaint volume is unusually high.

    Uses a rolling baseline (median) and a robust spread (MAD) over the previous
    ``window`` periods, so a spike does not inflate its own baseline.
    """
    counts = (
        df[df["is_complaint"] == 1]
        .set_index("created_at")
        .resample(freq)["feedback_id"]
        .count()
        .rename("complaints")
        .to_frame()
    )
    history = counts["complaints"].shift(1).rolling(window, min_periods=max(7, window // 4))
    counts["expected"] = history.median()
    mad = history.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    # robust spread, floored at the Poisson noise level sqrt(expected) so quiet days don't alarm
    counts["spread"] = np.maximum(1.4826 * mad, np.sqrt(counts["expected"].clip(lower=1.0)))
    counts["z_score"] = (counts["complaints"] - counts["expected"]) / counts["spread"]
    counts["is_anomaly"] = counts["z_score"] > z_threshold
    return counts.reset_index().rename(columns={"created_at": "period"})


def recurring_problems(df: pd.DataFrame, top_n: int = 15, ngram_range: tuple[int, int] = (2, 3)) -> pd.DataFrame:
    """Most frequent phrases in complaints - the 'what keeps coming up' list."""
    texts = df.loc[df["is_complaint"] == 1, "clean_text"]
    if len(texts) < 5:
        return pd.DataFrame(columns=["phrase", "mentions"])
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=STOP_WORDS, min_df=3)
    X = vec.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    phrases = vec.get_feature_names_out()
    # most frequent first; for ties prefer the longer, more specific phrase
    order = sorted(range(len(phrases)), key=lambda i: (-counts[i], -len(phrases[i].split())))
    kept: list[tuple[str, int]] = []
    for i in order:
        phrase, count = phrases[i], int(counts[i])
        words = set(phrase.split())
        # skip overlapping windows of the same sentence ("little data" vs "expensive little data")
        if any(words & set(p.split()) and c <= count * 1.25 for p, c in kept):
            continue
        kept.append((phrase, count))
        if len(kept) == top_n:
            break
    return pd.DataFrame(kept, columns=["phrase", "mentions"])


def summarize_findings(df: pd.DataFrame) -> str:
    """Deterministic natural-language summary. Also used as LLM fallback."""
    if df.empty:
        return "No feedback in the selected period."
    k = compute_kpis(df)
    cats = category_breakdown(df)
    seg = segment_table(df)
    anomalies = detect_anomalies(df)
    anomalies = anomalies[anomalies["is_anomaly"]]
    trend = monthly_trend(df)

    lines = [
        f"- **{k['total_feedback']:,} feedback items** analysed; average rating **{k['avg_rating']:.2f}/5**, "
        f"CSAT **{k['csat']:.0f}%**, simulated NPS **{k['nps']:+.0f}**.",
        f"- **{k['negative_share']:.0f}%** of feedback is negative.",
    ]
    if not cats.empty:
        top = cats.head(3)
        parts = [f"{r.complaint_category.replace('_', ' ')} ({r.share:.0f}%)" for r in top.itertuples()]
        lines.append(f"- Main complaint drivers: {', '.join(parts)}.")
    mom = category_month_over_month(df).dropna(subset=["change_pct"])
    rising = mom[(mom["change_pct"] > 25) & (mom["current"] >= 10)]
    if not rising.empty:
        r = rising.sort_values("change_pct", ascending=False).iloc[0]
        lines.append(
            f"- Fastest-growing issue last month: **{r.complaint_category.replace('_', ' ')}** "
            f"({r.change_pct:+.0f}% vs. previous month)."
        )
    if not seg.empty:
        worst = seg.iloc[0]
        lines.append(
            f"- Least satisfied segment: **{worst.customer_segment.replace('_', ' ')}** "
            f"(NPS {worst.nps:+.0f}, {worst.negative_share:.0f}% negative)."
        )
    if not anomalies.empty:
        top_a = anomalies.sort_values("z_score", ascending=False).iloc[0]
        lines.append(
            f"- {len(anomalies)} anomalous day(s) detected; biggest spike on "
            f"**{top_a.period:%Y-%m-%d}** with {int(top_a.complaints)} complaints "
            f"(expected ~{top_a.expected:.0f})."
        )
    if len(trend) >= 2:
        delta = trend["nps"].iloc[-1] - trend["nps"].iloc[-2]
        lines.append(
            f"- NPS moved **{delta:+.1f} points** in {trend['feedback_month'].iloc[-1]} vs. the previous month."
        )
    return "\n".join(lines)
