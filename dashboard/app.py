"""Streamlit dashboard for the Customer Experience AI platform.

Run from the repository root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import inspect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analytics, insights  # noqa: E402
from src.database import get_engine, read_feedback_full, run_query  # noqa: E402
from src.pipeline import run as run_pipeline  # noqa: E402

st.set_page_config(page_title="Customer Experience AI", page_icon="📊", layout="wide")

# Colours by job: sentiment is a polarity (red <-> gray <-> blue), magnitudes are one blue,
# anomalies use the reserved "critical" status colour together with a label.
SENTIMENT_COLORS = {"negative": "#e34948", "neutral": "#898781", "positive": "#2a78d6"}
SENTIMENT_ORDER = ["negative", "neutral", "positive"]
MAGNITUDE = "#2a78d6"
BASELINE = "#898781"
CRITICAL = "#d03b3b"


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=80, b=10),
        title_y=0.97,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title=None),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    return fig


# --- data ---------------------------------------------------------------------


@st.cache_resource
def engine():
    return get_engine()


def database_ready() -> bool:
    return inspect(engine()).has_table("feedback_analysis")


@st.cache_data(show_spinner="Loading feedback from the database...")
def load_data() -> pd.DataFrame:
    return read_feedback_full(engine())


@st.cache_data
def load_topics() -> pd.DataFrame:
    return run_query(engine(), "topic_overview")


@st.cache_data
def load_model_run() -> pd.DataFrame:
    return run_query(engine(), "latest_model_run")


if not database_ready():
    st.info("No analysed data found yet. Running the pipeline on the synthetic sample dataset...")
    with st.spinner("Preprocessing, training the sentiment model and discovering topics..."):
        run_pipeline(source="sample")
    st.cache_data.clear()

data = load_data()

# --- sidebar filters ------------------------------------------------------------

st.sidebar.title("Filters")
min_d, max_d = data["feedback_date"].min().date(), data["feedback_date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)


def multiselect(label: str, column: str) -> list[str]:
    options = sorted(data[column].dropna().unique())
    return st.sidebar.multiselect(label, options, default=[], placeholder="All")


channels = multiselect("Channel", "channel")
plans = multiselect("Plan", "plan")
regions = multiselect("Region", "region")
segments = multiselect("Customer segment", "customer_segment")

df = data
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df = df[(df["feedback_date"] >= start) & (df["feedback_date"] <= end)]
for column, selected in [("channel", channels), ("plan", plans), ("region", regions), ("customer_segment", segments)]:
    if selected:
        df = df[df[column].isin(selected)]

st.sidebar.caption(f"{len(df):,} of {len(data):,} feedback items selected")
if st.sidebar.button("Re-run pipeline (sample data)"):
    with st.spinner("Running pipeline..."):
        run_pipeline(source="sample")
    st.cache_data.clear()
    st.rerun()

# --- header & KPIs --------------------------------------------------------------

st.title("Customer Experience AI")
st.caption("Customer feedback → preprocessing → database → NLP → KPIs → insights")

if df.empty:
    st.warning("No feedback matches the current filters.")
    st.stop()

k = analytics.compute_kpis(df)
trend = analytics.monthly_trend(df)
prev = analytics.compute_kpis(df[df["feedback_month"] == trend["feedback_month"].iloc[-2]]) if len(trend) > 1 else None
last = analytics.compute_kpis(df[df["feedback_month"] == trend["feedback_month"].iloc[-1]])


def delta(key: str, fmt: str) -> str | None:
    return None if prev is None else format(last[key] - prev[key], fmt)


cols = st.columns(5)
cols[0].metric("Feedback items", f"{k['total_feedback']:,}")
cols[1].metric(
    "Avg. rating", f"{k['avg_rating']:.2f} / 5", delta("avg_rating", "+.2f"), help="Delta: last month vs. previous"
)
cols[2].metric("CSAT", f"{k['csat']:.0f}%", delta("csat", "+.1f"), help="Share of 4-5 star ratings")
cols[3].metric(
    "NPS (simulated)", f"{k['nps']:+.0f}", delta("nps", "+.1f"), help="5★ = promoter, 4★ = passive, 1-3★ = detractor"
)
cols[4].metric(
    "Negative feedback", f"{k['negative_share']:.0f}%", delta("negative_share", "+.1f"), delta_color="inverse"
)

tabs = st.tabs(["Overview", "Trends", "Complaints & topics", "Segments", "Anomalies", "Model", "Explorer", "Ask AI"])

# --- overview -------------------------------------------------------------------

with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Key findings")
        st.markdown(analytics.summarize_findings(df))
    with right:
        counts = df["sentiment_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0).reset_index()
        counts.columns = ["sentiment", "count"]
        fig = px.bar(
            counts,
            x="count",
            y="sentiment",
            orientation="h",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            text="count",
            title="Sentiment distribution",
        )
        fig.update_layout(showlegend=False, hovermode="closest")
        st.plotly_chart(style(fig, 300), width="stretch")

    by_channel = df.groupby(["channel", "sentiment_label"]).size().reset_index(name="count")
    by_channel["share"] = by_channel["count"] / by_channel.groupby("channel")["count"].transform("sum") * 100
    fig = px.bar(
        by_channel,
        x="share",
        y="channel",
        color="sentiment_label",
        orientation="h",
        category_orders={"sentiment_label": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment by channel (% of channel feedback)",
        labels={"share": "%", "sentiment_label": ""},
        hover_data={"count": True, "share": ":.1f"},
    )
    fig.update_layout(bargap=0.35, hovermode="closest")
    st.plotly_chart(style(fig, 320), width="stretch")

# --- trends ---------------------------------------------------------------------

with tabs[1]:
    c1, c2 = st.columns(2)
    fig = px.line(
        trend,
        x="feedback_month",
        y="nps",
        markers=True,
        title="Simulated NPS by month",
        labels={"feedback_month": "", "nps": "NPS"},
    )
    fig.update_traces(line_color=MAGNITUDE, line_width=2, marker_size=8)
    c1.plotly_chart(style(fig), width="stretch")

    fig = px.line(
        trend,
        x="feedback_month",
        y="avg_rating",
        markers=True,
        title="Average rating by month",
        labels={"feedback_month": "", "avg_rating": "Rating (1-5)"},
    )
    fig.update_traces(line_color=MAGNITUDE, line_width=2, marker_size=8)
    c2.plotly_chart(style(fig), width="stretch")

    weekly = df.groupby(["feedback_week", "sentiment_label"]).size().reset_index(name="count")
    fig = px.area(
        weekly,
        x="feedback_week",
        y="count",
        color="sentiment_label",
        category_orders={"sentiment_label": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        title="Weekly feedback volume by sentiment",
        labels={"feedback_week": "", "sentiment_label": ""},
    )
    st.plotly_chart(style(fig, 380), width="stretch")

    cat_month = (
        df[df["is_complaint"] == 1]
        .groupby(["feedback_month", "complaint_category"])
        .size()
        .reset_index(name="complaints")
    )
    pivot = cat_month.pivot(index="complaint_category", columns="feedback_month", values="complaints").fillna(0)
    fig = px.imshow(
        pivot,
        color_continuous_scale=["#cde2fb", "#2a78d6", "#0d366b"],
        aspect="auto",
        title="Complaints per category and month",
        labels={"x": "", "y": "", "color": "complaints"},
    )
    fig.update_layout(hovermode="closest")
    st.plotly_chart(style(fig, 360), width="stretch")

# --- complaints & topics ----------------------------------------------------------

with tabs[2]:
    c1, c2 = st.columns(2)
    cats = analytics.category_breakdown(df)
    fig = px.bar(
        cats.sort_values("complaints"),
        x="complaints",
        y="complaint_category",
        orientation="h",
        text="complaints",
        title="Complaints by category (rule-based taxonomy)",
        labels={"complaint_category": ""},
        hover_data={"share": ":.1f", "avg_rating": ":.2f"},
    )
    fig.update_traces(marker_color=MAGNITUDE)
    fig.update_layout(hovermode="closest")
    c1.plotly_chart(style(fig, 400), width="stretch")

    phrases = analytics.recurring_problems(df)
    fig = px.bar(
        phrases.sort_values("mentions"),
        x="mentions",
        y="phrase",
        orientation="h",
        title="Recurring problems (most frequent complaint phrases)",
        labels={"phrase": ""},
    )
    fig.update_traces(marker_color=MAGNITUDE)
    fig.update_layout(hovermode="closest")
    c2.plotly_chart(style(fig, 400), width="stretch")

    st.subheader("Month-over-month change by category")
    st.dataframe(
        analytics.category_month_over_month(df).style.format({"change_pct": "{:+.0f}%"}, na_rep="n/a"),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Complaint topics discovered by the topic model")
    st.caption("Unsupervised (NMF on TF-IDF, or BERTopic) - used to discover issues the taxonomy does not cover yet.")
    topic_counts = df[df["topic_id"].notna()].groupby("topic_label").size().rename("in_selection")
    topics = load_topics().merge(topic_counts, left_on="topic_label", right_index=True, how="left")
    st.dataframe(
        topics[["topic_label", "top_terms", "feedback_count", "in_selection"]],
        width="stretch",
        hide_index=True,
    )
    choice = st.selectbox("Read example complaints for a topic", topics["topic_label"])
    st.table(df[df["topic_label"] == choice][["created_at", "channel", "rating", "text"]].head(8))

# --- segments --------------------------------------------------------------------

with tabs[3]:
    dim = st.radio(
        "Segment by",
        ["customer_segment", "plan", "region", "channel"],
        horizontal=True,
        format_func=lambda s: s.replace("_", " "),
    )
    seg = analytics.segment_table(df, dim)
    c1, c2 = st.columns(2)
    fig = px.bar(
        seg,
        x="nps",
        y=dim,
        orientation="h",
        text=seg["nps"].round(0),
        title=f"Simulated NPS by {dim}",
        labels={dim: ""},
        hover_data={"feedback_count": True, "negative_share": ":.1f"},
    )
    fig.update_traces(marker_color=MAGNITUDE)
    fig.update_layout(hovermode="closest")
    c1.plotly_chart(style(fig), width="stretch")

    nps_mix = df.groupby([dim, "nps_group"]).size().reset_index(name="count")
    nps_mix["share"] = nps_mix["count"] / nps_mix.groupby(dim)["count"].transform("sum") * 100
    fig = px.bar(
        nps_mix,
        x="share",
        y=dim,
        color="nps_group",
        orientation="h",
        category_orders={"nps_group": ["detractor", "passive", "promoter"]},
        color_discrete_map={"detractor": "#e34948", "passive": "#898781", "promoter": "#2a78d6"},
        title="Promoters / passives / detractors (%)",
        labels={dim: "", "nps_group": "", "share": "%"},
    )
    fig.update_layout(hovermode="closest")
    c2.plotly_chart(style(fig), width="stretch")
    st.dataframe(
        seg.style.format({"avg_rating": "{:.2f}", "negative_share": "{:.1f}%", "nps": "{:+.1f}"}),
        width="stretch",
        hide_index=True,
    )

# --- anomalies ---------------------------------------------------------------------

with tabs[4]:
    c1, c2 = st.columns([1, 3])
    freq = c1.radio("Granularity", ["D", "W"], format_func={"D": "Daily", "W": "Weekly"}.get)
    z = c1.slider("Sensitivity (z-score threshold)", 2.0, 6.0, 3.0, 0.5)
    window = 28 if freq == "D" else 8
    anomalies = analytics.detect_anomalies(df, freq=freq, window=window, z_threshold=z)
    flagged = anomalies[anomalies["is_anomaly"]]

    fig = go.Figure()
    fig.add_bar(x=anomalies["period"], y=anomalies["complaints"], name="Complaints", marker_color=MAGNITUDE)
    fig.add_scatter(
        x=anomalies["period"],
        y=anomalies["expected"],
        name="Expected (rolling median)",
        line=dict(color=BASELINE, width=2, dash="dash"),
    )
    fig.add_scatter(
        x=flagged["period"],
        y=flagged["complaints"],
        mode="markers",
        name="⚠ Anomaly",
        marker=dict(color=CRITICAL, size=11, symbol="triangle-up", line=dict(width=2, color="white")),
    )
    fig.update_layout(title="Complaint volume with anomaly detection")
    c2.plotly_chart(style(fig, 400), width="stretch")

    c1.metric("Anomalous periods", len(flagged))
    if not flagged.empty:
        st.markdown("**Flagged periods** - what customers complained about:")
        for row in flagged.sort_values("z_score", ascending=False).head(5).itertuples():
            span = df["created_at"].dt.to_period(freq).dt.start_time == row.period.to_period(freq).start_time
            top = df[span & (df["is_complaint"] == 1)]["complaint_category"].value_counts().head(3)
            regions_hit = df[span & (df["is_complaint"] == 1)]["region"].value_counts().head(2)
            st.markdown(
                f"- ⚠ **{row.period:%Y-%m-%d}**: {int(row.complaints)} complaints vs. ~{row.expected:.0f} expected "
                f"(z = {row.z_score:.1f}). Top categories: {', '.join(f'{c} ({n})' for c, n in top.items())}; "
                f"regions: {', '.join(regions_hit.index)}."
            )

# --- model ---------------------------------------------------------------------------

with tabs[5]:
    run = load_model_run()
    if run.empty:
        st.info("No model evaluation stored (transformer backend or no run yet).")
    else:
        r = run.iloc[0]
        details = json.loads(r["metrics_json"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Model", r["model_name"])
        c2.metric("Hold-out accuracy", f"{r['accuracy']:.1%}")
        c3.metric("Macro F1", f"{r['macro_f1']:.3f}")
        st.caption(
            f"Trained on {r['n_train']:,} and evaluated on {r['n_test']:,} held-out rows. "
            "Labels are weak labels derived from star ratings (1-2 negative, 3 neutral, 4-5 positive)."
        )
        c1, c2 = st.columns(2)
        cm = pd.DataFrame(details["confusion"], index=details["labels"], columns=details["labels"])
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale=["#cde2fb", "#2a78d6", "#0d366b"],
            labels={"x": "Predicted", "y": "Actual", "color": "rows"},
            title="Confusion matrix",
        )
        fig.update_layout(hovermode="closest")
        c1.plotly_chart(style(fig, 380), width="stretch")
        report = pd.DataFrame(details["report"]).T.drop(index=["accuracy"], errors="ignore")
        c2.dataframe(report.style.format("{:.3f}"), width="stretch")

# --- explorer ------------------------------------------------------------------------

with tabs[6]:
    query = st.text_input("Search feedback text", placeholder="e.g. refund, outage, esim")
    view = df
    if query:
        view = view[view["text"].str.contains(query, case=False, na=False, regex=False)]
    sentiment_filter = st.multiselect("Sentiment", SENTIMENT_ORDER, default=SENTIMENT_ORDER)
    view = view[view["sentiment_label"].isin(sentiment_filter)]
    st.caption(f"{len(view):,} matching items")
    columns = [
        "created_at",
        "channel",
        "plan",
        "customer_segment",
        "rating",
        "sentiment_label",
        "sentiment_score",
        "complaint_category",
        "topic_label",
        "text",
    ]
    st.dataframe(view[columns].sort_values("created_at", ascending=False).head(1000), width="stretch", hide_index=True)
    st.download_button("Download selection as CSV", view[columns].to_csv(index=False), "feedback_selection.csv")

# --- ask AI --------------------------------------------------------------------------

with tabs[7]:
    st.subheader("Ask a question about your customers")
    if insights.llm_available():
        st.caption("Answers are generated by Claude from KPIs and representative (PII-masked) comments.")
    else:
        st.caption(
            "No `ANTHROPIC_API_KEY` set - answers use the rule-based summary. Add a key to `.env` to enable the LLM."
        )
    examples = [
        "What were the main reasons for negative feedback this month?",
        "Which customer segment is least satisfied and why?",
        "What happened during the complaint spike in 2025-09?",
    ]
    picked = st.selectbox("Example questions", ["(write your own)"] + examples)
    with st.form("ask"):
        question = st.text_area("Question", value="" if picked == "(write your own)" else picked)
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted and question.strip():
        with st.spinner("Analysing feedback..."):
            result = insights.ask(df, question)
        st.caption(f"Period: {result['period']} · Source: {result['source']}")
        st.markdown(result["answer"])
