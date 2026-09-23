"""Optional GenAI layer: answer questions about the feedback with Claude.

Retrieval is deliberately simple and transparent: we compute KPIs and pick
representative comments with pandas, then send that compact context to the LLM.
The model never sees the whole database, and PII was already masked during
preprocessing. Without ``ANTHROPIC_API_KEY`` the app falls back to the
rule-based summary from ``analytics.summarize_findings``.
"""

from __future__ import annotations

import os
import re

import pandas as pd

from src import analytics
from src.config import ANTHROPIC_MODEL

SYSTEM_PROMPT = (
    "You are a customer-experience analyst at a telecom company. You answer questions "
    "about customer feedback using only the KPIs and example comments provided. "
    "Be concise and concrete: lead with the answer, quote numbers from the context, "
    "name the main drivers, and end with 2-3 recommended actions. If the context does "
    "not contain enough information, say so."
)

MONTH_RE = re.compile(r"\b(20\d{2}-\d{2})\b")


def llm_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def select_period(df: pd.DataFrame, question: str) -> tuple[pd.DataFrame, str]:
    """Narrow the data to the month the question is about ('this month', '2025-09', ...)."""
    months = sorted(df["feedback_month"].unique())
    if not months:
        return df, "all data"
    q = question.lower()
    explicit = MONTH_RE.search(q)
    if explicit and explicit.group(1) in months:
        month = explicit.group(1)
    elif "last month" in q or "previous month" in q:
        month = months[-2] if len(months) > 1 else months[-1]
    elif "this month" in q or "current month" in q:
        month = months[-1]
    else:
        return df, f"{months[0]} to {months[-1]}"
    return df[df["feedback_month"] == month], month


def build_context(df: pd.DataFrame, n_examples: int = 25) -> str:
    k = analytics.compute_kpis(df)
    cats = analytics.category_breakdown(df).head(8)
    seg = analytics.segment_table(df)
    phrases = analytics.recurring_problems(df, top_n=10)
    anomalies = analytics.detect_anomalies(df)
    anomalies = anomalies[anomalies["is_anomaly"]]

    negatives = df[df["sentiment_label"] == "negative"].sort_values("sentiment_score")
    # spread the examples across categories instead of taking only the most extreme ones
    examples = (
        negatives.groupby("complaint_category", group_keys=False)
        .head(max(2, n_examples // max(negatives["complaint_category"].nunique(), 1)))
        .head(n_examples)
    )

    parts = [
        "## KPIs",
        "\n".join(f"- {name}: {value:.2f}" for name, value in k.items()),
        "## Complaints by category",
        cats.to_csv(index=False),
        "## Satisfaction by customer segment",
        seg.to_csv(index=False),
        "## Most frequent complaint phrases",
        phrases.to_csv(index=False),
        "## Complaint-volume anomalies (daily)",
        anomalies[["period", "complaints", "expected"]].to_csv(index=False) if not anomalies.empty else "none",
        "## Example negative comments",
        "\n".join(f"- [{r.complaint_category}, {r.channel}, {r.rating}*] {r.text}" for r in examples.itertuples()),
    ]
    return "\n\n".join(parts)


def ask(df: pd.DataFrame, question: str) -> dict:
    """Answer a natural-language question. Returns {'answer', 'period', 'source'}."""
    scoped, period = select_period(df, question)
    if scoped.empty:
        return {"answer": "No feedback found for that period.", "period": period, "source": "none"}

    if not llm_available():
        return {"answer": analytics.summarize_findings(scoped), "period": period, "source": "rule-based"}

    import anthropic

    client = anthropic.Anthropic()
    prompt = f'<context period="{period}">\n{build_context(scoped)}\n</context>\n\nQuestion: {question}'
    try:
        response = client.beta.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},
            # if a safety classifier declines, retry server-side on Anthropic's recommended fallback
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError:
        return {"answer": "Invalid Anthropic API key.", "period": period, "source": "error"}
    except anthropic.RateLimitError:
        return {"answer": "Rate limited by the LLM API, please retry shortly.", "period": period, "source": "error"}
    except anthropic.APIStatusError as e:
        return {"answer": f"LLM API error ({e.status_code}): {e.message}", "period": period, "source": "error"}
    except anthropic.APIConnectionError:
        return {"answer": "Could not reach the LLM API.", "period": period, "source": "error"}

    if response.stop_reason == "refusal":
        return {"answer": analytics.summarize_findings(scoped), "period": period, "source": "rule-based"}
    text = "".join(b.text for b in response.content if b.type == "text")
    return {"answer": text, "period": period, "source": response.model}
