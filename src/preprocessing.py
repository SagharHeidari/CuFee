"""Cleaning and feature preparation for raw customer feedback."""

from __future__ import annotations

import re

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s/-]{6,}\d(?!\w)")
MENTION_RE = re.compile(r"@\w+")
NON_TEXT_RE = re.compile(r"[^a-zäöüß0-9\s']")
SPACES_RE = re.compile(r"\s+")


def mask_pii(text: str) -> str:
    """Replace e-mails, phone numbers and URLs with placeholders (GDPR-friendly)."""
    text = EMAIL_RE.sub(" <email> ", text)
    text = URL_RE.sub(" <url> ", text)
    text = PHONE_RE.sub(" <phone> ", text)
    text = MENTION_RE.sub(" <user> ", text)
    return SPACES_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Normalise text for the NLP models: masked PII, lowercase, no punctuation."""
    if not isinstance(text, str):
        return ""
    text = mask_pii(text).lower()
    text = re.sub(r"<(email|url|phone|user)>", " ", text)
    text = NON_TEXT_RE.sub(" ", text)
    return SPACES_RE.sub(" ", text).strip()


def rating_to_sentiment(rating: float | None) -> str | None:
    """Weak label from star rating: 1-2 negative, 3 neutral, 4-5 positive."""
    if rating is None or pd.isna(rating):
        return None
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def rating_to_nps_group(rating: float | None) -> str | None:
    """Simulated NPS: a 1-5 rating is mapped onto the 0-10 NPS scale.

    5 stars -> promoter (9-10), 4 stars -> passive (7-8), 1-3 stars -> detractor (0-6).
    """
    if rating is None or pd.isna(rating):
        return None
    if rating >= 5:
        return "promoter"
    if rating == 4:
        return "passive"
    return "detractor"


def preprocess(df: pd.DataFrame, min_words: int = 2) -> pd.DataFrame:
    """Validate, deduplicate and enrich the raw feedback.

    Returns a new frame with ``text`` (PII-masked original), ``clean_text``,
    date helper columns, weak sentiment label and simulated NPS group.
    """
    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out["text"] = out["text"].fillna("").astype(str).map(mask_pii)

    out = out.dropna(subset=["created_at"])
    out = out.drop_duplicates(subset=["feedback_id"])
    out = out.drop_duplicates(subset=["created_at", "text"])

    out["clean_text"] = out["text"].map(clean_text)
    out["word_count"] = out["clean_text"].str.split().str.len().fillna(0).astype(int)
    out = out[out["word_count"] >= min_words]

    for col in ("channel", "plan", "region", "customer_segment"):
        out[col] = out[col].fillna("unknown").astype(str).str.strip().str.lower()

    # portable date columns, so SQL works the same on SQLite and PostgreSQL
    out["feedback_date"] = out["created_at"].dt.date
    out["feedback_week"] = out["created_at"].dt.to_period("W-SUN").dt.start_time.dt.date
    out["feedback_month"] = out["created_at"].dt.strftime("%Y-%m")

    out["rating_sentiment"] = out["rating"].map(rating_to_sentiment)
    out["nps_group"] = out["rating"].map(rating_to_nps_group)
    return out.reset_index(drop=True)
