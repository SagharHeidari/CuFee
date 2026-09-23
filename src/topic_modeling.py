"""Topic modeling and complaint categorisation.

* ``TopicModeler`` - unsupervised topic discovery. Default backend is NMF on
  TF-IDF (scikit-learn, fast and deterministic). With ``backend="bertopic"``
  it uses sentence-transformer embeddings + BERTopic (optional dependency).
* ``categorize`` - a transparent keyword taxonomy that maps each complaint to a
  business category (network, billing, ...). Business stakeholders need stable,
  named categories; topic models are for *discovering* new ones.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from src.config import RANDOM_STATE

DOMAIN_STOP_WORDS = {
    "please",
    "thanks",
    "fix",
    "really",
    "just",
    "time",
    "expected",
    "better",
    "recommend",
    "keep",
    "provider",
    "considering",
    "switching",
    "day",
    "days",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS)


class TopicModeler:
    def __init__(self, n_topics: int = 8, n_top_terms: int = 8, backend: str = "nmf") -> None:
        self.n_topics = n_topics
        self.n_top_terms = n_top_terms
        self.backend = backend
        self.topics_: pd.DataFrame | None = None

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """Fit the model and return one topic id per document."""
        if self.backend == "bertopic":
            return self._fit_bertopic(texts)
        return self._fit_nmf(texts)

    def _fit_nmf(self, texts: pd.Series) -> np.ndarray:
        self.vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=3, max_df=0.5)
        X = self.vectorizer.fit_transform(texts)
        n_topics = min(self.n_topics, X.shape[1] - 1, X.shape[0] - 1)
        self.model = NMF(n_components=n_topics, random_state=RANDOM_STATE, init="nndsvda", max_iter=400)
        W = self.model.fit_transform(X)
        assignments = W.argmax(axis=1)

        vocab = np.array(self.vectorizer.get_feature_names_out())
        rows = []
        for topic_id, weights in enumerate(self.model.components_):
            terms = vocab[np.argsort(weights)[::-1][: self.n_top_terms]].tolist()
            rows.append(
                {
                    "topic_id": topic_id,
                    "topic_label": _label_from_terms(terms),
                    "top_terms": ", ".join(terms),
                    "size": int((assignments == topic_id).sum()),
                }
            )
        self.topics_ = pd.DataFrame(rows)
        return assignments

    def _fit_bertopic(self, texts: pd.Series) -> np.ndarray:
        from bertopic import BERTopic  # optional dependency
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = BERTopic(embedding_model=embedder, nr_topics=self.n_topics, calculate_probabilities=False)
        assignments, _ = self.model.fit_transform(list(texts))
        info = self.model.get_topic_info()
        rows = []
        for _, r in info.iterrows():
            terms = [w for w, _ in (self.model.get_topic(r["Topic"]) or [])][: self.n_top_terms]
            rows.append(
                {
                    "topic_id": int(r["Topic"]),
                    "topic_label": "outliers" if r["Topic"] == -1 else _label_from_terms(terms),
                    "top_terms": ", ".join(terms),
                    "size": int(r["Count"]),
                }
            )
        self.topics_ = pd.DataFrame(rows)
        return np.asarray(assignments)


def _label_from_terms(terms: list[str]) -> str:
    """Readable label from the top 3 terms, skipping words already covered by a bigram."""
    picked: list[str] = []
    for term in terms:
        words = set(term.split())
        if any(words & set(p.split()) for p in picked):
            continue
        picked.append(term)
        if len(picked) == 3:
            break
    return " / ".join(picked)


# --- complaint categorisation -------------------------------------------------

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "network": [
        "signal",
        "network",
        "coverage",
        "outage",
        "connection",
        "internet",
        "5g",
        "3g",
        "4g",
        "lte",
        "data keeps",
        "slow",
        "speed",
        "calls drop",
        "down",
    ],
    "billing": ["bill", "invoice", "charge", "charged", "fee", "refund", "debit", "overcharged", "payment", "roaming"],
    "customer_service": ["support", "hotline", "agent", "customer service", "hold", "rude", "answer", "called", "chat"],
    "contract": ["contract", "cancel", "cancelling", "termination", "tariff", "plan", "extend"],
    "sim_activation": ["sim", "esim", "activation", "activated", "porting", "qr code"],
    "app_website": ["app", "website", "login", "log in", "crash", "error"],
    "price_value": ["expensive", "price", "cheap", "cheaper", "money", "value", "cost", "affordable"],
}
_CATEGORY_PATTERNS = {
    cat: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b") for cat, kws in CATEGORY_KEYWORDS.items()
}


def categorize(clean_text: str) -> str:
    """Return the category with the most keyword hits, or ``other``."""
    scores = {cat: len(p.findall(clean_text)) for cat, p in _CATEGORY_PATTERNS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def categorize_series(texts: pd.Series) -> pd.Series:
    return texts.map(categorize)
