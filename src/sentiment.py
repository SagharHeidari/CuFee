"""Sentiment classification.

Default model: TF-IDF (word 1-2 grams) + Logistic Regression, trained on weak
labels derived from star ratings and evaluated on a held-out test split. It is
fast, explainable and runs in CI.

Optional model: a pretrained transformer (``cardiffnlp/twitter-roberta-base-sentiment-latest``)
via Hugging Face, if ``transformers`` is installed (see requirements-optional.txt).
Comparing both is a good model-evaluation story for interviews.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE

LABELS = ["negative", "neutral", "positive"]


@dataclass
class EvaluationResult:
    accuracy: float
    macro_f1: float
    report: dict
    confusion: list[list[int]]
    n_train: int
    n_test: int
    labels: list[str] = field(default_factory=lambda: list(LABELS))


class SentimentClassifier:
    name = "tfidf_logreg"

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)),
            ]
        )

    def fit(self, texts: pd.Series, labels: pd.Series) -> SentimentClassifier:
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: pd.Series) -> pd.DataFrame:
        """Return label, score in [-1, 1] (P(pos) - P(neg)) and confidence."""
        proba = self.pipeline.predict_proba(texts)
        classes = list(self.pipeline.classes_)
        p = pd.DataFrame(proba, columns=classes).reindex(columns=LABELS, fill_value=0.0)
        return pd.DataFrame(
            {
                "sentiment_label": p.idxmax(axis=1).to_numpy(),
                "sentiment_score": (p["positive"] - p["negative"]).round(4).to_numpy(),
                "sentiment_confidence": p.max(axis=1).round(4).to_numpy(),
            }
        )

    def top_terms(self, n: int = 15) -> dict[str, list[str]]:
        """Most influential n-grams per class - useful for explaining the model."""
        vocab = np.array(self.pipeline.named_steps["tfidf"].get_feature_names_out())
        coefs = self.pipeline.named_steps["clf"].coef_
        return {label: vocab[np.argsort(coefs[i])[::-1][:n]].tolist() for i, label in enumerate(self.pipeline.classes_)}

    def save(self, path: str | Path) -> None:
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> SentimentClassifier:
        obj = cls()
        obj.pipeline = joblib.load(path)
        return obj


def train_and_evaluate(
    df: pd.DataFrame, text_col: str = "clean_text", label_col: str = "rating_sentiment", test_size: float = 0.2
) -> tuple[SentimentClassifier, EvaluationResult]:
    """Train on a stratified split, evaluate on the hold-out set, then refit on all labelled rows."""
    labelled = df.dropna(subset=[label_col])
    X_train, X_test, y_train, y_test = train_test_split(
        labelled[text_col],
        labelled[label_col],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=labelled[label_col],
    )
    model = SentimentClassifier().fit(X_train, y_train)
    y_pred = model.pipeline.predict(X_test)

    result = EvaluationResult(
        accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        report=classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        confusion=confusion_matrix(y_test, y_pred, labels=LABELS).tolist(),
        n_train=len(X_train),
        n_test=len(X_test),
    )
    model.fit(labelled[text_col], labelled[label_col])
    return model, result


class TransformerSentiment:
    """Optional pretrained transformer baseline (no training needed)."""

    name = "twitter_roberta"
    MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __init__(self) -> None:
        from transformers import pipeline  # optional dependency

        self._pipe = pipeline("sentiment-analysis", model=self.MODEL_ID, top_k=None, truncation=True)

    def predict(self, texts: pd.Series, batch_size: int = 32) -> pd.DataFrame:
        outputs = self._pipe(list(texts), batch_size=batch_size)
        rows = []
        for scores in outputs:
            p = {s["label"].lower(): s["score"] for s in scores}
            label = max(p, key=p.get)
            rows.append(
                {
                    "sentiment_label": label,
                    "sentiment_score": round(p.get("positive", 0) - p.get("negative", 0), 4),
                    "sentiment_confidence": round(p[label], 4),
                }
            )
        return pd.DataFrame(rows)
