import pandas as pd

from src.sentiment import SentimentClassifier, train_and_evaluate


def test_train_and_evaluate_beats_baseline(clean_df):
    model, result = train_and_evaluate(clean_df)
    majority = clean_df["rating_sentiment"].value_counts(normalize=True).max()
    assert result.accuracy > majority + 0.1
    assert 0 <= result.macro_f1 <= 1
    assert sum(map(sum, result.confusion)) == result.n_test


def test_predict_output_shape_and_ranges(clean_df):
    model, _ = train_and_evaluate(clean_df)
    pred = model.predict(pd.Series(["the network is down again terrible", "great fast friendly service"]))
    assert list(pred.columns) == ["sentiment_label", "sentiment_score", "sentiment_confidence"]
    assert pred["sentiment_label"].tolist() == ["negative", "positive"]
    assert pred["sentiment_score"].between(-1, 1).all()


def test_save_and_load_roundtrip(clean_df, tmp_path):
    model, _ = train_and_evaluate(clean_df)
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = SentimentClassifier.load(path)
    texts = clean_df["clean_text"].head(20)
    assert loaded.predict(texts).equals(model.predict(texts))
