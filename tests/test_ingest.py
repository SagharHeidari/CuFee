import json

import pandas as pd
import pytest

from src.ingest import CANONICAL_COLUMNS, load_file


def test_load_amazon_style_jsonl(tmp_path):
    path = tmp_path / "reviews.jsonl"
    rows = [
        {"rating": 1.0, "title": "Bad", "text": "Stopped working after a week", "timestamp": 1588687728923},
        {"rating": 5.0, "title": "Great", "text": "Works perfectly, great value", "timestamp": 1600000000000},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    df = load_file(path, {"timestamp": "created_at"})
    assert list(df.columns) == CANONICAL_COLUMNS
    assert df["created_at"].dt.year.tolist() == [2020, 2020]
    assert (df["channel"] == "unknown").all()


def test_load_twitter_style_csv_with_limit(tmp_path):
    path = tmp_path / "twcs.csv"
    pd.DataFrame(
        {
            "tweet_id": [1, 2, 3],
            "created_at": ["Tue Oct 31 22:10:47 +0000 2017"] * 3,
            "text": ["@sprintcare my phone has no signal", "@sprintcare thanks!", "@sprintcare still down"],
        }
    ).to_csv(path, index=False)
    df = load_file(path, {"tweet_id": "feedback_id"}, limit=2)
    assert len(df) == 2
    assert df["feedback_id"].tolist() == ["1", "2"]
    assert df["created_at"].dt.tz is None
    assert df["rating"].isna().all()


def test_missing_columns_raise_helpful_error(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"review": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Use --map"):
        load_file(path)
