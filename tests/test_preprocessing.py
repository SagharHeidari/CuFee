import pandas as pd

from src.preprocessing import clean_text, mask_pii, preprocess, rating_to_nps_group, rating_to_sentiment


def test_mask_pii_removes_contact_details():
    text = "Mail me at max.mustermann@example.com or call +49 151 2345 6789, see https://x.io/a"
    masked = mask_pii(text)
    assert "example.com" not in masked
    assert "2345" not in masked
    assert "https" not in masked
    assert "<email>" in masked and "<phone>" in masked and "<url>" in masked


def test_clean_text_normalises():
    assert clean_text("  The APP crashes!!! Again...  ") == "the app crashes again"
    assert clean_text(None) == ""


def test_rating_mappings():
    assert [rating_to_sentiment(r) for r in (1, 2, 3, 4, 5)] == [
        "negative",
        "negative",
        "neutral",
        "positive",
        "positive",
    ]
    assert [rating_to_nps_group(r) for r in (1, 3, 4, 5)] == ["detractor", "detractor", "passive", "promoter"]
    assert rating_to_sentiment(float("nan")) is None


def test_preprocess_deduplicates_and_drops_empty(raw_df, clean_df):
    assert clean_df["feedback_id"].is_unique
    assert len(clean_df) < len(raw_df)
    assert (clean_df["word_count"] >= 2).all()
    assert clean_df["feedback_month"].str.match(r"^\d{4}-\d{2}$").all()


def test_preprocess_handles_bad_dates():
    df = pd.DataFrame(
        {
            "feedback_id": ["a", "b"],
            "created_at": ["2025-01-01", "not a date"],
            "channel": ["email", None],
            "plan": ["prepaid", "prepaid"],
            "region": ["x", "y"],
            "customer_segment": ["new_customer", "new_customer"],
            "rating": [5, 1],
            "text": ["great service thanks", "bad bad service"],
        }
    )
    out = preprocess(df)
    assert out["feedback_id"].tolist() == ["a"]
