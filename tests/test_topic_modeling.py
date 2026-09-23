import pytest

from src.topic_modeling import TopicModeler, categorize


@pytest.mark.parametrize(
    "text,expected",
    [
        ("no signal and the network is down", "network"),
        ("i was charged twice on my invoice", "billing"),
        ("the hotline agent was rude", "customer_service"),
        ("cannot cancel my contract", "contract"),
        ("esim activation failed", "sim_activation"),
        ("the app crashes on login", "app_website"),
        ("too expensive for the money", "price_value"),
        ("hello there", "other"),
    ],
)
def test_categorize(text, expected):
    assert categorize(text) == expected


def test_topic_model_assigns_every_document(clean_df):
    texts = clean_df["clean_text"]
    modeler = TopicModeler(n_topics=6)
    ids = modeler.fit_transform(texts)
    assert len(ids) == len(texts)
    assert set(ids) <= set(modeler.topics_["topic_id"])
    assert modeler.topics_["size"].sum() == len(texts)
    assert modeler.topics_["topic_label"].str.len().gt(0).all()
