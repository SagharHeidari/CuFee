from types import SimpleNamespace

from src import insights
from src.database import get_engine, read_feedback_full


def test_select_period(pipeline_db):
    df = read_feedback_full(get_engine(pipeline_db[0]))
    scoped, period = insights.select_period(df, "What went wrong this month?")
    assert period == df["feedback_month"].max()
    assert scoped["feedback_month"].eq(period).all()
    _, period = insights.select_period(df, "What happened in 2025-09?")
    assert period == "2025-09"
    whole, _ = insights.select_period(df, "Overall, what do customers hate?")
    assert len(whole) == len(df)


def test_ask_falls_back_without_api_key(pipeline_db, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    df = read_feedback_full(get_engine(pipeline_db[0]))
    result = insights.ask(df, "main reasons for negative feedback this month?")
    assert result["source"] == "rule-based"
    assert "feedback items" in result["answer"]


def test_ask_uses_llm_when_configured(pipeline_db, monkeypatch):
    import anthropic

    calls = {}

    class FakeMessages:
        def create(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(
                stop_reason="end_turn",
                model=kwargs["model"],
                content=[SimpleNamespace(type="text", text="Network outages drove complaints.")],
            )

    class FakeClient:
        def __init__(self, *a, **k):
            self.beta = SimpleNamespace(messages=FakeMessages())

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    df = read_feedback_full(get_engine(pipeline_db[0]))
    result = insights.ask(df, "What happened in 2025-09?")
    assert result["answer"] == "Network outages drove complaints."
    prompt = calls["messages"][0]["content"]
    assert 'period="2025-09"' in prompt and "Example negative comments" in prompt
