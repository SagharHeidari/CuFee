import pandas as pd

from src.database import get_engine, load_named_queries, run_query


def test_named_queries_are_parsed():
    queries = load_named_queries()
    for name in ("kpi_overview", "monthly_trend", "complaints_by_category", "daily_complaints", "feedback_full"):
        assert name in queries
        assert queries[name].lstrip().upper().startswith("SELECT")


def test_every_query_runs_against_pipeline_db(pipeline_db):
    url, result = pipeline_db
    engine = get_engine(url)
    months = run_query(engine, "monthly_trend")
    for name in load_named_queries():
        params = (
            {"month": months["feedback_month"].iloc[-1], "limit": 5} if ":month" in load_named_queries()[name] else {}
        )
        assert isinstance(run_query(engine, name, **params), pd.DataFrame)


def test_sql_kpis_match_pandas_kpis(pipeline_db):
    url, result = pipeline_db
    sql = run_query(get_engine(url), "kpi_overview").iloc[0]
    k = result["kpis"]
    assert int(sql["total_feedback"]) == k["total_feedback"]
    assert abs(sql["nps"] - k["nps"]) < 1e-6
    assert abs(sql["csat"] * 100 - k["csat"]) < 1e-6
