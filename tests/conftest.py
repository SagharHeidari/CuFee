import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generator import generate_feedback  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return generate_feedback(n=1500, seed=7)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return preprocess(raw_df)


@pytest.fixture(scope="session")
def pipeline_db(tmp_path_factory):
    """Run the full pipeline once into a temporary SQLite database.

    Set TEST_DATABASE_URL to run the same tests against PostgreSQL (CI does this).
    """
    from src.pipeline import run

    url = os.getenv("TEST_DATABASE_URL") or f"sqlite:///{tmp_path_factory.mktemp('db') / 'test.db'}"
    result = run(source="sample", n=1500, database_url=url, save_artifacts=False)
    return url, result
