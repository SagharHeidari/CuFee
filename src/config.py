"""Central configuration, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SQL_DIR = PROJECT_ROOT / "sql"


def _load_dotenv() -> None:
    """Minimal .env loader so the project works without python-dotenv."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# SQLite is the zero-setup default; point this at PostgreSQL for the "real" setup, e.g.
# postgresql+psycopg2://cx_user:cx_password@localhost:5432/cx_analytics
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'cx_analytics.db'}")

# Optional LLM layer. Without a key the app falls back to rule-based summaries.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")

RANDOM_STATE = 42
