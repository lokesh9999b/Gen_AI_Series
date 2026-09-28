"""Central configuration for the SQL agent.

Everything the agent needs lives here so developers integrate with a
new database by changing settings, not code. Values can be overridden
with environment variables (see .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
# Keep the historical location so existing demo data carries over.
DEFAULT_DB_PATH = PACKAGE_DIR.parent / "my_tasks.db"


@dataclass
class SQLAgentConfig:
    """Settings for one SQL agent instance."""

    db_uri: str = f"sqlite:///{DEFAULT_DB_PATH}"
    init_demo_schema: bool = True
    allowed_tables: tuple = ("tasks",)
    table_columns: tuple = ("id", "title", "description", "status", "created_at")
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.0
    thread_id: str = "1"
    title: str = "SQL Task Agent"
    chat_placeholder: str = "Ask Anything"
    read_limit: int = 10

    @property
    def canon_header(self) -> str:
        """Canonical markdown table header derived from the columns."""
        return "| " + " | ".join(self.table_columns) + " |"


def _env_tables(name: str, default: tuple) -> tuple:
    raw = os.getenv(name)
    if not raw:
        return default
    return tuple(t.strip() for t in raw.split(",") if t.strip())


def load_config() -> SQLAgentConfig:
    """Build config from environment with safe demo defaults."""
    cfg = SQLAgentConfig()
    cfg.db_uri = os.getenv("SQL_AGENT_DB_URI", cfg.db_uri)
    cfg.model = os.getenv("SQL_AGENT_MODEL", cfg.model)
    cfg.temperature = float(os.getenv("SQL_AGENT_TEMPERATURE", str(cfg.temperature)))
    cfg.thread_id = os.getenv("SQL_AGENT_THREAD_ID", cfg.thread_id)
    cfg.title = os.getenv("SQL_AGENT_TITLE", cfg.title)
    cfg.allowed_tables = _env_tables("SQL_AGENT_TABLES", cfg.allowed_tables)
    cfg.table_columns = _env_tables("SQL_AGENT_COLUMNS", cfg.table_columns)
    cfg.read_limit = int(os.getenv("SQL_AGENT_READ_LIMIT", str(cfg.read_limit)))
    cfg.init_demo_schema = os.getenv("SQL_AGENT_INIT_DEMO", "1") == "1"
    return cfg
