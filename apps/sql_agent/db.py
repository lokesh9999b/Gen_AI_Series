"""Database helpers: connect to any SQLAlchemy-supported DB, optional demo schema."""
from __future__ import annotations

from langchain_community.utilities import SQLDatabase

DEMO_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK (status IN ('pending', 'in_progress', 'completed')) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def get_db(db_uri: str) -> SQLDatabase:
    """Connect to any database LangChain supports.

    Examples:
        sqlite:      "sqlite:///my_tasks.db"
        postgres:    "postgresql://user:pass@localhost:5432/mydb"
        mysql:       "mysql+pymysql://user:pass@localhost:3306/mydb"
    """
    return SQLDatabase.from_uri(db_uri)


def init_demo_schema(db: SQLDatabase) -> None:
    """Create the demo `tasks` table. Skip this for real databases."""
    db.run(DEMO_SCHEMA)
