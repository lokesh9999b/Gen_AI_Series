"""SQL agent package: a DB-agnostic LangChain SQL agent with guardrails.

Quick start (SQLite demo)::

    streamlit run apps/sql_agent/app.py

Integrate with any database::

    from sql_agent import build_agent, load_config
    from langchain_community.utilities import SQLDatabase

    config = load_config()
    config.db_uri = "postgresql://user:pass@localhost:5432/mydb"
    config.allowed_tables = ("orders", "customers")
    config.init_demo_schema = False
    agent, _ = build_agent(SQLDatabase.from_uri(config.db_uri), config)
"""

from sql_agent.agent import build_agent, create_guard_state, default_system_prompt
from sql_agent.config import SQLAgentConfig, load_config

__all__ = ["build_agent", "create_guard_state", "default_system_prompt", "SQLAgentConfig", "load_config"]
