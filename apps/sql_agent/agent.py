"""Agent assembly: model + guarded tools + prompt -> compiled agent."""
from __future__ import annotations

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

from sql_agent.config import SQLAgentConfig
from sql_agent.guardrails import Guardrails
from sql_agent.tools import wrap_query_tool


def create_guard_state() -> dict:
    """Fresh confirm-on-write state. Plain dict (see tools.py for why)."""
    return {"confirmed_writes": set(), "pending_write": None}


def default_system_prompt(config: SQLAgentConfig) -> str:
    """Prompt with strict display + confirmation rules for the task table."""
    tables = ", ".join(f"'{t}'" for t in config.allowed_tables)
    cols = ", ".join(config.table_columns)
    return f"""
You are a task management assistant that interacts with a SQL database containing {tables}.

TASK RULES:
1. Limit SELECT queries to {config.read_limit} results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser.
4. Add the description like you are telling to me

DISPLAY RULES (strict):
- Never show SQL queries, tool names, or raw database tuples like [(1, ...)].
- Every task list MUST be a markdown table starting with exactly this header row:
  {config.canon_header}
  then one row per task.
- Copy every value EXACTLY as the query returned it. Never shorten, truncate, rewrite or reformat values — keep timestamps character-for-character (e.g. 2026-09-28 13:16:57).
- Then one summary line, e.g. "Displayed the 2 most recent tasks."
- No SQL, no code fences, no raw tuples.

CONFIRMATION RULE:
- INSERT/UPDATE/DELETE need the user's approval, but the system enforces it. Just call the tool normally — if approval is needed you will be told so, then ask the user to reply yes or no. Never ask for approval before calling the tool, and never show the SQL.

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT {config.read_limit}
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: {cols}.
"""


def build_agent(db: SQLDatabase, config: SQLAgentConfig, state: dict,
                checkpointer, system_prompt: str | None = None):
    """Assemble model + guarded toolkit and return (agent, guard).

    Works with any SQLDatabase (sqlite/postgres/mysql). Pass your own
    `checkpointer` (e.g. MemorySaver) and a per-user thread id at
    invoke time via {"configurable": {"thread_id": ...}}.
    """
    from langchain.agents import create_agent

    model = ChatGroq(model=config.model, temperature=config.temperature, streaming=True)
    guard = Guardrails(allowed_tables=config.allowed_tables, read_limit=config.read_limit)
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = [wrap_query_tool(t, state, guard) if t.name == "sql_db_query" else t
             for t in toolkit.get_tools()]
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt or default_system_prompt(config),
        checkpointer=checkpointer,
    )
    return agent, guard
