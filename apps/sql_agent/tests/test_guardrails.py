"""Unit tests for guardrails + tool wrapper. No API keys needed."""
import concurrent.futures

import pytest
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq

from sql_agent.agent import create_guard_state
from sql_agent.guardrails import Guardrails
from sql_agent.tools import wrap_query_tool


@pytest.fixture()
def guard():
    return Guardrails(allowed_tables=("tasks",), read_limit=10)


@pytest.fixture()
def memory_db():
    db = SQLDatabase.from_uri("sqlite:///:memory:")
    db.run("CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, status TEXT DEFAULT 'pending')")
    return db


@pytest.fixture()
def guarded_tool(memory_db):
    dummy = ChatGroq(model="openai/gpt-oss-120b", api_key="dummy-test-key")
    raw = [t for t in SQLDatabaseToolkit(db=memory_db, llm=dummy).get_tools()
           if t.name == "sql_db_query"][0]
    return wrap_query_tool(raw, create_guard_state(), Guardrails())


def test_block_schema_changes(guard):
    assert guard.check_blocked("DROP TABLE tasks") is not None
    assert guard.check_blocked("TRUNCATE TABLE tasks") is not None
    assert guard.check_blocked("ALTER TABLE tasks ADD COLUMN x TEXT") is not None


def test_block_multi_statement_and_comments(guard):
    assert guard.check_blocked("DELETE FROM tasks WHERE id=1; DELETE FROM tasks WHERE id=2") is not None
    assert guard.check_blocked("SELECT * FROM tasks -- all") is not None
    assert guard.check_blocked("SELECT /* x */ * FROM tasks") is not None


def test_block_other_tables(guard):
    assert guard.check_blocked("SELECT * FROM sqlite_master") is not None
    assert guard.check_blocked("SELECT * FROM users") is not None
    assert guard.check_blocked("DELETE FROM users WHERE id=1") is not None


def test_allow_tasks_crud(guard):
    assert guard.check_blocked("SELECT * FROM tasks LIMIT 10") is None
    assert guard.check_blocked("INSERT INTO tasks(title) VALUES ('a')") is None
    assert guard.check_blocked("UPDATE tasks SET status='completed' WHERE id=1") is None
    assert guard.check_blocked("DELETE FROM tasks WHERE id=1") is None


def test_reject_unknown_statement(guard):
    assert guard.check_blocked("VACUUM") is not None


def test_cap_select(guard):
    assert guard.cap_select("SELECT * FROM tasks").rstrip().endswith("LIMIT 10")
    assert guard.cap_select("SELECT * FROM tasks LIMIT 5").rstrip().endswith("LIMIT 5")


def test_scope_is_configurable():
    g = Guardrails(allowed_tables=("orders", "customers"), read_limit=5)
    assert g.check_blocked("SELECT * FROM orders") is None
    assert g.check_blocked("SELECT * FROM tasks") is not None
    assert g.cap_select("SELECT * FROM orders").rstrip().endswith("LIMIT 5")


def test_tool_blocks_drop(guarded_tool):
    assert "BLOCKED" in str(guarded_tool.invoke({"query": "DROP TABLE tasks"}))


def test_tool_confirm_flow(guarded_tool, memory_db):
    state = {"confirmed_writes": set(), "pending_write": None}
    # rebuild wrapper bound to our observable state
    from langchain_groq import ChatGroq as _C
    dummy = _C(model="openai/gpt-oss-120b", api_key="dummy-test-key")
    raw = [t for t in SQLDatabaseToolkit(db=memory_db, llm=dummy).get_tools()
           if t.name == "sql_db_query"][0]
    tool = wrap_query_tool(raw, state, Guardrails())

    sql = "INSERT INTO tasks(title) VALUES ('test')"
    assert "confirmation" in str(tool.invoke({"query": sql})).lower()
    assert state["pending_write"] == sql
    state["confirmed_writes"].add(sql)
    tool.invoke({"query": sql})
    assert "test" in str(memory_db.run("SELECT title FROM tasks"))
    assert "test" in str(tool.invoke({"query": "SELECT * FROM tasks"}))


def test_tool_thread_safe(guarded_tool):
    """Regression: tools run in a worker thread without Streamlit context."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        result = ex.submit(
            guarded_tool.invoke, {"query": "DELETE FROM tasks WHERE id IN (4,1)"}
        ).result(timeout=60)
    assert "confirmation" in str(result).lower()
