from dotenv import load_dotenv
load_dotenv()

import re
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
import streamlit as st


if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []
if "confirmed_writes" not in st.session_state:
    st.session_state.confirmed_writes = set()
if "pending_write" not in st.session_state:
    st.session_state.pending_write = None


DB_PATH = Path(__file__).parent / "my_tasks.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

db.run("""
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK (status IN ('pending', 'in_progress', 'completed')) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
WRITE_RE = re.compile(r"^\s*(insert|update|delete|replace)\b", re.IGNORECASE)
SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)
LIMIT_RE = re.compile(r"\blimit\s+\d+", re.IGNORECASE)
FROM_TASKS_RE = re.compile(r"\bfrom\s+tasks\b", re.IGNORECASE)
TASKS_WORD_RE = re.compile(r"\btasks\b", re.IGNORECASE)

# Statements that are never allowed through the agent.
BLOCKED_PATTERNS = [
    (re.compile(r"\b(drop|truncate|alter|attach|detach|pragma|vacuum|reindex)\b", re.IGNORECASE),
     "schema-changing commands (DROP/TRUNCATE/ALTER/...) are not allowed"),
    (re.compile(r";\s*\S", re.DOTALL), "multi-statement queries are not allowed"),
    (re.compile(r"--", re.DOTALL), "SQL comments are not allowed"),
    (re.compile(r"/\*", re.DOTALL), "SQL block comments are not allowed"),
    # Any table other than tasks (system tables included) is off limits.
    (re.compile(r"\b(sqlite_master|sqlite_sequence|sqlite_stat1|users?|accounts?|passwords?)\b", re.IGNORECASE),
     "only the 'tasks' table may be accessed"),
]

YES_RE = re.compile(r"^\s*(yes|yeah|yep|confirm|confirmed|proceed|ok|okay)\b", re.IGNORECASE)
NO_RE = re.compile(r"^\s*(no|nope|cancel|cancelled|canceled|stop|don't|dont)\b", re.IGNORECASE)


def is_display_chunk(chunk) -> bool:
    """True only for final-answer text.

    stream_mode="messages" yields every intermediate step: the model's
    draft SQL, tool-call messages and raw DB tuples. Rendering all of
    that leaks SQL into the chat, so only plain assistant text qualifies.
    """
    if getattr(chunk, "type", "") != "ai":
        return False
    if getattr(chunk, "tool_calls", None):
        return False
    content = getattr(chunk, "content", "")
    return isinstance(content, str) and bool(content.strip())


def check_blocked(sql: str) -> str | None:
    """Return a reason if the SQL must never run, else None."""
    for pattern, reason in BLOCKED_PATTERNS:
        if pattern.search(sql):
            return reason
    if SELECT_RE.match(sql):
        if not FROM_TASKS_RE.search(sql):
            return "SELECT queries may only read the 'tasks' table"
    elif WRITE_RE.match(sql):
        if not TASKS_WORD_RE.search(sql):
            return "writes may only touch the 'tasks' table"
    else:
        return "only SELECT/INSERT/UPDATE/DELETE statements are allowed"
    return None


def cap_select(sql: str) -> str:
    """Enforce the 10-row cap on SELECTs missing a LIMIT."""
    if SELECT_RE.match(sql) and not LIMIT_RE.search(sql):
        return sql.rstrip().rstrip(";") + " LIMIT 10"
    return sql


from langchain_core.tools import StructuredTool


def wrap_query_tool(tool, get_state):
    """Guard the agent's SQL-execution tool with a same-named replacement.

    - Blocked statements are rejected before touching the DB.
    - SELECTs are scoped to tasks and capped at 10 rows.
    - INSERT/UPDATE/DELETE run only after the user confirmed them once.
    """
    orig_run = tool._run

    def guarded(query: str) -> str:
        state = get_state()
        blocked = check_blocked(query)
        if blocked:
            return f"BLOCKED BY GUARDRAIL: {blocked}. Tell the user briefly without showing any SQL."
        query = cap_select(query)

        if WRITE_RE.match(query):
            if query not in state["confirmed_writes"]:
                state["pending_write"] = query
                return ("GUARDRAIL: this INSERT/UPDATE/DELETE needs the user's confirmation. "
                        "Ask the user to reply yes to confirm or no to cancel. "
                        "Do not show the SQL query.")
        return orig_run(query)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=guarded,
    )


def _state():
    return st.session_state


model = ChatGroq(model="openai/gpt-oss-120b", temperature=0, streaming=True)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = [wrap_query_tool(t, _state) if t.name == "sql_db_query" else t
         for t in toolkit.get_tools()]
memory = st.session_state.memory


system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table.

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser.
4. Add the description like you are telling to me

DISPLAY RULES (strict):
- Never show SQL queries, tool names, or raw database tuples like [(1, ...)].
- Show only a clean markdown table plus a one-line summary.

CONFIRMATION RULE:
- INSERT/UPDATE/DELETE need the user's approval. If a tool says confirmation is needed, ask the user to reply yes or no. Never show the SQL.

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=memory,
)
st.title("SQL Task Agent")


def run_agent(user_text: str):
    """Send one user turn to the agent and stream only the final answer."""
    st.chat_message("user").markdown(user_text)
    st.session_state.history.append({"role": "user", "content": user_text})

    response = agent.stream(
        {"messages": [{"role": "user", "content": user_text}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode="messages",
    )

    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""
        for chunk, _meta in response:
            if is_display_chunk(chunk):
                message = message + chunk.content
                space.markdown(message)
        st.session_state.history.append({"role": "AI", "content": message})


for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

if st.session_state.pending_write:
    st.warning("A database change is waiting for your approval. Confirm to run it, or cancel.")
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("Confirm"):
        st.session_state.confirmed_writes.add(st.session_state.pending_write)
        st.session_state.pending_write = None
        run_agent("Yes, confirmed. Proceed with the approved change.")
        st.rerun()
    if cancel_col.button("Cancel"):
        st.session_state.pending_write = None
        st.session_state.history.append({"role": "AI", "content": "Cancelled. No changes were made."})
        st.rerun()

query = st.chat_input("Ask Anything")

if query:
    pending = st.session_state.pending_write
    if pending and YES_RE.match(query):
        st.session_state.confirmed_writes.add(pending)
        st.session_state.pending_write = None
        run_agent("Yes, confirmed. Proceed with the approved change.")
    elif pending and NO_RE.match(query):
        st.session_state.pending_write = None
        st.session_state.history.append({"role": "AI", "content": "Cancelled. No changes were made."})
        st.rerun()
    else:
        run_agent(query)
