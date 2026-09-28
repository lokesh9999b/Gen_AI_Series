"""SQL Task Agent — Streamlit frontend (thin UI layer, logic lives in modules)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

from sql_agent import build_agent, create_guard_state, load_config
from sql_agent.agent import run_approved
from sql_agent.db import get_db, init_demo_schema
from sql_agent.display import (
    NO_RE,
    YES_RE,
    chunk_text,
    finalize,
    is_display_chunk,
)

config = load_config()

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

# Plain module-level dict, NOT session state: tools run in a LangGraph
# worker thread without Streamlit context. Persists across reruns.
GUARD_STATE = create_guard_state()

db = get_db(config.db_uri)
if config.init_demo_schema:
    init_demo_schema(db)

agent, _guard = build_agent(db, config, GUARD_STATE, st.session_state.memory)

st.title(config.title)


def render_answer(user_text: str, thread_id: str):
    """Stream one agent turn, showing only the repaired final answer.

    Retries once on transient model errors (Groq occasionally rejects a
    malformed gpt-oss tool call containing `<|channel|>` tokens); the app
    never dies with a traceback.
    """
    attempts = 0
    while True:
        try:
            response = agent.stream(
                {"messages": [{"role": "user", "content": user_text}]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )
            box = st.chat_message("ai")
            with box:
                space = st.empty()
                message = ""
                for chunk, _meta in response:
                    if is_display_chunk(chunk):
                        message = message + chunk_text(chunk)
                        space.markdown(finalize(message, config.canon_header) or message)
                final = finalize(message, config.canon_header)
                return final or "I couldn't produce an answer. Please rephrase and try again."
        except Exception:
            attempts += 1
            if attempts > 1:
                return ("The model hit a transient error and couldn't answer. "
                        "Please try again — no database change was made.")


def run_agent(user_text: str, send_text: str | None = None):
    """Send one user turn to the agent and record both sides in history.

    `send_text` overrides what the agent receives (used to nudge it after
    a bare yes), while the chat bubble always shows `user_text`.
    """
    st.chat_message("user").markdown(user_text)
    st.session_state.history.append({"role": "user", "content": user_text})
    answer = render_answer(send_text or user_text, config.thread_id)
    st.session_state.history.append({"role": "AI", "content": answer})


def execute_pending() -> str:
    """Run the staged write directly and return a summary note for the agent.

    Approval executes in code (never via a regenerated model query), so
    reworded SQL can't restart the confirmation loop. The agent only
    re-queries (reads) and summarizes.
    """
    return run_approved(db, GUARD_STATE)


def confirm_and_run(user_text: str):
    """Record the user's confirmation, execute the staged write, summarize."""
    st.chat_message("user").markdown(user_text)
    st.session_state.history.append({"role": "user", "content": user_text})
    answer = render_answer(execute_pending(), config.thread_id)
    st.session_state.history.append({"role": "AI", "content": answer})


for message in st.session_state.history:
    st.chat_message(message["role"]).markdown(message["content"])

if GUARD_STATE["pending_write"]:
    st.warning("A database change is waiting for your approval. Confirm to run it, or cancel.")
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("Confirm"):
        confirm_and_run("Confirmed. Proceed with the approved change.")
        st.rerun()
    if cancel_col.button("Cancel"):
        GUARD_STATE["pending_write"] = None
        st.session_state.history.append({"role": "AI", "content": "Cancelled. No changes were made."})
        st.rerun()

query = st.chat_input(config.chat_placeholder)

if query:
    pending = GUARD_STATE["pending_write"]
    if pending and YES_RE.match(query):
        confirm_and_run(query)
    elif pending and NO_RE.match(query):
        GUARD_STATE["pending_write"] = None
        st.session_state.history.append({"role": "AI", "content": "Cancelled. No changes were made."})
        st.rerun()
    elif not pending and YES_RE.match(query):
        # Model asked for approval without staging a tool call: nudge it
        # to call the tool now so the guard can stage and confirm it.
        run_agent(query, send_text=(
            f'User says: "{query}". If you were waiting for approval for a '
            "database change, call the appropriate tool now; otherwise respond normally."))
    else:
        run_agent(query)
