"""SQL Task Agent — Streamlit frontend (thin UI layer, logic lives in modules)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

from sql_agent import build_agent, create_guard_state, load_config
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
    """Stream one agent turn, showing only the repaired final answer."""
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
        return finalize(message, config.canon_header) or message


def run_agent(user_text: str):
    """Send one user turn to the agent and record both sides in history."""
    st.chat_message("user").markdown(user_text)
    st.session_state.history.append({"role": "user", "content": user_text})
    answer = render_answer(user_text, config.thread_id)
    st.session_state.history.append({"role": "AI", "content": answer})


for message in st.session_state.history:
    st.chat_message(message["role"]).markdown(message["content"])

if GUARD_STATE["pending_write"]:
    st.warning("A database change is waiting for your approval. Confirm to run it, or cancel.")
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("Confirm"):
        GUARD_STATE["confirmed_writes"].add(GUARD_STATE["pending_write"])
        GUARD_STATE["pending_write"] = None
        run_agent("Yes, confirmed. Proceed with the approved change.")
        st.rerun()
    if cancel_col.button("Cancel"):
        GUARD_STATE["pending_write"] = None
        st.session_state.history.append({"role": "AI", "content": "Cancelled. No changes were made."})
        st.rerun()

query = st.chat_input(config.chat_placeholder)

if query:
    pending = GUARD_STATE["pending_write"]
    if pending and YES_RE.match(query):
        GUARD_STATE["confirmed_writes"].add(pending)
        GUARD_STATE["pending_write"] = None
        run_agent("Yes, confirmed. Proceed with the approved change.")
    elif pending and NO_RE.match(query):
        GUARD_STATE["pending_write"] = None
        st.session_state.history.append({"role": "AI", "content": "Cancelled. No changes were made."})
        st.rerun()
    else:
        run_agent(query)
