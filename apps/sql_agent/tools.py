"""Wrap the agent's SQL-execution tool with guardrails.

The wrapper keeps the original tool name/description/schema so the model
sees no difference; unsafe or unconfirmed SQL is rejected inside `func`
and the rejection text is returned to the model as a normal tool result.

IMPORTANT: `state` must be a plain dict (module-level), never
st.session_state -- tools execute in a LangGraph worker thread where
Streamlit session state is unavailable (raises KeyError).
"""
from __future__ import annotations

from langchain_core.tools import StructuredTool

from sql_agent.guardrails import Guardrails

CONFIRM_MESSAGE = (
    "GUARDRAIL: this INSERT/UPDATE/DELETE needs the user's confirmation. "
    "Ask the user to reply yes to confirm or no to cancel. "
    "Do not show the SQL query."
)


def wrap_query_tool(tool, state: dict, guard: Guardrails):
    """Return a same-named StructuredTool guarded by `guard`."""
    orig_run = tool._run

    def guarded(query: str) -> str:
        blocked = guard.check_blocked(query)
        if blocked:
            return f"BLOCKED BY GUARDRAIL: {blocked}. Tell the user briefly without showing any SQL."
        query = guard.cap_select(query)

        if guard.is_write(query):
            if query not in state["confirmed_writes"]:
                state["pending_write"] = query
                return CONFIRM_MESSAGE
        return orig_run(query)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=guarded,
    )
