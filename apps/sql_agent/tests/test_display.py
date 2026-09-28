"""Unit tests for the display pipeline. No API keys needed."""
from langchain_core.messages import AIMessageChunk, ToolMessage

from sql_agent.display import (
    chunk_text,
    clean_display,
    ensure_table_header,
    finalize,
    fix_timestamps,
    is_display_chunk,
)

HEADER = "| id | title | description | status | created_at |"


def test_ai_text_shown():
    assert is_display_chunk(AIMessageChunk(content="hello")) is True


def test_ai_tool_call_hidden():
    chunk = AIMessageChunk(
        content="", tool_call_chunks=[{"name": "q", "args": "{}", "id": "1", "index": 0}])
    assert is_display_chunk(chunk) is False


def test_tool_message_hidden():
    assert is_display_chunk(ToolMessage(content="[(1, 'a')]", tool_call_id="1")) is False


def test_empty_hidden():
    assert is_display_chunk(AIMessageChunk(content="  ")) is False


def test_list_content_text():
    chunk = AIMessageChunk(content=[{"type": "text", "text": "hi"}])
    assert is_display_chunk(chunk) is True
    assert chunk_text(chunk) == "hi"


def test_clean_strips_glued_sql():
    assert clean_display("SELECT COUNT(*) AS t FROM tasks;| total |\n|---|---|") == "| total |\n|---|---|"


def test_clean_keeps_table():
    assert "pending" in clean_display("| a |\n|---|\n| pending |")


def test_header_repaired():
    assert ensure_table_header("|----|\n|4 | x |", HEADER).startswith(HEADER)


def test_header_kept():
    text = "| id | t |\n|---|---|\n|1|a|"
    assert ensure_table_header(text, HEADER) == text


def test_no_table_untouched():
    assert ensure_table_header("hello", HEADER) == "hello"


def test_timestamps():
    assert fix_timestamps("2026-09-2813:43:05") == "2026-09-28 13:43:05"
    assert fix_timestamps("2026-09-28 13:43:05") == "2026-09-28 13:43:05"


def test_finalize_pipeline():
    out = finalize("SELECT * FROM tasks;\n|----|\n|1|a|", HEADER)
    assert out.startswith(HEADER)
    assert "SELECT" not in out
