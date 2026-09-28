"""Display pipeline: stream only final-answer text, then repair model formatting.

Two problems are handled:
1. `stream_mode="messages"` yields every intermediate step (draft SQL,
   tool calls, raw DB tuples). `is_display_chunk` keeps only plain
   assistant text so SQL never leaks into the chat.
   (Streamed chunks are AIMessageChunk, not "ai"; tool results are ToolMessage.)
2. The model itself sometimes pastes SQL into its reply, drops the table
   header, or retypes timestamps without the space. The `finalize`
   pipeline repairs all three deterministically.
"""
from __future__ import annotations

import re

from langchain_core.messages import AIMessageChunk

FENCE_SQL_RE = re.compile(r"(?is)```(?:sql)?\s*(select|insert|update|delete)\b.*?```")
FULL_LINE_SQL_RE = re.compile(r"(?im)^\s*(select|insert|update|delete)\b.*$")
LEADING_SQL_RE = re.compile(r"(?is)^\s*(select|insert|update|delete)\b.*?;\s*")
TABLE_SEP_RE = re.compile(r"^\|[\s\-\:|]+\|\s*$")
# The model often retypes timestamps and drops the separating space.
SMASHED_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}(?::\d{2})?)")

YES_RE = re.compile(r"^\s*(yes|yeah|yep|confirm|confirmed|proceed|ok|okay)\b", re.IGNORECASE)
NO_RE = re.compile(r"^\s*(no|nope|cancel|cancelled|canceled|stop|don't|dont)\b", re.IGNORECASE)


def chunk_text(chunk) -> str:
    """Extract plain text from a message chunk (str or list of blocks)."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif hasattr(block, "text") and isinstance(getattr(block, "text"), str):
                parts.append(block.text)
        return "".join(parts)
    return ""


def is_display_chunk(chunk) -> bool:
    """True only for final-answer text chunks."""
    if not isinstance(chunk, AIMessageChunk):
        return False
    if getattr(chunk, "tool_calls", None):
        return False
    return bool(chunk_text(chunk).strip())


def clean_display(text: str) -> str:
    """Strip SQL the model echoes into its final answer."""
    text = FENCE_SQL_RE.sub("", text)
    text = LEADING_SQL_RE.sub("", text)
    text = FULL_LINE_SQL_RE.sub("", text)
    return text.strip()


def fix_timestamps(text: str) -> str:
    """Restore the space the model drops when retyping timestamps."""
    return SMASHED_TS_RE.sub(r"\1 \2", text)


def ensure_table_header(text: str, header: str) -> str:
    """Prepend the header if a markdown table lost its header row."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if TABLE_SEP_RE.match(line.strip()):
            prev = lines[i - 1].strip() if i > 0 else ""
            if not (prev.startswith("|") and not TABLE_SEP_RE.match(prev)):
                lines.insert(i, header)
            break
    return "\n".join(lines)


def finalize(text: str, header: str) -> str:
    """Full repair pipeline applied to the final streamed answer."""
    return fix_timestamps(ensure_table_header(clean_display(text), header))
