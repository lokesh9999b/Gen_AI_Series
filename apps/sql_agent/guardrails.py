"""SQL guardrails: pure functions, no Streamlit, no DB access.

Rules enforced before any SQL reaches the database:
1. Blocklist  - schema-changing / dangerous statements never run.
2. Scope      - only configured tables may be touched.
3. Read cap   - SELECTs without LIMIT get one appended.
4. Confirm    - writes run only after the user approved them (see tools.py).
"""
from __future__ import annotations

import re


WRITE_RE = re.compile(r"^\s*(insert|update|delete|replace)\b", re.IGNORECASE)
SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)
LIMIT_RE = re.compile(r"\blimit\s+\d+", re.IGNORECASE)

# Tables that are never reachable, whatever the allow-list says.
SYSTEM_TABLE_RE = re.compile(
    r"\b(sqlite_master|sqlite_sequence|sqlite_stat1"
    r"|pg_catalog|pg_tables|information_schema"
    r"|users?|accounts?|passwords?)\b",
    re.IGNORECASE,
)
SCHEMA_CHANGE_RE = re.compile(
    r"\b(drop|truncate|alter|attach|detach|pragma|vacuum|reindex|grant|revoke)\b",
    re.IGNORECASE,
)
MULTI_STMT_RE = re.compile(r";\s*\S", re.DOTALL)
LINE_COMMENT_RE = re.compile(r"--", re.DOTALL)
BLOCK_COMMENT_RE = re.compile(r"/\*", re.DOTALL)


class Guardrails:
    """Parameterizable SQL policy for one agent instance."""

    def __init__(self, allowed_tables: tuple = ("tasks",), read_limit: int = 10):
        self.allowed_tables = tuple(t.lower() for t in allowed_tables)
        self.read_limit = read_limit
        tables = "|".join(re.escape(t) for t in self.allowed_tables) or r"(?!)"
        self._from_allowed_re = re.compile(rf"\bfrom\s+(?:\w+\.)?({tables})\b", re.IGNORECASE)
        self._touch_allowed_re = re.compile(rf"\b({tables})\b", re.IGNORECASE)

    @staticmethod
    def is_write(sql: str) -> bool:
        return bool(WRITE_RE.match(sql))

    @staticmethod
    def normalize(sql: str) -> str:
        """Canonical form so reworded-but-identical SQL matches approvals."""
        return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()

    def check_blocked(self, sql: str) -> str | None:
        """Return a reason if the SQL must never run, else None."""
        if SCHEMA_CHANGE_RE.search(sql):
            return "schema-changing commands (DROP/TRUNCATE/ALTER/...) are not allowed"
        if MULTI_STMT_RE.search(sql):
            return "multi-statement queries are not allowed"
        if LINE_COMMENT_RE.search(sql):
            return "SQL comments are not allowed"
        if BLOCK_COMMENT_RE.search(sql):
            return "SQL block comments are not allowed"
        if SYSTEM_TABLE_RE.search(sql):
            return f"only {sorted(self.allowed_tables)} may be accessed"
        if SELECT_RE.match(sql):
            if not self._from_allowed_re.search(sql):
                return f"SELECT queries may only read {sorted(self.allowed_tables)}"
        elif WRITE_RE.match(sql):
            if not self._touch_allowed_re.search(sql):
                return f"writes may only touch {sorted(self.allowed_tables)}"
        else:
            return "only SELECT/INSERT/UPDATE/DELETE statements are allowed"
        return None

    def cap_select(self, sql: str) -> str:
        """Append LIMIT when a SELECT has none."""
        if SELECT_RE.match(sql) and not LIMIT_RE.search(sql):
            return sql.rstrip().rstrip(";") + f" LIMIT {self.read_limit}"
        return sql
