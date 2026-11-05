"""Read-only SQL queries against the app's Postgres database."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

MAX_ROWS = 200


class DbQueryInput(BaseModel):
    sql: str = Field(description="A single read-only SELECT (or WITH ... SELECT) statement")


class DbQueryOutput(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool


class ReadOnlyDatabase:
    def __init__(self, url: str | None = None, engine: Engine | None = None) -> None:
        if engine is None and url is None:
            raise ValueError("pass a database url or engine")
        self._url = url
        self._engine = engine

    @property
    def engine(self) -> Engine:
        # Connect lazily so building the registry doesn't need Postgres running.
        if self._engine is None:
            assert self._url is not None
            self._engine = create_engine(self._url, pool_pre_ping=True)
        return self._engine

    def query(self, args: DbQueryInput) -> DbQueryOutput:
        sql = args.sql.strip().rstrip(";")
        first = sql.split(None, 1)[0].upper() if sql else ""
        if first not in ("SELECT", "WITH") or ";" in sql:
            raise PermissionError("only a single SELECT statement is allowed")
        with self.engine.connect() as conn:
            # Second line of defense: the database itself rejects writes.
            if conn.dialect.name == "postgresql":
                conn.execute(text("SET TRANSACTION READ ONLY"))
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [list(r) for r in result.fetchmany(MAX_ROWS + 1)]
            conn.rollback()
        return DbQueryOutput(
            columns=columns,
            rows=[[_jsonable(v) for v in r] for r in rows[:MAX_ROWS]],
            truncated=len(rows) > MAX_ROWS,
        )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
