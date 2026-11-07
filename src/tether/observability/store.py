"""Finished spans, stored in Postgres (SQLite in tests) for the trace explorer and metrics."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode
from sqlalchemy import JSON, BigInteger, Float, Integer, String, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from tether.hitl.store import Base


class SpanRecord(Base):
    __tablename__ = "spans"

    span_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(32), index=True)  # == task id
    parent_span_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(8))  # ok | error
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    start_ns: Mapped[int] = mapped_column(BigInteger, index=True)
    end_ns: Mapped[int] = mapped_column(BigInteger)
    duration_ms: Mapped[float] = mapped_column(Float)
    # Denormalized for fast metrics.
    role: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON)


def to_record(s: ReadableSpan) -> SpanRecord:
    attrs = dict(s.attributes or {})
    ctx = s.context
    error = s.status.description if s.status.status_code == StatusCode.ERROR else None
    return SpanRecord(
        span_id=f"{ctx.span_id:016x}",
        trace_id=f"{ctx.trace_id:032x}",
        parent_span_id=f"{s.parent.span_id:016x}" if s.parent else None,
        name=s.name,
        kind=str(attrs.get("tether.kind", "other")),
        status="error" if s.status.status_code == StatusCode.ERROR else "ok",
        error=error[:2000] if error else None,
        start_ns=s.start_time or 0,
        end_ns=s.end_time or 0,
        duration_ms=((s.end_time or 0) - (s.start_time or 0)) / 1e6,
        role=attrs.get("tether.role"),
        model=attrs.get("gen_ai.request.model"),
        tool=attrs.get("tether.tool"),
        input_tokens=int(attrs.get("gen_ai.usage.input_tokens", 0)),
        output_tokens=int(attrs.get("gen_ai.usage.output_tokens", 0)),
        cost_usd=float(attrs.get("tether.cost_usd", 0.0)),
        attributes=json.loads(json.dumps(attrs, default=str)),
    )


class SpanStore:
    def __init__(self, engine: Engine) -> None:
        Base.metadata.create_all(engine, tables=[SpanRecord.__table__])
        self._session = sessionmaker(engine, expire_on_commit=False)

    def add(self, records: list[SpanRecord]) -> None:
        with self._session.begin() as s:
            for r in records:
                s.merge(r)  # idempotent if a span is exported twice

    def trace(self, trace_id: str) -> list[SpanRecord]:
        q = select(SpanRecord).where(SpanRecord.trace_id == trace_id).order_by(SpanRecord.start_ns)
        with self._session() as s:
            return list(s.scalars(q))

    def since(self, start_ns: int) -> list[SpanRecord]:
        q = select(SpanRecord).where(SpanRecord.start_ns >= start_ns)
        with self._session() as s:
            return list(s.scalars(q))


class SpanStoreExporter(SpanExporter):
    def __init__(self, store: SpanStore) -> None:
        self.store = store

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            self.store.add([to_record(s) for s in spans])
            return SpanExportResult.SUCCESS
        except Exception:  # noqa: BLE001 - never crash the app over telemetry
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass
