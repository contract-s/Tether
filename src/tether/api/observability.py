"""Traces, per-task cost, and metrics across tasks."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from tether.observability import metrics
from tether.observability.store import SpanStore
from tether.observability.tracing import flush

router = APIRouter()
EXPLORER = Path(__file__).parent / "static" / "trace_explorer.html"


@lru_cache
def get_span_store() -> SpanStore:
    from tether.config import settings
    from tether.hitl.store import make_engine

    return SpanStore(make_engine(settings.database_url))


Spans = Annotated[SpanStore, Depends(get_span_store)]


@router.get("/tasks/{task_id}/trace")
def task_trace(task_id: str, spans: Spans) -> dict[str, Any]:
    """The task's spans as a tree, for the trace explorer."""
    flush()
    records = spans.trace(task_id)
    if not records:
        raise HTTPException(404, "no trace for this task (yet)")
    return {"task_id": task_id, "spans": metrics.build_tree(records),
            "cost": metrics.task_cost(records)}


@router.get("/tasks/{task_id}/cost")
def task_cost(task_id: str, spans: Spans) -> dict[str, Any]:
    flush()
    records = spans.trace(task_id)
    if not records:
        raise HTTPException(404, "no trace for this task (yet)")
    return metrics.task_cost(records)


@router.get("/metrics")
def metrics_summary(spans: Spans, days: Annotated[int, Query(ge=1, le=365)] = 7) -> dict[str, Any]:
    """Cost per task type, most expensive agents, tool patterns, escalation rate."""
    flush()
    since = int((time.time() - days * 86400) * 1e9)
    return {"days": days, **metrics.summary(spans.since(since))}


@router.get("/dashboard/traces", include_in_schema=False)
def trace_explorer() -> FileResponse:
    return FileResponse(EXPLORER)
