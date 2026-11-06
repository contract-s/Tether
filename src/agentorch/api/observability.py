"""Traces, per-task cost, and metrics across tasks."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from agentorch.observability import metrics
from agentorch.observability.store import SpanStore
from agentorch.observability.tracing import flush

router = APIRouter()


@lru_cache
def get_span_store() -> SpanStore:
    from agentorch.config import settings
    from agentorch.hitl.store import make_engine

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
