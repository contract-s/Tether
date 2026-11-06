"""Turn stored spans into a trace tree, per-task cost, and cross-task metrics."""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from agentorch.observability.store import SpanRecord


def span_dict(r: SpanRecord) -> dict[str, Any]:
    return {
        "span_id": r.span_id, "parent_span_id": r.parent_span_id, "name": r.name,
        "kind": r.kind, "status": r.status, "error": r.error,
        "start_ms": r.start_ns / 1e6, "duration_ms": round(r.duration_ms, 2),
        "role": r.role, "model": r.model, "tool": r.tool,
        "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
        "cost_usd": r.cost_usd, "attributes": r.attributes, "children": [],
    }


def build_tree(spans: list[SpanRecord]) -> list[dict[str, Any]]:
    """Nest spans under their parents. Spans whose parent isn't stored (the task's
    virtual root) become top-level. Each node also gets its subtree's total cost."""
    nodes = {r.span_id: span_dict(r) for r in spans}
    roots = []
    for r in sorted(spans, key=lambda s: s.start_ns):
        node = nodes[r.span_id]
        parent = nodes.get(r.parent_span_id or "")
        (parent["children"] if parent else roots).append(node)

    def total(node: dict[str, Any]) -> float:
        node["subtree_cost_usd"] = node["cost_usd"] + sum(total(c) for c in node["children"])
        return node["subtree_cost_usd"]

    for root in roots:
        total(root)
    return roots


def _add(bucket: dict[str, float], key: str, amount: float) -> None:
    bucket[key] = bucket.get(key, 0.0) + amount


def task_cost(spans: list[SpanRecord]) -> dict[str, Any]:
    """Tokens and dollars by agent and model, tool usage, wall-clock and human time."""
    llm = [s for s in spans if s.kind == "llm"]
    tools = [s for s in spans if s.kind == "tool"]
    segments = [s for s in spans if s.kind == "segment"]
    humans = [s for s in spans if s.kind == "human"]
    by_role: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_model: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for s in llm:
        for bucket in (by_role[s.role or "unknown"], by_model[s.model or "unknown"]):
            bucket["calls"] += 1
            bucket["input_tokens"] += s.input_tokens
            bucket["output_tokens"] += s.output_tokens
            bucket["cost_usd"] += s.cost_usd
    tool_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for s in tools:
        t = tool_counts[s.tool or s.name]
        t["calls"] += 1
        t["errors"] += s.status == "error"
        t["total_ms"] += s.duration_ms
    plan = next((s for s in spans if s.name == "supervisor.plan"), None)
    return {
        "task_type": (plan.attributes.get("agentorch.task_type") if plan else None),
        "total_cost_usd": round(sum(s.cost_usd for s in llm), 6),
        "llm_calls": len(llm),
        "input_tokens": sum(s.input_tokens for s in llm),
        "output_tokens": sum(s.output_tokens for s in llm),
        "tool_calls": len(tools),
        "agent_seconds": round(sum(s.duration_ms for s in segments) / 1000, 2),
        "human_wait_seconds": round(sum(s.duration_ms for s in humans) / 1000, 2),
        "escalations": len(humans),
        "by_role": {k: dict(v) for k, v in by_role.items()},
        "by_model": {k: dict(v) for k, v in by_model.items()},
        "tools": {k: dict(v) for k, v in tool_counts.items()},
    }


def summary(spans: list[SpanRecord]) -> dict[str, Any]:
    """Across tasks: cost per task type, most expensive agents, tool patterns,
    escalation rate by day."""
    by_task: dict[str, list[SpanRecord]] = defaultdict(list)
    for s in spans:
        by_task[s.trace_id].append(s)
    tasks = []
    for task_id, task_spans in by_task.items():
        if not any(s.kind == "segment" for s in task_spans):
            continue  # spans from scripts/tests that don't belong to a task
        cost = task_cost(task_spans)
        started = min(s.start_ns for s in task_spans)
        tasks.append({"task_id": task_id, "started_at": started / 1e9, **{
            k: cost[k] for k in ("task_type", "total_cost_usd", "llm_calls", "tool_calls",
                                 "agent_seconds", "human_wait_seconds", "escalations")}})

    types: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        types[t["task_type"] or "unknown"].append(t)
    by_type = {k: {"tasks": len(v),
                   "avg_cost_usd": round(statistics.mean(t["total_cost_usd"] for t in v), 6),
                   "avg_agent_seconds": round(statistics.mean(t["agent_seconds"] for t in v), 2)}
               for k, v in types.items()}

    task_ids = {t["task_id"] for t in tasks}
    agents: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    tool_latency: dict[str, list[float]] = defaultdict(list)
    tool_errors: dict[str, int] = defaultdict(int)
    for s in spans:
        if s.trace_id not in task_ids:
            continue
        if s.kind == "llm":
            a = agents[s.role or "unknown"]
            a["calls"] += 1
            a["cost_usd"] += s.cost_usd
            a["input_tokens"] += s.input_tokens
            a["output_tokens"] += s.output_tokens
        elif s.kind == "tool":
            tool_latency[s.tool or s.name].append(s.duration_ms)
            tool_errors[s.tool or s.name] += s.status == "error"
    tools = {name: {"calls": len(lat), "errors": tool_errors[name],
                    "error_rate": round(tool_errors[name] / len(lat), 3),
                    "avg_ms": round(statistics.mean(lat), 1),
                    "p95_ms": round(_percentile(lat, 95), 1)}
             for name, lat in tool_latency.items()}

    days: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        days[datetime.fromtimestamp(t["started_at"], UTC).date().isoformat()].append(t)
    escalation_rate = [{"day": day, "tasks": len(ts),
                        "escalated": sum(1 for t in ts if t["escalations"]),
                        "rate": round(sum(1 for t in ts if t["escalations"]) / len(ts), 3)}
                       for day, ts in sorted(days.items())]

    return {
        "tasks": sorted(tasks, key=lambda t: t["started_at"], reverse=True),
        "total_cost_usd": round(sum(t["total_cost_usd"] for t in tasks), 6),
        "cost_by_task_type": by_type,
        "agents_by_cost": dict(sorted(((k, dict(v)) for k, v in agents.items()),
                                      key=lambda kv: kv[1]["cost_usd"], reverse=True)),
        "tools": tools,
        "escalation_rate_by_day": escalation_rate,
    }


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered)) - 1))
    return ordered[index]
