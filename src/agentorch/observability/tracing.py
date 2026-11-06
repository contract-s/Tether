"""OpenTelemetry tracing for every agent decision.

One task = one trace. The trace id is the task id (both are 128-bit hex), so every run
segment of a task (before and after each human pause, even across restarts) lands in
the same trace. Spans carry an `agentorch.kind` attribute:

    segment  one stretch of execution between pauses
    node     a LangGraph node (plan, work, human_plan, ...)
    agent    supervisor.plan, specialist.run, reviewer.review, supervisor.synthesize
    llm      one model call: model, tokens, cost, prompt and response
    tool     one tool call: input, output, latency, success
    memory   retrieval and saving of long-term memories
    human    a human review, from the moment it was requested to the decision

Spans go to two places: our own span store (for the trace explorer and cost metrics)
and, when OTEL_EXPORTER_OTLP_ENDPOINT is set, an OTLP collector such as Jaeger.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, Status, StatusCode, TraceFlags

from agentorch.observability.cost import llm_cost

TRACER_NAME = "agentorch"
MAX_TEXT = 20_000  # characters of prompt/response kept on a span

_provider: TracerProvider | None = None


def setup_tracing(
    exporters: list[SpanExporter], service_name: str = "agentorch", otlp_endpoint: str = ""
) -> TracerProvider:
    """Install the global tracer provider once. Later calls just return it."""
    global _provider
    if _provider is not None:
        return _provider
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    for exporter in exporters:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)))
    trace.set_tracer_provider(provider)
    _provider = provider
    return provider


def flush() -> None:
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


@contextmanager
def span(name: str, kind: str, **attributes: Any) -> Iterator[trace.Span]:
    """A span that records exceptions as errors. Attribute values are made OTel-safe."""
    with tracer().start_as_current_span(name, record_exception=True,
                                        set_status_on_exception=True) as s:
        s.set_attribute("agentorch.kind", kind)
        set_attributes(s, attributes)
        yield s


def set_attributes(s: trace.Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            s.set_attribute(key, value)
        else:
            s.set_attribute(key, json.dumps(value, default=str)[:MAX_TEXT])


def clip(text: str | None) -> str | None:
    if text is None:
        return None
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + " [truncated]"


def record_llm_usage(
    s: trace.Span, model: str, input_tokens: int, output_tokens: int,
    cache_read_tokens: int = 0, extra_cost: float = 0.0,
) -> float | None:
    """Put token counts and dollar cost on a span; returns the cost."""
    cost = llm_cost(model, input_tokens, output_tokens, cache_read_tokens)
    if cost is not None:
        cost += extra_cost
    set_attributes(s, {
        "gen_ai.system": "anthropic",
        "gen_ai.request.model": model,
        "gen_ai.usage.input_tokens": input_tokens,
        "gen_ai.usage.output_tokens": output_tokens,
        "agentorch.cache_read_tokens": cache_read_tokens,
        "agentorch.cost_usd": cost,
    })
    return cost


def mark_error(s: trace.Span, message: str) -> None:
    s.set_status(Status(StatusCode.ERROR, message))


# --- task traces ---------------------------------------------------------------

def root_span_id(task_id: str) -> int:
    """A stable id for the (virtual) root of a task's trace."""
    return int(task_id[:16], 16) or 1


def task_context(task_id: str) -> otel_context.Context:
    """Context whose parent is the task's root, so spans join the task's trace."""
    parent = SpanContext(
        trace_id=int(task_id, 16), span_id=root_span_id(task_id), is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return trace.set_span_in_context(NonRecordingSpan(parent))
