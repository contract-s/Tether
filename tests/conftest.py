import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# One tracer provider for the whole test session, exporting synchronously to memory.
_EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_provider)


@pytest.fixture
def finished_spans() -> InMemorySpanExporter:
    """Spans finished during the test (cleared before it starts)."""
    _EXPORTER.clear()
    return _EXPORTER
