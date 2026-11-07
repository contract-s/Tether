"""Wire the real agents (Claude, default tools, Redis + ChromaDB memory) into the workflow."""

from __future__ import annotations

import logging

import redis
from langgraph.checkpoint.base import BaseCheckpointSaver

from agentorch.agents.reviewer import Reviewer
from agentorch.agents.specialist import SpecialistAgent
from agentorch.agents.supervisor import Supervisor
from agentorch.config import settings
from agentorch.graph.workflow import Agents, build_workflow, checkpoint_serde
from agentorch.llm import AnthropicClient, LLMClient
from agentorch.memory.extractor import MemoryExtractor
from agentorch.memory.long_term import ChromaDefaultEmbedder, LongTermMemory
from agentorch.memory.service import MemoryServices
from agentorch.memory.working import WorkingMemory
from agentorch.observability.llm_tracing import TracingLLM
from agentorch.tools.defaults import build_default_registry
from agentorch.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


def default_working_memory() -> MemoryServices:
    """Only the per-task Redis working memory (no LLM or vector store needed)."""
    services = MemoryServices()
    client = redis.Redis.from_url(settings.redis_url)
    try:
        client.ping()
        services.working = lambda task_id: WorkingMemory(client, task_id)
    except redis.RedisError as e:
        log.warning("Redis unavailable, running without working memory: %s", e)
    return services


def default_memory(llm: LLMClient) -> MemoryServices:
    """Connect whichever memory stores are running; the workflow works without them."""
    services = default_working_memory()

    try:
        import chromadb

        chroma = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        services.long_term = LongTermMemory(chroma, ChromaDefaultEmbedder())
        services.extractor = MemoryExtractor(llm)
    except Exception as e:  # noqa: BLE001 - any connection problem means "no memory"
        log.warning("ChromaDB unavailable, running without long-term memory: %s", e)
    return services


def default_agents(llm: LLMClient | None = None, registry: ToolRegistry | None = None) -> Agents:
    llm = llm or TracingLLM(AnthropicClient())
    registry = registry or build_default_registry()
    return Agents(
        supervisor=Supervisor(llm),
        reviewer=Reviewer(llm),
        make_specialist=lambda s: SpecialistAgent(s, llm, registry),
        memory=default_memory(llm),
    )


def default_workflow(checkpointer: BaseCheckpointSaver | None = None):
    return build_workflow(default_agents(), checkpointer=checkpointer)


def setup_default_tracing(engine=None) -> None:
    """Spans go to the span store (trace explorer, cost metrics) and, if configured, OTLP."""
    from agentorch.hitl.store import make_engine
    from agentorch.observability.store import SpanStore, SpanStoreExporter
    from agentorch.observability.tracing import setup_tracing

    engine = engine or make_engine(settings.database_url)
    setup_tracing([SpanStoreExporter(SpanStore(engine))],
                  service_name=settings.otel_service_name,
                  otlp_endpoint=settings.otel_exporter_otlp_endpoint)


def postgres_checkpointer(database_url: str | None = None) -> BaseCheckpointSaver:
    """Durable checkpoints, so a run paused for review survives restarts."""
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    url = (database_url or settings.database_url).replace("postgresql+psycopg://", "postgresql://")
    pool = ConnectionPool(url, max_size=10, open=True,
                          kwargs={"autocommit": True, "prepare_threshold": 0,
                                  "row_factory": dict_row})
    saver = PostgresSaver(pool, serde=checkpoint_serde())
    saver.setup()
    return saver


def default_task_service():
    from agentorch.hitl.assistant import ReviewAssistant
    from agentorch.hitl.notifier import LogNotifier, WebhookNotifier
    from agentorch.hitl.service import TaskService
    from agentorch.hitl.store import HitlStore, make_engine

    notifier = (WebhookNotifier(settings.review_webhook_url) if settings.review_webhook_url
                else LogNotifier())
    engine = make_engine(settings.database_url)
    setup_default_tracing(engine)
    dispatch = None
    if settings.task_runner == "celery":
        from agentorch.workers import celery_dispatch

        dispatch = celery_dispatch
    elif settings.task_runner != "thread":
        raise ValueError(f"TASK_RUNNER must be 'thread' or 'celery', not {settings.task_runner!r}")
    return TaskService(
        graph=default_workflow(checkpointer=postgres_checkpointer()),
        store=HitlStore(engine),
        notifier=notifier,
        assistant=ReviewAssistant(TracingLLM(AnthropicClient())),
        dispatch=dispatch,
    )
