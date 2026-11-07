"""End-to-end fixtures: the real Postgres, Redis and ChromaDB from docker-compose.

    make up && make e2e          # scripted model, no API cost
    make e2e-live                # also the tests that call Claude

Tests use their own database (agentorch_e2e) and Chroma collection, so they don't mix
with tasks in the dashboards. Everything is skipped when the infrastructure isn't up.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import redis
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from sqlalchemy import create_engine, text

from agentorch.agents.reviewer import Reviewer
from agentorch.agents.specialist import SpecialistAgent
from agentorch.agents.supervisor import Supervisor
from agentorch.api import memory as memory_api
from agentorch.api import observability as obs_api
from agentorch.api import tasks as tasks_api
from agentorch.api.main import app
from agentorch.config import settings
from agentorch.graph.factory import postgres_checkpointer
from agentorch.graph.workflow import Agents, build_workflow
from agentorch.hitl.service import TaskService
from agentorch.hitl.store import HitlStore, make_engine
from agentorch.llm import LLMClient
from agentorch.memory.extractor import MemoryExtractor
from agentorch.memory.long_term import ChromaDefaultEmbedder, LongTermMemory
from agentorch.memory.service import MemoryServices
from agentorch.memory.working import WorkingMemory
from agentorch.observability.llm_tracing import TracingLLM
from agentorch.observability.store import SpanStore, SpanStoreExporter
from agentorch.tools.registry import ToolRegistry

E2E_DB = "agentorch_e2e"
COLLECTION = "agentorch_e2e"


def _infra_error() -> str | None:
    try:
        with create_engine(settings.database_url).connect() as c:
            c.execute(text("select 1"))
        redis.Redis.from_url(settings.redis_url).ping()
        import chromadb

        chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port).heartbeat()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    return None


def pytest_collection_modifyitems(config, items) -> None:
    here = os.path.dirname(__file__)
    e2e = [i for i in items if str(i.fspath).startswith(here)]
    if not e2e:
        return
    if error := _infra_error():
        if os.environ.get("AGENTORCH_REQUIRE_INFRA") == "1":  # CI: missing services is a failure
            raise pytest.UsageError(f"end-to-end infrastructure not reachable: {error}")
        skip = pytest.mark.skip(reason=f"infrastructure not running (make up): {error}")
        for item in e2e:
            item.add_marker(skip)
        return
    live_ok = os.environ.get("AGENTORCH_LIVE") == "1" and settings.anthropic_api_key
    for item in e2e:
        if "live" in item.keywords and not live_ok:
            item.add_marker(pytest.mark.skip(
                reason="calls Claude: set AGENTORCH_LIVE=1 and ANTHROPIC_API_KEY"))


@pytest.fixture(scope="session")
def database_url() -> str:
    """A database of our own on the compose Postgres, created on first use."""
    admin = create_engine(settings.database_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        if not c.execute(text("select 1 from pg_database where datname = :n"),
                         {"n": E2E_DB}).scalar():
            c.execute(text(f"create database {E2E_DB}"))
    admin.dispose()
    return settings.database_url.rsplit("/", 1)[0] + f"/{E2E_DB}"


@pytest.fixture(scope="session")
def engine(database_url):
    engine = make_engine(database_url)
    # Spans also go to this database's span store, as in production (trace API, cost).
    trace.get_tracer_provider().add_span_processor(
        SimpleSpanProcessor(SpanStoreExporter(SpanStore(engine))))
    return engine


@pytest.fixture(scope="session")
def checkpointer(database_url):
    return postgres_checkpointer(database_url)


@pytest.fixture(scope="session")
def redis_client():
    return redis.Redis.from_url(settings.redis_url)


@pytest.fixture(scope="session")
def long_term() -> LongTermMemory:
    import chromadb

    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return LongTermMemory(client, ChromaDefaultEmbedder(), collection=COLLECTION)


@pytest.fixture
def user(long_term) -> Iterator[str]:
    """A fresh user per test, so long-term memories don't leak between tests."""
    user_id = f"e2e-{uuid.uuid4().hex[:8]}"
    yield user_id
    long_term.delete_user(user_id)


class Stack:
    """The real services wired around a given model, driven through the HTTP API."""

    def __init__(self, llm: LLMClient, engine, checkpointer, redis_client, long_term,
                 registry: ToolRegistry | None = None, **workflow: Any) -> None:
        self.llm = llm
        self.engine, self.checkpointer, self.redis = engine, checkpointer, redis_client
        traced = TracingLLM(llm)
        registry = registry or ToolRegistry()
        self.memory = MemoryServices(
            long_term=long_term, extractor=MemoryExtractor(traced),
            working=lambda tid: WorkingMemory(redis_client, tid))
        self.agents = Agents(
            supervisor=Supervisor(traced), reviewer=Reviewer(traced, threshold=0.7),
            make_specialist=lambda s: SpecialistAgent(s, traced, registry), memory=self.memory)
        self.workflow = {"max_attempts": 2, "plan_confidence_threshold": 0.7, **workflow}
        self.service = self.new_service()
        self.spans = SpanStore(engine)
        self.client = TestClient(app)

    def new_service(self) -> TaskService:
        """A fresh service over the same databases, as after a restart."""
        graph = build_workflow(self.agents, checkpointer=self.checkpointer, **self.workflow)
        return TaskService(graph, HitlStore(self.engine),
                           submit=ThreadPoolExecutor(max_workers=4).submit)

    def use(self, service: TaskService) -> None:
        self.service = service
        app.dependency_overrides[tasks_api.get_service] = lambda: service

    def start(self, task: str, user_id: str, **kw: Any) -> str:
        res = self.client.post("/tasks", json={"task": task, "user_id": user_id, **kw})
        assert res.status_code == 202, res.text
        return res.json()["id"]

    def wait(self, task_id: str, *statuses: str, timeout: float = 60) -> dict[str, Any]:
        statuses = statuses or ("done", "failed", "rejected")
        deadline = time.time() + timeout
        while time.time() < deadline:
            task = self.client.get(f"/tasks/{task_id}").json()
            if task["status"] in statuses:
                time.sleep(0.2)  # let the run's segment span end and be stored
                return task
            time.sleep(0.1)
        raise AssertionError(f"task stuck in {task['status']!r}, wanted {statuses}")


@pytest.fixture
def make_stack(engine, checkpointer, redis_client, long_term) -> Iterator[Callable[..., Stack]]:
    def make(llm: LLMClient, **kw: Any) -> Stack:
        stack = Stack(llm, engine, checkpointer, redis_client, long_term, **kw)
        stack.use(stack.service)
        app.dependency_overrides[obs_api.get_span_store] = lambda: stack.spans
        app.dependency_overrides[memory_api.get_store] = lambda: long_term
        return stack

    yield make
    app.dependency_overrides.clear()
