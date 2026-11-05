"""Wire the real agents (Claude, default tools, Redis + ChromaDB memory) into the workflow."""

from __future__ import annotations

import logging

import redis
from langgraph.checkpoint.base import BaseCheckpointSaver

from agentorch.agents.reviewer import Reviewer
from agentorch.agents.specialist import SpecialistAgent
from agentorch.agents.supervisor import Supervisor
from agentorch.config import settings
from agentorch.graph.workflow import Agents, build_workflow
from agentorch.llm import AnthropicClient, LLMClient
from agentorch.memory.extractor import MemoryExtractor
from agentorch.memory.long_term import ChromaDefaultEmbedder, LongTermMemory
from agentorch.memory.service import MemoryServices
from agentorch.memory.working import WorkingMemory
from agentorch.tools.defaults import build_default_registry
from agentorch.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


def default_memory(llm: LLMClient) -> MemoryServices:
    """Connect whichever memory stores are running; the workflow works without them."""
    services = MemoryServices()

    client = redis.Redis.from_url(settings.redis_url)
    try:
        client.ping()
        services.working = lambda task_id: WorkingMemory(client, task_id)
    except redis.RedisError as e:
        log.warning("Redis unavailable, running without working memory: %s", e)

    try:
        import chromadb

        chroma = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        services.long_term = LongTermMemory(chroma, ChromaDefaultEmbedder())
        services.extractor = MemoryExtractor(llm)
    except Exception as e:  # noqa: BLE001 - any connection problem means "no memory"
        log.warning("ChromaDB unavailable, running without long-term memory: %s", e)
    return services


def default_agents(llm: LLMClient | None = None, registry: ToolRegistry | None = None) -> Agents:
    llm = llm or AnthropicClient()
    registry = registry or build_default_registry()
    return Agents(
        supervisor=Supervisor(llm),
        reviewer=Reviewer(llm),
        make_specialist=lambda s: SpecialistAgent(s, llm, registry),
        memory=default_memory(llm),
    )


def default_workflow(checkpointer: BaseCheckpointSaver | None = None):
    return build_workflow(default_agents(), checkpointer=checkpointer)
