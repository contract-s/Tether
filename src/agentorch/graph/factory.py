"""Wire the real agents (Claude + default tools) into the workflow."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver

from agentorch.agents.reviewer import Reviewer
from agentorch.agents.schemas import Specialist
from agentorch.agents.specialist import SpecialistAgent
from agentorch.agents.supervisor import Supervisor
from agentorch.graph.workflow import Agents, build_workflow
from agentorch.llm import AnthropicClient, LLMClient
from agentorch.tools.defaults import build_default_registry
from agentorch.tools.registry import ToolRegistry


def default_agents(llm: LLMClient | None = None, registry: ToolRegistry | None = None) -> Agents:
    llm = llm or AnthropicClient()
    registry = registry or build_default_registry()
    return Agents(
        supervisor=Supervisor(llm),
        reviewer=Reviewer(llm),
        make_specialist=lambda s: SpecialistAgent(s, llm, registry),
    )


def default_workflow(checkpointer: BaseCheckpointSaver | None = None):
    return build_workflow(default_agents(), checkpointer=checkpointer)


__all__ = ["Specialist", "default_agents", "default_workflow"]
