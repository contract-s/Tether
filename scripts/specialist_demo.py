"""Run one specialist on a subtask with real tools, then have the reviewer check it.

    uv run python scripts/specialist_demo.py research
    uv run python scripts/specialist_demo.py analysis
"""

import sys

from agentorch.agents.reviewer import Reviewer
from agentorch.agents.schemas import Complexity, Specialist, Subtask
from agentorch.agents.specialist import SpecialistAgent
from agentorch.llm import AnthropicClient
from agentorch.tools.defaults import build_default_registry

SUBTASKS = {
    "research": Subtask(
        id="t1",
        description="Find the current latest stable release versions of Qdrant, Weaviate and "
        "Milvus, with release dates.",
        specialist=Specialist.RESEARCH,
        required_inputs=[],
        depends_on=[],
        expected_output="A markdown table: database, version, release date, source URL.",
        complexity=Complexity.LOW,
    ),
    "analysis": Subtask(
        id="t1",
        description="Compute how much RAM 10 million 768-dimension float32 vectors need, "
        "raw and with HNSW overhead (assume 32 links/node at 4 bytes each).",
        specialist=Specialist.ANALYSIS,
        required_inputs=[],
        depends_on=[],
        expected_output="The numbers in GB with the calculation shown.",
        complexity=Complexity.LOW,
    ),
}

kind = sys.argv[1] if len(sys.argv) > 1 else "research"
subtask = SUBTASKS[kind]
llm = AnthropicClient()
registry = build_default_registry()

run = SpecialistAgent(subtask.specialist, llm, registry).run(subtask)
print(f"== {subtask.specialist.value} specialist: {run.turns} turns, "
      f"{len(run.tool_calls)} tool calls, succeeded={run.result.succeeded}")
for rec in run.tool_calls:
    print(f"   - {rec.tool}: {'ok' if rec.success else rec.error} ({rec.latency_ms:.0f} ms)")
print(f"\n{run.result.output or run.result.error}\n")

review = Reviewer(llm).review(subtask, run.result, evidence=run.tool_calls)
v = review.verdict
print(f"== reviewer: passed={review.passed} quality={v.quality_score:.2f}\n{v.feedback}")

tokens = sum(u.input_tokens + u.output_tokens for u in [*run.usage, review.usage])
print(f"\ntotal tokens: {tokens}")
