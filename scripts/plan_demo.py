"""Ask the real supervisor to plan a task. Needs ANTHROPIC_API_KEY in .env.

    uv run python scripts/plan_demo.py "Compare the top 3 vector databases for a RAG app"
"""

import sys

from agentorch.agents.supervisor import Supervisor
from agentorch.llm import AnthropicClient

task = " ".join(sys.argv[1:]) or (
    "Research the three most popular open-source vector databases, compare their "
    "performance and pricing, and write a one-page recommendation."
)
result = Supervisor(AnthropicClient()).plan(task)
plan = result.plan

print(f"Goal: {plan.goal}\nConfidence: {plan.confidence:.2f}\nReasoning: {plan.reasoning}\n")
for i, batch in enumerate(result.batches, 1):
    print(f"Batch {i} (parallel):")
    for st in batch:
        deps = f" <- {', '.join(st.depends_on)}" if st.depends_on else ""
        print(f"  [{st.id}] {st.specialist.value:<8} {st.description}{deps}")
tokens = sum(u.input_tokens + u.output_tokens for u in result.usage)
print(f"\nAttempts: {result.attempts}, tokens: {tokens}, model: {result.usage[-1].model}")
