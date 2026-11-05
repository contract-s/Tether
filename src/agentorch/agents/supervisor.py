"""Supervisor agent: decomposes a complex request into an execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentorch.agents.planning import InvalidPlanError, validate_plan
from agentorch.agents.schemas import ExecutionPlan, Specialist, Subtask
from agentorch.llm import LLMClient, LLMUsage

SPECIALIST_DESCRIPTIONS = {
    Specialist.RESEARCH: "finds information: web search, reading sources, collecting facts",
    Specialist.ANALYSIS: "works with data: extraction, comparison, calculations, tables",
    Specialist.WRITING: "produces prose deliverables: summaries, reports, emails",
    Specialist.CODE: "writes and runs code in a sandbox",
}

SYSTEM_PROMPT = f"""You are the supervisor of a team of specialist agents. Break the \
user's request into the smallest set of subtasks that fully accomplishes it, and assign \
each subtask to exactly one specialist.

Specialists:
{chr(10).join(f"- {s.value}: {d}" for s, d in SPECIALIST_DESCRIPTIONS.items())}

Rules:
- Use `depends_on` only when a subtask genuinely needs another's output. Independent \
subtasks run in parallel, so don't chain them unnecessarily.
- Dependencies must form a DAG: no cycles, and only reference ids that exist in the plan.
- Set `confidence` honestly. Below 0.7 means a human will review the plan before any \
work starts, so lower it when the request is ambiguous, risky, or needs information \
you don't have.
- If past tasks are provided, reuse approaches that worked and avoid ones that failed."""


@dataclass
class PlanningResult:
    plan: ExecutionPlan
    batches: list[list[Subtask]]
    usage: list[LLMUsage] = field(default_factory=list)
    attempts: int = 1


class Supervisor:
    def __init__(self, llm: LLMClient, max_attempts: int = 2) -> None:
        self.llm = llm
        self.max_attempts = max_attempts

    def plan(self, task: str, memories: list[str] | None = None) -> PlanningResult:
        prompt = self._build_prompt(task, memories or [])
        usage: list[LLMUsage] = []
        last_error: InvalidPlanError | None = None

        for attempt in range(1, self.max_attempts + 1):
            if last_error is not None:
                prompt += (
                    f"\n\nYour previous plan was invalid: {last_error}. "
                    "Produce a corrected plan."
                )
            result = self.llm.structured("supervisor", SYSTEM_PROMPT, prompt, ExecutionPlan)
            usage.append(result.usage)
            try:
                batches = validate_plan(result.output)
            except InvalidPlanError as e:
                last_error = e
                continue
            return PlanningResult(result.output, batches, usage, attempt)

        raise InvalidPlanError(
            f"no valid plan after {self.max_attempts} attempts: {last_error}"
        )

    @staticmethod
    def _build_prompt(task: str, memories: list[str]) -> str:
        parts = [f"<request>\n{task}\n</request>"]
        if memories:
            joined = "\n".join(f"- {m}" for m in memories)
            parts.append(f"<past_tasks>\n{joined}\n</past_tasks>")
        return "\n\n".join(parts)
