"""Supervisor agent: decomposes a complex request into an execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field

from tether.agents.planning import InvalidPlanError, validate_plan
from tether.agents.schemas import ExecutionPlan, Specialist, Subtask
from tether.llm import LLMClient, LLMUsage
from tether.observability.tracing import set_attributes, span

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


SYNTHESIS_PROMPT = """You are the supervisor delivering the final result to the user. \
Your specialists have completed and reviewed their subtasks. Combine their outputs into \
one deliverable that answers the original request.

- The user only sees your answer, not the subtask outputs, so include everything they need.
- Keep sources and caveats the specialists reported. Do not add facts that aren't in \
their outputs.
- If any subtasks are listed as missing, say plainly what couldn't be completed."""


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
        """Plan the task (see `_plan`), traced as one `supervisor.plan` span."""
        with span("supervisor.plan", "agent", **{
            "tether.role": "supervisor", "tether.memories_given": len(memories or []),
        }) as s:
            result = self._plan(task, memories)
            set_attributes(s, {
                "tether.attempts": result.attempts,
                "tether.confidence": result.plan.confidence,
                "tether.subtasks": len(result.plan.subtasks),
                # A coarse task type for cost metrics: which specialists the plan used.
                "tether.task_type": "+".join(sorted({st.specialist.value
                                                        for st in result.plan.subtasks})),
                "tether.batches": [[st.id for st in b] for b in result.batches],
            })
            return result

    def _plan(self, task: str, memories: list[str] | None = None) -> PlanningResult:
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

    def synthesize(
        self, task: str, plan: ExecutionPlan, outputs: dict[str, str]
    ) -> tuple[str, LLMUsage]:
        """Combine the approved subtask outputs into the final deliverable."""
        with span("supervisor.synthesize", "agent", **{
            "tether.role": "supervisor", "tether.outputs": len(outputs),
        }):
            return self._synthesize(task, plan, outputs)

    def _synthesize(
        self, task: str, plan: ExecutionPlan, outputs: dict[str, str]
    ) -> tuple[str, LLMUsage]:
        parts = [f"<request>\n{task}\n</request>"]
        for st in plan.subtasks:
            if st.id in outputs:
                parts.append(
                    f"<subtask_output id=\"{st.id}\" specialist=\"{st.specialist.value}\">\n"
                    f"{st.description}\n---\n{outputs[st.id]}\n</subtask_output>"
                )
        missing = [st.id for st in plan.subtasks if st.id not in outputs]
        if missing:
            parts.append(f"<missing_subtasks>{', '.join(missing)}</missing_subtasks>")
        messages = [{"role": "user", "content": "\n\n".join(parts)}]
        turn = self.llm.chat("supervisor", SYNTHESIS_PROMPT, messages, [])
        return turn.text.strip(), turn.usage

    @staticmethod
    def _build_prompt(task: str, memories: list[str]) -> str:
        parts = [f"<request>\n{task}\n</request>"]
        if memories:
            joined = "\n".join(f"- {m}" for m in memories)
            parts.append(f"<past_tasks>\n{joined}\n</past_tasks>")
        return "\n\n".join(parts)
